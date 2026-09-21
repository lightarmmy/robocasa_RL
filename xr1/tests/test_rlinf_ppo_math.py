# Copyright (C) 2026 Xiaomi Corporation.
"""CPU checks for the RLinf embodied PPO math path used by XR-1."""

import torch
import numpy as np

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.envs.sim.robocasa.venv import RobocasaSubprocEnvWorker
from rlinf.envs.sim.robocasa365.robocasa365_env import (
    Robocasa365Env,
    classify_robocasa365_step_boundaries,
)


def test_embodied_gae_and_actor_critic_loss_backpropagate():
    # Rollout tensors use RLinf's [time, batch, action_chunks] layout.
    steps, batch_size, chunks, action_dim = 3, 1, 1, 4
    rewards = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])
    dones = torch.zeros(steps + 1, batch_size, chunks, dtype=torch.bool)
    dones[-1] = True
    values = torch.zeros(steps + 1, batch_size, chunks)
    rollout_mask = torch.ones(steps, batch_size, chunks, dtype=torch.bool)

    advantages = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="gae",
        rewards=rewards,
        dones=dones,
        values=values,
        num_action_chunks=chunks,
        gamma=0.99,
        gae_lambda=0.95,
        group_size=1,
        reward_type="chunk_level",
        loss_mask=rollout_mask,
        normalize_advantages=True,
    )
    assert advantages["advantages"].shape == (steps, batch_size, chunks)
    assert advantages["returns"].shape == (steps, batch_size, chunks)
    assert torch.isfinite(advantages["advantages"]).all()

    logprobs = torch.randn(steps, action_dim, requires_grad=True)
    # Keep values inside the PPO value-clip interval so this regression test
    # always exercises the critic gradient rather than randomly clipping it.
    current_values = torch.full((steps,), 0.1, requires_grad=True)
    old_logprobs = logprobs.detach() + 0.01
    loss, metrics = policy_loss(
        loss_type="actor_critic",
        task_type="embodied",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=action_dim,
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages["advantages"],
        values=current_values,
        returns=advantages["returns"],
        prev_values=torch.zeros_like(current_values),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=torch.ones(steps, 1, dtype=torch.bool),
        max_episode_steps=steps,
        critic_warmup=False,
    )
    assert metrics
    assert torch.isfinite(loss)
    loss.backward()
    assert logprobs.grad is not None and torch.isfinite(logprobs.grad).all()
    assert current_values.grad is not None and torch.isfinite(current_values.grad).all()
    assert float(logprobs.grad.abs().sum()) > 0.0
    assert float(current_values.grad.abs().sum()) > 0.0


def test_worker_restart_is_truncation_not_success():
    terminations, truncations = classify_robocasa365_step_boundaries(
        successes=np.array([False, True, False]),
        raw_dones=np.array([True, True, False]),
        worker_restarted=np.array([True, False, False]),
        elapsed_steps=np.array([10, 20, 30]),
        task_horizons=np.array([100, 100, 30]),
    )
    assert terminations.tolist() == [False, True, False]
    assert truncations.tolist() == [True, False, True]


def test_native_worker_exitcode_keeps_signal_name():
    assert RobocasaSubprocEnvWorker._format_exitcode(None) == "None"
    assert RobocasaSubprocEnvWorker._format_exitcode(0) == "0"
    assert RobocasaSubprocEnvWorker._format_exitcode(-6) == "-6 (SIGABRT)"


def test_gae_does_not_cross_restart_boundary():
    rewards = torch.zeros(2, 1, 1)
    dones = torch.zeros(3, 1, 1, dtype=torch.bool)
    dones[1] = True
    # A deliberately huge value after the boundary must not affect step 0.
    values = torch.tensor([[[1.0]], [[1000.0]], [[1000.0]]])
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="gae",
        rewards=rewards,
        dones=dones,
        values=values,
        num_action_chunks=1,
        gamma=0.99,
        gae_lambda=0.95,
        group_size=1,
        reward_type="chunk_level",
        loss_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        normalize_advantages=False,
    )
    assert torch.allclose(result["advantages"][0], torch.tensor([[-1.0]]))


def test_ordered_task_sampling_covers_every_task_once():
    env = Robocasa365Env.__new__(Robocasa365Env)
    env.task_sampling_strategy = "ordered"
    env.cfg = {"is_eval": False, "use_ordered_reset_state_ids": False}
    env._ordered_task_cursor = 0
    env.num_tasks = 50
    env.total_num_processes = 1
    sampled = [int(env._sample_task_ids(1)[0]) for _ in range(50)]
    assert sampled == list(range(50))
    assert int(env._sample_task_ids(1)[0]) == 0
