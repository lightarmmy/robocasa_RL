import torch
from types import SimpleNamespace

from mibot.rlinf.ppo_policy import XR1PPOPolicy
from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder


class DummyXR1(torch.nn.Module):
    state_shape = (1, 4)

    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Linear(4, 4)
        self.dit = torch.nn.Linear(4, 4)
        self.state_projector = torch.nn.Linear(4, 4)

    def generate(self, batch, noise=None):
        return batch["action"] * 0.0 + 1.0

    def generate_with_grad(self, batch, noise=None):
        return self.generate(batch, noise)


def test_modes_and_ppo_contract():
    policy = XR1PPOPolicy(DummyXR1(), action_dim=4, action_horizon=2)
    assert torch.count_nonzero(policy.value_head.weight) == 0
    assert torch.count_nonzero(policy.value_head.bias) == 0
    policy.set_trainable_mode("action_expert")
    assert not policy.xr1_model.vlm.weight.requires_grad
    assert policy.xr1_model.dit.weight.requires_grad
    batch = {"action": torch.zeros(2, 2, 4), "state": torch.zeros(2, 1, 4)}
    actions, result = policy.predict_action_batch({"xr1_batch": batch}, mode="eval")
    assert actions.shape == (2, 2, 4)
    assert result["prev_logprobs"].shape == (2, 2, 4)
    out = policy.default_forward(result["forward_inputs"], compute_values=True)
    assert out["logprobs"].shape == (2, 2, 4)
    assert out["values"].shape == (2, 1)


class ToyFlowVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Linear(1, 1)
        self.visual = torch.nn.Linear(1, 1)

    def forward(self, input_ids, **_):
        batch_size, sequence_length = input_ids.shape
        return SimpleNamespace(
            attention_mask=torch.ones(batch_size, sequence_length, dtype=torch.bool),
            position_ids=torch.zeros(3, batch_size, sequence_length, dtype=torch.long),
            past_key_values=None,
        )


class ToyFlowXR1(torch.nn.Module):
    state_shape = (1, 2)

    def __init__(self):
        super().__init__()
        self.vlm = ToyFlowVLM()
        self.dit = torch.nn.Linear(1, 1)
        self.state_projector = torch.nn.Linear(2, 2)
        self.velocity_scale = torch.nn.Parameter(torch.tensor(0.1))

    def rotary_emb(self, action_mask, position_ids):
        return action_mask, position_ids

    def dit_forward(self, noisy_action, **_):
        return noisy_action * self.velocity_scale


def test_probability_flow_logprob_reuses_rollout_probes_and_backpropagates():
    policy = XR1PPOPolicy(
        ToyFlowXR1(),
        action_dim=2,
        action_horizon=2,
        likelihood_mode="flow_hutchinson",
        clip_normalized_action=None,
    )
    batch = {
        "action": torch.zeros(1, 2, 2),
        "action_mask": torch.ones(1, 2, 2),
        "state": torch.zeros(1, 1, 2),
        "input_ids": torch.ones(1, 3, dtype=torch.long),
    }
    policy.eval()
    _, rollout = policy.predict_action_batch({"xr1_batch": batch}, mode="train")
    # All rollout forward-input tensors are batch-first so RLinf can route
    # them through its generic dim-0 concat/split path.
    assert rollout["forward_inputs"]["xr1_trace_probes"].shape == (1, 5, 1, 2, 2)
    replay = policy.default_forward(rollout["forward_inputs"], compute_values=True)
    assert torch.allclose(replay["logprobs"], rollout["prev_logprobs"], atol=1e-5)
    policy.train()
    updated = policy.default_forward(rollout["forward_inputs"], compute_values=True)
    (-updated["logprobs"].sum() + updated["values"].sum()).backward()
    assert policy.xr1_model.velocity_scale.grad is not None
    assert torch.isfinite(policy.xr1_model.velocity_scale.grad)


def test_xr1_trajectory_dynamically_pads_different_task_prompts():
    builder = EmbodiedTrajectoryBuilder()
    builder.forward_inputs = [{
        "input_ids": torch.arange(476).reshape(1, 476),
        "attention_mask": torch.ones(1, 476, dtype=torch.long),
        "xr1_text_pad_token_id": torch.tensor([151643]),
    }, {
        "input_ids": torch.arange(486).reshape(1, 486),
        "attention_mask": torch.ones(1, 486, dtype=torch.long),
        "xr1_text_pad_token_id": torch.tensor([151643]),
    }]
    stacked = builder.to_trajectory().forward_inputs
    assert stacked["input_ids"].shape == (2, 1, 486)
    assert stacked["attention_mask"].shape == (2, 1, 486)
    assert torch.all(stacked["input_ids"][0, :, 476:] == 151643)
    assert torch.count_nonzero(stacked["attention_mask"][0, :, 476:]) == 0
    assert torch.count_nonzero(stacked["attention_mask"][1]) == 486
