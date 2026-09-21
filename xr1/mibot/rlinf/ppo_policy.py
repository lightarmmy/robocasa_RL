"""RLinf PPO policy wrapper for the Xiaomi-Robotics-1 action model.

XR-1 samples actions by integrating a conditional ODE from Gaussian noise.
Besides the legacy diagonal-Gaussian surrogate, this wrapper supports a
probability-flow likelihood: it inverts the same Euler solver from an executed
action and estimates the ODE divergence with Hutchinson probes.  The latter is
an estimated likelihood (the released five-step Euler solver is itself an
approximation), but it describes the actual XR-1 sampler rather than adding a
second action-space Gaussian policy.
"""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
except ImportError:  # Allows importing the wrapper before RLinf is installed.
    class BasePolicy:  # type: ignore[no-redef]
        pass

    class ForwardType(Enum):  # type: ignore[no-redef]
        DEFAULT = "default"


class XR1PPOPolicy(nn.Module, BasePolicy):
    """Adapt an instantiated XR-1 model to RLinf's embodied PPO interface.

    ``forward_inputs`` must contain either ``xr1_batch`` (the native XR-1
    model batch) or the native batch fields directly.  Rollout callers should
    pass an ``obs_to_batch`` callable that turns an environment observation
    into that native batch.  This keeps RoboCasa365 token/image processing out
    of RLinf's actor worker.
    """

    _ACTION_MODULES = (
        "dit", "state_projector", "action_projector", "action_output_layer",
        "t_embedder", "t_projector", "sink",
    )
    # Let RLinf's generic FSDP policy discover the custom Transformer blocks
    # nested below the wrapper without hard-coding Xiaomi classes in RLinf.
    _no_split_modules = ["DecoderLayer"]
    _LANGUAGE_MODULES = (
        "vlm", "state_projector_choice", "action_projector_choice",
        "score_projector_choice",
    )
    _PACKED_MODALITY_FIELDS = (
        ("pixel_values", "pixel_values_lens"),
        ("image_grid_thw", "image_grid_thw_lens"),
        ("pixel_values_videos", "pixel_values_videos_lens"),
        ("video_grid_thw", "video_grid_thw_lens"),
        ("second_per_grid_ts", "second_per_grid_ts_lens"),
    )
    # Rollout-only diagnostics are persisted in ``forward_inputs`` so they
    # survive RLinf's time/batch flattening.  They must never be forwarded to
    # the HuggingFace processor/model as regular multimodal inputs.
    _DIAGNOSTIC_FIELDS = {
        "xr1_rollout_mean", "xr1_rollout_logprobs", "xr1_trace_probes",
        "xr1_text_pad_token_id",
    }

    def __init__(
        self,
        xr1_model: nn.Module,
        action_dim: int = 60,
        action_horizon: int = 30,
        add_value_head: bool = True,
        trainable_mode: str = "full",
        obs_to_batch: Optional[Any] = None,
        action_std: float = 0.2,
        learn_action_std: bool = False,
        action_to_env: Optional[Any] = None,
        env_action_dim: Optional[int] = None,
        clip_normalized_action: Optional[float] = None,
        likelihood_mode: str = "gaussian",
        flow_trace_samples: int = 1,
    ) -> None:
        super().__init__()
        self.xr1_model = xr1_model
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.obs_to_batch = obs_to_batch
        self.action_to_env = action_to_env
        self.env_action_dim = int(env_action_dim) if env_action_dim is not None else None
        self.clip_normalized_action = clip_normalized_action
        self.likelihood_mode = str(likelihood_mode).strip().lower()
        if self.likelihood_mode not in {"gaussian", "flow_hutchinson"}:
            raise ValueError("likelihood_mode must be 'gaussian' or 'flow_hutchinson'")
        self.flow_trace_samples = int(flow_trace_samples)
        if self.flow_trace_samples < 1:
            raise ValueError("flow_trace_samples must be at least one")
        if self.likelihood_mode == "flow_hutchinson" and clip_normalized_action is not None:
            raise ValueError(
                "clip_normalized_action changes the sampled action density; "
                "set it to null when likelihood_mode='flow_hutchinson'"
            )
        if float(action_std) <= 0.0:
            raise ValueError("action_std must be positive")
        self.learn_action_std = bool(learn_action_std)
        # Keep wrapper-owned parameters in the checkpoint dtype.  RLinf wraps
        # the complete policy in FSDP, which requires a uniform parameter dtype
        # even when the underlying HF checkpoint is loaded in bf16.
        model_dtype = next(xr1_model.parameters()).dtype
        self.log_std = nn.Parameter(
            torch.full(
                (self.action_horizon, self.action_dim),
                float(np.log(action_std)),
                dtype=model_dtype,
            )
        )
        # A learned variance is not part of XR-1's native policy and is very
        # easy to collapse when the Gaussian surrogate is used with sparse
        # rewards.  Keep it fixed unless explicitly requested.
        self.log_std.requires_grad_(self.learn_action_std)
        self.value_head = (
            nn.Linear(self._infer_hidden_size(), 1, dtype=model_dtype)
            if add_value_head
            else None
        )
        if self.value_head is not None:
            # With sparse binary task rewards, a random value head creates
            # non-zero GAE targets before the policy has observed any reward.
            # Zero initialization keeps both actor and critic signals at zero
            # until rollout data contains an actual learning signal.
            nn.init.zeros_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)
        self.set_trainable_mode(trainable_mode)

    @property
    def language_model(self):
        """Expose the nested VLM language stack to RLinf FSDP wrapping."""
        return self.xr1_model.vlm.language_model

    @property
    def visual(self):
        """Expose the nested VLM vision stack to RLinf FSDP wrapping."""
        return self.xr1_model.vlm.visual

    def _infer_hidden_size(self) -> int:
        state_projector = getattr(self.xr1_model, "state_projector", None)
        if state_projector is not None:
            linear_layers = [m for m in state_projector.modules() if isinstance(m, nn.Linear)]
            if linear_layers:
                return int(linear_layers[-1].out_features)
        state_shape = getattr(self.xr1_model, "state_shape", None)
        if state_shape:
            return int(state_shape[-1])
        return 60

    def set_trainable_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        if mode not in {"full", "action_expert"}:
            raise ValueError("trainable_mode must be 'full' or 'action_expert'")
        for param in self.xr1_model.parameters():
            param.requires_grad_(mode == "full")
        if mode == "action_expert":
            for name in self._LANGUAGE_MODULES:
                module = getattr(self.xr1_model, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(False)
            for name in self._ACTION_MODULES:
                module = getattr(self.xr1_model, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(True)
        if self.value_head is not None:
            for param in self.value_head.parameters():
                param.requires_grad_(True)
        self.trainable_mode = mode

    def parameter_report(self) -> dict[str, int | str]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"mode": self.trainable_mode, "total": total, "trainable": trainable}

    @staticmethod
    def _batch_from_inputs(forward_inputs: dict[str, Any]) -> dict[str, Any]:
        batch = forward_inputs.get("xr1_batch", forward_inputs)
        if not isinstance(batch, dict):
            raise TypeError("forward_inputs['xr1_batch'] must be a dict")
        return {
            key: value
            for key, value in batch.items()
            if key not in XR1PPOPolicy._DIAGNOSTIC_FIELDS
        }

    @staticmethod
    def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
        # XR-1.forward pops bookkeeping fields; always hand it a disposable map.
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    @classmethod
    def _restore_packed_batch(cls, batch: dict[str, Any]) -> dict[str, Any]:
        """Restore batch-first multimodal tensors before calling the VLM.

        RLinf stores variable-length video/image tensors as ``[B,N,...]`` plus
        a per-sample length field. The XR-1/Qwen processor expects the original
        flattened leading dimension, so this conversion is kept local to the
        model call and never mutates replay storage.
        """
        restored = dict(batch)
        for key, lens_key in cls._PACKED_MODALITY_FIELDS:
            tensor = restored.get(key)
            lens = restored.pop(lens_key, None)
            if not isinstance(tensor, torch.Tensor) or not isinstance(lens, torch.Tensor):
                continue
            if tensor.ndim < 2 or tensor.shape[0] != lens.numel():
                continue
            lengths = lens.reshape(-1).to(dtype=torch.long)
            if torch.any(lengths <= 0):
                raise ValueError(f"Invalid non-positive lengths in '{lens_key}'")
            if torch.all(lengths == lengths[0]):
                per_sample = int(lengths[0].item())
                if tensor.shape[1] < per_sample:
                    raise ValueError(
                        f"Packed '{key}' second dimension {tensor.shape[1]} "
                        f"is smaller than {per_sample}"
                    )
                restored[key] = tensor[:, :per_sample].reshape(
                    tensor.shape[0] * per_sample, *tensor.shape[2:]
                )
            else:
                if tensor.shape[1] < int(lengths.max().item()):
                    raise ValueError(f"Packed '{key}' does not contain all requested lengths")
                restored[key] = torch.cat(
                    [tensor[index, : int(length)] for index, length in enumerate(lengths.tolist())],
                    dim=0,
                )
        return restored

    def _call_model(self, batch: dict[str, Any], *, differentiable: bool, noise: Any = None) -> torch.Tensor:
        batch = self._restore_packed_batch(batch)
        # Use one and the same XR-1 generation entry point for rollout and
        # actor update.  The method itself toggles eval() and restores the
        # previous state; under rollout's outer no_grad context this is also a
        # no-grad call, removing generate()/generate_with_grad() drift.
        if hasattr(self.xr1_model, "generate_with_grad"):
            return self._actions(self.xr1_model.generate_with_grad(batch, noise=noise))
        # ``MiBoTForActionGeneration`` inherits a text ``generate`` method from
        # Transformers; only call a native action generator defined by the
        # concrete class itself.
        if not differentiable and "generate" in type(self.xr1_model).__dict__:
            return self._actions(self.xr1_model.generate(batch, noise=noise))
        # The released HF custom-code model puts its public forward under
        # @torch.no_grad(). Reuse its submodules directly for PPO gradients.
        if all(hasattr(self.xr1_model, name) for name in ("vlm", "dit", "dit_forward", "rotary_emb")):
            return self._actions(self._hf_generate(batch, noise=noise))
        if hasattr(self.xr1_model, "generate"):
            return self._actions(self.xr1_model.generate(batch, noise=noise))
        try:
            return self._actions(self.xr1_model(**batch))
        except TypeError:
            return self._actions(self.xr1_model(batch, return_loss=False))

    def _hf_generate(self, batch: dict[str, Any], noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Differentiable ODE path for the released MiBoT HF model."""
        velocity = self._hf_velocity(batch)
        action_mask = batch["action_mask"]
        x = torch.randn_like(action_mask) if noise is None else noise.to(
            device=action_mask.device, dtype=action_mask.dtype
        )
        num_steps = int(batch.get("num_steps", 5))
        dt = 1.0 / num_steps
        for step in range(num_steps):
            timestep = torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * step / num_steps
            x = x + velocity(x, timestep) * dt
        return x

    def _hf_velocity(self, batch: dict[str, Any]) -> Any:
        """Build the conditional XR-1 ODE velocity function for one batch."""
        model = self.xr1_model
        state = batch["state"]
        action_mask = batch["action_mask"]
        vlm_drop = {
            "state", "action_mask", "action", "xr1_noise", "_xr1_noise",
            "num_steps", "prefix_length",
        }
        vlm_inputs = {
            key: value for key, value in batch.items()
            if key not in vlm_drop and isinstance(value, torch.Tensor)
        }
        vlm_outputs = model.vlm(**vlm_inputs, use_cache=True)
        action_bs, action_length, _ = action_mask.shape
        _, state_length, _ = state.shape
        query_length = action_length + state_length + 1
        position_ids = (
            torch.arange(0, query_length, device=action_mask.device)
            .view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None] + 1
        )
        position_embeds = model.rotary_emb(action_mask, position_ids)
        dit_mask = torch.tril(torch.ones((action_bs, query_length, query_length), device=action_mask.device))
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, query_length, -1)
        attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        state_embed = model.state_projector(state)

        def dit_forward_fn(noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return model.dit_forward(
                noisy_action=noisy_action, t=timestep, action_mask=action_mask,
                state_embed=state_embed, position_embeds=position_embeds,
                past_key_values=vlm_outputs.past_key_values, attn_mask=attn_mask,
            )

        return dit_forward_fn

    @staticmethod
    def _standard_normal_logprob(value: torch.Tensor) -> torch.Tensor:
        return -0.5 * (value.float().square() + float(np.log(2.0 * np.pi)))

    def _flow_logprobs(
        self,
        batch: dict[str, Any],
        action: torch.Tensor,
        *,
        differentiable: bool,
        trace_probes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate log p(action|observation) for XR-1's probability flow.

        Starting from the fixed executed action, integrate the released Euler
        ODE backwards and accumulate ``div(v)``.  The action is deliberately
        held fixed: PPO needs the likelihood of the rollout action under the
        current policy, not the likelihood along a newly generated trajectory.
        ``trace_probes`` are cached with a rollout so old/new log-probability
        comparisons use the same Hutchinson estimator.
        """
        if not all(hasattr(self.xr1_model, name) for name in ("vlm", "dit", "dit_forward", "rotary_emb")):
            raise RuntimeError("flow_hutchinson likelihood requires the released HuggingFace XR-1 model")
        batch = self._restore_packed_batch(batch)
        active_mask = self._mask_for(action, batch)
        if active_mask is None:
            active_mask = torch.ones_like(action, dtype=torch.bool)
        active = active_mask.to(dtype=action.dtype)
        active_count = active.float().sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        num_steps = int(batch.get("num_steps", 5))
        expected_probe_shape = (num_steps, self.flow_trace_samples, *action.shape)
        if trace_probes is None:
            probes = torch.empty(expected_probe_shape, device=action.device, dtype=action.dtype)
            probes.bernoulli_(0.5).mul_(2.0).sub_(1.0)
            probes = probes * active.unsqueeze(0).unsqueeze(0)
        else:
            probes = trace_probes.to(device=action.device, dtype=action.dtype)
            # RLinf routes every ``forward_inputs`` tensor by dimension 0.
            # Store probe caches batch-first at the policy boundary, then
            # restore the estimator's [step, trace, batch, ...] layout here.
            # The legacy layout is accepted for already materialized local
            # trajectories, but must not be emitted by new rollouts.
            batch_first_shape = (
                action.shape[0],
                num_steps,
                self.flow_trace_samples,
                *action.shape[1:],
            )
            if tuple(probes.shape) == batch_first_shape:
                probes = probes.movedim(0, 2)
            if tuple(probes.shape) != expected_probe_shape:
                raise ValueError(
                    "xr1_trace_probes shape "
                    f"{tuple(trace_probes.shape)} is neither batch-first "
                    f"{batch_first_shape} nor legacy {expected_probe_shape}"
                )
        # A rollout calls this method from a no_grad policy method, but the
        # VJP with respect to x is still required to estimate divergence.
        # Hutchinson divergence requires a second derivative through the VLM
        # attention stack. CUDA's flash/memory-efficient SDP kernels do not
        # implement that backward-of-backward, so force the math kernel only
        # for this likelihood path. The normal rollout generation remains on
        # the fast kernel.
        sdp_context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
            if action.is_cuda
            else nullcontext()
        )
        with torch.enable_grad(), sdp_context:
            velocity = self._hf_velocity(batch)
            x = action.detach().to(dtype=batch["action_mask"].dtype).requires_grad_(True)
            log_det = torch.zeros(action.shape[0], device=action.device, dtype=torch.float32)
            dt = 1.0 / num_steps
            for step in reversed(range(num_steps)):
                timestep = torch.full(
                    (x.shape[0], 1, 1), step / num_steps,
                    device=x.device, dtype=x.dtype,
                )
                v = velocity(x, timestep)
                trace = torch.zeros_like(log_det)
                for probe in probes[step]:
                    vjp = torch.autograd.grad(
                        outputs=v,
                        inputs=x,
                        grad_outputs=probe,
                        retain_graph=True,
                        create_graph=differentiable,
                    )[0]
                    trace = trace + (vjp.float() * probe.float() * active.float()).sum(dim=(-2, -1))
                log_det = log_det + dt * trace / self.flow_trace_samples
                x = x - v * dt
        logprobs = self._standard_normal_logprob(x) * active.float()
        # RLinf expects an action-shaped log-prob tensor.  The Jacobian term is
        # scalar per action chunk, so distribute it over active coordinates;
        # summing coordinates exactly recovers the change-of-variables density.
        logprobs = logprobs - log_det[:, None, None] * active.float() / active_count
        return logprobs, probes.detach()

    def _distribution(self, mean: torch.Tensor) -> Normal:
        if mean.ndim == 2:
            mean = mean.unsqueeze(0)
        if mean.shape[-2:] != self.log_std.shape:
            raise ValueError(f"XR-1 action shape {tuple(mean.shape[-2:])} != configured {tuple(self.log_std.shape)}")
        log_std = self.log_std.to(device=mean.device, dtype=torch.float32)
        return Normal(mean.float(), log_std.exp())

    @staticmethod
    def _mask_for(mean: torch.Tensor, batch: dict[str, Any]) -> Optional[torch.Tensor]:
        mask = batch.get("action_mask")
        if not isinstance(mask, torch.Tensor):
            return None
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and mean.shape[0] > 1:
            mask = mask.expand(mean.shape[0], *mask.shape[1:])
        if mask.shape != mean.shape:
            try:
                mask = mask.expand_as(mean)
            except RuntimeError as error:
                raise ValueError(f"action_mask shape {tuple(mask.shape)} cannot broadcast to {tuple(mean.shape)}") from error
        return mask.to(device=mean.device, dtype=mean.dtype).bool()

    @staticmethod
    def _actions(output: Any) -> torch.Tensor:
        """Normalize native XR-1 tensors and HF ``ActionGenerationOutput``."""
        if isinstance(output, torch.Tensor):
            return output
        actions = getattr(output, "actions", None)
        if isinstance(actions, torch.Tensor):
            return actions
        raise TypeError("XR-1 model output must be a Tensor or expose .actions")

    @staticmethod
    def _fit_feature(feature: torch.Tensor, width: int) -> torch.Tensor:
        feature = feature.float().reshape(feature.shape[0], -1)
        if feature.shape[-1] < width:
            feature = torch.nn.functional.pad(feature, (0, width - feature.shape[-1]))
        return feature[:, :width]

    def _value(self, forward_inputs: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        if self.value_head is None:
            return torch.zeros(batch_size, device=device)
        feature = forward_inputs.get("value_features")
        used_state_projector = feature is None
        if feature is None:
            state = forward_inputs.get("state")
            if state is None and isinstance(forward_inputs.get("xr1_batch"), dict):
                state = forward_inputs["xr1_batch"].get("state")
            if state is None:
                return torch.zeros(batch_size, device=device)
            feature = state
        if feature.shape[0] == 1 and batch_size > 1:
            feature = feature.expand(batch_size, *feature.shape[1:])
        hidden = self.value_head.in_features
        state_projector = getattr(self.xr1_model, "state_projector", None)
        if used_state_projector and state_projector is not None:
            projector_params = list(state_projector.parameters())
            # FSDP may expose a parameterless local view after flattening;
            # the value head dtype is the correct model dtype fallback.
            projector_dtype = (
                projector_params[0].dtype
                if projector_params
                else self.value_head.weight.dtype
            )
            projected = state.to(dtype=projector_dtype)
            feature = state_projector(projected)
            if feature.ndim > 2:
                feature = feature.mean(dim=1)
        else:
            feature = self._fit_feature(feature.to(device), hidden)
        value_dtype = self.value_head.weight.dtype
        return self.value_head(feature.to(dtype=value_dtype))

    def default_forward(self, forward_inputs: dict[str, Any], compute_logprobs: bool = True,
                        compute_entropy: bool = False, compute_values: bool = True, **_: Any) -> dict[str, Any]:
        batch = self._clone_batch(self._batch_from_inputs(forward_inputs))
        action = forward_inputs.get("action")
        noise = forward_inputs.get("xr1_noise")
        if noise is not None:
            batch["_xr1_noise"] = noise
        if self.likelihood_mode == "flow_hutchinson":
            if action is None:
                raise ValueError("flow_hutchinson requires the fixed rollout action")
            action = action.to(next(self.parameters()).device)
            if action.ndim == 2:
                action = action.unsqueeze(0)
            logprobs = entropy = None
            probes = forward_inputs.get("xr1_trace_probes")
            if compute_logprobs:
                logprobs, _ = self._flow_logprobs(
                    batch,
                    action,
                    differentiable=self.training,
                    trace_probes=probes,
                )
            values = self._value(forward_inputs, action.shape[0], action.device) if compute_values else None
            output = {"logprobs": logprobs, "entropy": entropy, "values": values}
            rollout_logprobs = forward_inputs.get("xr1_rollout_logprobs")
            if isinstance(rollout_logprobs, torch.Tensor) and logprobs is not None:
                rlp = rollout_logprobs.to(device=logprobs.device, dtype=logprobs.dtype)
                if rlp.shape == logprobs.shape:
                    output["xr1_logprob_abs_diff"] = (logprobs.float() - rlp.float()).abs().mean()
                    output["xr1_logprob_max_diff"] = (logprobs.float() - rlp.float()).abs().max()
            return output
        with torch.set_grad_enabled(self.training):
            mean = self._call_model(batch, differentiable=True, noise=noise)
        if mean.ndim == 2:
            mean = mean.unsqueeze(0)
        dist = self._distribution(mean)
        action = mean if action is None else action.reshape_as(mean).to(mean.device)
        active_mask = self._mask_for(mean, batch)
        logprobs = dist.log_prob(action) if compute_logprobs else None
        entropy = dist.entropy() if compute_entropy else None
        if active_mask is not None:
            if logprobs is not None:
                logprobs = logprobs.masked_fill(~active_mask, 0.0)
            if entropy is not None:
                entropy = entropy.masked_fill(~active_mask, 0.0)
        values = self._value(forward_inputs, mean.shape[0], mean.device) if compute_values else None
        output = {"logprobs": logprobs, "entropy": entropy, "values": values}
        # During rollout, compare the second (gradient-capable) model call with
        # the no-grad action generation call.  A non-zero gap here proves the
        # issue is in the policy wrapper, before any optimizer update occurs.
        rollout_mean = forward_inputs.get("xr1_rollout_mean")
        if isinstance(rollout_mean, torch.Tensor):
            rm = rollout_mean.to(device=mean.device, dtype=mean.dtype)
            if rm.ndim == 2:
                rm = rm.unsqueeze(0)
            if rm.shape == mean.shape:
                output["xr1_mean_abs_diff"] = (mean.float() - rm.float()).abs().mean()
                output["xr1_mean_max_diff"] = (mean.float() - rm.float()).abs().max()
        rollout_logprobs = forward_inputs.get("xr1_rollout_logprobs")
        if isinstance(rollout_logprobs, torch.Tensor) and logprobs is not None:
            rlp = rollout_logprobs.to(device=logprobs.device, dtype=logprobs.dtype)
            if rlp.shape == logprobs.shape:
                output["xr1_logprob_abs_diff"] = (logprobs.float() - rlp.float()).abs().mean()
                output["xr1_logprob_max_diff"] = (logprobs.float() - rlp.float()).abs().max()
        return output

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs: Any) -> dict[str, Any]:
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(f"XR1PPOPolicy only supports {ForwardType.DEFAULT}")
        return self.default_forward(**kwargs)

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict[str, Any], calculate_logprobs: bool = True,
                             calculate_values: bool = True, mode: str = "train", **_: Any):
        if self.obs_to_batch is None:
            if not isinstance(env_obs, dict) or "xr1_batch" not in env_obs:
                raise ValueError("Provide obs_to_batch or env_obs['xr1_batch'] for XR-1 rollout")
            batch = env_obs["xr1_batch"]
        else:
            batch = self.obs_to_batch(env_obs)
        batch = self._clone_batch(dict(batch))
        # Environment adapters emit CPU tensors; rollout inference may run on
        # CUDA.  Normalize device placement at the policy boundary so the
        # processor output and the HF model cannot diverge across devices.
        model_device = next(self.parameters()).device
        batch = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        template = batch.get("action")
        if template is None:
            action_mask = batch.get("action_mask")
            if action_mask is None:
                raise ValueError("XR-1 rollout batch needs 'action' or 'action_mask' to define the action shape")
            template = action_mask.float()
            batch["action"] = template
        noise = torch.randn_like(template)
        action = self._call_model(self._clone_batch(batch), differentiable=False, noise=noise)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        dist = self._distribution(action)
        if self.likelihood_mode == "flow_hutchinson":
            # The initial ODE noise is the policy's exploration source.  Do
            # not add a second Gaussian sample or clip the result: both would
            # invalidate the probability-flow density.
            sampled = action.float()
        else:
            sampled = dist.sample() if str(mode).lower() in {"train", "training"} else action.float()
        if self.clip_normalized_action is not None:
            sampled = sampled.clamp(
                -float(self.clip_normalized_action),
                float(self.clip_normalized_action),
            )
        # Keep forward_inputs flat: RLinf's trajectory splitter concatenates and
        # splits tensor fields, but intentionally does not handle nested dicts.
        forward_inputs = {
            key: value for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        forward_inputs.update({"action": sampled, "xr1_noise": noise})
        tokenizer = getattr(getattr(self.obs_to_batch, "processor", None), "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.obs_to_batch is not None:
            if pad_token_id is None:
                raise ValueError("XR-1 tokenizer must define pad_token_id")
            # The trajectory builder uses this rollout-only field to dynamically
            # right-pad variable task prompts to the longest prompt in this
            # rollout.  It is filtered before every model invocation.
            forward_inputs["xr1_text_pad_token_id"] = torch.full(
                (sampled.shape[0],),
                int(pad_token_id),
                dtype=torch.long,
                device=sampled.device,
            )
        # Keep the exact no-grad generation result for a direct consistency
        # check in the actor update.  These fields are filtered before model
        # invocation and are split/flattened by RLinf like other tensors.
        forward_inputs["xr1_rollout_mean"] = action.detach().cpu()
        if self.likelihood_mode == "flow_hutchinson":
            # Reuse trace probes during the actor update so a stochastic
            # divergence estimate does not itself appear as a PPO ratio.
            _, probes = self._flow_logprobs(
                self._clone_batch(batch),
                sampled,
                differentiable=False,
                trace_probes=None,
            )
            # Recompute once with the cached probes; this is the value that
            # becomes prev_logprobs and is compared during the update.
            # Keep batch as dim 0.  RLinf concatenates/splits every
            # forward-input tensor on that dimension when routing rollouts.
            forward_inputs["xr1_trace_probes"] = probes.movedim(2, 0).cpu()
        result = self.default_forward(
            forward_inputs,
            compute_logprobs=calculate_logprobs,
            compute_entropy=False,
            compute_values=calculate_values,
        )
        if result["logprobs"] is not None:
            forward_inputs["xr1_rollout_logprobs"] = result["logprobs"].detach().cpu()
        forward_inputs["action"] = sampled.detach().cpu()
        forward_inputs["xr1_noise"] = noise.detach().cpu()
        env_action = sampled
        if self.action_to_env is not None:
            env_action = self.action_to_env(sampled)
        if self.env_action_dim is not None:
            env_action = env_action[..., : self.env_action_dim]
        return env_action.detach().cpu().numpy(), {
            "prev_logprobs": result["logprobs"].cpu() if result["logprobs"] is not None else None,
            "prev_values": result["values"].cpu() if result["values"] is not None else None,
            "forward_inputs": forward_inputs,
        }
