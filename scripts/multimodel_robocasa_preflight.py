#!/usr/bin/env python3
"""Validate model resources and algorithm invariants before RoboCasa training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "multimodel_robocasa" / "registry.json"


def validate_grpo_contract(config: dict[str, Any]) -> list[str]:
    """Return violations of the grouped RoboCasa trajectory contract."""
    errors: list[str] = []
    group_size = int(config.get("group_size", 1))
    total_num_envs = int(config.get("total_num_envs", 0))
    episode_steps = int(config.get("max_episode_steps", 0))
    rollout_steps = int(config.get("max_steps_per_rollout_epoch", 0))

    if group_size < 2:
        errors.append("GRPO group_size must be at least 2")
    if total_num_envs <= 0 or total_num_envs % group_size:
        errors.append("total_num_envs must be positive and divisible by group_size")
    if config.get("seed_strategy") not in {"same", "openpi"}:
        errors.append("group members must use the same simulator seed")
    if not bool(config.get("ignore_terminations")):
        errors.append("ignore_terminations must be true to prevent asynchronous success resets")
    if not bool(config.get("auto_reset")):
        errors.append("auto_reset must be true for the fixed-horizon truncation boundary")
    if episode_steps <= 0 or rollout_steps != episode_steps:
        errors.append("GRPO rollout length must equal the positive training episode horizon")
    if not bool(config.get("rotate_tasks_on_rollout")):
        errors.append("tasks must rotate after each complete grouped rollout")
    if bool(config.get("rotate_tasks_on_auto_reset")):
        errors.append("task rotation on individual auto-reset must be disabled")
    return errors


def resolve_checkpoint(checkpoint: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a registry checkpoint without copying it into the repository."""
    path = Path(checkpoint).expanduser()
    return path if path.is_absolute() else repo_root / path


def validate_model_entry(
    model: str,
    entry: dict[str, Any],
    algorithm: str,
    norm_stats: Path | None = None,
    expected_action_dim: int = 12,
    checkpoint_override: Path | None = None,
) -> list[str]:
    """Validate one registry entry and its local resources."""
    errors: list[str] = []
    if algorithm not in entry.get("algorithms", []):
        errors.append(f"{model} does not declare support for {algorithm}")

    checkpoint = (
        checkpoint_override.expanduser()
        if checkpoint_override is not None
        else resolve_checkpoint(str(entry["checkpoint"]))
    )
    if not checkpoint.is_dir():
        errors.append(f"checkpoint directory does not exist: {checkpoint}")
    elif model == "pi05":
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "model.safetensors"
        if not config_path.is_file():
            errors.append(f"pi0.5 config is missing: {config_path}")
        if not weights_path.is_file():
            errors.append(f"pi0.5 weights are missing: {weights_path}")

        if norm_stats is None:
            errors.append("pi0.5 RoboCasa norm_stats.json is required")
        elif not norm_stats.is_file():
            errors.append(f"pi0.5 norm stats do not exist: {norm_stats}")
        else:
            try:
                norm_data = json.loads(norm_stats.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"pi0.5 norm stats are not valid JSON: {exc}")
            else:
                if not isinstance(norm_data, dict) or not norm_data:
                    errors.append("pi0.5 norm stats must be a non-empty JSON object")
                else:
                    stats = norm_data.get("norm_stats", norm_data)
                    actions = stats.get("actions", {}) if isinstance(stats, dict) else {}
                    action_mean = actions.get("mean") if isinstance(actions, dict) else None
                    if not isinstance(action_mean, list):
                        errors.append("pi0.5 norm stats must contain actions.mean")
                    elif len(action_mean) != expected_action_dim:
                        errors.append(
                            "pi0.5 norm stats action dimension mismatch: "
                            f"expected {expected_action_dim}, got {len(action_mean)}"
                        )

    if entry.get("status") not in {"trainable", "requires_external_assets"}:
        requirements = "; ".join(entry.get("requirements", []))
        errors.append(f"{model} is not trainable yet: {requirements}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", required=True, choices=("ppo", "grpo"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the registry checkpoint with an external local path",
    )
    parser.add_argument("--expected-action-dim", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = json.loads(args.registry.read_text())
    entry = registry.get("models", {}).get(args.model)
    if entry is None:
        print(f"ERROR: unknown model {args.model!r}")
        return 2

    errors = validate_model_entry(
        args.model,
        entry,
        args.algorithm,
        args.norm_stats,
        args.expected_action_dim,
        args.checkpoint,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    checkpoint = (
        args.checkpoint.expanduser()
        if args.checkpoint is not None
        else resolve_checkpoint(str(entry["checkpoint"]))
    )
    print(
        f"READY model={args.model} algorithm={args.algorithm} "
        f"checkpoint={checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
