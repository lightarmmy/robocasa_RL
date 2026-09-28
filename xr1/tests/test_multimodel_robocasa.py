import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "scripts" / "multimodel_robocasa_preflight.py"
SPEC = importlib.util.spec_from_file_location("multimodel_preflight", PREFLIGHT_PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def _valid_grpo_config():
    return {
        "group_size": 4,
        "total_num_envs": 8,
        "seed_strategy": "same",
        "ignore_terminations": True,
        "auto_reset": True,
        "max_episode_steps": 768,
        "max_steps_per_rollout_epoch": 768,
        "rotate_tasks_on_rollout": True,
        "rotate_tasks_on_auto_reset": False,
    }


def test_grpo_contract_accepts_grouped_complete_trajectories():
    assert PREFLIGHT.validate_grpo_contract(_valid_grpo_config()) == []


def test_grpo_contract_rejects_unpaired_partial_rollouts():
    config = _valid_grpo_config()
    config.update(
        group_size=1,
        seed_strategy="global_unique",
        ignore_terminations=False,
        max_steps_per_rollout_epoch=64,
        rotate_tasks_on_rollout=False,
        rotate_tasks_on_auto_reset=True,
    )
    errors = PREFLIGHT.validate_grpo_contract(config)
    assert len(errors) == 6


def _pi05_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "pytorch_pi05_base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").touch()
    return checkpoint


def test_pi05_base_checkpoint_requires_robocasa_norm_stats(tmp_path):
    registry = json.loads(
        (REPO_ROOT / "multimodel_robocasa" / "registry.json").read_text()
    )
    errors = PREFLIGHT.validate_model_entry(
        "pi05",
        registry["models"]["pi05"],
        "ppo",
        checkpoint_override=_pi05_checkpoint(tmp_path),
    )
    assert errors == ["pi0.5 RoboCasa norm_stats.json is required"]


def test_pi05_rejects_norm_stats_from_another_robot(tmp_path):
    registry = json.loads(
        (REPO_ROOT / "multimodel_robocasa" / "registry.json").read_text()
    )
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text(json.dumps({"norm_stats": {"actions": {"mean": [0.0] * 7}}}))
    errors = PREFLIGHT.validate_model_entry(
        "pi05",
        registry["models"]["pi05"],
        "ppo",
        norm_stats,
        checkpoint_override=_pi05_checkpoint(tmp_path),
    )
    assert errors == [
        "pi0.5 norm stats action dimension mismatch: expected 12, got 7"
    ]
