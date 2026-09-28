# Multi-model RoboCasa PPO/GRPO platform

This directory defines the shared contract for training XR-1, pi0.5, FastWAM, and DreamZero with PPO or GRPO on RoboCasa365. RLinf owns the distributed rollout and optimization loop; each model owns observation processing, stochastic action sampling, replayable log-probabilities, and checkpoint conversion.

## Current support

| Model | PPO | GRPO | Current state |
| --- | --- | --- | --- |
| XR-1 | validated training path | configuration and contract tests | trainable |
| pi0.5 | configuration ready | configuration ready | base weights found; RoboCasa norm stats still required |
| FastWAM | not connected | not connected | model adapter required |
| DreamZero | inference/SFT only | inference/SFT only | replayable log-probability forward is not implemented |

The pi0.5 base checkpoint is referenced in `registry.json`; it is not copied into this repository. Its `config.json` describes a 32-dimensional, 10-step action model. RLinf already contains the `pi05_robocasa365_pretrain_human300` observation/action transforms, but the base checkpoint does not contain their RoboCasa normalization statistics. The preflight therefore rejects training until matching statistics are supplied with `--norm-stats` or `PI05_ROBOCASA_NORM_STATS`.

## Algorithm contract

Every model adapter must return the executed environment action together with `prev_logprobs` and replayable `forward_inputs`. During optimization, `default_forward()` must recompute the probability of that same fixed action. PPO additionally uses a value prediction. GRPO does not require a value head, but it requires at least two stochastic trajectories for the same task and initial state.

XR-1 GRPO uses grouped RoboCasa environments with the following invariants:

- `group_size >= 2`, and the environment count is divisible by the group size;
- every member of a group receives the same task and simulator seed;
- the rollout covers one fixed training episode, and task rotation occurs only after that rollout;
- success does not asynchronously reset one member of the group;
- homogeneous all-failure or all-success groups are filtered because their relative advantage is zero.

The short smoke config verifies data flow and backward propagation; it is not a learning result. The full config uses a 768-simulator-step training episode, matching the stable XR-1 simulator process lifetime established by the PPO runs.

## Commands

Run the dependency and contract preflight:

```bash
cd /path/to/robocasa_RL
.conda-robocasa365/bin/python scripts/multimodel_robocasa_preflight.py --model xr1 --algorithm grpo
```

Submit a one-update XR-1 GRPO integration smoke:

```bash
XR1_RLINF_CONFIG=xr1_robocasa365_grpo_smoke \
XR1_RLINF_MAX_STEPS=1 \
sbatch scripts/run_xr1_rlinf_ppo.sbatch
```

The pi0.5 preflight reports the missing benchmark statistics instead of launching an invalid experiment:

```bash
.conda-robocasa365/bin/python scripts/multimodel_robocasa_preflight.py \
  --model pi05 --algorithm ppo \
  --checkpoint /path/to/pytorch_pi05_base \
  --norm-stats /path/to/robocasa/norm_stats.json
```

Once matching statistics are available, submit PPO or GRPO through the same external base checkpoint:

```bash
PI05_ROBOCASA_NORM_STATS=/path/to/robocasa/norm_stats.json \
PI05_BASE_CKPT=/path/to/pytorch_pi05_base \
PI05_RLINF_CONFIG=pi05_robocasa365_ppo \
sbatch scripts/run_pi05_rlinf_robocasa.sbatch
```
