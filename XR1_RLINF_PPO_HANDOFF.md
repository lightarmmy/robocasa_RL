# XR-1 + RLinf PPO：交接与实验状态（2026-09-18）

这份文档是给新对话/新维护者的快速入口。它记录当前代码实际实现的
XR-1 RoboCasa365 PPO 框架、已验证/未验证的结论、结果位置与后续实验准则。
不要把“作业正常结束”误解成“性能已提升”：当前长跑尚在进行，最终只能以
完整的独立评测和训练曲线判断。

## 0. 项目背景与本分支范围

上游 `README.md` 的 Xiaomi-Robotics-1（XR-1）是一个 Vision-Language-Action
foundation model：Qwen3-VL VLM 与较小的 DiT/MoT action generator 结合，原始
训练分为大规模 pre-training 和 embodiment/instruction post-training。上游发布了
RoboCasa365 对齐 checkpoint `checkpoints/Xiaomi-Robotics-1-RoboCasa365`，它是本
PPO 实验的起点；不是从随机初始化训练。

本分支增加的是**在线、环境回报驱动的 RLinf PPO post-training adapter**，目标是
回答“在固定 XR-1 checkpoint、RoboCasa365 稀疏成功回报和给定算力下，单纯 PPO
还能带来多少提升”。它不改写上游监督式 post-training 配方，也不应把 README 中的
官方 benchmark 数字当成当前 PPO checkpoint 的分数。

## 1. 当前结论和正在运行的实验

### 已验证

- XR-1 能通过 RLinf 的 embodied PPO 接口训练，当前策略是**全量训练**
  (`actor.model.trainable_mode: full`)，且 value head 一同训练。
- `flow_hutchinson` 的 rollout、缓存 likelihood、PPO actor backward 和 FSDP
  均已在 GPU 上跑通。关键是关闭 FSDP 的自动嵌套包装
  (`XR1_RLINF_DISABLE_AUTO_WRAP=1`)，保留最外层 FSDP。
- 1-update 和 10-update 的 single-task integration run 都以 `COMPLETED (0:0)`
  结束，且没有 `Non-finite grad norm`/`Skipping optimizer step`。
- RoboCasa365 评测的 EGL/MuJoCo `read_pixels` 原生 abort 可以用“每个 episode
  后重启 Python worker”绕过。OpenDrawer 完整 5/5 评测成功（见第 9 节）。

### 不能据此宣称的结论

- 10 updates、且 reward 为零的 smoke **不证明** PPO 提升了成功率。
- `flow_hutchinson` 是发布版五步 Euler ODE 的离散反积分 + Hutchinson trace
  估计，不是连续 ODE 的精确解析 likelihood。
- 单任务 smoke 的 KL/clip/EV 仅是链路健康指标，不是全任务泛化能力证据。

### 进行中的主实验

| 项目 | 值 |
| --- | --- |
| Slurm job | `106535` (`xr1-flow-target50-1k-r256`) |
| 目标 | 纯 PPO 在 XR-1 + RoboCasa365 `target50` 的能力边界 |
| 状态（最近更新时） | `RUNNING`，节点 `gnho020` |
| 更新数 | 1000 (`runner.max_steps`，即 1000 次 PPO update；不是 1000 个 action/episode) |
| 任务 | 所有 `target50`，随机采样，无 task filter |
| 策略训练范围 | full model + value head |
| actor LR / critic LR | `1e-9` / `1e-4` |
| rollout / episode 配置 | 1 environment，64 / 64 |
| checkpoint | 250、500、750、1000 update |
| 输出目录 | `results/xr1_ppo_flow_target50_full_1000u_lr1e9_rootfsdp/` |

状态查询：

```bash
squeue -j 106535 -o "%.18i %.12T %.30j %.10M %.20R %.20b"
tail -n 120 logs/xr1_rlinf_ppo_106535.log
```

## 2. 仓库与关键入口

仓库根目录是本文件所在目录（下文用 `$REPO` 表示）。核心文件：

| 作用 | 文件 |
| --- | --- |
| XR-1 到 RLinf 的 PPO policy adapter | `xr1/mibot/rlinf/ppo_policy.py` |
| adapter 使用与约束 | `xr1/mibot/rlinf/README.md` |
| 受版本控制的 RLinf 配置镜像 | `xr1/mibot/rlinf/configs/` |
| 标准 RoboCasa365 PPO 配置 | `configs/xr1_robocasa365_ppo_full.yaml` |
| flow likelihood 模型覆盖 | `configs/xr1_ppo_flow.yaml` |
| flow smoke 组合配置 | `configs/xr1_robocasa365_ppo_flow_smoke.yaml` |
| Slurm 训练提交脚本 | `scripts/run_xr1_rlinf_ppo.sbatch` |
| policy / PPO 数学单测 | `xr1/tests/test_rlinf_ppo_policy.py`、`xr1/tests/test_rlinf_ppo_math.py` |
| 任务调度式评测 worker | `eval_robocasa365/dynamic_eval.py` |
| 评测 worker 启动/重启逻辑 | `scripts/launch_robocasa365.sh` |
| 隔离评测的 Slurm 提交 | `scripts/run_robocasa365_policy_eval_isolated.sbatch` |

依赖的 Python 环境是 `$REPO/.conda-robocasa365/bin/python`。RLinf 是
`$REPO/third_party/RLinf`；运行本地测试时必须显式设置 `PYTHONPATH`。RLinf 是外部
checkout（不会随本仓库上传）；训练 sbatch 在启动时把 `configs/` 内的四个配置镜像同步
到其 Hydra config 路径。

本次主仓库改动依赖独立 RLinf commit `6731138`（基于上游 `09a7404`）。发布时必须同时将
该 commit 推送到可访问的 RLinf fork；只上传 Xiaomi-Robotics-1 主仓库无法复现训练路径。

## 3. 数据流与 PPO 结构

```text
RoboCasa365 observation
  -> obs_to_batch / XR-1 multimodal processor
  -> released XR-1 VLM + DiT conditional five-step ODE
  -> normalized 60-D action chunks -> RoboCasa environment
  -> rollout batch (action, native XR-1 batch, old logprob, value, reward)
  -> RLinf PPO (GAE, ratio clipping, actor + value losses)
  -> XR1PPOPolicy.default_forward() recomputes current logprob/value
```

`XR1PPOPolicy` 的职责：

1. 将 RLinf 的 flat `forward_inputs` 还原为 XR-1 native multimodal batch；
2. 以 XR-1 生成的 5-step ODE action 进行 rollout；
3. 为 PPO 提供每个 action coordinate 的 logprob 和每个 chunk 的 value；
4. 保存/恢复必要的 rollout diagnostics，使 RLinf 在时间和 batch 维度切分后仍可
   计算 old/new policy ratio；
5. 处理 action horizon、mask、60D normalized action 到环境 action 的适配。

当前 `full` 模式会将 `xr1_model` 全部参数设为 trainable；`action_expert`
模式才会冻结 language/choice 模块并仅保留动作相关模块和 value head。两种模式
由 `set_trainable_mode()` 控制。主实验是 **full**，不是 action-expert-only。

## 4. 为什么需要 flow likelihood

### 旧的 Gaussian surrogate

旧实现把 XR-1 ODE 输出当作均值，并另加一个 diagonal Gaussian action policy。
这方便计算 `log_prob`，但并不是 XR-1 实际 sampling distribution：XR-1 本身从
Gaussian noise 经过 conditional ODE 生成 action，再在 action space 额外采样
Gaussian 会改变所执行的策略。旧 100-update run 的典型症状是
`approx_kl≈0.264`、clip fraction `≈0.261`、最大 logprob 差 `≈12.678`，critic EV
接近零或为负，不能作为可信提升基线。

### `flow_hutchinson`

当前模式把 sampler 视为 probability-flow ODE：

1. rollout 从标准 Gaussian 初始噪声，经发布的 5-step Euler ODE 得到 action；
2. actor update 固定已执行 action，反向 Euler 积分回 base noise；
3. log density = 标准 Gaussian base logprob + change-of-variables Jacobian；
4. Jacobian divergence 用 Rademacher Hutchinson probes 估计；
5. probes 缓存在 rollout 内，actor 重算时复用，避免 estimator noise 被误当成 PPO
   policy shift。

重要约束：

- flow 模式必须 `clip_normalized_action: null`，否则 clipping 改变了密度；
- `algorithm.entropy_bonus: 0.0`，因为当前没有可用的 flow entropy estimator；
- `flow_trace_samples: 1` 是当前吞吐/方差折中；增加 probes 前先确认长跑健康；
- 流 likelihood 需要 Hessian-vector product（divergence 对模型参数反传），代价比
  surrogate 更高。

## 5. 已修复的训练问题与原因

### A. trace probe 的 RLinf batch 路由错误

RLinf 对每个 `forward_inputs` tensor 按第 0 维切分。初版 probes 是
`[ode_step, trace, batch, ...]`，会被误认为 batch 为 ode step，导致
`split_with_sizes ... got [1]`。现在落盘时转成 batch-first：

```python
forward_inputs["xr1_trace_probes"] = probes.movedim(2, 0).cpu()
# [batch, ode_step, trace, ...]
```

读取时在 `_flow_logprobs()` 恢复内部 `[step, trace, batch, ...]` 布局。

### B. CUDA efficient/flash attention 没有二阶导

Hutchinson divergence 的训练反传触发 attention backward-of-backward；CUDA
memory-efficient/flash SDP 会报：

```text
RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
```

`_flow_logprobs()` 仅在 likelihood 路径强制 math SDP：

```python
torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_mem_efficient=False, enable_math=True
)
```

普通 rollout generation 仍可使用快速 attention kernel。

### C. 嵌套 FSDP 的二阶 backward 状态错误

math SDP 后，自动包装的 Transformer child FSDP 仍会在二阶导时报：

```text
ValueError: expected to be in states [FORWARD_BACKWARD] but current state is IDLE
```

训练提交时设置 `XR1_RLINF_DISABLE_AUTO_WRAP=1`。这让 RLinf 的 outer FSDP 仍存在，
但不再把每个 Transformer 层套成 child FSDP；1/10 update GPU 运行已验证这条路径。

### D. 小于 episode 的 rollout 曾产生 NaN

一次 `rollout_steps=16` 试验没有收集完整轨迹，出现 `num_trajectories=0`、NaN
advantages，并跳过 optimizer。主实验使用 64。注意运行中 `num_trajectories=0` 仍
可能只是“本 update 没刚好结束 episode”的统计；真正的健康判据是 advantages、loss
和 grad norm 必须是有限数，且没有 `Skipping optimizer step`。

### E. 长跑中的 renderer worker 重启上限

第一次全任务长跑 `106403` 在 step 114 退出，但 PPO 数值并未失败。RoboCasa renderer
子进程大约每 13 updates 因 pipe EOF 退出，现有 venv 能重启并继续；默认
`RLINF_ROBOCASA_MAX_RESTARTS=8` 在第 9 次重启时抛出 `worker exceeded restart limit`。
重跑 `106535` 明确设为 `RLINF_ROBOCASA_MAX_RESTARTS=256`，以覆盖 1k-update 期间
预期的约 77 次可恢复 renderer restart。仍须监控是否出现非 renderer/PPO 错误；这不是
把数值训练异常忽略掉。

## 6. 配置语义：特别是“100 步”

`runner.max_steps=100` 表示 **100 次 PPO update iteration**，不是 100 个动作，
也不是 100 个 episode。每个 update 再收集
`env.train.max_steps_per_rollout_epoch` 个环境控制步骤（本次为 64），每个控制决策
由 XR-1 生成 action chunk。因此实际执行的低层 action 数还会乘以 chunk horizon。

主配置的关键默认值：

| 参数 | 当前主实验值/来源 |
| --- | --- |
| `task_soup` | `target50` |
| `total_num_envs` | 1 |
| `max_episode_steps` | 64 |
| `max_steps_per_rollout_epoch` | 64 |
| PPO epochs | flow smoke 覆盖为 1 |
| clip ratio | 0.2 |
| GAE | gamma 0.99，lambda 0.95 |
| actor precision | bf16 |
| FSDP | no_shard + orig params；训练时禁用 auto wrap |
| model ODE steps | 5 |

## 7. 已有 smoke 结果

| 作业 | 内容 | 结论 |
| --- | --- | --- |
| `106179` | flow 1 update，默认嵌套 FSDP | FAILED：FSDP `IDLE` 二阶 backward 问题 |
| `106182` / `106183` | 1/10 update，rollout 16 | 正常结束但无效：NaN advantage，optimizer skipped |
| `106184` | 1 update，TurnOnMicrowave，rollout 64 | 完成有效 backward；KL 0.102，clip frac 0.25，grad norm 1858，EV -29.006 |
| `106185` | 10 updates，TurnOnMicrowave，rollout 64 | 完成 10/10，最后 KL -0.039，clip frac 0，logprob abs diff `5.92e-4`，EV -0.953，return 0 |

smoke checkpoint：

```text
results/xr1_ppo_flow_turnonmicrowave_10u_r64_rootfsdp/
  xr1_robocasa365_ppo_full/checkpoints/global_step_10/
```

解释：logprob diff 在约 `1e-4` 到 `1e-3` 是当前 rollout/update 重算的一致性
诊断；KL 的单批估计可能轻微为负。EV 长期为负、reward 始终为零、KL 持续偏大或
clip fraction 持续过高，才是需要停止或调参的信号。

## 8. 训练命令与安全操作

### 单元测试

```bash
cd "$REPO"
PYTHONPATH=xr1:third_party/RLinf:. .conda-robocasa365/bin/python -m pytest -q \
  -p no:cacheprovider xr1/tests/test_rlinf_ppo_policy.py xr1/tests/test_rlinf_ppo_math.py
```

已验证输出为 `3 passed`。修改 likelihood、probe shape 或 PPO loss 前必须重跑。

### 复现主训练提交

```bash
cd "$REPO"
env \
  XR1_RLINF_CONFIG=xr1_robocasa365_ppo_flow_smoke \
  XR1_RLINF_MAX_STEPS=1000 \
  XR1_RLINF_SAVE_INTERVAL=250 \
  XR1_RLINF_TRAIN_ENVS=1 \
  XR1_RLINF_ROLLOUT_STEPS=64 \
  XR1_RLINF_EPISODE_STEPS=64 \
  XR1_RLINF_TASK_SAMPLING=random \
  XR1_RLINF_DISABLE_AUTO_WRAP=1 \
  XR1_RLINF_LR=1.0e-9 \
  RLINF_ROBOCASA_MAX_RESTARTS=256 \
  XR1_RLINF_LOG_DIR="$REPO/results/xr1_ppo_flow_target50_full_1000u_lr1e9_rootfsdp_r256" \
  sbatch --job-name=xr1-flow-target50-1k-r256 scripts/run_xr1_rlinf_ppo.sbatch
```

`scripts/run_xr1_rlinf_ppo.sbatch` 支持的常用覆盖包括：

- `XR1_RLINF_MAX_STEPS`、`XR1_RLINF_SAVE_INTERVAL`、`XR1_RLINF_RESUME_DIR`；
- task/rollout：`TRAIN_ENVS`、`EPISODE_STEPS`、`ROLLOUT_STEPS`、
  `TASK_FILTER_INCLUDE`、`TASK_SAMPLING`；
- optimization：`LR`、`MICRO_BATCH`、`GLOBAL_BATCH`、`UPDATE_EPOCH`、
  `CLIP_GRAD`、`NORMALIZE_ADVANTAGES`；
- model：`TRAINABLE_MODE`、`ACTION_STD`、`CLIP_ACTION`；
- flow/FSDP：配置名使用 flow config，并设置 `DISABLE_AUTO_WRAP=1`。

每个 checkpoint 当前约 **40 GiB**。因此 1000-update job 设为 250-step 保存间隔；
不要不加评估地将 `save_interval=10` 用于长跑（约 4 TiB）。

### 监控清单

在日志或 TensorBoard 追踪：

1. `actor/approx_kl`：不应持续急剧上升；单批小负值并不异常；
2. `actor/clip_fraction`：持续接近 1 表明更新太大，长期为 0 且 reward 无变化则
   可能学习太慢/信号太弱；
3. `actor/xr1_logprob_abs_diff` 与 max diff：应维持很小，突增优先检查 policy
   replay consistency；
4. `actor/grad_norm`、policy/total loss：必须有限，绝不能出现 skipped optimizer；
5. `critic/explained_variance`、value loss：稀疏奖励初期可能差，但长期必须观察是否
   改善；
6. `rewards`、`return`、task-level success：这是最关键的性能信号。

发生下面任一情况，应停止继续烧算力并保留日志/checkpoint 诊断：NaN/Inf、FSDP
错误、logprob diff 明显恶化、KL/clip 持续失控，或长段训练无任何非零回报。

## 9. 独立 RoboCasa365 评测

训练日志不是成功率。独立评测必须检查聚合结果中：

```text
completed_tasks == expected_tasks
episodes == expected_episodes
```

否则 success rate 无效。旧 target50 eval `105451` 只完成 2/50 tasks、10/250
episodes，不能作为模型分数；根因是 render path 的
`robosuite ... read_pixels -> Fatal Python error: Aborted`。

修复机制：`launch_robocasa365.sh` 默认 `WORKER_MAX_JOBS=1`，
`dynamic_eval.py worker --max-jobs` 每完成一个 episode 后退出，shell 为下一个 job
重启新的 Python/MuJoCo/EGL process。单纯 `env.close()` 不足以释放该进程级资源。

已验证的完整小评测：

| 作业 | checkpoint/task | 结果 |
| --- | --- | --- |
| `105593` | pretrain step100，OpenDrawer，5 fixed seeds | 5/5，100%，completed tasks/episodes 均完整 |

结果目录：
`eval_results/robocasa365-full_step100-procfresh-openDrawer-105593/`。

主长跑至少在 250/500/1000 checkpoint 各做一次小型固定任务评测，再对最优 checkpoint
运行完整 target50 × 5 episodes。示例（单 GPU、小评测）：

```bash
NUM_LANES=1 EVAL_TASKS=OpenDrawer NUM_TRIALS=5 \
  sbatch --gres=gpu:1 scripts/run_robocasa365_policy_eval_isolated.sbatch
```

实际评测时必须将脚本的 checkpoint/output 参数指向待评 checkpoint，并为每次评测使用
新的结果目录。不要覆盖 pretrain 或其他 checkpoint 的 aggregate。

## 10. 推荐的实验决策树

1. **现在**：让 `106535` 至少跑到 step 250，确认 checkpoint、有限指标和是否出现
   非零 reward；不要只看作业状态。
2. **step 250**：对一个固定小任务集合做隔离评测；若训练指标恶化或 success 不低于
   pretrain，则暂停 1k 并分析 LR、reward coverage 和 critic。
3. **step 500/1000**：同一评测协议复测；仅对有趋势的 checkpoint 跑全 target50。
4. **若全程 reward 接近零**：这是纯 PPO + 稀疏二元成功 reward 的探索上限证据，
   不应仅把步数继续放大。下一轮应改变 reward/探索或采用 action-expert-only 的
   对照，而不是把“PPO 无效”归因给 flow 代码。
5. **若 KL/clip 不稳**：先降 actor LR、检查 logprob consistency、保持 trace probes
   固定；不要恢复 Gaussian surrogate 来掩盖目标分布不一致。
6. **若有成功趋势但 full fine-tune 不稳**：做同预算 `trainable_mode=action_expert`
   对照，分开回答“更稳定”与“更高成功率”。

## 11. 新对话最小上下文提示词

可以把下面内容直接给新对话：

```text
请先阅读 XR1_RLINF_PPO_HANDOFF.md，再检查 job 106535 的 squeue/sacct、
logs/xr1_rlinf_ppo_106535.log 和 results/xr1_ppo_flow_target50_full_1000u_lr1e9_rootfsdp_r256。
这是 XR-1 RoboCasa365 的 RLinf PPO：当前用 flow_hutchinson likelihood、full model、
XR1_RLINF_DISABLE_AUTO_WRAP=1。不要把作业 COMPLETED 当成功；报告 KL、clip fraction、
logprob diff、grad finite、critic EV、return，并以完整隔离评测的 aggregate 判定 success。
不要改用 Gaussian surrogate，除非明确作为独立 ablation。
```

## 12. 2026-09-20：job 106535 完成后的只读审计

> 本节是对前文“长跑进行中”状态的后续审计。前文保留作为实验历史，但涉及
> `106535` 当前状态、target50 覆盖和训练有效性的判断，以本节为准。
>
> 本次审计只读取配置、代码、日志、metrics 和 checkpoint，没有修改训练代码或配置，
> 也没有启动新训练。本节用于在后续对话或代码改动丢失时保留诊断依据。

### 12.1 总结：作业完成不等于训练有效

job `106535` 在工程执行层面完成了 1000 个 PPO update：

- `metrics.log` 包含 step 1--1000；
- 在 step 250/500/750/1000 保存了 checkpoint；
- 没有发现 NaN/Inf、OOM 或 `Skipping optimizer step`；
- XR-1 flow likelihood/replay consistency 指标总体有限；
- PPO policy/math 单元测试仍为 `3 passed`。

但是，这不能被视为一次有效的“50-task PPO 训练”，也没有证明策略性能提升。审计确认
了三个会改变训练含义的问题：

1. `target50` 实际退化为只训练 `MakeIceLemonade`；
2. RoboCasa 子进程发生 78 次周期性 EOF，重启后的 reset 没有被上层标成 episode
   boundary，导致跨 reset 的伪连续 trajectory；
3. 训练期间记录到的 21 个完整 episode 全部 reward/return/success 为 0。在这种情况下，
   actor 仍可被随机 critic、GAE advantage normalization 和错误 transition 驱动，非零
   PPO loss/KL/clip fraction 不能证明学到了任务。

因此应把 `106535` 定性为：

> **1000-update 执行与 checkpoint 保存成功，但任务覆盖、trajectory 正确性和任务性能均
> 未通过验收。该 checkpoint 只能作为诊断产物，不能作为有效 target50 PPO 结果发布。**

### 12.2 实际只训练了一个任务

训练配置为：

```yaml
env:
  train:
    total_num_envs: 1
    split: target
    task_soup: target50
    task_sampling_strategy: random
    rotate_tasks_on_rollout: false
    auto_reset: true
    rotate_tasks_on_auto_reset: false
```

对应文件：
`xr1/mibot/rlinf/configs/xr1_robocasa365_ppo_full.yaml:63-74`。

`target50` 只指定了 50-task 候选池，并不保证实际遍历 50 个任务。由于只有一个 env，且
rollout boundary 和 auto-reset 两种轮换都关闭，首次随机采样后任务就永久固定。日志中
只出现过一次任务创建：

```text
Creating MakeIceLemonade with split=target
```

没有其他 task 被创建。因此准确描述应是“从 target50 中随机选到
`MakeIceLemonade` 后进行 1000 updates 单任务训练”，不是 50-task 训练。

`third_party/RLinf/rlinf/workers/env/env_worker.py:782-796` 表明只有
`rotate_tasks_on_rollout=true` 时，rollout 结束才会调用 `update_reset_state_ids()`。
关闭该开关的原始动机是避免任务切换反复关闭/创建 EGL context，降低
`read_pixels` native abort 风险；但它同时造成了上述单任务副作用。

### 12.3 78 次 EOF 的周期与可能根因

`logs/xr1_rlinf_ppo_106535.log` 中共有 78 条：

```text
[RoboCasa] restarted worker after EOF; count=N exitcode=None
```

重启对应的 PPO update 基本是：

```text
12, 25, 38, 50, 63, 76, 89, 101, ... , 993
```

即几乎严格每 12--13 updates 一次。当前每个 update 的 rollout 为 64 simulator
steps，所以约每 `12.5 * 64 = 800` simulator steps 发生一次 EOF。如此稳定的周期更像
renderer/context 生命周期或资源累计阈值，不像 PPO 数值发散或随机 action 偶发错误。

历史同一路径已有明确 native crash 栈，例如
`logs/xr1_rlinf_ppo_104310.log`：

```text
Fatal Python error: Aborted
robosuite/utils/binding_utils.py:174 in read_pixels
...
robosuite/environments/base.py in step
RLinf/.../robocasa/venv.py in _worker
```

因此当前首要假设是 MuJoCo/EGL 在 `read_pixels` 中 SIGABRT。不过，不能把这个历史栈
直接当成 `106535` 的精确死因：当前 restart 实现没有保存旧进程的退出状态或 stderr。

`third_party/RLinf/rlinf/envs/sim/robocasa/venv.py:231-271` 中，
`_restart_after_eof()` 在读取/打印 exit code 前已经用新进程覆盖 `self.process`，所以日志里
的 `exitcode=None` 是新进程尚在运行的状态，不是旧进程的真实退出码。现有 `106535`
产物无法严格区分 SIGABRT、SIGKILL/OOM 或其他 native crash。

把 `RLINF_ROBOCASA_MAX_RESTARTS` 从 8 提到 256 只使训练能越过故障并跑完，不是根因
修复，也不应作为稳定性验收标准。

### 12.4 EOF recovery 会静默破坏 PPO trajectory

这是比“环境偶尔重启”更严重的数据正确性问题。

`third_party/RLinf/rlinf/envs/sim/robocasa/venv.py:263-271` 在 pipe EOF 后重建子进程、
reset 新环境，并返回：

```python
return (obs, 0.0, True, {"worker_restarted": True})
```

它试图把丢失的 transition 表示成 zero-reward terminal。但是
`third_party/RLinf/rlinf/envs/sim/robocasa365/robocasa365_env.py:687-700` 随后执行：

```python
raw_obs, rewards, dones, info_lists = self.env.step(env_actions)
del rewards, dones
terminations = np.array(
    [info.get("success", False) for info in info_lists]
).astype(bool)
```

这会产生以下结果：

- subprocess worker 返回的 `done=True` 被直接丢弃；
- `worker_restarted` 没有映射为 termination/truncation；
- RoboCasa365 外层 `_elapsed_steps` 没有因内部 reset 清零；
- reset 前的 state/action 与 reset 后的 observation 被拼成同一条 trajectory；
- GAE 和 critic bootstrap 把不连续的两个 MDP state 当作正常相邻状态。

所以 78 次 restart 至少引入了 78 个未标注的 MDP discontinuity。即使训练不中断，
returns、advantages 和 critic targets 也已经被污染。后续不能仅通过“restart 后继续收集”
来宣称容错成功；必须保证 episode boundary 在完整数据通路中被保留。

### 12.5 完整 episode 的真实表现：21/21 失败

`MakeIceLemonade` registry horizon 为 3000 simulator steps。`metrics.log` 共记录 21 次
完整 episode，全部为：

```text
episode_len=3008.0
num_trajectories=1
return=0.0
reward=0.0
success_once=0.0
```

`3008` 是 16-step action chunk 对 3000 horizon 向上对齐的结果。由此可明确判定：

- 训练中观察到的完整 episode 为 21 个；
- 21/21 均失败；
- 没有环境成功回报；
- 没有证据显示策略学会了 `MakeIceLemonade`，更不能外推到 target50。

### 12.6 为什么零成功回报时 actor 仍有更新

`xr1/mibot/rlinf/ppo_policy.py:121-125` 使用普通 `nn.Linear` 新建 value head，没有零
初始化。训练开始时 critic 因而会对状态产生随机 value。

GAE 位于 `third_party/RLinf/rlinf/algorithms/advantages.py:24-84`：

```python
delta = rewards[t] + gamma * values[t + 1] * (~dones[t + 1]) - values[t]
advantages = returns - values[:-1]
```

即使真实 reward 全为 0，只要 `V(s_t)` 与 `V(s_{t+1})` 不同，advantage 就不为 0。
该函数的 `normalize_advantages` 默认值为 `True`；当前 embodied actor 调用没有显式传入
该参数，所以使用默认值。`safe_normalize()` 又会执行：

```python
(advantage - mean) / (std + 1e-5)
```

这意味着很小的随机 critic 差分也能被缩放成非零 actor 学习信号。EOF recovery 产生的
跨 reset value jump 会进一步污染该信号。此外，rollout truncation 时
`compute_bootstrap_rewards()` 会把 `gamma * bootstrap_value` 加到最后一步 reward；因此
表格中某些细小正负 `rewards` 不等于真实环境成功奖励。

这解释了为什么外部 episode return 全为 0，但日志仍有非零 policy loss、KL 和
clip fraction。它们只说明 PPO optimizer 在更新，不能说明更新方向与任务成功有关。

指标汇总中 actor likelihood/replay 数值总体有限，但 critic explained variance 中位数约
`-49.65`，最差约 `-510000`。在零真实回报和损坏 transition 的背景下，这应视为 critic
target/拟合失败的信号，而不是稀疏奖励训练的正常成功证据。

## 13. 后续改进方案（尚未实施）

下面按依赖顺序列出推荐修改。必须先保证数据语义正确，再讨论增加训练步数、调学习率或
扩大任务数；否则只会更长时间地优化错误 trajectory。

### 13.1 第一优先级：保留 worker restart 的 episode boundary

目标：任何 subprocess crash/reset 都不能跨 reset 计算 GAE。

推荐实现原则：

1. 在 `RoboCasa365Env.step()` 中保留底层 `dones`，不要无条件 `del dones`；
2. 将 `worker_restarted=True` 显式映射到 truncation（更符合外部故障而非任务自然
   termination），并确保该 transition 后启动新 episode；
3. 对发生 restart 的 env 重置外层 `_elapsed_steps` 和 episode metrics；
4. 丢失 action 对应的真实 transition 无法恢复，不能把 reset 后 observation 当成它的
   next observation。实现时应选择明确的 fault-transition policy：丢弃受影响的 partial
   trajectory，或用 truncation boundary 隔开并正确 bootstrap/mask；
5. 检查 RLinf `dones` 的时间索引。当前 GAE 使用 `dones[step + 1]`，修复后必须用合成测试
   证明 boundary 前的 advantage 不会传播到 reset 后；
6. auto-reset 路径要返回 terminal observation 还是 reset observation，必须与
   `compute_bootstrap_rewards()` 的约定一致，避免在错误 observation 上 bootstrap。

建议新增最小回归测试：构造一个确定在第 N 步 EOF 的 fake subprocess env，令 reset 前后
observation/value 差异很大，然后断言：

- restart 被标为 truncation/done；
- episode step counter 清零；
- metrics 分成两个 episode；
- reset 后 value 不进入 reset 前 trajectory 的 GAE；
- batch 中没有跨 reset 的 `(s_t, a_t, s_{t+1})`。

在此测试通过前，不应继续正式 PPO 长跑。

### 13.2 第二优先级：保存旧 worker 的真实故障证据

修改 `_restart_after_eof()` 时应在覆盖 `self.process` 前：

1. 保存 `old_process`；
2. 对其做有超时的 `join()`；
3. 记录 `old_process.pid`、`exitcode`、signal 名称和 restart 前累计 simulator steps；
4. 若仍存活，区分“pipe 断开但进程活着”和“native process 已退出”，再做受控 terminate；
5. 将 child stderr 重定向到按 PID/restart count 区分的文件，避免 native traceback 丢失；
6. 同时记录 GPU、RSS、文件描述符数量和 EGL/MuJoCo renderer/context 标识，以验证是否有
   约 800-step 的资源增长；
7. restart counter 应按 env/worker 分开，并输出发生时的 PPO update、episode step 和
   task，而不是只记录全局 count。

只有拿到 `106535` 同等路径上的旧进程 exit code/native stderr，才能确认究竟是
`read_pixels` SIGABRT、OOM/SIGKILL、driver error 还是其他故障。

### 13.3 第三优先级：根治或隔离约 800-step renderer crash

建议按以下顺序做短时诊断，不要直接再跑 1000 updates：

1. 固定 `MakeIceLemonade`、固定 seed 和 action source，做超过 1600 simulator steps 的
   environment-only reproduction，确认是否在约 800/1600 steps 重现；
2. 分别测试有/无 camera `read_pixels`，确认故障是否只在渲染路径；
3. 按固定间隔记录 RSS、GPU memory、FD 数和 EGL context/renderer 重建次数；
4. 使用进程级隔离：在接近已知阈值前由外层调度器结束完整 Python/MuJoCo/EGL worker，
   再启动新进程。该边界必须作为 episode truncation/trajectory boundary，而不能在 rollout
   内静默换 observation；
5. 将 evaluator 已采用的 `WORKER_MAX_JOBS=1` / 每 episode 新进程思路作为训练侧参考，
   但需要评估启动开销，并保证 PPO rollout 数据协议正确；
6. 若确定是具体 MuJoCo/robosuite/EGL 版本问题，再做最小版本或 renderer backend 对照。

验收不是“restart limit 没耗尽”，而应至少满足以下其一：

- 连续多次超过原故障周期且无 native crash；或
- 进程级预防性轮换稳定运行，且每次轮换均形成正确 truncation，回归测试证明无跨 reset
  trajectory。

### 13.4 第四优先级：建立真正的 50-task sampling/scheduling

不能简单地把 `rotate_tasks_on_rollout=true` 打开后直接长跑，因为此前关闭它就是为了规避
频繁重建 EGL context。应把“任务覆盖”和“renderer 稳定性”一起设计。

推荐优先方案是进程级 task isolation/scheduling：

- 外层维护 target50 的 ordered 或可复现 shuffled task queue；
- 每个 worker/process 在其生命周期内只运行一个任务或有限数量 episode；
- 完成一个明确的 trajectory/episode 后退出整个 simulator process，再由调度器取下一个
  task；
- 保存 sampler state、task index 和 per-task rollout/episode count，使 checkpoint resume
  后不会重新只采某一小部分任务；
- 训练日志必须记录每个 update 的 task name，聚合中输出 unique tasks、每任务 simulator
  steps、episodes、successes 和 rewards。

如果仍选择 rollout boundary 轮换，至少必须先验证反复 task reconfiguration 不再触发 EGL
abort，并确认每次任务切换都在合法 trajectory boundary。`random` sampling 只提供概率覆盖，
不能证明 50 个任务都训练过；为了验收，建议首轮使用 ordered/round-robin 或带 coverage
约束的 shuffled sampler。

50-task 训练的最低覆盖验收应包括：

```text
unique_tasks_seen == 50
每个 task 的 simulator_steps > 0
每个 task 的完整 episode 或明确 truncation 数可核对
日志中的 task-count 总和与实际 rollout steps 一致
resume 前后 sampler state 连续
```

### 13.5 第五优先级：避免零回报下的伪 actor 更新

在 trajectory 修复后，再单独处理稀疏 reward/critic 问题：

1. 加入 telemetry，将 `environment_reward`、`bootstrap_value_adjustment`、reward model 输出和
   最终 `adjusted_reward` 分开记录；不能继续用一个 `rewards` 字段混合解释；
2. 分别记录 raw advantage 与 normalized advantage 的 mean/std/min/max，以及有非零环境
   reward 的 batch 比例；
3. 当整个 rollout 没有真实任务奖励时，做明确策略选择并作为实验变量：跳过 actor update、
   仅训练 critic、使用非稀疏但经过验证的 shaped reward，或引入 expert/demo warm start；
   不要默认把随机 critic noise 标准化后更新 actor；
4. value head 初始化、critic warm-up 和 `normalize_advantages` 应做受控 ablation。关闭
   normalization 本身不能修复错误 trajectory，也不应被当作单点根治；
5. 保留 `flow_hutchinson` 的固定 executed action、缓存 trace probes、math SDP attention、
   `clip_normalized_action: null`、`entropy_bonus: 0.0` 和
   `XR1_RLINF_DISABLE_AUTO_WRAP=1`，除非作为明确的独立 ablation；
6. actor LR、critic LR、clip range 等超参只应在环境数据正确且出现可解释 reward 后调整。

性能判断必须以独立 fixed-seed evaluation 为准，不能用非零 policy loss、KL 或训练时
bootstrap reward 代替 success rate。

### 13.6 推荐修复与验证顺序

严格按以下阶段推进，每阶段失败就停止，不要直接扩大预算：

#### 阶段 A：纯单元测试

- restart/EOF 能保留 truncation boundary；
- 不存在跨 reset GAE；
- elapsed steps、episode metrics 和 terminal/reset observation 语义正确；
- 旧进程 exit code/stderr 能被保存。

#### 阶段 B：environment-only 稳定性复现

- 固定单任务运行至少 2--3 倍原 800-step 故障周期；
- 得到明确 native crash 证据，或证明预防性 process rotation 稳定；
- 记录资源随 simulator steps 的变化。

#### 阶段 C：64/64 单任务 PPO smoke

- 只验证 flow/FSDP/GAE 数据通路；
- 人工注入一次 worker failure，确认 batch boundary 正确；
- 无 NaN/Inf、无 skipped optimizer、logprob replay diff 维持很小。

#### 阶段 D：多任务 coverage smoke

- 用 2--5 个任务做 ordered/round-robin；
- 日志确认每个任务实际被创建和采样；
- task switch/process rotation 只发生在合法 boundary；
- sampler state 可保存并 resume。

#### 阶段 E：target50 短跑

- 在增加 update 数前先证明 `unique_tasks_seen == 50`；
- 输出每任务 steps/episodes/reward/success；
- 无未处理 EOF；若有受控 restart，数量、exit code 和 truncation 均可审计；
- 训练 reward 与 bootstrap adjustment 分开。

#### 阶段 F：正式训练与独立评测

- 保存 pretrain baseline 与各 PPO checkpoint 的同协议 fixed-seed 结果；
- 先做小任务集合 gate，再做完整 target50 × 固定 trials；
- 必须满足 `completed_tasks == expected_tasks`、
  `episodes == expected_episodes`；
- 只有完整评测显著优于 baseline，才能声称 PPO 有性能收益。

### 13.7 后续接手者的首要检查清单

新对话开始时先执行只读检查：

1. 阅读本节和 `logs/xr1_rlinf_ppo_106535.log`；
2. 确认当前代码是否仍在 `RoboCasa365Env.step()` 中丢弃底层 `dones`；
3. 确认 `_restart_after_eof()` 是否仍先覆盖 `self.process` 再打印 exit code；
4. 检查配置中两个 `rotate_tasks_*` 是否仍为 `false`；
5. 不要复用 `106535` 目录覆盖旧证据；任何修复实验使用新 log/result 目录；
6. 在修改前保存 git diff/commit hash，在修改后逐阶段验证；
7. 不要把提高 restart limit、Slurm `COMPLETED` 或 checkpoint 存在当作训练正确性证明。

截至本节写入时，尚未实施上述代码修复，也没有提交或启动新的 PPO 作业。

## 14. 2026-09-20：修复实施、验证与新长跑

本节记录第 12--13 节诊断后的实际实现。第 13 节保留为修复设计历史；当前代码和作业
状态以本节为准。

### 14.1 已实施修复

1. **native worker 故障改为 fail-fast**
   - `RLINF_ROBOCASA_EOF_POLICY` 默认是 `raise`；
   - lost transition 不再默认伪装成可训练 terminal；
   - 在覆盖 worker handle 前保存旧 PID 和 exit code，负 exit code 显示 signal 名；
   - 完整训练设置 `RLINF_ROBOCASA_MAX_RESTARTS=0`，任何意外 EOF 都立即停止，避免生成
     静默损坏的 trajectory。
2. **底层 boundary 不再丢失**
   - `RoboCasa365Env.step()` 保留 subprocess `raw_dones`；
   - task success 映射为 termination，其他底层 done、horizon 和诊断 restart 映射为
     truncation；
   - `worker_restarted` 和 `transition_valid` 进入 info，便于测试和审计。
3. **renderer 生命周期隔离**
   - 训练 episode 使用 chunk 对齐的 768 simulator steps，低于旧作业约 800-step 的
     周期性 crash 阈值；
   - episode/task boundary 不在同一进程内 reconfigure EGL，而是正常关闭整个旧
     Python/MuJoCo/EGL process，再用新 task callable 启动新 PID；
   - 日志记录 `old_pid`、`old_exitcode`、`new_pid`、old/new task ID 和 task name。
4. **真正的 target50 ordered coverage**
   - `task_sampling_strategy=ordered`；
   - `rotate_tasks_on_rollout=false`，避免每 64-step rollout 截断 episode；
   - `rotate_tasks_on_auto_reset=true`，每个 768-step episode 后轮换到下一 task；
   - task rotation 日志可核验 `old_task_id -> new_task_id`；metrics 的完整 episode 增加
     `task_id`。
5. **零回报下不再由随机 critic 制造 actor 信号**
   - XR-1 value head 的 weight/bias 改为零初始化；
   - 在无真实 reward 的 smoke 中，advantages/returns/policy loss/value loss/grad norm 均为
     0；这比把 critic 随机差分标准化成 PPO 信号更符合稀疏奖励语义。
6. **训练与正式评测 horizon 分离**
   - 训练使用 768-step 安全隔离 horizon；
   - eval 仍使用 registry `task_horizon` 和 3008-step rollout budget，不能用训练 truncation
     替代正式 benchmark 评测。

### 14.2 测试结果

CPU 回归命令：

```bash
PYTHONPATH=xr1:third_party/RLinf:. .conda-robocasa365/bin/python -m pytest -q \
  -p no:cacheprovider \
  xr1/tests/test_rlinf_ppo_policy.py xr1/tests/test_rlinf_ppo_math.py
```

结果：`7 passed`。测试覆盖 value head 零初始化、worker signal 格式、restart truncation、
GAE 不跨 boundary、ordered 50-task sampling，以及既有 flow likelihood/backprop contract。
`bash -n scripts/run_xr1_rlinf_ppo.sbatch`、Python compile 和两个 worktree 的
`git diff --check` 也通过。

GPU/环境集成验证：

| job | 目的 | 结果 |
| --- | --- | --- |
| `107893` | 14 updates，跨过旧约 800-step 故障周期 | `COMPLETED`，14/14；768-step boundary 正常换 PID，旧进程 exit 0，无 EOF |
| `107899` | 2 updates，强制每 64 steps ordered task rotation | `COMPLETED`；发现并暴露旧 callable 一拍延迟，促使增加 worker identity 取证 |
| `107905` | 1 update，验证实际新 worker 的 task identity | `COMPLETED`；PID `263841` 为 task 0，轮换后 PID `265918` 明确为 task 1，旧进程 exit 0 |

`107905` 的关键证据：

```text
[RoboCasa] worker ready; pid=263841 task_id=0 task=CloseBlenderLid
[RoboCasa365] task rotation: old_task_id=0 new_task_id=1 ...
[RoboCasa] replaced worker; ... old_exitcode=0 new_pid=265918
[RoboCasa] worker ready; pid=265918 task_id=1 task=CloseFridge
```

### 14.3 新完整训练

通过上述 gate 后已提交全新长跑，旧 `106535` 目录未被覆盖：

```text
job: 107910
name: xr1-target50-fixed-1k
updates: 1000
rollout: 64 simulator steps/update
training episode/process lifetime: 768 simulator steps
task schedule: ordered target50, rotate on auto-reset
actor LR: 1e-9
update epochs: 1
save interval: 250
EOF policy: raise
unexpected restart allowance: 0
```

结果目录：

```text
results/xr1_ppo_flow_target50_full_1000u_fixed_20260920/
```

日志：

```text
logs/xr1_rlinf_ppo_107910.log
```

监控时必须检查：

- 每 12 updates 左右出现的是 `task_reconfigure old_exitcode=0`，而不是 EOF restart；
- worker-ready task ID 与 task-rotation new task ID 一致；
- 约 600 updates 内（50 tasks × 12 updates/task）至少完成一轮 `task_id 0..49`；
- 无 `RoboCasa simulator worker died`、NaN/Inf、skipped optimizer；
- 零 reward rollout 的 advantage/policy loss 不再来自随机 value head；
- checkpoint 在 250/500/750/1000 正常保存。

新作业完成只能证明正确覆盖并完成训练；性能收益仍须按第 9 节使用完整 fixed-seed
target50 独立评测确认。

## 15. 2026-09-21：`107910` 失败、变长 prompt 修复与重新提交

### 15.1 `107910` 的实际失败原因

`107910` 最终为 `FAILED (ExitCode 127:0)`，运行 9 分 35 秒，只完成到 global step 30；
MaxRSS 约 46.1 GiB，不是系统内存 OOM。没有生成 step-250 checkpoint，因此不能从该
作业恢复，必须使用新结果目录重新训练。

首个致命异常是：

```text
RuntimeError: stack expects each tensor to be equal size,
but got [1, 486] at entry 0 and [1, 476] at entry 1
```

调用链为 `EmbodiedTrajectoryBuilder.to_trajectory()` ->
`stack_list_of_dict_tensor(self.forward_inputs)` -> `torch.stack(v_list)`。根因是成功 episode
可在一个 64-step rollout 中途结束，auto-reset 随即切到 ordered target50 的下一任务；同一
rollout buffer 因此包含不同官方 task prompt。XR-1 processor 为它们生成不同长度的
`input_ids`/`attention_mask`，而 RLinf 原实现直接 `torch.stack`，不支持变长序列。

该作业在失败前确实记录了真实成功 episode（`task_id=1`、`episode_len=192`、
`return=1.0`、`success_once=1.0`）；task boundary 的旧 worker 均为 `exitcode=0`，所以这次
故障不是 renderer EOF、native crash 或错误 restart。

### 15.2 最终修复：trajectory 内动态右填充

最初尝试在 observation adapter 中把所有 prompt 固定右填充到 512。CPU 和真实 processor
shape 测试虽然通过，但 GPU job `107938` 在第一个 update 反向传播时 OOM：需要额外申请
9.41 GiB，而当时只剩 9.31 GiB。该方案还会让每次 rollout inference 都承担 512-token
长度，因此已撤销，不能重新采用。

最终实现只在 trajectory 构建、即确实需要 stack 时动态填充到该 rollout 的实际最大长度：

- `XR1PPOPolicy.predict_action_batch()` 将 checkpoint tokenizer 的 `pad_token_id` 作为
  `xr1_text_pad_token_id` rollout-only 元数据保存；
- 该字段加入 `_DIAGNOSTIC_FIELDS`，在所有 XR-1 model 调用前过滤，不会成为模型参数；
- `EmbodiedTrajectoryBuilder.to_trajectory()` 在 stack 前检查 `input_ids`、
  `attention_mask` 和 pad-token 元数据；
- 只对短样本末尾补 checkpoint pad token，并把对应 attention mask 补 0；
- 最大长度来自当前 trajectory，不设置全局上限、不截断 prompt，也不增加单步 rollout
  inference 的序列长度；
- 其他模型/forward inputs 没有 `xr1_text_pad_token_id` 时完全保持原始 RLinf 行为。

真实 checkpoint processor 验证中，两种 prompt 的原始长度为 473 和 478，构建后 shape
为 `[2,1,478]`；短样本有效 token 原样保留，尾部为 pad token `151643`，mask 为 0。

### 15.3 回归和 GPU gate

回归命令覆盖 XR-1 policy/math 以及 RLinf data tests：

```bash
PYTHONPATH=xr1:third_party/RLinf:. .conda-robocasa365/bin/python -m pytest -q \
  -p no:cacheprovider \
  xr1/tests/test_rlinf_ppo_policy.py \
  xr1/tests/test_rlinf_ppo_math.py \
  third_party/RLinf/tests/unit_tests/test_data.py
```

结果为 `36 passed, 7 skipped`。Python compile、`bash -n`、主仓库和 RLinf checkout 的
`git diff --check` 均通过。

动态 padding GPU smoke：

```text
job: 107945
state: COMPLETED (0:0)
updates: 32/32
elapsed: 11:06 (训练 metrics elapsed 09:53)
MaxRSS: 58,047,792 KiB
checkpoint: global_step_32
result: results/xr1_ppo_flow_target50_promptpad_dynamic_smoke_32u_20260921/
```

该 smoke 实际完成了 task 0--3 并创建 task 4，所有 task replacement 的旧进程均
`exitcode=0`。其中 task 1 和 task 3 都成功：

```text
task_id=1 episode_len=192 return=1.0 success_once=1.0
task_id=3 episode_len=208 return=1.0 success_once=1.0
```

这意味着一个 rollout 中途 success -> auto-reset -> 不同 prompt 的关键路径被真实覆盖；
全程没有 `stack expects each tensor`、OOM、EOF、native crash 或 traceback。训练指标有限，
step 32 的 `xr1_logprob_abs_diff=5.37e-4`、`xr1_logprob_max_diff=0.025`。零回报早期的
`critic/explained_variance=nan` 只是目标方差为零时统计量未定义；出现成功回报后 value
loss、policy loss 和梯度均为有限值。

### 15.4 新完整训练 `107949`

smoke 通过后，已用新目录提交完整训练，未覆盖 `107910` 或其他历史结果：

```text
job: 107949
name: xr1-t50-dynpad-1k
updates: 1000
checkpoint interval: 250 (预期 250/500/750/1000)
rollout: 64 simulator steps/update
episode/process horizon: 768 simulator steps
task schedule: ordered target50, rotate only on auto-reset
actor LR: 1e-9
update epochs: 1
trainable mode: full
flow likelihood: flow_hutchinson
EOF policy: raise
unexpected restart allowance: 0
allocator: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
result: results/xr1_ppo_flow_target50_full_1000u_dynamicpad_20260921/
log: logs/xr1_rlinf_ppo_107949.log
```

`expandable_segments` 只降低 actor/rollout 同卡共存时的显存碎片，不改变 PPO 数学、模型
输入或 optimizer。验收仍需确认作业完成、四个 checkpoint 存在、task 0--49 至少完整覆盖、
所有 boundary replacement 均为正常 `exitcode=0`、无 stack/OOM/EOF/NaN 参数或 skipped
optimizer。即使这些条件全部通过，也只能证明训练健康；PPO 性能收益仍需独立完整
fixed-seed target50 evaluation。

提交后启动检查：`107949` 已在 `gnho008` 运行到 global step `8/1000`，ETA 约 4 小时
36 分；rollout、actor backward 和 weight sync 均正常，未出现 stack/OOM/EOF/traceback。
## 16. fixed-seed target50 独立评测：MuJoCo/EGL readback 修复与 v5 作业（2026-09-21）

### 16.1 评测协议与有效性条件

当前只比较三组策略：原始预训练模型、PPO step 500、PPO step 1000。三组必须使用完全
相同的协议：RoboCasa365 `target50`、`split=pretrain`、base seed 7、每任务 5 trials、
官方 task horizon，共 50 tasks / 250 episodes。只有 aggregate 同时满足
`completed_tasks=expected_tasks=50` 和 `episodes=expected_episodes=250` 才能作为结果；训练
rollout 中的 35/106 success telemetry 不是 benchmark 分数。

### 16.2 已确认的 renderer 根因和最终 readback 路径

原始 MuJoCo 3.3.1 `mjr_readPixels` 在长失败 episode（最小复现为
`CloseBlenderLid / seed=8`，约 step 800 后）会从 native code `SIGABRT`。OSMesa、Xvfb/GLX、
定期重建 `MjrContext`、将相机更新频率减半均不能解决。直接调用 client-memory
`OpenGL.glReadPixels` 对 PPO 轨迹仍会在相同位置 abort，说明问题位于 NVIDIA EGL 的直接
CPU readback 路径，而不是 Python/MuJoCo 包装本身。

最终实现位于 `eval_robocasa365/entry.py`：先将 multisample offscreen FBO resolve 到
`offFBO_r`，然后让 `glReadPixels` 写入 `GL_PIXEL_PACK_BUFFER`，再用
`glGetBufferSubData` 拷回 CPU。模型输入、相机、每步渲染、物理 stepping、success check 和
官方 horizon 均不变。每个新 context 的首帧还会同时走原始 MuJoCo readback，并要求
`exact_pixel_match=true`；不一致会立即使作业失败。

验证证据：

- `108142`：预训练 `CloseBlenderLid / seed=8`，恢复每步渲染后 900/900，首帧 exact match；
- `108180`：预训练 CloseBlenderLid seeds 7--11，5/5 均到 900 steps；
- `108196` / `108199`：PickPlaceCounterToCabinet、NavigateKitchen 各 5/5 完整；
- `108270`：step500 必现 PPO 轨迹完整到 900/900，exact match，无 abort；
- `108271` / `108276`：step500、step1000 各 5 seeds，共 10 个 900-step episode，全部
  `COMPLETED (0:0)`、errors=0。

`dynamic_eval.py recover` 和 `scripts/launch_robocasa365.sh` 还增加了 native abort 后将
`running/*.json` 原子移回 pending 的恢复逻辑和 attempt 上限；但最终性能结果仍不允许缺失
任何 episode。

### 16.3 结果目录取舍、v5 lane 映射故障和当前 v6 作业

以下目录都是诊断/部分运行，**不得用于性能比较**：所有早期五 checkpoint 目录、v2、v3、
v4，以及名称包含 `smoke` 或 `gate` 的目录。v4 曾被隐藏的并行提交 `108208` 和后续作业
同时写入，已明确废弃；相关作业全部取消。

v5 作业已经结束，但三组都是 Slurm `FAILED (1:0)`，且只能产生 13/50 tasks、65/250
episodes。根因不是 PBO renderer：lane 0 正常完成且无 renderer error；lane 1--3 在导入
robosuite 时因 `CUDA_VISIBLE_DEVICES=1/2/3`、`MUJOCO_EGL_DEVICE_ID=0` 不一致触发 assertion，
未进入 simulator。因此 v5 的 0/65 success 和 0% 都是无效部分结果。

修复后每条 lane 使用相同的 job-local ordinal：

```bash
CUDA_VISIBLE_DEVICES=${lane}
MUJOCO_EGL_DEVICE_ID=${lane}
```

`108372` 专门将 `CloseFridge` 分配到 lane 1，完整达到 900 steps 并生成 1/1 aggregate，
验证映射正确。当前唯一候选正式结果是全新 v6：

```text
108377 pretrained -> eval_results/fixedseed-target50-pretrained-5trials-v6-20260921
108378 step500    -> eval_results/fixedseed-target50-step500-5trials-v6-20260921
108379 step1000   -> eval_results/fixedseed-target50-step1000-5trials-v6-20260921
```

本节更新时三组 v6 均已运行，所有 lane 0--3 都实际进入评测，初始 scheduler results 为
18/25/25、errors=0。作业结束后必须重新读取三个 `aggregate.json`，核验 50/250 完整
分母、checkpoint 路径、seed/split/trials，之后才能报告整体成功率和 per-task paired
fixed-seed 差异。

## 17. v6 的 0% eval 回归、修复与 v7 作业（2026-09-22）

### 17.1 v6 结果无效，0% 不是模型性能

v6 最终暴露了两个独立回归。`108377` 以 `FAILED (1:0)` 结束，只完成 30/50 tasks、
150/250 episodes；`108378` 和 `108379` 在分别只有 16/50、8/50 个有效 task 时被取消，
因为已经有 29、34 个 task 连续三次失败，不可能再形成完整 50/250 分母。三组当时的
success 都是 0，但不得作为模型结果。

旧结果证明 evaluator/model/action 主链路本来可以成功：

- `robocasa365-two-policy-isolated-105135` 完整 50×5 结果分别为 154/250（61.6%）和
  153/250（61.2%）；
- 原始模型旧的部分运行 `robocasa365-target50-5trials-single` 为 76/125（60.8%）；
- `105451` aggregate 本身只有 2/50 tasks，不能报告 100% 总分，但同 seed 的 success
  视频能用于行为回归。例如 `CloseFridge / seed=12` 中旧 policy 明显运动并成功，v6
  则跑满 900 steps 仍近乎静止。

视频帧审计进一步定位到 readback：旧 CloseFridge success 视频有 142 帧且 142 个不同
frame；v6 failure 视频有 451 帧却只有 38 个不同 frame，单个陈旧 frame 最多连续重复
165 次。策略因此长期收到 stale observation，而不是正常闭环图像。

### 17.2 direct GL hook 的具体错误和修复

v6 将 `direct_gl_readback` 默认打开。旧实现把 `GL_READ_FRAMEBUFFER` /
`GL_DRAW_FRAMEBUFFER` 绑定到 MuJoCo resolve FBO 后没有恢复，还遗留 read buffer、pack
alignment 和 pixel-pack-buffer binding。它只在修改 GL binding 后调用原始
`mjr_readPixels` 做首帧比较，因此两个 reader 可以同时读取同一个错误/stale target，
`exact_pixel_match=true` 不能证明后续帧正确。相同 context 还可能在每个 episode 被重复
包装。运行中同时出现 `GLError 1282 glBindFramebuffer` 和 native abort。

修复位于 `eval_robocasa365/entry.py`：

- 正式评测默认和 sbatch 都显式使用 `--no-direct-gl-readback`，恢复旧的 MuJoCo readback；
- 仍保留 PBO 路径用于诊断，但它现在在任何 GL 修改前取 reference，完整保存并恢复 read/
  draw FBO、read buffer、pack alignment 和 PBO binding；
- context 用安装标记防止重复 hook。

原生 MuJoCo readback 的偶发 abort 继续由既有的每 episode worker replacement、native-abort
queue recovery 和每 task 三次 process-level retry 隔离，不能再用会改变 policy observation
语义的 readback 替代路径掩盖。

### 17.3 Slurm CUDA/EGL 映射修复

v6 把每条 lane 的 `CUDA_VISIBLE_DEVICES` 缩成 `0`、`1`、`2`、`3`。该集群的 sbatch
进程已经将物理分配重映射成 job-local `CUDA_VISIBLE_DEVICES=0,1,2,3`；再次在 child 中
用物理 `SLURM_JOB_GPUS` 或缩窄错误 ordinal 都可能得到 `No CUDA GPUs are available`。

当前脚本保留 Slurm 提供的完整 job-local visibility list。`deploy/server.py` 新增 `--device`
并显式把四条 lane 放到 `cuda:0..3`；`MUJOCO_EGL_DEVICE_ID` 同样使用 job-local lane，满足
robosuite 要求该值必须出现在 `CUDA_VISIBLE_DEVICES` 中。脚本仍记录物理
`SLURM_JOB_GPUS`，便于审计非连续分配，但不再把物理 id 当作 torch ordinal。

### 17.4 测试、gate 和 v7 完整评测

新增 `xr1/tests/test_robocasa365_eval.py`，覆盖 readback 默认关闭、PBO GL 状态恢复、防重复
hook 和 model server job-local device 选择。完整轻量回归为 `11 passed`，Python compile、
shell syntax 和 `git diff --check` 均通过。

早期 gate `108492` / `108496` 分别确认了“物理 id 不能作为 child CUDA ordinal”和
“物理 EGL id 不在 job-local CUDA list”两种错误，均在进入 episode 前失败，不含性能
结果。当前验收 gate 为：

```text
108504  pretrained / OpenStandMixerHead / seeds 42--46 / 5 episodes
         完整 5/5 且 successes >= 1 才返回 success
```

三组完整 v7 已提交为 `afterok:108504`，gate 不通过就不会启动：

```text
108506  pretrained -> eval_results/fixedseed-target50-pretrained-5trials-v7-20260922
108507  step500    -> eval_results/fixedseed-target50-step500-5trials-v7-20260922
108509  step1000   -> eval_results/fixedseed-target50-step1000-5trials-v7-20260922
```

`108504` 随后在 `gnho008` 以 `COMPLETED (0:0)` 结束：1/1 task、5/5 episodes、5/5
success，steps 分别为 98/186/98/121/106，且 summary 明确记录
`direct_gl_readback=false`。这恢复了旧 evaluator 对同一任务/seeds 的成功行为，排除了
原始权重真实为 0% 的解释。依赖解除后 `108506` 已在 `gnho008` 启动，四条 lane 的 model
server 均实际进入运行；物理分配为非连续 `0,1,5,6`，job-local CUDA list 为
`0,1,2,3`，验证新映射覆盖了此前失败的非连续分配。`108507` / `108509` 当时因 Priority
等待资源。

正式报告仍必须等待每组
`completed_tasks=expected_tasks=50`、`episodes=expected_episodes=250`，并检查 job
exit code、checkpoint、split、seed 和 trials；任何部分 aggregate 或 gate 都不是最终分数。
