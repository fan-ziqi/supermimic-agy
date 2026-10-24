# SuperMimic Reproduction — Google Antigravity Pro 订阅与模型评测项目 (`mjlab` + PyTorch)

> [!IMPORTANT]
> **项目开源说明与免责声明 (Project Purpose & Disclaimer)**
> 1. **开源目的**：本仓库（[`fan-ziqi/supermimic-agy`](https://github.com/fan-ziqi/supermimic-agy)）旨在**测试 Google Antigravity Pro 版订阅及大模型（`Claude Sonnet 5.5 (Medium)` 与 `Gemini 4 Argon (Medium)`）在无人工代码干预、零第三方复现代码参考条件下的端到端科研论文复现能力**。
> 2. **非官方说明**：本项目**仅用于测试智能体与模型能力，并非为了开源 *SuperMimic*，亦非 *SuperMimic* 的官方实现**。*SuperMimic* 算法与论文的全部学术贡献归原作者所有（请参考原论文 [arXiv:2610.28480](https://arxiv.org/pdf/2610.28480) 与官方网站 [https://supermimic.github.io/](https://supermimic.github.io/)）。
> 3. **透明性记录**：复现全过程的详细时间线、模型切换记录（`Claude Sonnet 5.5` $\to$ `Gemini 4 Argon`）以及**全部 8 条用户提示词原文**均完整合并收录于本 `README.md` 中。

Clean-room reproduction of **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model** ([arXiv:2610.28480](https://arxiv.org/pdf/2610.28480), [Project Website](https://supermimic.github.io/), CoRL 2026) using [`mjlab`](https://github.com/mujocolab/mjlab) (MuJoCo Warp) on Unitree G1, conducted autonomously as an evaluation benchmark for **Google Antigravity Pro**.

- **目标**：基于 `mjlab` 复现 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* (arXiv 2610.28480, CoRL 2026)。
- **约束**：只参考论文 (`refs/paper.pdf`) 与官网 (https://supermimic.github.io/)，**不参考任何第三方复现代码**。
- **硬件**：1× NVIDIA GeForce RTX 4090 (48 GB)，32 核 CPU，60 GB 内存（原论文使用 H100，以下结果反映单卡 RTX 4090 上的受控对比与收敛曲线）。

---

## 1. 复现全流程时间线与 3D 物理仿真演示视频 (Overview & 3D Demo Video)

<video src="docs/supermimic_reproduction_overview.mp4" controls width="100%" poster="docs/figures/video_poster.png"></video>

[![SuperMimic Reproduction Overview Video (Click to Play / Download HD MP4)](docs/figures/video_preview.gif)](docs/supermimic_reproduction_overview.mp4)

- **高清完整视频 (`1280×720 HD`, `30 fps`, `16.8 MB`)**：[`docs/supermimic_reproduction_overview.mp4`](docs/supermimic_reproduction_overview.mp4)
- **视频章节结构**：
  - **Chapter 1**：标题、开源评测说明、硬件环境 (`1× RTX 4090`)、核心架构与定量成果总览
  - **Chapter 2**：6 阶段复现时间线逐项详解（展示每个阶段具体做了什么、全部用户提示词与模型切换记录）
  - **Chapter 3**：3D MuJoCo 实时物理仿真演示 1 — `SuperMimic (H=32)` 在 4 类多样化 LAFAN1 动作（`dance1_subject2` 连续舞蹈、`run1_subject5` 高速奔跑急转、`jumps1_subject1` 跳跃落地恢复、`walk1_subject1` 自然步态）上的全身跟踪表现
  - **Chapter 4**：3D MuJoCo 对照实验演示 2 — 同一起始帧下 3 路同屏对比：`SuperMimic (H=32 + Critic)` 稳健行走 vs. `消融 H=16 无 Critic` 贪婪失衡摔倒 vs. `无模型基线 PPO-Sim`
  - **Chapter 5**：3D MuJoCo 下游任务演示 3 — 复合转向速度阶跃响应对比：通过世界模型 BPTT 训练的 `RL 速度编码器` vs. `动作蒸馏基线`
  - **Chapter 6**：四大核心实验结果图表巡礼 (`docs/figures/fig1..fig4.png`)

---

## 2. 用户提示词全量记录（按时间顺序逐字记录）

| 序号 | 时间 (UTC+8, 2026-10-24) | 当时模型 | 用户提示词原文 |
|---|---|---|---|
| 1 | 20:15:59 | Claude Sonnet 5.5 (Medium) | `复现这篇论文https://arxiv.org/pdf/2610.28480，官网：https://supermimic.github.io/ 使用mjlab 不要参考别人的复现代码只看论文和官网 文档中要记录复现的时间线` |
| 2 | 20:17:20 | Claude Sonnet 5.5 (Medium) | `我不会干预 你自己决定` |
| 3 | 20:19:08 | Claude Sonnet 5.5 (Medium) | `拷贝其他项目已经建好的环境 避免下载等待` |
| 4 | 20:20:32 | Claude Sonnet 5.5 (Medium) | `没做完不要停` |
| 5 | 20:41:27 | Gemini 4 Argon (Medium) | `没有额度了，现在从 sonnet5.5 切换到 gemini4argon 都是medium 这个要记录到复现记录中，继续复现` |
| 6 | 20:58:24 | Gemini 4 Argon (Medium) | `我的所有提示词都要写到时间线文档中` |
| 7 | 22:21:20 | Gemini 4 Argon (Medium) | `全部做完之后要生成一个视频，展示整个时间线，每个阶段做了什么，以及如果有的话还要有必要的视频演示` |
| 8 | 23:02:40 | Gemini 4 Argon (Medium) | `好了吗` |

---

## 3. 复现全过程详细时间线

| 时间 (UTC+8, 2026-10-24) | 事件 / 提示词 | 结果 / 备注 |
|---|---|---|
| 20:15:59 | **用户提示词 #1**: `复现这篇论文https://arxiv.org/pdf/2610.28480，官网：https://supermimic.github.io/ 使用mjlab 不要参考别人的复现代码只看论文和官网 文档中要记录复现的时间线` | 开始执行任务，当前模型为 Claude Sonnet 5.5 (Medium) |
| 20:16 | 读取官网与论文全文(含附录 A–G),保存到 `refs/` | 提取了方法、超参 (Table A.5)、奖励 (A.2)、DR (A.3)、网络 (A.4)、重置采样 (D.1) |
| 20:16 | 检查环境:RTX 4090 可用,PyPI / GitHub / HuggingFace 可访问 | |
| 20:16 | 尝试 `uv pip install mjlab` 到新 `.venv` | 下载依赖较慢，后台运行中 |
| 20:17:20 | **用户提示词 #2**: `我不会干预 你自己决定` | 自主推进全部实验设计与决策 |
| 20:19:08 | **用户提示词 #3**: `拷贝其他项目已经建好的环境 避免下载等待` | 取消 `uv pip install`，改为复制本机已建好的虚拟环境至 `.venv`（仅复制环境依赖，不复制任何外部项目源码），耗时 5 s；移除指向原项目的 `.pth` 和 dist-info |
| 20:20 | 验证环境:Python 3.12, mjlab 1.5.0, mujoco/mujoco-warp 3.10, torch 2.12.1+cu130, CUDA 可用 | 可正常 import mjlab 与 mjlab.tasks，无第三方代码泄露 |
| 20:20:32 | **用户提示词 #4**: `没做完不要停` | 持续自主推进代码实现与实验训练 |
| 20:20 | 阅读 mjlab `tasks/tracking`(env cfg / MotionCommand / rewards)与 `ManagerBasedRlEnv.step`,确认 `auto_reset=False` 可拿到终止瞬间的状态,可作为世界模型的数据源 | 设计决策:用 mjlab 做数据收集模拟器,世界模型/策略/训练循环自己用 torch 写 |
| 20:22 | 从 HuggingFace 下载 LAFAN1 G1 重定向 CSV(40 个 clip) | 有 1 个文件(dance1_subject3)首次下载失败,重试后补齐 |
| 20:24 | `scripts/convert_lafan.py`:CSV→50Hz,用 MuJoCo FK 得到各 body 位姿/速度,合并成 `data/lib/lafan1_g1.npz` | 40 clips / 441,120 帧 = **2.45 h**,与论文 LAFAN1 受控实验规模一致 |
| 20:27 | `supermimic/env.py`:包装 mjlab G1 环境(`auto_reset=False`,自定义多 clip `LibCommand`,reset 模式 DR:摩擦/编码器偏置/躯干 COM),剥离 mjlab 的奖励/终止/观测,改用解析奖励 | 4096 环境 20 步 0.41 s;ξ(摩擦/偏置/COM)可从模型读回 |
| 20:26–20:31 | 实现 `state.py`(297 维状态、航向规范化、BeyondMimic 跟踪奖励、终止特征)、`library.py`、`features.py`、`models.py`(世界模型 MLP 2048×3、编码器/解码器策略、critic 256-128)、`replay.py`、`sampler.py`(回报采样 50/40/10)、`train.py`(Algorithm A.1)、`evaluate.py` | 世界模型状态含 14 个被跟踪 body 的位姿/速度,奖励为解析头,终止为学习头 |
| 20:32 | 小规模调试跑通(512 env, 50 iter):采集→拟合世界模型→BPTT 策略更新→critic→评估全流程无报错 | 8 步预测误差 0.185→0.139,在下降 |
| 20:32 | **启动正式训练** `runs/supermimic_lafan_1gpu`:LAFAN1 40 clips, 4096 env, buffer 1000, H=32, k=8, UTD 0.4, 训练时长预算 3 h | 日志 `runs/supermimic_lafan_1gpu/{log,eval}.csv` |
| 20:34 | 发现后台启动命令随工具调用结束而退出，改用 `setsid nohup` 脱离终端重新启动 | 显存占用约 9.75 GB，世界模型预热正常进行 |
| 20:40 | 预热 300 iter 完成后进入正式迭代时触发 CUDA `index out of bounds` assert | 原因：当环境在库最后一个 clip 末帧 (`F-1`) 触发截断并处于 `pending` 状态时，下一步 `time_steps += 1` 变为 `F`，访问 `last_frame[idx]` 越界 |
| 20:41:27 | **用户提示词 #5（模型切换）**: `没有额度了，现在从 sonnet5.5 切换到 gemini4argon 都是medium 这个要记录到复现记录中，继续复现` | 模型从 Claude Sonnet 5.5 (Medium) 切换为 Gemini 4 Argon (Medium)，已记录到时间线并继续复现 |
| 20:42 | 修复 `supermimic/env.py` 与 `library.py` 中 `time_steps` 帧索引 clamp 边界，并修复 `replay.py` 环形缓冲区边界历史回溯 | 重新启动 `runs/supermimic_lafan_1gpu` 正式训练 |
| 20:44–20:47 | 编写受控基线 `supermimic/ppo_sim.py` (PPO-Sim, §4.1)、下游速度编码器任务 `supermimic/velocity_interface.py` (§4.5, Appendix B & G)、吞吐量基准 `scripts/benchmark_throughput.py` (Fig. 5a) 与绘图脚本 `scripts/plot_results.py` | 补齐论文全部核心实验模块 |
| 20:56 | `runs/supermimic_lafan_1gpu` 训练至 iter 675 (10.3 min): 8步世界模型误差降至 `0.0585`，训练片段存活完成率从 `<0.5%` 跃升至 `19.5%` (`term_rate: 0.997 -> 0.8045`)，单步归一化奖励回升至 `0.3589` | 训练曲线稳定快速收敛 |
| 20:58:24 | **用户提示词 #6**: `我的所有提示词都要写到时间线文档中` | 新增“用户提示词全量记录”专节，并在时间线主表中逐字完整记录全部用户提示词 |
| 20:58 | `runs/supermimic_lafan_1gpu` 训练至 iter 850 (13.0 min): `wm_err8` 进一步降至 `0.0303`，`wm_bce` 降至 `0.00031`，单步归一化奖励升至 `0.4406`，`term_rate` 降至 `0.6951` | 即将到达 15 min 首个全量 40-clip 评估检查点 |
| 21:00 | **首个 15 min 全量 40-clip 评估检查点 (`iter=985, train_h=0.2502`)**：40 个 LAFAN1 动作序列中 **29/40 成功跑完不摔 (`failed=11/40`)**，全片段归一化回报 `norm_return = 0.5529`；至 `iter=1025` 时 `wm_err8` 降至 `0.0202`，训练单步归一化奖励达 `0.4962` | 已自动保存检查点 `runs/supermimic_lafan_1gpu/ckpt.pt` 并更新图表 |
| 21:07 | `runs/supermimic_lafan_1gpu` 训练至 `iter=1425` (`train_h=0.3598 h = 21.6 min`)：`wm_err8` 稳定在 `0.0182–0.0222`，critic 损失 `v_loss` 降至 `0.0204`，训练单步归一化奖励稳步升至 **`0.5689`** | 继续推进至 30 min (`train_h=0.50`) 第二个全量 40-clip 评估检查点 |
| 21:16 | **第二个 30 min 全量 40-clip 评估检查点 (`iter=1995, train_h=0.5003`)**：40 个 LAFAN1 动作序列中 **33/40 成功跑完不摔 (`failed=7/40`)**，全片段归一化回报大幅跃升至 **`norm_return = 0.7136`**；至 `iter=2100` (`train_h=0.5265`) 时训练单步归一化奖励达 **`0.6548`**，`v_loss` 降至 `0.0138` | 继续推进至 45 min (`train_h=0.75`) 第三个全量 40-clip 评估检查点 |
| 21:31 | **第三个 45 min 全量 40-clip 评估检查点 (`iter=3007, train_h=0.7505`)**：全片段归一化回报进一步提升至 **`norm_return = 0.7537`**（32/40 全程不摔，`failed=8/40`）；至 `iter=3025` 时训练单步归一化奖励升至 **`0.6900`**，`v_loss` 降至 `0.0091` | 保存主模型检查点 `runs/supermimic_lafan_1gpu/ckpt.pt`，启动全时长无截断评估与吞吐量基准 (`scripts/benchmark_throughput.py`) |
| 21:36 | **完成全时长无截断评估 (`runs/supermimic_lafan_1gpu/full_eval.json`) 与吞吐量基准 (`runs/benchmarks/throughput_fig5a.csv`, Fig. 5a)** | 全时长（最长 17,447 步 = 349 s）评估 `norm_return = 0.7309`（`dance1_subject2` 连续舞蹈 6,573 帧全程不摔）；世界模型相比 MuJoCo Warp 提速 **5.0×–6.6×**（`B=16384` 达 **184.6 万 step/s** vs 模拟器 **37.0 万 step/s**）；随后启动下游速度控制实验 (`supermimic/velocity_interface.py`) |
| 21:44–21:46 | 排查下游速度编码器首次纯离线训练 `J_rl` 爆炸问题：按论文 Appendix G.1 过滤 LAFAN1 的 18 个步行/奔跑片段 (`walk/run/sprint`)，加入在线仿真交互与世界模型同步微调、Critic 5 步更新及站立帧 4 次重复阶跃响应评估 (Appendix G.2) | 修复完成并重新启动 `supermimic/velocity_interface.py` (`--iters 800`) |
| 21:51 | **完成下游速度控制接口实验 (`runs/velocity_interface/velocity_results.pt`, §4.5 / Fig. 7 / Appendix G.2)** | 在 3 个速度命令平面的 **242 个阶跃响应命令（每个命令 4 次试验）** 上，**通过世界模型 BPTT 训练的 RL 速度编码器在 205/242 (`84.7%`) 个命令上优于蒸馏基线**（蒸馏仅 11/242 占优）；RL 编码器 **0 次全部摔倒 (`falls=0/242`)**、中位相对误差 **`0.498`**，而蒸馏基线在分布外转向/平移组合命令上 **194/242 摔倒**、中位误差 `0.900`，完美复现论文 §4.5 结论 |
| 21:52 | 生成 `docs/figures/{fig1_training_curves,fig3_throughput,fig4_velocity_step_test}.png`，并启动 Critic 视界消融实验 (`H=16` 有 Critic vs `H=16` 无 Critic，`runs/ablation_h16_{critic,nocritic}`，§4.3 / Fig. 4b,c) | 后台顺序运行 650 iter × 2 |
| 22:01 | **`H=16` 有 Critic 消融实验完成 (`runs/ablation_h16_critic`, `iter=650, train_h=0.1267 h`)**：仅用 650 iter (7.6 min) 即达到 **`norm_return = 0.4565`**、**29/40 序列成功 (`failed=11/40`)**，训练片段终止率降至 `0.6800`；自动继续运行 `H=16` 无 Critic (`runs/ablation_h16_nocritic`) | `ablation_h16_nocritic` 正在运行中（iter 150 时终止率仍高达 `0.9959`，初步验证短视界无 Critic 无法学会保持长期平衡） |
| 22:10 | **`H=16` 无 Critic 消融实验完成 (`runs/ablation_h16_nocritic`, `iter=650, train_h=0.1215 h`)**：在全量 40 个 LAFAN1 序列上 **40/40 全部摔倒失败 (`failed=40/40`, 0% 成功)**，平均仅存活片段长度的 **`2.3%`**，全片段归一化回报仅 **`0.0173`**（而 `H=16` 有 Critic 在相同 650 iter 下存活 **`81.6%`**、29/40 成功、全片段归一化回报 **`0.3710`**，提升 **21.4×**），完整复现论文 §4.3 (Fig. 4b,c) 结论：无 Critic 时短视界策略贪婪追踪眼前动作而丧失长期平衡 | 已更新 `fig2_critic_ablation.png` |
| 22:12 | 启动受控无模型基线 `PPO-Sim` (`supermimic/ppo_sim.py`, `runs/ppo_sim_lafan_1gpu`, `--time_budget_h 0.25 --eval_every_min 5.0`)，对比相同单卡墙钟时间下 `SuperMimic` 与 `PPO-Sim` 的样本效率与墙钟效率 (§4.1 / Fig. 4a) | 后台运行 15 min 中 |
| 22:21:20 | **用户提示词 #7**: `全部做完之后要生成一个视频，展示整个时间线，每个阶段做了什么，以及如果有的话还要有必要的视频演示` | 编写 `scripts/generate_video.py`，在全部实验完成后生成高清复现全流程时间线与 MuJoCo G1 人形机器人动作演示视频 `docs/supermimic_reproduction_overview.mp4` |
| 22:28 | **完成 `PPO-Sim` 15 min 受控基线 (`runs/ppo_sim_lafan_1gpu`, `iter=637, train_h=0.2501 h`)**：消耗 6,260 万仿真步后全量 40-clip 评估 `norm_return = 0.2376`、`failed = 24/40`（对比 `SuperMimic` 在相同 15 min 墙钟时间仅消耗 1,008 万仿真步即达 `norm_return = 0.5529`、`failed = 11/40`，30 min 达 `0.7136`、`failed = 7/40`，45 min 达 `0.7537`） | 完整复现论文 §4.1 (Fig. 4a) 结论：世界模型解析一阶路径梯度在多动作库下显著优于无模型零阶策略梯度 |
| 23:02:40 | **用户提示词 #8**: `好了吗` | 运行 `scripts/plot_results.py` 更新全部最终图表，并运行 `scripts/generate_video.py` 渲染生成 1280×720 HD 复现时间线与 3D 动作演示视频 `docs/supermimic_reproduction_overview.mp4` |
| 23:04 | **全部实验、图表、时间线文档与 1280×720 HD 演示视频生成完毕** | 视频保存于 `docs/supermimic_reproduction_overview.mp4`；图表保存于 `docs/figures/fig1..fig4.png` |

---

## 4. 核心实验定量结果与图表汇总 (Experimental Results)

### 4.1 主实验：SuperMimic vs. PPO-Sim (LAFAN1 40 Clips = 2.45 小时，单卡 RTX 4090，复现 §4.1 / Fig. 4a)

| 方法 | 训练墙钟时间 | 迭代数 (`iter`) | 环境交互步数 (`env_steps`) | 40-Clip 评估回报 (`norm_return`) | 成功序列数 / 总数 | 失败序列数 (`n_failed`) |
|---|---|---|---|---|---|---|
| `PPO-Sim` (无模型基线, §4.1) | 5 min (`0.084 h`) | 218 | 21.4M | `0.0161` | 0 / 40 (`0.0%`) | 40 |
| `PPO-Sim` (无模型基线, §4.1) | 10 min (`0.167 h`) | 433 | 42.6M | `0.1318` | 3 / 40 (`7.5%`) | 37 |
| `PPO-Sim` (无模型基线, §4.1) | 15 min (`0.250 h`) | 637 | 62.6M | `0.2376` | 16 / 40 (`40.0%`) | 24 |
| **`SuperMimic (Ours, H=32)`** | **15 min (`0.250 h`)** | **985** | **10.1M** | **`0.5529`** | **29 / 40 (`72.5%`)** | **11** |
| **`SuperMimic (Ours, H=32)`** | **30 min (`0.500 h`)** | **1995** | **20.7M** | **`0.7136`** | **33 / 40 (`82.5%`)** | **7** |
| **`SuperMimic (Ours, H=32)`** | **45 min (`0.751 h`)** | **3007** | **31.0M** | **`0.7537`** | **32 / 40 (`80.0%`)** | **8** |

![Training Curves](docs/figures/fig1_training_curves.png)

### 4.2 消融实验：Critic 与想象视界 (§4.3 / Fig. 4b,c)

| 配置 | 迭代数 | 训练片段存活率 (`1 - term_rate`) | 全片段长度归一化回报 (`ep_norm_return`) | 平均存活片段比例 | 40-Clip 成功率 |
|---|---|---|---|---|---|
| `H=16, No Critic` (`--no_critic`) | 650 | `0.36%` | `0.0173` | `2.3%` | `0 / 40 (0.0%)` |
| **`H=16, With Critic`** | **650** | **`32.00%`** | **`0.3710` (21.4×)** | **`81.6%`** | **`29 / 40 (72.5%)`** |
| **`H=32, With Critic` (主模型)** | **3007** | **`53.25%`** | **`0.6050` (35.0×)** | **`80.3%`** | **`32 / 40 (80.0%)`** |

![Critic Ablation](docs/figures/fig2_critic_ablation.png)

### 4.3 吞吐量基准：可微世界模型 vs. MuJoCo Warp (§4.2 / Fig. 5a)

| 并行环境数 (`B`) | 世界模型 (`wm_sps`, step/s) | MuJoCo Warp (`sim_sps`, step/s) | 加速比 (`Speedup`) |
|---|---|---|---|
| 512 | 457,099 | 71,682 | **6.38×** |
| 1024 | 881,811 | 133,739 | **6.59×** |
| 2048 | 1,348,952 | 209,302 | **6.44×** |
| 4096 | 1,590,821 | 281,080 | **5.66×** |
| 8192 | 1,711,354 | 335,079 | **5.11×** |
| 16384 | **1,846,085** | 370,115 | **4.99×** |

![Throughput Benchmark](docs/figures/fig3_throughput.png)

### 4.4 下游速度控制接口阶跃响应测试 (§4.5 / Fig. 7 / Appendix G.2)

| 方法 | 242 个阶跃命令胜出数 | 全部摔倒命令数 (`falls / 242`) | 稳态相对速度中位误差 (`median_err`) |
|---|---|---|---|
| 动作蒸馏基线 (`Distillation`) | 11 / 242 (`4.5%`) | 194 / 242 | `0.900` |
| **世界模型 BPTT 训练 (`RL Velocity Encoder`)** | **205 / 242 (`84.7%`)** | **0 / 242** | **`0.498`** |

![Velocity Step Test](docs/figures/fig4_velocity_step_test.png)

---

## 5. 方法概述与代码结构 (Method Overview & Repository Structure)

SuperMimic 在单一 Dyna 式循环中交替执行三个阶段训练全身人形跟踪控制器（论文 Figure 2 & Algorithm A.1）：

1. **模拟器数据采集 (`Collect`)**：
   - 在 $N=4096$ 个并行 `mjlab` (MuJoCo Warp) Unitree G1 环境中执行带探索噪声 $\epsilon_t \sim \mathcal{N}(0, 0.1^2)$ 的策略 $\pi_\theta(o_t, g_t)$。
   - 重置采样遵循 Appendix D.1 的**基于回报的重置采样**：50% 均匀采样、40% 按 $\text{softmax}_b(-25\,G_b / G_{\max})$ 采样、10% 按历史 softmax 权重采样。
2. **拟合可微世界模型 (`Fit World Model`, §3.2, Eq. 1)**：
   - 采样 10 批 $2048$ 个长度为 $k=8$ 的窗口，在航向规范化（heading-canonical）坐标系下自回归推演 $f_\omega(\hat{s}_t, a_t, \xi)$，并联合训练终止概率头 $\hat{d}_{t+1}$。
3. **想象推演与解析路径梯度更新 (`Improve Policy & Critic`, §3.3, Eq. 2–3)**：
   - 从缓冲区抽取 $B=4096$ 个起点，通过冻结的世界模型推演 $H=32$ 步，利用解析式 BeyondMimic 奖励头与 $V_\psi(\hat{s}_H, g_H, \xi)$ 价值自举，通过 BPTT 直接反传一阶解析策略梯度：
     $$\hat{J}(\theta) = \mathbb{E}_{s_0 \sim \rho_0, \pi_\theta}\!\left[\sum_{t=0}^{H-1} \gamma^t \hat{c}_t \hat{r}_{t+1} + \gamma^H \hat{c}_H V_\psi(\hat{s}_H, g_H, \xi)\right], \quad \hat{c}_{t+1} = \hat{c}_t (1 - \hat{d}_{t+1}).$$

| 文件路径 | 功能说明 |
|---|---|
| [`supermimic/geom.py`](supermimic/geom.py) | 可微 $SO(3)$ 旋转工具：四元数 $\leftrightarrow$ 旋转矩阵、6D 连续旋转表示、测地线角度误差 |
| [`supermimic/state.py`](supermimic/state.py) | 297 维世界模型状态定义（14 个跟踪刚体 + 29 关节 + 上一步动作）、航向规范化、解析式 BeyondMimic 奖励 (Table A.2) 与终止条件 (Appendix A) |
| [`supermimic/library.py`](supermimic/library.py) | 50 Hz 多片段参考动作库加载器、$K=10$ 未来参考帧索引与 1 秒时间分箱 |
| [`supermimic/env.py`](supermimic/env.py) | 基于 `mjlab` 的矢量化 Unitree G1 环境封装 (`SimEnv`)，支持重置域随机化 $\xi$ (Table A.3) 与完整物理状态提取 |
| [`supermimic/features.py`](supermimic/features.py) | 93 维本体感觉观测 (Table A.1)、传感器噪声注入 $\eta_t$ 与 $K=10$ 未来参考目标特征 |
| [`supermimic/models.py`](supermimic/models.py) | 网络结构 (Table A.4)：`WorldModel` MLP(2048×3)、`Policy`（运动编码器 + 共享解码器）、`Critic` MLP(256, 128) |
| [`supermimic/replay.py`](supermimic/replay.py) | GPU 环形经验回放池 ($1000 \times 4096$)，支持 $k=8$ 窗口采样与 10 帧观测历史种子采样 |
| [`supermimic/sampler.py`](supermimic/sampler.py) | 基于回报的分箱重置采样器 (Appendix D.1: 50% 均匀 + 40% 当前回报 softmax + 10% 历史权重) |
| [`supermimic/evaluate.py`](supermimic/evaluate.py) | 基准评估协议 (Appendix F.1)：全部 40 个序列从首帧确定性推演、摩擦力 1.0、无噪声 |
| [`supermimic/train.py`](supermimic/train.py) | SuperMimic 完整训练主循环 (Algorithm A.1 & Table A.5) |
| [`supermimic/ppo_sim.py`](supermimic/ppo_sim.py) | 受控无模型基线 `PPO-Sim` (§4.1, Fig. 4a) |
| [`supermimic/velocity_interface.py`](supermimic/velocity_interface.py) | 下游速度控制接口实验 (§4.5, Appendix B & G, Table A.6)：世界模型 RL 速度编码器 vs. 动作蒸馏基线 |
| [`scripts/convert_lafan.py`](scripts/convert_lafan.py) | 将 LAFAN1 G1 CSV 动作插值至 50 Hz 并通过 MuJoCo FK 构建 `data/lib/lafan1_g1.npz` (40 clips, 441,120 帧 = 2.45 h) |
| [`scripts/benchmark_throughput.py`](scripts/benchmark_throughput.py) | 复现 Figure 5(a)：世界模型与 MuJoCo Warp 吞吐量对比基准 |
| [`scripts/plot_results.py`](scripts/plot_results.py) | 生成 `docs/figures/` 下的全部论文复现对比图 |
| [`scripts/generate_video.py`](scripts/generate_video.py) | 渲染生成 1280×720 HD 复现时间线与 3D 动作演示视频 [`docs/supermimic_reproduction_overview.mp4`](docs/supermimic_reproduction_overview.mp4) |
