# A6_POST：方法与实现

## 1. 输入和基础动作

输入是两帧观测：世界坐标点云 `[B, 2, 4096, 6]`（XYZRGB）与机器人状态 `[B, 2, 16]`（9 维关节状态 + 7 维 TCP 位姿），以及点有效性掩码。数据读取器从已有 Zarr 读取 `data/point_cloud`、`data/state`、`data/action`、`data/point_valid_mask` 和 `meta/episode_ends`。

`AQRDP3` 继承 `SimpleDP3`。`_conditional_sample_global_only()` 用全局观测条件完成 DDIM 采样，产生归一化动作序列 `[B, 12, 8]`。其中第 0 槽用于观测对齐，第 1–8 槽为可执行动作槽。历史在线评估每次只执行前 4 步，再重新规划。

## 2. 动作决定几何查询位置

`_build_local_query()` 将最终基础动作作为查询条件。`ActionToTool` 和 `TorchCandidateTrajectoryLift` 结合动作归一化约定、控制器线性响应及 Panda 正向运动学，得到候选工具轨迹。

`RelativeMultiScaleQuery` 在 1 / 3 / 5 / 10 cm 邻域内提取局部信息。关系特征包含相对位置、距离、表面法向，以及表面朝向和运动方向的关系：`r_face = -n · d_hat`、`r_motion = n · v_hat`。锚点身份保留用于区分当前/候选位置、左右指尖及扫掠点。

当前版使用 convex 关系融合，未开启动作条件 attention，也没有额外 relation gain。代码里的可选模块并不等于本版全部启用。

## 3. 一次性动作修正

`PostDiffusionActionRefiner.forward()` 编码基础动作与时间槽身份，并结合全局条件、局部关系 token。残差采用：

`delta = head(base_context + local_context) - head(base_context)`

该结构使零局部输入产生零残差，另有显式的局部存在掩码。置信度通过 sigmoid 得到。

`bound_action_residual()` 将机械臂动作修正限制在归一化空间的最大范数 0.30 内，保留至少 0.50 的基础动作前进比例，并保护非机械臂动作维度。这些是动作约束，不构成任务成功保证。随后执行：

`corrected = base + confidence * bounded_delta`

AQR 在完整扩散序列完成后只调用一次；不会在每一个去噪步都重新执行该修正。

## 4. 训练路径

`_compute_post_diffusion_loss()` 使用与推理一致的采样流程生成候选动作，但不通过采样循环反向传播。训练时可以在完成采样后加入动作扰动，候选动作仍经过同一套查询与修正路径。

训练目标组合残差目标损失、修正动作损失与置信度损失，权重分别为 1.0、1.0、0.10。本配置 `post_base_frozen=false`，因此还优化全局扩散损失。置信度目标的构造和有效槽掩码以该函数实现为准。

## 5. 工程入口

训练入口为 `contactflow/tools/train_aqr_dp3.py`，在线交互入口为 `contactflow/tools/evaluate_aqr_dp3_maniskill.py`。它们用于展示数据批次、模型、优化器、检查点和环境交互之间的连接。

从原数据准备脚本抽出的运行时函数现在位于：

- `contactflow/dp3/aqr_dp3/observation.py`：在线点云裁剪、采样与掩码对齐。
- `contactflow/dp3/aqr_bc/urdf.py`：从安装的 ManiSkill 资源加载 Panda URDF。

这两部分是推理所需的观测与运动学处理，不生成训练数据集。
