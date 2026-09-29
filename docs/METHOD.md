#：方法与实现

## 1. 输入和基础动作

输入是历史观测：世界坐标 XYZRGB 点云与关节状态、TCP 位姿，以及点有效性掩码。

`AQRDP3` 继承 `SimpleDP3`。`_conditional_sample_global_only()` 用全局观测条件完成 DDIM 采样，产生归一化动作序列。序列区分观测对齐槽与可执行槽；执行部分动作后再重新规划。

## 2. 动作决定几何查询位置

`_build_local_query()` 将最终基础动作作为查询条件。`ActionToTool` 和 `TorchCandidateTrajectoryLift` 结合动作归一化约定、控制器线性响应及 Panda 正向运动学，得到候选工具轨迹。

`RelativeMultiScaleQuery` 在多个尺度的邻域内提取局部信息。关系特征包含相对位置、距离、表面法向，以及表面朝向和运动方向的关系：`r_face = -n · d_hat`、`r_motion = n · v_hat`。锚点身份保留用于区分当前/候选位置、左右指尖及扫掠点。

当前版使用 convex 关系融合，未开启动作条件 attention，也没有额外 relation gain。代码里的可选模块并不等于本版全部启用。

## 3. 一次性动作修正

`PostDiffusionActionRefiner.forward()` 编码基础动作与时间槽身份，并结合全局条件、局部关系 token。残差采用：

`delta = head(base_context + local_context) - head(base_context)`

该结构使零局部输入产生零残差，另有显式的局部存在掩码。置信度通过 sigmoid 得到。

`bound_action_residual()` 将机械臂动作修正限制在归一化空间的最大范数内，并约束基础动作的前进比例，并保护非机械臂动作维度。这些是动作约束，不构成任务成功保证。随后执行：

`corrected = base + confidence * bounded_delta`

AQR 在完整扩散序列完成后只调用一次；不会在每一个去噪步都重新执行该修正。

## 4. 训练路径

`_compute_post_diffusion_loss()` 使用与推理一致的采样流程生成候选动作，但不通过采样循环反向传播。训练时可以在完成采样后加入动作扰动，候选动作仍经过同一套查询与修正路径。

训练目标组合残差目标损失、修正动作损失与置信度损失。全局模型可通过扩散损失参与联合训练。置信度目标的构造和有效槽掩码以该函数实现为准。
