# 第二阶段阶段性训练与分场景评估总结

更新时间：2026-07-08

## 1. 本轮目标

本轮目标不是直接把所有信号类型一次性强行训练到最好，而是按稳定性逐步推进：

1. 先从较简单、较稳定的信号开始，例如线性 chirp、二次多项式 chirp、三次多项式 chirp。
2. 如果简单场景下 IF-Net + 可微 ICCD 层能够稳定重构，再扩展到全部类型。
3. 对全部类型做分场景评估，找出哪些类型已经满足第二阶段继续推进的需要，哪些类型还需要单独优化。
4. 对困难类型尝试专家模型或专门配置，判断是否能通过更长训练解决。

## 2. 已完成的代码工作

本轮在第二阶段代码中补充了分场景评估和多种训练配置：

- 新增 `src/stage2_iccd/eval_scenarios.py`，用于对指定 checkpoint 按信号类型分别评估。
- 新增 `configs/separated_frozen.yaml`，用于简单分离场景的课程训练。
- 新增 `configs/all_multiexpert.yaml`，用于多 IF-Net 专家候选输入的全类型训练实验。
- 新增 `configs/local_jump_frozen.yaml`，用于 local_jump 专门训练。
- 新增 `configs/sinusoidal_frozen.yaml`，用于 sinusoidal_fm 专门训练。
- 修改 `model.py` 中的候选 IF 混合器：
  - 原先是全局可学习平均，容易把不同专家输出的 IF 曲线直接平均成不存在的轨迹。
  - 现在增加基于 ICCD 初步重构残差的动态候选权重，让每个样本根据重构误差选择更合适的候选。
- 修改训练日志，记录候选权重均值和候选选择温度。

## 3. 训练路径与主要结果

### 3.1 简单分离场景训练

训练对象：linear、quadratic、cubic。

使用配置：`configs/separated_frozen.yaml`

训练后 checkpoint：`stage2_iccd/runs/separated_frozen/latest.pt`

分场景评估结果：

| 场景 | 重构 SNR / dB | IF MAE / Hz |
| --- | ---: | ---: |
| linear | 23.40 | 2.38 |
| quadratic | 23.02 | 2.47 |
| cubic | 21.64 | 3.55 |
| aggregate | 22.69 | 2.80 |

结论：简单场景训练是稳定的。说明第二阶段的可微 ICCD 层、alpha 学习、候选 IF 权重和小型 IF refinement head 都能正常参与训练，并且没有出现明显梯度或重构不稳定问题。

### 3.2 从简单场景扩展到全部类型

使用 `separated_frozen` 作为初始化，继续训练全部类型。

训练后 checkpoint：`stage2_iccd/runs/all_from_separated/latest.pt`

这是目前最稳的通用第二阶段 checkpoint。

分场景评估结果：

| 场景 | 重构 SNR / dB | IF MAE / Hz | 判断 |
| --- | ---: | ---: | --- |
| linear | 21.18 | 3.26 | 可接受 |
| quadratic | 21.76 | 2.40 | 较好 |
| cubic | 21.52 | 3.22 | 可接受 |
| sinusoidal_fm | 17.67 | 5.58 | 偏弱 |
| crossing | 19.97 | 3.03 | 可接受 |
| near_parallel | 23.35 | 1.71 | 最稳定 |
| local_jump | 16.17 | 6.70 | 明显偏弱 |
| tangent_or_overlap | 21.11 | 3.13 | 可接受 |
| aggregate | 20.34 | 3.63 | 总体可用 |

结论：全类型一起推进是可行的，但不是所有类型同步受益。当前通用模型对 linear、quadratic、cubic、crossing、near_parallel、tangent_or_overlap 的表现已经比较稳定；主要短板集中在 local_jump 和 sinusoidal_fm。

## 4. 困难类型的补充实验

### 4.1 sinusoidal_fm 专门模型

训练后 checkpoint：`stage2_iccd/runs/sinusoidal_frozen/latest.pt`

评估结果：

| 场景 | 重构 SNR / dB | IF MAE / Hz |
| --- | ---: | ---: |
| sinusoidal_fm | 20.28 | 4.05 |

相对通用模型：

- 通用模型：17.67 dB，5.58 Hz
- sinusoidal 专门模型：20.28 dB，4.05 Hz

结论：sinusoidal_fm 可以通过专门训练明显改善。后续可以保留通用模型，也可以在路由器足够稳定后为 sinusoidal_fm 使用专门分支。

### 4.2 local_jump 专门模型

训练后 checkpoint：`stage2_iccd/runs/local_jump_frozen/latest.pt`

评估结果：

| 场景 | 重构 SNR / dB | IF MAE / Hz |
| --- | ---: | ---: |
| local_jump | 15.15 | 8.34 |

相对通用模型：

- 通用模型：16.17 dB，6.70 Hz
- local_jump 专门模型：15.15 dB，8.34 Hz

结论：local_jump 不是简单换成 local_jump IF-Net checkpoint 或继续训练 ICCD 层就能解决。它的误差主要来自跳变位置和跳变前后 IF 段的结构没有被第二阶段显式建模。当前的 ICCD refinement head 更擅长连续、小幅、平滑修正，对突变点附近的非平滑 IF 变化表达能力不足。

## 5. 多专家候选混合实验

本轮尝试了多 IF-Net 专家候选输入，包括 balanced、polynomial、sinusoidal、local_jump 等多个第一阶段 checkpoint。

### 5.1 全局平均式候选混合

结果较差：

- validation 重构 SNR 约 13.45 dB
- validation IF MAE 约 14.40 Hz

主要原因：不同专家输出的 IF 曲线不一定是同一个物理分量的同一个候选。直接做全局可学习平均时，可能把几条合理曲线平均成一条不真实的中间曲线，尤其在 crossing、local_jump、相切或短时重合场景中更明显。

### 5.2 基于 ICCD 残差的动态候选混合

动态混合后，模型会根据每个候选 IF 的初步 ICCD 重构残差来给权重。

结果仍不适合作为主路径：

- aggregate 重构 SNR 约 16.54 dB
- aggregate IF MAE 约 11.45 Hz
- crossing 出现明显退化，重构 SNR 约 4.32 dB，IF MAE 约 30.43 Hz

原因：ICCD 残差本身并不总能区分“物理身份正确的 IF”和“短时间内也能解释能量但身份错误的 IF”。在 crossing 场景中，候选曲线可能局部重构误差不大，但分量身份已经交换，导致后续 ICCD 展开层沿错误轨迹分解。

结论：多专家不是不能用，但不能只靠残差选择。后续如果使用多专家，需要加入场景判别器或身份一致性约束，让路由器先判断信号结构，再决定使用哪个专家或怎样融合 top-k 候选。

## 6. 当前最佳使用建议

当前第二阶段最稳组合：

- 通用模型：`stage2_iccd/runs/all_from_separated/latest.pt`
- sinusoidal_fm 专门模型：`stage2_iccd/runs/sinusoidal_frozen/latest.pt`

建议：

1. 一般场景优先使用 `all_from_separated`。
2. 如果前端判别器较确定是 sinusoidal_fm，可以切换到 `sinusoidal_frozen`。
3. 暂时不要把 `all_multiexpert_dynamic` 作为主模型，因为它在 crossing 上不稳定。
4. local_jump 暂时继续使用通用模型，直到第二阶段加入显式跳变位置建模。

## 7. 对第二阶段任务的影响

第二阶段的核心目标是用可微 ICCD 展开层把第一阶段的 IF 初始估计转化为可重构、可反传、可联合优化的分量分解过程。

从当前结果看：

- 简单场景和多数连续 IF 场景已经能支撑第二阶段继续推进。
- crossing 当前通用模型表现可接受，但多专家路由会引入身份交换风险，因此后续必须保留 top-2 候选和身份一致性检查。
- local_jump 会影响第二阶段的端到端精修，因为跳变点附近 IF 误差会直接导致相位积分误差，进而影响 ICCD 字典构造和分量重构。
- sinusoidal_fm 需要专门路由或更强的周期性 IF refinement，但问题比 local_jump 更容易通过专门模型缓解。

## 8. 下一步优化方向

优先级建议如下：

1. 为 local_jump 增加跳变位置辅助输入或辅助头。
   - 第一阶段已经有跳变事件定位思路，但第二阶段目前没有显式使用它。
   - 应把 jump position、jump confidence 或 jump mask 输入到 refinement head。
   - refinement head 在跳变点附近允许更大、更局部的 IF 修正。

2. 把 ICCD 展开层改成局部/分段式 IF refinement。
   - 对 local_jump，不能只用全局平滑修正。
   - 可以在跳变点前后分别建模 IF 曲线或分别设置正则强度。

3. 为 crossing 加身份一致性约束。
   - 保留 top-2 候选。
   - 在时间轴上限制分量身份频繁交换。
   - 训练损失中加入轨迹连续性或最优匹配约束。

4. 重新设计多专家路由器。
   - 不建议只用 ICCD 残差。
   - 应结合信号类型判别器、候选置信度、身份一致性和局部重构误差。

5. 在通用模型稳定后，再逐步解冻 IF-Net。
   - 目前仍建议先固定第一阶段 IF-Net，只训练 alpha、候选权重和 refinement head。
   - 等 local_jump 和 crossing 的第二阶段机制补足后，再做端到端联合训练。

## 9. 当前是否可以继续第二阶段

可以继续第二阶段，但需要分层推进：

- 对 linear、quadratic、cubic、near_parallel、tangent_or_overlap、crossing：可以继续使用当前通用模型推进可微 ICCD 展开层和端到端训练框架。
- 对 sinusoidal_fm：可以使用专门模型作为临时分支，并继续优化路由。
- 对 local_jump：不建议直接进入大规模联合训练，应先补充跳变位置辅助头或分段式 refinement，否则端到端训练可能把误差传回 IF-Net，造成不稳定的错误修正。

因此，当前判断是：第二阶段框架已经成立，通用模型基本可用；但 local_jump 需要作为下一轮重点结构优化，而不是只依赖更长训练。
