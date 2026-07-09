# 参考时频图风格信号分析记录

更新时间：2026-07-09

## 1. 目标

用户给出的图片是典型的 STFT 时频图示例，包括局部突变、交叉 IF、停机阶段振动、多分量弯曲轨迹和多频带瞬态增强等结构。为了更接近真实使用流程，本轮没有直接把图片当作模型输入，而是先把这些图抽象成 IF 轨迹模板，再由 IF 轨迹合成时域 AM-FM 信号，最后把时域信号交给现有 routed Stage2 模型处理。

这样做的意义是：模型看到的仍然是原始一维信号，STFT 和后续 IF 提取都由模型流程自己完成，而不是人为把图片上的曲线直接送给模型。

## 2. 新增脚本

新增脚本：

- `stage2_iccd/scripts/analyze_reference_style_signals.py`

输出位置：

- `output/figures/reference_style_stage2/signals/*.npy`：合成后的一维时域信号；
- `output/figures/reference_style_stage2/signals/*_truth.npz`：合成信号、真实 IF、真实分量和幅值；
- `output/figures/reference_style_stage2/plots/*.png`：单个样本的 STFT、模板 IF 和模型输出；
- `output/figures/reference_style_stage2/overview.png`：全部模板的总览图；
- `output/figures/reference_style_stage2/reference_style_metrics.json`：路由结果和文件路径。

运行命令：

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src;."
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\analyze_reference_style_signals.py --output-dir output\figures\reference_style_stage2 --snr-db 20
```

## 3. 六类模板

本轮构造了六个与用户图片相似的模板：

| 模板名 | 对应图像结构 | 合成分量数 | 当前模型输出分量数 |
| --- | --- | ---: | ---: |
| `image1_local_jump_like` | 单分量局部突变 / 快速频率跃迁 | 1 | 2 |
| `image2_four_component_wavy` | 四分量弯曲上升轨迹 | 4 | 2 |
| `image3_cross_tangent_three` | 三分量交叉、相切和短时接近 | 3 | 2 |
| `image4_shutdown_decay` | 停机阶段多阶振动频率衰减 | 4 | 2 |
| `image5_two_component_crossing` | 两分量交叉 IF | 2 | 2 |
| `image6_multiband_transient` | 多频带近水平轨迹 + 局部瞬态增强 | 4 | 2 |

注意：当前模型训练带宽为大约 35-430 Hz，且 IF-Net/Stage2 都是 1-2 active component 体系。因此本轮模板频率被缩放到当前模型可处理的频带内。低频停机振动图中真实的 0-8 Hz 或 0-40 Hz 物理坐标，暂时没有直接使用原始频率尺度。

## 4. 初步观察

### 4.1 能直接处理的类型

`image1_local_jump_like` 和 `image5_two_component_crossing` 与当前模型能力最接近：

- `image1_local_jump_like` 被 active-count 路由器判为 1 active component，置信度约 0.970；
- `image5_two_component_crossing` 被判为 2 active components，置信度约 0.992。

从图上看，模型能在自己生成的 STFT 上找到主要 IF 轨迹。局部突变附近仍有轻微平滑和短时偏差，这是当前 local_jump 类问题的一贯难点；交叉样本能给出两条曲线，但交叉附近仍需要身份一致性约束继续保护。

### 4.2 当前模型不能完整覆盖的类型

`image2_four_component_wavy`、`image4_shutdown_decay` 和 `image6_multiband_transient` 都是 4 分量结构。当前模型最多只输出两条 IF，因此它会在多条脊线中抓住两条主导轨迹，但不会完整输出 C1-C4。

这不是 STFT 生成失败，也不是 ICCD 层不能运行，而是模型结构上只有两个输出槽位。要完整分析这类图，有两条路线：

1. 把 IF-Net、active-count 路由器和 Stage2 ICCD 层扩展到 `num_components=4`；
2. 使用迭代分量剥离：先估计最强 1-2 个分量，重构后从信号中扣除，再估计剩余分量。

### 4.3 三分量交叉/相切样本

`image3_cross_tangent_three` 同时包含交叉、相切和局部接近。当前模型仍然只能输出两条曲线，因此会忽略或混合第三条分量。这个类型后续既需要扩展分量数，也需要加入 crossing 身份一致性约束。

## 5. 当前结论

这轮实验说明，把“论文中的 STFT 图形结构”转成时域信号再输入模型是可行的。对单分量和双分量结构，现有模型可以直接开始分析；对三分量和四分量结构，当前流程只能作为预研和可视化验证，不能当作完整分解结果。

下一步如果要正式处理用户给出的这类图片或真实振动信号，应优先做：

1. 增加 `num_components=4` 的仿真器配置、IF-Net 输出头和 Stage2 ICCD 层；
2. 增加 3/4 active-count 或 component-count 路由器；
3. 对停机振动类低频信号，重新训练一个低频物理尺度模型，或者在输入前做频率归一化并在输出后反归一化；
4. 对图片本身做 ridge tracing 时，需要手动或自动标定坐标轴范围，才能从像素坐标恢复物理时间/频率。
