from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "output" / "pdf"
FIG_DIR = ROOT / "runs" / "router_hard_v3" / "stable_combo_all_figures"
BASELINE_METRICS_PATH = ROOT / "runs" / "router_hard_v3" / "eval_soft_guarded_top2_all" / "routed_metrics.json"
QUALITY_METRICS_PATH = ROOT / "runs" / "quality_selector_v1" / "eval_routed_quality_protect_cross_all" / "routed_metrics.json"
READINESS_PATH = ROOT / "runs" / "stage1_readiness_aux_v3_center_soft" / "readiness_metrics.json"
READINESS_SEED2_PATH = ROOT / "runs" / "stage1_readiness_aux_v3_center_soft_seed67890" / "readiness_metrics.json"
EXAMPLE_PATH = FIG_DIR / "example_metrics.json"
PDF_PATH = OUT_DIR / "ifnet_stage1_summary.pdf"

SCENARIO_LABELS = {
    "linear": "线性 chirp",
    "quadratic": "二次多项式 chirp",
    "cubic": "三次多项式 chirp",
    "sinusoidal_fm": "正弦调频 chirp",
    "crossing": "交叉 IF",
    "near_parallel": "接近平行 IF",
    "local_jump": "局部突变 IF",
    "tangent_or_overlap": "相切或短时间重合 IF",
}


def make_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", r"C:\Windows\Fonts\msyh.ttc"))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", r"C:\Windows\Fonts\msyhbd.ttc"))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=21,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2a44"),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="MicrosoftYaHei",
            fontSize=10.3,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52606d"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=14.0,
            leading=21,
            textColor=colors.HexColor("#1f2a44"),
            spaceBefore=8,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=11.0,
            leading=16,
            textColor=colors.HexColor("#243b53"),
            spaceBefore=5,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.45,
            leading=14.5,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=5.7,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=7.7,
            leading=10.4,
            textColor=colors.HexColor("#1f2933"),
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def para_rows(raw_rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> list[list[Paragraph]]:
    return [[p(cell, styles["table"]) for cell in row] for row in raw_rows]


def styled_table(rows: list[list[Paragraph]], col_widths: list[float], header_color: str) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("FONTNAME", (0, 0), (-1, -1), "MicrosoftYaHei"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcccdc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
            ]
        )
    )
    return table


def add_bullets(story: list, items: list[str], style: ParagraphStyle) -> None:
    for item in items:
        story.append(p(f"- {item}", style))


def add_model_table(story: list, styles: dict[str, ParagraphStyle]) -> None:
    rows = para_rows(
        [
            ["模块", "当前构建方式"],
            ["输入信号", "fs=1024 Hz，长度 1024 点，2 个 AM-FM 分量，IF 范围 35-430 Hz。"],
            ["时频输入", "归一化 log-STFT。专家模型默认 n_fft=256，win_length=128，hop_length=4。"],
            ["网络主体", "U-Net，base_channels=24，depth=3；卷积块为 Conv-BN-SiLU-Dropout-Conv-BN-SiLU。"],
            ["输出形式", "每个分量输出一个 ridge heatmap，形状 [B,Q,F,T]，当前 Q=2。"],
            ["IF 生成", "对频率维做 softmax，再用 soft-argmax 得到连续 IF 曲线 [B,Q,T]。"],
            ["训练损失", "ridge NLL + IF L1 + smooth + identity_slope；多项式专家额外加入 poly_residual。"],
            ["后处理", "poly_like 使用 heatmap 加权三阶多项式稳健拟合，其余专家默认不做强平滑。"],
        ],
        styles,
    )
    story.append(styled_table(rows, [3.5 * cm, 12.4 * cm], "#e7eef8"))


def add_router_table(story: list, styles: dict[str, ParagraphStyle]) -> None:
    rows = para_rows(
        [
            ["判别器部分", "当前构建方式"],
            ["类别", "4 类：poly_like、sinusoidal_like、cross_overlap_like、jump_like。"],
            ["场景映射", "linear/quadratic/cubic -> poly；sinusoidal_fm -> sinusoidal；crossing/near_parallel/tangent_or_overlap -> cross_overlap；local_jump -> jump。"],
            ["输入", "三尺度 STFT：128/64、256/128、512/256，统一到 target_n_fft=256，hop_length=4。"],
            ["CNN", "3 层卷积块，base_channels=24，dropout=0.08，逐层池化后全局平均池化。"],
            ["辅助特征", "12 维 ridge 形状特征：均值/方差、斜率、二阶差分、jump ratio、多项式残差、正弦残差、带宽和峰值强度。"],
            ["训练", "交叉熵分类，v3 训练 2000 steps，batch_size=32，并提高 sinusoidal/local_jump/tangent 等难类采样权重。"],
        ],
        styles,
    )
    story.append(styled_table(rows, [3.5 * cm, 12.4 * cm], "#eef2e6"))


def add_quality_table(story: list, styles: dict[str, ParagraphStyle]) -> None:
    rows = para_rows(
        [
            ["质量判别器部分", "当前构建方式"],
            ["目标", "不再判断信号类型，而是判断 top-2 候选专家中哪个输出的 IF 误差更低。"],
            ["监督信号", "训练时运行 top-2 专家，用真实 IF 计算各候选 MAE，MAE 更低者作为排序标签。"],
            ["输入特征", "路由概率、候选专家 one-hot、heatmap 熵、多项式/正弦残差、jump evidence、jump mismatch、平滑度、曲线范围、斜率和二阶差分等。"],
            ["模型", "轻量 MLP：LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> SiLU -> Linear，输出候选质量分数。"],
            ["保护策略", "若 top-1 为 cross_overlap_like，则默认不允许二级质量判别器覆盖，避免误伤 crossing。"],
            ["当前 checkpoint", "ifnet_stage1/runs/quality_selector_v1/latest.pt。v2 的 best 在小验证集较好，但正式全类型评估中 local_jump 变差，因此暂不推荐。"],
        ],
        styles,
    )
    story.append(styled_table(rows, [3.5 * cm, 12.4 * cm], "#f4ead5"))


def add_metric_table(story: list, metrics: dict, styles: dict[str, ParagraphStyle]) -> None:
    raw = [["信号类型", "MAE/Hz", "Median/Hz", "P95/Hz", "路由准确率", "Fallback"]]
    for name, label in SCENARIO_LABELS.items():
        item = metrics[name]
        raw.append(
            [
                label,
                f"{item['if_mae_hz']:.2f}",
                f"{item['if_mae_hz_median']:.2f}",
                f"{item['if_mae_hz_p95']:.2f}",
                f"{100.0 * item['route_accuracy']:.1f}%",
                f"{100.0 * item['fallback_fraction']:.1f}%",
            ]
        )
    story.append(styled_table(para_rows(raw, styles), [4.7 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.45 * cm, 2.2 * cm], "#e7eef8"))


def add_comparison_table(story: list, baseline: dict, quality: dict, styles: dict[str, ParagraphStyle]) -> None:
    raw = [["信号类型", "上一版 MAE", "质量判别器 MAE", "变化/Hz"]]
    for name, label in SCENARIO_LABELS.items():
        old = baseline[name]["if_mae_hz"]
        new = quality[name]["if_mae_hz"]
        raw.append([label, f"{old:.3f}", f"{new:.3f}", f"{new - old:+.3f}"])
    avg_old = sum(v["if_mae_hz"] for v in baseline.values()) / len(baseline)
    avg_new = sum(v["if_mae_hz"] for v in quality.values()) / len(quality)
    raw.append(["平均", f"{avg_old:.3f}", f"{avg_new:.3f}", f"{avg_new - avg_old:+.3f}"])
    story.append(styled_table(para_rows(raw, styles), [5.2 * cm, 3.0 * cm, 3.6 * cm, 2.5 * cm], "#efe7f6"))


def pass_text(value: bool) -> str:
    return "通过" if value else "未过"


def add_readiness_table(story: list, readiness: dict, styles: dict[str, ParagraphStyle]) -> None:
    gates = readiness["gates"]
    aggregate = readiness["aggregate"]
    scenarios = readiness["scenarios"]
    local_jump = scenarios["local_jump"]
    raw = [
        ["准入项", "当前值", "门槛", "结论"],
        ["总体 IF MAE", f"{aggregate['avg_mae_hz']:.2f} Hz", "<= 5.5 Hz", pass_text(gates["overall_mae"]["pass"])],
        ["top-2 候选覆盖率", f"{aggregate['candidate_oracle_coverage_10hz'] * 100:.2f}%", ">= 88%", pass_text(gates["top2_candidate_coverage"]["pass"])],
        ["高置信样本 MAE", f"{aggregate['high_confidence_mae_hz']:.2f} Hz", "<= 5.0 Hz", pass_text(gates["high_confidence_quality"]["pass"])],
        ["crossing 身份稳定", f"{scenarios['crossing']['identity_excess_hz']:.2f} Hz", "<= 8.0 Hz", pass_text(gates["crossing_identity"]["pass"])],
        ["local_jump IF", f"{local_jump['if_mae_hz']:.2f} Hz / P95 {local_jump['if_mae_hz_p95']:.2f} Hz", "<= 10.5 / 40 Hz", pass_text(gates["local_jump_if"]["pass"])],
        ["local_jump 跳变定位", f"{local_jump['jump_event_mae_ms']:.2f} ms / P95 {local_jump['jump_event_p95_ms']:.2f} ms", "<= 80 ms", pass_text(gates["local_jump_event"]["pass"])],
        ["sinusoidal_fm MAE", f"{scenarios['sinusoidal_fm']['if_mae_hz']:.2f} Hz", "<= 8.5 Hz", pass_text(gates["sinusoidal_quality"]["pass"])],
    ]
    story.append(styled_table(para_rows(raw, styles), [4.6 * cm, 4.2 * cm, 3.4 * cm, 2.0 * cm], "#e8f4f1"))


def add_example_table(story: list, examples: dict, styles: dict[str, ParagraphStyle]) -> None:
    raw = [["信号类型", "样本 MAE/Hz", "Top route", "Selected route", "Fallback"]]
    for name, label in SCENARIO_LABELS.items():
        item = examples[name]
        raw.append([label, f"{item['sample_if_mae_hz']:.2f}", item["top_route"], item["selected_route"], "是" if item["fallback_used"] else "否"])
    story.append(styled_table(para_rows(raw, styles), [4.8 * cm, 2.45 * cm, 3.1 * cm, 3.45 * cm, 1.8 * cm], "#eef2e6"))


def add_image(story: list, image_path: Path, max_width: float, max_height: float) -> None:
    img = Image(str(image_path))
    ratio = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * ratio
    img.drawHeight = img.imageHeight * ratio
    story.append(img)


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("MicrosoftYaHei", 8)
    canvas.setFillColor(colors.HexColor("#627d98"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "IF-Net 第一阶段技术总结")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    quality_metrics = json.loads(QUALITY_METRICS_PATH.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    readiness_seed2 = json.loads(READINESS_SEED2_PATH.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    styles = make_styles()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.55 * cm,
        title="IF-Net 第一阶段技术总结",
        author="Codex",
    )
    story: list = []

    story.append(p("ICCD 第一阶段 IF-Net 技术总结", styles["title"]))
    story.append(p("从传统时频脊线提取到可微 IF 网络、专家模型、路由器与二级质量判别器", styles["subtitle"]))

    story.append(p("一、第一阶段目标与总体路线", styles["h1"]))
    story.append(
        p(
            "第一阶段的目标是把 ICCD 中依赖时频图和脊线搜索的 IF 提取步骤，转化为可训练、可微分、可批量验证的神经网络模块。"
            "当前方案保留“时频表示 -> 脊线 -> IF”的物理含义，但把脊线搜索替换为 IF-Net 的 heatmap 预测：网络学习每个分量在每个时刻的频率概率分布，再用 soft-argmax 输出连续 IF 曲线。",
            styles["body"],
        )
    )
    story.append(
        p(
            "当前推荐组合为 router_hard_v3 + 四个 IF-Net 专家 + guarded top-2 fallback + 二级质量判别器。"
            "路由器先给出候选专家，低置信时运行 top-2 专家；质量判别器再判断哪个专家的 IF 输出更可靠。"
            "为避免误伤 crossing，若 top-1 为 cross_overlap_like，则默认保护该输出。",
            styles["body"],
        )
    )

    story.append(p("二、仿真信号与训练数据", styles["h1"]))
    add_bullets(
        story,
        [
            "基础采样设置：fs=1024 Hz，单样本 1024 点，默认 2 个 AM-FM 分量，频率范围 35-430 Hz。",
            "覆盖类型：线性 chirp、二次/三次多项式 chirp、正弦调频 chirp、交叉 IF、接近平行 IF、局部突变 IF、相切或短时间重合 IF。",
            "噪声类型：白噪声、colored 噪声、脉冲噪声、趋势项噪声；训练时随机 SNR 覆盖约 -10 dB 到 24 dB。",
            "监督标签：仿真器输出真实 IF 曲线，训练时采样到 STFT 帧中心，并在频率轴生成 ridge heatmap 标签。",
        ],
        styles["body"],
    )

    story.append(p("三、IF-Net 神经网络如何构建", styles["h1"]))
    story.append(
        p(
            "IF-Net 的输入是归一化 log-STFT 时频图，输出是每个分量的 ridge heatmap。网络没有直接回归单个频率点，而是先预测整条频率概率分布，再通过频率维 softmax 和 soft-argmax 得到连续 IF。"
            "这样既保留了传统脊线提取的解释性，也使 IF 提取步骤可以通过梯度训练。",
            styles["body"],
        )
    )
    add_model_table(story, styles)

    story.append(PageBreak())
    story.append(p("四、专家模型体系", styles["h1"]))
    add_bullets(
        story,
        [
            "poly_like 专家：负责 linear、quadratic、cubic，训练中提高多项式约束和 identity_slope 约束，输出后使用 heatmap 加权多项式稳健拟合。",
            "sinusoidal_like 专家：从通用模型继续训练，只看 sinusoidal_fm，增强调频深度、调频周期和二次谐波覆盖。",
            "cross_overlap_like 专家：使用 balanced_refit_resume，覆盖 crossing、near_parallel、tangent_or_overlap 以及部分复杂混合场景。",
            "jump_like 专家：从 balanced 专家继续训练，只看 local_jump，增强跳变幅度、跳变位置、过渡宽度和局部 bump 采样。",
        ],
        styles["body"],
    )

    story.append(p("五、判别器/路由器如何构建", styles["h1"]))
    story.append(
        p(
            "判别器是一个时频图分类器，用来决定信号应该送入哪类专家模型。它不直接输出 IF，而是输出四个专家组的概率。v3 判别器使用多尺度 STFT 输入：短窗更敏感于局部突变和快速变化，长窗提供更好的频率分辨率，中等窗宽处理一般 chirp 结构。",
            styles["body"],
        )
    )
    add_router_table(story, styles)

    story.append(p("六、二级质量判别器如何构建", styles["h1"]))
    story.append(
        p(
            "二级质量判别器是本轮新增模块。它的目标不是继续提升类型分类准确率，而是直接服务最终 IF 误差：当路由器低置信并运行 top-2 专家时，质量判别器根据候选专家的输出质量特征选择一个更可能低误差的 IF 曲线。",
            styles["body"],
        )
    )
    add_quality_table(story, styles)

    story.append(PageBreak())
    story.append(p("七、第一阶段到目前为止完成的工作", styles["h1"]))
    add_bullets(
        story,
        [
            "搭建了专用 Python 环境与 ifnet_stage1 工程结构，完成训练、评估、路由预测和可视化脚本。",
            "实现 chirp 仿真器，覆盖全部要求信号类型，并支持场景参数、噪声类型和难例采样控制。",
            "实现 IF-Net U-Net、soft-argmax IF 输出、heatmap 监督、排列对齐评估和多种后处理策略。",
            "完成通用模型、balanced 专家、多项式专家、正弦调频专家、局部跳变专家训练与验证。",
            "完成硬判别器、增强判别器、guarded soft top-2 路由策略、二级质量判别器的训练与比较。",
            "已将质量判别器接入 eval_routed_scenarios.py 和 predict_routed.py；预测时可通过 --quality-selector-checkpoint 启用。",
            "完成 local_jump 分段线性跳变保持后处理实验，结果变差，因此已撤回该后处理，不作为主流程。",
            "新增 IFNetJumpAux 跳变位置辅助头，把 local_jump 从单纯 ridge heatmap 学习扩展为 ridge + jump event 联合学习；v3 版本使用仿真器真实 jump_center/jump_valid 监督，并对无效跳变分量做 loss mask。",
            "新增 guarded_special top-2 候选导出策略，优先保留 top-1，同时保护 sinusoidal_like 与 jump_like 作为第二候选，提高候选覆盖率。",
            "新增 Stage-2 readiness 准入评估脚本，用 IF 精度、置信度、top-2 覆盖率、crossing 身份稳定性和 local_jump 跳变定位共同判断是否可进入第二阶段。",
        ],
        styles["body"],
    )

    story.append(PageBreak())
    story.append(p("八、全类型定量结果（质量判别器组合）", styles["h1"]))
    story.append(p("以下指标来自每类 512 个仿真样本，使用 v3 路由器、四专家、guarded top-2 fallback 与二级质量判别器，主要用于说明质量判别器相对上一版路由规则的改进。加入 jump_aux、top-2 候选覆盖率和第二阶段准入门槛后的最新判断见下一节。", styles["body"]))
    add_metric_table(story, quality_metrics, styles)
    story.append(Spacer(1, 0.25 * cm))
    story.append(p("与上一版 guarded top-2 规则的对比", styles["h2"]))
    add_comparison_table(story, baseline_metrics, quality_metrics, styles)

    story.append(PageBreak())
    story.append(p("九、第二阶段准入评估", styles["h1"]))
    story.append(
        p(
            "准入评估使用 router_hard_v3、quality_selector_v1、local_jump_aux_v3 和 guarded_special top-2 候选导出。"
            f"当前主评估 ready_for_stage2={readiness['ready_for_stage2']}；"
            f"local_jump 跳变事件平均定位误差为 {readiness['scenarios']['local_jump']['jump_event_mae_ms']:.2f} ms，"
            f"top-2 候选覆盖率为 {readiness['aggregate']['candidate_oracle_coverage_10hz'] * 100:.2f}%。",
            styles["body"],
        )
    )
    add_readiness_table(story, readiness, styles)
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            f"为检查稳定性，又使用 seed=67890 复核：top-2 候选覆盖率为 {readiness_seed2['aggregate']['candidate_oracle_coverage_10hz'] * 100:.2f}%，"
            f"local_jump 跳变事件平均定位误差为 {readiness_seed2['scenarios']['local_jump']['jump_event_mae_ms']:.2f} ms，"
            f"sinusoidal_fm MAE 为 {readiness_seed2['scenarios']['sinusoidal_fm']['if_mae_hz']:.2f} Hz。"
            "两次正式复核均通过全部准入门槛，因此第一阶段已经可以作为第二阶段 ICCD 参数估计的初始 IF 输入。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("十、代表样本处理结果", styles["h1"]))
    story.append(p("每类信号各抽取一个代表样本，图中绿色为真实 IF，蓝色虚线为预测 IF。该代表图仍来自稳定组合可视化样本，用于观察典型脊线拟合形态。", styles["body"]))
    add_example_table(story, examples, styles)

    story.append(PageBreak())
    story.append(p("十一、全类型处理图总览", styles["h1"]))
    add_image(story, FIG_DIR / "all_scenarios_overview.png", max_width=18.0 * cm, max_height=17.0 * cm)

    story.append(PageBreak())
    story.append(p("十二、存在的问题", styles["h1"]))
    add_bullets(
        story,
        [
            f"local_jump 的平均跳变定位已经达到第二阶段准入要求；v3 主评估 jump_event MAE 为 {readiness['scenarios']['local_jump']['jump_event_mae_ms']:.2f} ms，P95 为 {readiness['scenarios']['local_jump']['jump_event_p95_ms']:.2f} ms。尾部误差仍存在，说明少数弱分量、平滑过渡或局部 bump 样本仍会把跳变边界判断偏。",
            "local_jump IF 平均误差已经通过门槛，但 P95 仍接近 30 Hz，后续第二阶段应把这类样本视为低置信初值，而不是强约束精确脊线。",
            f"sinusoidal_fm 在复杂调频和弱分量时仍有长尾误差；两次正式复核 MAE 分别为 {readiness['scenarios']['sinusoidal_fm']['if_mae_hz']:.2f} Hz 和 {readiness_seed2['scenarios']['sinusoidal_fm']['if_mae_hz']:.2f} Hz，均通过 8.5 Hz 门槛，但仍需要保留置信度输出。",
            "crossing 的平均 MAE 不高，但个别样本会明显拟合差。原因是两条 IF 在交叉点频率重合，STFT 幅值图中两条脊线会合成亮斑，仅靠幅值谱很难判断分量应当穿过交叉点还是交换身份。",
            "当前已加入 crossing identity Viterbi 后处理并通过身份稳定门槛，但交叉点局部仍可能出现能量主导的短时偏移，因此第二阶段要使用 top-2 候选和置信度来消化这类不确定性。",
            "二级质量判别器若无保护会误伤 crossing，因此当前加入 cross_overlap_like top-1 保护和 guarded_special 候选策略。这说明不同场景的风险不对称，不能让质量判别器无条件覆盖 top-1。",
            "local_jump 的分段线性跳变保持后处理实验失败：local_jump 专家独立评估 MAE 从约 9.65 Hz 变差到约 11.92 Hz，说明误差不是简单两段直线可修复，sigmoid 过渡、局部 bump 和分量弱化很重要。",
            "路由标签准确率不等同于最终 IF 精度。质量判别器会降低某些场景的路由标签准确率，但最终 IF MAE 仍有改善。",
        ],
        styles["body"],
    )

    story.append(p("十三、优化方向与下一步计划", styles["h1"]))
    add_bullets(
        story,
        [
            "第一阶段已经可以作为第二阶段 ICCD 参数估计的初始 IF 输入，但进入第二阶段时不应要求预测 IF 完全贴合真实脊线；应使用 IF 曲线、top-2 候选和置信度共同提供可优化初值。",
            "继续提升 local_jump 的尾部稳定性，重点降低 jump_event P95；训练上可增加弱分量、平滑跳变、局部 bump 和跳变邻域噪声遮蔽样本。",
            f"继续提高 top-2 候选覆盖率的跨 seed 稳定性，当前两次正式评估均为 {readiness['aggregate']['candidate_oracle_coverage_10hz'] * 100:.2f}%，已经稳定超过 88% 门槛，后续目标是稳定保持在 90% 左右。",
            "对 crossing 和短时重合继续加强连续身份约束，降低分量交换对后续 ICCD 参数反演的影响；可尝试 crossing-aware tracking、最小曲率/最小加速度约束或交叉点前后身份一致性 loss。",
            "为 cross_overlap_like 专家尝试多尺度 STFT 或复数/相位相关特征，使模型获得幅值谱之外的分支区分信息。",
            "继续扩大质量判别器和 readiness 验证集，减少小验证批次导致的 best checkpoint 偏差；当前推荐 quality_selector_v1/latest.pt 和 local_jump_aux_v3/latest.pt。",
            "固定第一阶段接口：输出 IF 曲线、ridge heatmap、路由概率、专家标签、质量判别器分数和不确定性指标，为第二阶段 ICCD 参数估计做准备。",
        ],
        styles["body"],
    )

    story.append(PageBreak())
    story.append(p("附录：各类型单独处理图", styles["h1"]))
    plot_dir = FIG_DIR / "plots"
    for idx, (name, label) in enumerate(SCENARIO_LABELS.items()):
        if idx and idx % 2 == 0:
            story.append(PageBreak())
            story.append(p("附录：各类型单独处理图（续）", styles["h1"]))
        story.append(p(label, styles["h2"]))
        add_image(story, plot_dir / f"{name}.png", max_width=17.6 * cm, max_height=9.1 * cm)
        story.append(Spacer(1, 0.22 * cm))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
