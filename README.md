# signal-analysis

This repository manages the code for a three-stage ICCD-oriented signal analysis pipeline.

## Stage Layout

- `ifnet_stage1/`: neural IF initial estimation. This stage replaces hard ridge extraction with differentiable STFT features, IF-Net heatmaps, router/expert models, top-2 fallback, and quality selection.
- `stage2_iccd/`: reserved for ICCD parameter estimation and reconstruction initialized by stage-1 IF curves.
- `stage3_pipeline/`: reserved for end-to-end integration, evaluation, visualization, and application-level workflows.

## Current Status

Stage 1 is the active implementation. It contains:

- chirp simulator covering linear, polynomial, sinusoidal FM, crossing, near-parallel, local-jump, tangent, and short-overlap IF cases;
- IF-Net U-Net estimator;
- hard router and expert models;
- guarded top-2 candidate routing;
- secondary quality selector;
- confidence and top-2 candidate export;
- crossing identity-continuity postprocessing;
- local-jump event auxiliary head;
- Stage-1 readiness gate for deciding when Stage 2 can start;
- evaluation and prediction scripts;
- PDF summary generation script.

Current preferred Stage-1 handoff combination:

- `router_hard_v3`;
- `quality_selector_v1`;
- guarded-special top-2 candidate export;
- `local_jump_aux_v3` for local-jump event timing with simulator jump-center supervision.

Large training outputs and model checkpoints under `ifnet_stage1/runs/` are intentionally ignored. If trained weights need to be shared later, use Git LFS or GitHub Releases rather than committing them directly.

## Development

From this workspace:

```powershell
.\.venv_ifnet\Scripts\Activate.ps1
python -m pip install -e .\ifnet_stage1
```

Smoke test:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.smoke_test
```

Current stage-1 technical report:

```text
output/pdf/ifnet_stage1_summary.pdf
```
