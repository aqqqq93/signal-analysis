# IF-Net Stage 1

This folder contains the first-stage neural replacement for the ICCD ridge
detector. The goal is narrow: learn instantaneous-frequency curves from a
differentiable time-frequency representation, then pass those curves to the
existing ICCD reconstruction code.

Traditional ICCD path:

```text
STFT/GPTFT -> hard ridge detector -> IF curves -> ICCD reconstruction
```

Stage-1 neural path:

```text
differentiable STFT -> IF-Net heatmaps -> soft-argmax IF curves -> ICCD reconstruction
```

## Environment

From `D:\signal analysis`:

```powershell
.\.venv_ifnet\Scripts\Activate.ps1
python -m pip install -e .\ifnet_stage1
```

The workspace environment was created with:

```powershell
python -m venv --system-site-packages .venv_ifnet
```

This keeps a project-local Python entry point while reusing the installed
PyTorch stack.

## Quick check

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.smoke_test
```

## Train

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.train --config .\ifnet_stage1\configs\default.yaml
```

The simulator explicitly covers these signal families:

- linear chirp
- quadratic chirp
- cubic chirp
- sinusoidal FM chirp
- crossing IF
- near-parallel IF
- local-jump IF
- tangent or short-overlap IF

Outputs are written under `ifnet_stage1/runs/` by default.

## Hard Router

The hard router first classifies a signal into one of four route groups, then
selects the matching IF-Net expert:

```text
poly_like           -> polynomial_refit_resume
sinusoidal_like     -> sinusoidal_refit
cross_overlap_like  -> balanced_refit_resume
jump_like           -> local_jump_refit
```

Train the router:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.train_router --config .\ifnet_stage1\configs\router_hard_v3.yaml
```

Evaluate route accuracy:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.eval_router --checkpoint .\ifnet_stage1\runs\router_hard_v3\latest.pt --output-dir .\ifnet_stage1\runs\router_hard_v3\eval_all
```

Evaluate end-to-end routed IF estimation:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.eval_routed_scenarios --router-checkpoint .\ifnet_stage1\runs\router_hard_v3\latest.pt --output-dir .\ifnet_stage1\runs\router_hard_v3\eval_routed_all_specialists
```

Predict IF from a `.npy` signal with automatic routing:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.predict_routed --router-checkpoint .\ifnet_stage1\runs\router_hard_v3\latest.pt --input signal.npy --output routed_if.npz
```

Low-confidence top-2 fallback is implemented, but it is optional because the
current hard route plus specialists is usually cleaner:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.eval_routed_scenarios --router-checkpoint .\ifnet_stage1\runs\router_hard_v3\latest.pt --output-dir .\ifnet_stage1\runs\router_hard_v3\eval_routed_all_specialists_fallback --fallback --fallback-confidence 0.78 --fallback-margin 0.18
```

## Stage-1 Readiness Gate

Stage 1 should not be frozen for Stage 2 only because the average IF MAE looks
acceptable. The handoff check also verifies confidence, top-2 candidate
coverage, crossing identity continuity, and local-jump event timing:

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.eval_stage1_readiness `
  --router-checkpoint .\ifnet_stage1\runs\router_hard_v3\latest.pt `
  --output-dir .\ifnet_stage1\runs\stage1_readiness_v1 `
  --batch-size 16 --batches 8 `
  --quality-selector-checkpoint .\ifnet_stage1\runs\quality_selector_v1\latest.pt `
  --quality-selector-margin 0.10 `
  --quality-protect-top-routes cross_overlap_like `
  --jump-aux-checkpoint .\ifnet_stage1\runs\local_jump_aux_v2\latest.pt `
  --candidate-policy guarded_special `
  --candidate-special-boost 0.12
```

The script writes `readiness_metrics.json` and returns `ready_for_stage2`.
The current acceptance gates are:

- overall selected IF MAE <= 5.5 Hz;
- top-2 oracle coverage at 10 Hz >= 88%;
- high-confidence subset MAE <= 5 Hz with at least 55% coverage;
- crossing fixed-identity MAE <= 12 Hz and identity excess <= 8 Hz;
- local-jump IF MAE <= 10.5 Hz and P95 <= 40 Hz;
- local-jump event MAE <= 80 ms;
- sinusoidal-FM MAE <= 8.5 Hz.

## Local-Jump Auxiliary Head

`IFNetJumpAux` adds a temporal jump-location head on top of IF-Net. It is
intended to provide event timing for Stage 2 while the existing local-jump
expert remains responsible for the IF curve itself.

```powershell
.\.venv_ifnet\Scripts\python.exe -m ifnet_stage1.train_jump_aux `
  --config .\ifnet_stage1\configs\local_jump_aux.yaml
```

For now, use the auxiliary checkpoint through `--jump-aux-checkpoint` in the
readiness evaluator instead of replacing the main jump expert.

The current preferred combination is:

- router: `ifnet_stage1/runs/router_hard_v3/latest.pt`;
- quality selector: `ifnet_stage1/runs/quality_selector_v1/latest.pt`;
- IF experts: existing poly/sinusoidal/cross/local-jump experts;
- local-jump event head: `ifnet_stage1/runs/local_jump_aux_v2/latest.pt`;
- candidate export policy: `guarded_special`, boost `0.12`.

With this combination, the `stage1_readiness_aux_v2_guarded_candidates`
evaluation passed all readiness gates on seed `13579`. A second seed `67890`
kept top-2 candidate coverage above the 88% gate; its sinusoidal-FM MAE
fluctuation was resolved when rechecked with 512 sinusoidal samples.
