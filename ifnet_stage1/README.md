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
