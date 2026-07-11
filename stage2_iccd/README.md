# Stage 2: Differentiable ICCD Unfolding

This folder contains the second-stage prototype: frozen stage-1 IF estimates are used to initialize a differentiable ICCD reconstruction layer. The first training phase only updates:

- the ICCD Tikhonov parameter `alpha`;
- the soft weights over top-k IF candidates;
- a small 1D IF refinement head.

The stage-1 IF-Net checkpoint stays frozen during this phase. Once reconstruction is stable, the same code can be extended to unfreeze IF-Net and train the whole pipeline end to end.

## Principle

ICCD assumes each component can be written as

```text
x_m(t) = a_m(t) cos(phi_m(t)) + b_m(t) sin(phi_m(t))
phi_m(t) = 2 pi integral IF_m(t) dt
```

The envelopes `a_m(t)` and `b_m(t)` are represented by a low-order Fourier basis. Given IF curves, the code builds the ICCD dictionary and solves a batched Tikhonov least-squares problem:

```text
theta = (H^T H + alpha I)^(-1) H^T x
x_hat_m = H_m theta_m
```

Because the dictionary is built with PyTorch tensors and the solve uses `torch.linalg.solve`, gradients can flow from reconstruction loss back to the refined IF curves, candidate weights, and `alpha`.

## Important Files

- `src/stage2_iccd/differentiable_iccd.py`: differentiable real-valued ICCD layer.
- `src/stage2_iccd/model.py`: candidate mixer, lightweight IF refinement head, and full stage-2 model.
- `src/stage2_iccd/candidates.py`: frozen IF-Net candidate provider plus an oracle-perturbed debug provider.
- `src/stage2_iccd/train_stage2.py`: training loop for the frozen-IF-Net stage.
- `src/stage2_iccd/quality_selector.py`: supervised branch-quality selector used to compare default and specialist stage-2 branches.
- `src/stage2_iccd/quality_context.py`: optional Stage1 router and active-count confidence features for the supervised quality selector.
- `src/stage2_iccd/eval_scenarios.py`: per-scenario reconstruction and IF evaluation.
- `src/stage2_iccd/active_count.py`: lightweight active-component count classifier.
- `src/stage2_iccd/train_active_count.py`: active-count router training.
- `src/stage2_iccd/eval_active_count.py`: active-count router evaluation.
- `src/stage2_iccd/eval_active_routed_stage2.py`: routed stage-2 evaluation using the active-count router.
- `scripts/plot_old_new_stage2_comparison.py`: visual comparison between an older checkpoint and the current routed stage-2 output.
- `scripts/compare_stage2_checkpoints.py`: sample-wise IF/SNR comparison between two stage-2 checkpoints.
- `scripts/evaluate_stage2_quality_gate.py`: diagnostic gate between the default and polynomial-specialist stage-2 checkpoints.
- `scripts/sweep_stage2_quality_gate.py`: offline sweep for quality-gate score penalties and margins.
- `scripts/train_stage2_quality_selector.py`: supervised quality-selector training; labels are generated from which branch has lower IF MAE on simulated data.
- `scripts/eval_stage2_quality_selector.py`: independent per-scenario evaluation for the supervised quality selector.
- `scripts/analyze_reference_style_signals.py`: synthesizes time-domain signals shaped like external STFT examples, then runs the routed stage-2 model on those signals.
- `scripts/build_stage2_summary_pdf.py`: rebuilds `output/pdf/stage2_iccd_summary.pdf` from `results_summary_zh.md`.
- `results_summary_zh.md`: current Chinese training summary, per-scenario metrics, and next-step diagnosis.
- `configs/active_count_simple.yaml`: active-count router for linear/quadratic/cubic one-vs-two active components.
- `configs/active_count_simple_near_parallel.yaml`: active-count router extended with near_parallel samples.
- `configs/separated_frozen.yaml`: easier separated two-component curriculum before all scenarios.
- `configs/simple_multicomponent_long.yaml`: longer simple separated two-component training; current best simple checkpoint source.
- `configs/simple_multicomponent_robust.yaml`: robustness probe for simple separated signals; useful for diagnosis, not the current best.
- `configs/poly_multicomponent_refine.yaml`: polynomial two-component refinement probe; diagnostic specialist, not the default checkpoint.
- `configs/balanced_multicomponent_refine.yaml`: balanced two-component refinement probe; diagnostic only because it regressed on several robust cases.
- `configs/simple_single_component.yaml`: active-component masked single-component training.
- `configs/simple_active_mixed.yaml`: mixed one/two active-component probe; diagnostic only, not the current best.
- `configs/simple_active_mixed_p0_conservative.yaml`: conservative active-mask training recipe that reduces single-component leakage while limiting two-component regression.
- `configs/all_multiexpert.yaml`: all-scenario training with several frozen IF-Net experts as candidate IF sources.
- `configs/local_jump_frozen.yaml`: focused local-jump stage-2 specialist.
- `configs/local_jump_segmented_p1.yaml`: P1 segmented local-jump refinement; uses jump-mask conditioning and separate smooth/jump refinement branches.
- `configs/sinusoidal_frozen.yaml`: focused sinusoidal-FM stage-2 specialist.
- `configs/default.yaml`: quick debug training with perturbed ground-truth IF candidates.
- `configs/frozen_ifnet.yaml`: real stage-2 training initialized by a frozen stage-1 checkpoint.

## Quick Checks

From the repository root:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.smoke_test
```

Short debug training:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.train_stage2 --config stage2_iccd/configs/default.yaml --steps 20 --batch-size 2 --run-dir stage2_iccd/runs/debug
```

Frozen IF-Net training:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.train_stage2 --config stage2_iccd/configs/frozen_ifnet.yaml
```

Per-scenario evaluation:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.eval_scenarios --checkpoint stage2_iccd/runs/frozen_ifnet/latest.pt --output-dir stage2_iccd/runs/frozen_ifnet/eval_scenarios
```

Evaluation with explicit noise/SNR overrides:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.eval_scenarios --checkpoint stage2_iccd/runs/simple_multicomponent_long/latest.pt --output-dir stage2_iccd/runs/simple_multicomponent_long/eval_simple_robust --scenarios linear quadratic cubic --snr-db-min -2 --snr-db-max 24 --noise-types-json "{white:0.55,colored:0.25,impulsive:0.10,trend:0.10}"
```

Active-count router training:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.train_active_count --config stage2_iccd/configs/active_count_simple_near_parallel.yaml
```

Active-count router evaluation:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.eval_active_count --checkpoint stage2_iccd/runs/active_count_simple_near_parallel/latest.pt --output-dir stage2_iccd/runs/active_count_simple_near_parallel/eval_simple_easy --scenarios linear quadratic cubic --active-components 1 2
```

Routed stage-2 evaluation:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.eval_active_routed_stage2 --active-checkpoint stage2_iccd/runs/active_count_simple_near_parallel/latest.pt --single-checkpoint stage2_iccd/runs/simple_single_component/latest.pt --multi-checkpoint stage2_iccd/runs/simple_multicomponent_long/latest.pt --output-dir stage2_iccd/runs/active_count_simple_near_parallel/eval_routed_easy --scenarios linear quadratic cubic near_parallel --active-components 1 2
```

Checkpoint-vs-checkpoint diagnostic comparison:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src;."
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\compare_stage2_checkpoints.py --scenario quadratic --num-samples 160 --output-dir stage2_iccd/runs/poly_multicomponent_refine/compare_quadratic_easy
```

Quality-gate diagnostic and sweep:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src;."
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\evaluate_stage2_quality_gate.py --output-dir stage2_iccd/runs/poly_multicomponent_refine/quality_gate_easy --num-samples 80
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\sweep_stage2_quality_gate.py --csv stage2_iccd/runs/poly_multicomponent_refine/quality_gate_easy/quality_gate.csv --output-json stage2_iccd/runs/poly_multicomponent_refine/quality_gate_easy/sweep.json
```

Supervised quality-selector training and evaluation:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src;."
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\train_stage2_quality_selector.py --run-dir stage2_iccd/runs/stage2_quality_selector_poly_balanced --steps 500 --batch-size 8 --balance-classes --margin-scale-hz 0.8
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\eval_stage2_quality_selector.py --selector-checkpoint stage2_iccd/runs/stage2_quality_selector_poly_balanced/best.pt --output-dir stage2_iccd/runs/stage2_quality_selector_poly_balanced/eval_best
```

P0 context-aware quality-selector training:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src;."
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\train_stage2_quality_selector.py --run-dir stage2_iccd/runs/stage2_quality_selector_p0_context --steps 800 --batch-size 8 --val-batches 28 --hidden 96 --dropout 0.10 --balance-classes --margin-scale-hz 0.65
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\eval_stage2_quality_selector.py --selector-checkpoint stage2_iccd/runs/stage2_quality_selector_p0_context/best.pt --output-dir stage2_iccd/runs/stage2_quality_selector_p0_context/eval_best
```

Conservative active-mask training and single-active evaluation:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.train_stage2 --config stage2_iccd/configs/simple_active_mixed_p0_conservative.yaml
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.eval_scenarios --checkpoint stage2_iccd/runs/simple_active_mixed_p0_conservative/latest.pt --output-dir stage2_iccd/runs/simple_active_mixed_p0_conservative/eval_single_active --scenarios linear quadratic cubic near_parallel --active-components 1
```

P1 segmented local-jump refinement:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src"
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.train_stage2 --config stage2_iccd/configs/local_jump_segmented_p1.yaml
.\.venv_ifnet\Scripts\python.exe -m stage2_iccd.eval_scenarios --checkpoint stage2_iccd/runs/local_jump_segmented_p1/latest.pt --output-dir stage2_iccd/runs/local_jump_segmented_p1/eval_p1_compare --scenarios local_jump --batches 64 --batch-size 6 --snr-db-min -4 --snr-db-max 22 --noise-types-json "{white:0.55,colored:0.25,impulsive:0.10,trend:0.10}"
```

Reference-style external signal analysis:

```powershell
$env:PYTHONPATH="stage2_iccd/src;ifnet_stage1/src;."
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\analyze_reference_style_signals.py --output-dir output\figures\reference_style_stage2
```

Rebuild the Chinese PDF summary:

```powershell
.\.venv_ifnet\Scripts\python.exe stage2_iccd\scripts\build_stage2_summary_pdf.py
```

`default.yaml` is intentionally easier than the real setting because it uses perturbed true IF curves. It is for validating the ICCD layer, alpha learning, candidate weighting, and refinement-head gradients before using real IF-Net outputs.

For a closer match to the stage-1 soft top-2 workflow, `configs/frozen_ifnet.yaml` can use either one checkpoint or a `checkpoints:` list. With multiple checkpoints, the candidate mixer learns soft weights across frozen expert IF outputs; with one checkpoint, it falls back to raw-plus-smoothed candidates for debugging.

## Current P0 Optimization Hooks

- Active-component component loss now uses the ground-truth active mask before component matching and adds an inactive-slot energy penalty. This prevents a one-component signal from being silently split across two reconstructed output slots.
- `Stage2ModelConfig.refine_extra_channels` and `Stage2ICCDModel.forward(..., refinement_extra=...)` provide the condition-input interface for `jump_mask` or `jump_prob` conditioning. The default remains zero extra channels, so existing checkpoints keep loading normally.
- The supervised quality selector is implemented as a lightweight MLP over deployment-safe branch features: reconstruction residuals, branch residual ratios, IF difference statistics, smoothness, range, curvature, candidate entropy, Stage1 router confidence, active-count confidence, identity-consistency cues, and local curvature-jump cues. It is currently a diagnostic branch chooser, not yet the default production route.

## Current P1 Optimization Hooks

- `IFRefinementHead` supports `refinement_mode: segmented`. In this mode, the ordinary smooth branch handles non-jump regions and a separate jump branch handles the `jump_mask` region.
- `train_stage2.build_refinement_extra` can convert simulator `jump_center`/`jump_valid` labels into Gaussian jump masks for the refinement head. This keeps local-jump supervision explicit without changing the frozen stage-1 IF-Net.
- Old stage-2 checkpoints can initialize the segmented model: existing refinement weights are copied into the smooth branch, new condition channels are zero-initialized, and the jump branch is trained from scratch.
