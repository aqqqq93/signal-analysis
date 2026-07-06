from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch

from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .postprocess import apply_if_postprocess
from .simulation import sim_config_from_dict
from .tf import feature_channels, log_spectrogram, stft_config_from_dict


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict IF curves from a .npy signal and save an .npz file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to .npy array shaped [N] or [B, N].")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--postprocess",
        default="none",
        choices=["none", "poly", "polynomial", "poly3", "poly_prob", "poly_heatmap", "prob_poly"],
    )
    parser.add_argument("--poly-topk", type=int, default=7)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--poly-robust-iters", type=int, default=2)
    parser.add_argument("--poly-huber-hz", type=float, default=12.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    arr = np.load(args.input).astype("float32")
    if arr.ndim == 1:
        arr = arr[None, :]
    signal = torch.from_numpy(arr).to(device)
    model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    feats, freq_grid = log_spectrogram(signal, stft_cfg, sim_cfg.fs)
    logits = model(feats)
    pred_if, probs = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
    pred_if = apply_if_postprocess(
        pred_if,
        mode=args.postprocess,
        degree=args.poly_degree,
        robust_iters=args.poly_robust_iters,
        huber_hz=args.poly_huber_hz,
        probs=probs,
        freq_grid=freq_grid,
        topk=args.poly_topk,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "if_hz": pred_if.cpu().numpy(),
        "freq_grid": freq_grid.cpu().numpy(),
        "ridge_probs": probs.cpu().numpy(),
    }
    if Path(args.output).suffix.lower() == ".mat":
        savemat(args.output, payload)
    else:
        np.savez(args.output, **payload)


if __name__ == "__main__":
    main()
