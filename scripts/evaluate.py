"""Prediction error of trained models on the test split.

    python scripts/evaluate.py                       # all systems and modes in checkpoints/
    python scripts/evaluate.py --horizons 1 5 30

Errors are mean squared errors on the normalized states ([-1, 1] per component),
averaged over all test trajectories and start times. A forecast that overflows
makes the mean infinite or NaN. The last column counts such forecasts at the
longest horizon.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_koopman import MODES, SYSTEMS, HybridKoopman, Normalizer, forecast, get_config, horizon_mse  # noqa: E402
from hybrid_koopman.data import csv_path, load_trajectories  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--data", default="data/csv")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 10, 30])
    a = ap.parse_args()

    H = max(a.horizons)
    print(f"{'system':24s} {'mode':20s} " + " ".join(f"{'MSE@' + str(h):>10s}" for h in a.horizons) + f" {'diverged':>10s}")
    for s in a.systems:
        cfg = get_config(s)
        raw = load_trajectories(csv_path(a.data, s, "test"), cfg.len_time)
        for mode in MODES:
            path = os.path.join(a.checkpoints, f"{s}_{mode}.pt")
            if not os.path.exists(path):
                continue
            model, meta = HybridKoopman.load(path)
            x = Normalizer.from_dict(meta["normalizer"])(raw)
            mse = horizon_mse(model, x, H)
            diverged = (~torch.isfinite(forecast(model, x, H)[:, :, -1]).all(-1)).sum().item()
            print(f"{s:24s} {mode:20s} " + " ".join(f"{mse[h - 1].item():10.2e}" for h in a.horizons)
                  + f" {diverged:10d}")


if __name__ == "__main__":
    main()
