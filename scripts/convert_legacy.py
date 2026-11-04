"""Convert checkpoints written by the original research code to the release format.

Usage:
    python scripts/convert_legacy.py --models "<dir with original .pth files>" --data data/csv --out checkpoints

Each original file name has the form {System}_{legacy mode}{run name}_nc{.}_nr{.}_mem{.}_epoch_{.}.pth.
The release format stores the network configuration, the weights and the
training-set normalizer, so a checkpoint is enough to run predictions.
"""

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_koopman import HybridKoopman, Normalizer, get_config, load_sdp, load_trajectories  # noqa: E402
from hybrid_koopman.configs import LEGACY_MODE_NAMES, SYSTEMS  # noqa: E402
from hybrid_koopman.data import csv_path  # noqa: E402

PATTERN = re.compile(r"^(?P<system>[A-Za-z]+)_(?P<mode>lusch_params|no_hot_start|hot_start)(?P<run>.*)_nc\d+_nr\d+_mem\d+_epoch_(?P<epoch>\d+)\.pth$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", default="data/csv")
    ap.add_argument("--sdp", default="data/sdp")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    normalizers = {}
    for f in sorted(glob.glob(os.path.join(args.models, "*.pth"))):
        m = PATTERN.match(os.path.basename(f))
        if not m or m["system"] not in SYSTEMS:
            print("skip", f)
            continue
        system, mode = m["system"], LEGACY_MODE_NAMES[m["mode"]]
        cfg = get_config(system)
        if system not in normalizers:
            normalizers[system] = Normalizer.fit(load_trajectories(csv_path(args.data, system, "train1"), cfg.len_time))
        sdp = load_sdp(os.path.join(args.sdp, f"{system}.mat"))
        model = HybridKoopman.for_system(system, mode, sdp)
        model.load_legacy_state(torch.load(f, map_location="cpu", weights_only=False))
        out = os.path.join(args.out, f"{system}_{mode}.pt")
        model.save(out, system=system, mode=mode, epoch=int(m["epoch"]), run=m["run"],
                   normalizer=normalizers[system].to_dict())
        print(f"{os.path.basename(f)} -> {out}")


if __name__ == "__main__":
    main()
