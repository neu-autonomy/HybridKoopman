"""Train one model.

    python scripts/train.py --system Pendulum --mode hybrid
    python scripts/train.py --system Lorenz --mode lusch --epochs 50     # short run

Modes: lusch, hybrid_no_pretrain, hybrid. The defaults reproduce the paper's settings.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_koopman import MODES, SYSTEMS, get_config, load_sdp, load_split  # noqa: E402
from hybrid_koopman.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--data", default="data/csv")
    ap.add_argument("--sdp", default="data/sdp")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--recon-epochs", type=int)
    ap.add_argument("--freeze-epochs", type=int)
    ap.add_argument("--pretrain-epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-shifts", type=int)
    ap.add_argument("--max-train-traj", type=int, help="use only the first N training trajectories")
    ap.add_argument("--no-legacy", action="store_true", help="use the corrected loss details (see losses.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    cfg = get_config(a.system, epochs=a.epochs, recon_epochs=a.recon_epochs, freeze_epochs=a.freeze_epochs,
                     pretrain_epochs=a.pretrain_epochs, batch_size=a.batch_size, num_shifts=a.num_shifts)
    train_traj, norm = load_split(a.data, a.system, "train1", cfg.len_time)
    test_traj, _ = load_split(a.data, a.system, "test", cfg.len_time, norm)
    if a.max_train_traj:
        train_traj = train_traj[:a.max_train_traj]
    sdp = load_sdp(os.path.join(a.sdp, f"{a.system}.mat"))
    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project="HybridKoopman", config={**cfg.__dict__, "mode": a.mode})
    _, path = train(cfg, a.mode, train_traj, test_traj, sdp=sdp, normalizer=norm, out_dir=a.out,
                    legacy=not a.no_legacy, seed=a.seed, wandb_run=run)
    print("saved", path)


if __name__ == "__main__":
    main()
