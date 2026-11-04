"""Training schedule used in the paper.

1. Reconstruction phase (recon_epochs): train the encoder and decoder as an autoencoder.
2. hybrid mode only: pretrain the auxiliary networks on the SDP eigenvalues
   (pretrain_epochs, separate optimizer), then keep them frozen for freeze_epochs
   while the encoder and decoder adapt.
3. Train all networks on the full Koopman loss until `epochs`.

In the hybrid mode the epochs that start and end the frozen phase take no
optimizer step, as in the original code, so epoch counts match the paper's checkpoints.
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from .configs import canonical_mode
from .data import TrajectoryWindows
from .losses import koopman_loss, pretrain_loss, reconstruction_loss
from .model import HybridKoopman
from .predict import one_step_mse


def _set_aux_trainable(model, flag):
    for p in model.auxiliary.parameters():
        p.requires_grad_(flag)


def pretrain_auxiliary(model, loader, sdp, cfg, legacy=True, log=print):
    opt = torch.optim.Adam(model.auxiliary.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.pretrain_epochs):
        for windows in loader:
            opt.zero_grad()
            loss = pretrain_loss(model, windows, sdp, legacy=legacy)
            loss.backward()
            opt.step()
        if epoch % 50 == 0 or epoch == cfg.pretrain_epochs - 1:
            log(f"  pretrain epoch {epoch}: loss {loss.item():.3e}")
    return loss.item()


def train(cfg, mode, train_traj, test_traj=None, sdp=None, normalizer=None, out_dir="runs",
          legacy=True, seed=0, eval_every=10, log=print, wandb_run=None):
    """Train one model. Trajectories are normalized arrays of shape (N, T, dim).

    Returns the trained model and the path of the final checkpoint.
    """
    mode = canonical_mode(mode)
    torch.manual_seed(seed)
    model = HybridKoopman.for_system(cfg.name, mode, sdp)
    loader = DataLoader(TrajectoryWindows(train_traj, cfg.num_shifts, model.memory),
                        batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    hybrid = mode == "hybrid"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{cfg.name}_{mode}")
    extra = {"system": cfg.name, "mode": mode,
             "normalizer": normalizer.to_dict() if normalizer is not None else None}

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        if hybrid and epoch == cfg.recon_epochs:
            log(f"epoch {epoch}: pretraining auxiliary networks on the SDP eigenvalues")
            pretrain_auxiliary(model, loader, sdp, cfg, legacy=legacy, log=log)
            _set_aux_trainable(model, False)
            continue
        if hybrid and epoch == cfg.recon_epochs + cfg.freeze_epochs:
            log(f"epoch {epoch}: unfreezing auxiliary networks")
            _set_aux_trainable(model, True)
            continue

        total, parts = 0.0, {}
        for windows in loader:
            opt.zero_grad(set_to_none=True)
            if epoch < cfg.recon_epochs:
                loss = reconstruction_loss(model, windows, cfg.recon_lam)
            else:
                loss, parts = koopman_loss(model, windows, cfg.recon_lam, cfg.mid_shift_lam,
                                           cfg.Linf_lam, legacy=legacy)
            loss.backward()
            opt.step()
            total += loss.item()
        total /= len(loader)

        msg = {"epoch": epoch, "train_loss": total, **{k: v.item() for k, v in parts.items()}}
        if test_traj is not None and epoch >= cfg.recon_epochs and epoch % eval_every == 0:
            msg["test_one_step_mse"] = one_step_mse(model, test_traj)
        if wandb_run is not None:
            wandb_run.log(msg)
        if epoch % eval_every == 0:
            log(" ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in msg.items())
                + f" ({time.time() - t0:.0f} s)")
        if epoch % cfg.checkpoint_every == 0:
            model.save(f"{stem}_epoch{epoch}.pt", epoch=epoch, **extra)

    path = f"{stem}.pt"
    model.save(path, epoch=cfg.epochs - 1, **extra)
    return model, path
