"""Training losses.

The Koopman loss follows Lusch et al. (2018): reconstruction, linearity of the
latent dynamics over k steps, prediction in state space over k steps, and an
infinity-norm penalty on the worst reconstruction and prediction errors.

`legacy=True` reproduces three details of the code used for the paper:
  * The infinity-norm reconstruction penalty compares each sample with the
    worst reconstruction error over the whole batch (a broadcasting error),
    so the term grows with the batch size.
  * The per-horizon weights lam**h are tiled by sample rather than by horizon,
    so the weight of a residual depends on its flat index modulo k and not on
    its horizon. With mid_shift_lam = 1 (all systems except Lorenz) both choices
    give the same loss for the linearity term.
  * The pretraining targets for several complex pairs are ordered
    [mu_1, ..., mu_n, omega_1, ..., omega_n], while the network outputs are
    ordered [mu_1, omega_1, ..., mu_n, omega_n]. This only affects Lorenz,
    the one system with two complex pairs.
With `legacy=False` the penalty is computed per sample, the weights follow the
horizon and the targets follow the output order.
"""

import torch


def reconstruction_loss(model, windows, recon_lam):
    """Autoencoder loss used in the first training phase (summed, as in the paper's code).

    windows: (B, P, D, k + 1) with D = state_dim * memory.
    """
    x0 = windows[..., 0].reshape(-1, windows.shape[2])
    return recon_lam * ((model.decoder(model.encoder(x0)) - x0) ** 2).sum()


def _horizon_weights(lam, k, n, legacy, device):
    w = lam ** torch.arange(1, k + 1, dtype=torch.float32, device=device)
    if legacy:
        return w.repeat(n)               # index j -> lam**(1 + j % k)
    return w.repeat_interleave(n)        # index j -> lam**(1 + j // n), the horizon of row j


def koopman_loss(model, windows, recon_lam, mid_shift_lam, Linf_lam, legacy=True):
    """Full loss over k prediction steps. Returns (total, parts)."""
    D, k = windows.shape[2], windows.shape[3] - 1
    flat = windows.reshape(-1, D, k + 1)
    n = flat.shape[0]
    x0 = flat[..., 0]
    targets = flat[..., 1:].permute(2, 0, 1)            # (k, n, D), horizon-major

    z0 = model.encoder(x0)
    z = z0
    zs = []
    for _ in range(k):
        z = model.advance(z)
        zs.append(z)
    zs = torch.stack(zs)                                  # (k, n, latent)

    x0_hat = model.decoder(z0)
    preds = model.decoder(zs.reshape(k * n, -1)).reshape(k, n, D)
    z_targets = model.encoder(targets.reshape(k * n, D)).reshape(k, n, -1)

    recon = recon_lam * ((x0_hat - x0) ** 2).mean()
    lin_rows = ((z_targets - zs) ** 2).mean(-1).reshape(-1)
    fwd_rows = ((preds - targets) ** 2).mean(-1).reshape(-1)
    linearity = lin_rows @ _horizon_weights(mid_shift_lam, k, n, legacy, windows.device)
    forward = fwd_rows @ _horizon_weights(recon_lam, k, n, legacy, windows.device)

    eps = 1e-8
    # Worst relative reconstruction error per sample, and worst relative prediction error per horizon.
    if legacy:
        # The original code compares each sample with the reconstructions of all samples in the
        # batch (a broadcast), so this term is sum_i max_{j,d} |x0_hat[j, d] - x0[i, d]| / max_d |x0[i, d]|.
        # max_j |a_j - x| = max(max_j a_j - x, x - min_j a_j), which avoids the (n, n, D) tensor.
        num = torch.maximum(x0_hat.amax(0) - x0, x0 - x0_hat.amin(0)).amax(-1)
    else:
        num = (x0_hat - x0).abs().amax(-1)
    linf_recon = (num / (x0.abs().amax(-1) + eps)).sum()
    linf_pred = ((preds - targets).abs().amax((1, 2)) / (targets.abs().amax((1, 2)) + eps)).sum()
    linf = Linf_lam * (linf_recon + linf_pred)

    total = recon + linearity + forward + linf
    return total, {"reconstruction": recon, "linearity": linearity, "forward": forward, "linf": linf}


def pretrain_loss(model, windows, sdp, legacy=True):
    """Match the auxiliary network outputs to the SDP eigenvalues at every training state."""
    x0 = windows[..., 0].reshape(-1, windows.shape[2])
    with torch.no_grad():
        z = model.encoder(x0)
    cplx, real = model.omegas(z)
    loss = z.new_zeros(())
    if model.num_complex_pairs:
        ce = torch.as_tensor(sdp.complex_log_eigs, dtype=torch.complex64)
        if legacy:
            target = torch.cat([ce.real, ce.imag])                    # [mu_1..mu_n, omega_1..omega_n]
        else:
            target = torch.stack([ce.real, ce.imag], dim=1).reshape(-1)
        target = target.float().to(z.device)
        loss = loss + ((cplx.reshape(z.shape[0], -1) - target) ** 2).mean()
    if model.num_real:
        target = torch.as_tensor(sdp.real_log_eigs, dtype=torch.float32, device=z.device)
        loss = loss + ((real - target) ** 2).mean()
    return loss
