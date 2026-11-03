"""Deep Koopman network with a state-dependent block-diagonal Koopman operator.

The encoder maps a memory-stacked state to latent coordinates. Each conjugate
pair of eigenvalues acts on two latent coordinates, and each real eigenvalue on
one. An auxiliary network per eigenvalue maps the latent coordinates it acts on
(the radius for a complex pair) to the eigenvalue's continuous-time parameters,
which form the Koopman matrix for one step. The decoder maps the advanced
latent state back to a memory-stacked state.
"""

import torch
import torch.nn as nn

from .configs import canonical_mode, get_config


def mlp(widths):
    layers = []
    for a, b in zip(widths[:-1], widths[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class HybridKoopman(nn.Module):
    def __init__(self, encoder_widths, decoder_widths, hidden_widths_omega,
                 num_complex_pairs, num_real, memory=1, parameterization="exp", delta_t=1.0):
        """
        encoder_widths, decoder_widths: layer widths for one state (memory 1).
            The input, latent and output widths are set from memory and the eigenvalue counts.
        parameterization: 'exp' builds blocks exp(mu dt) R(omega dt) and exp(lambda dt), as in the paper.
            'direct' uses the network outputs as the eigenvalue magnitudes.
        delta_t: step used inside the Koopman matrix. The paper uses 1.0, so the auxiliary
            networks learn eigenvalues per sample rather than per unit time.
        """
        super().__init__()
        self.num_complex_pairs = int(num_complex_pairs)
        self.num_real = int(num_real)
        self.memory = int(memory)
        self.latent_dim = 2 * self.num_complex_pairs + self.num_real
        self.parameterization = parameterization
        self.delta_t = delta_t
        self.state_dim = encoder_widths[0]

        enc = list(encoder_widths)
        dec = list(decoder_widths)
        enc[0] = enc[0] * self.memory
        dec[-1] = dec[-1] * self.memory
        enc[-1] = self.latent_dim
        dec[0] = self.latent_dim
        self.encoder = mlp(enc)
        self.decoder = mlp(dec)

        hidden = list(hidden_widths_omega)
        # Complex-pair networks first, then real ones. This order matches the original checkpoints.
        self.auxiliary = nn.ModuleList(
            [mlp([1] + hidden + [2]) for _ in range(self.num_complex_pairs)]
            + [mlp([1] + hidden + [1]) for _ in range(self.num_real)]
        )
        self.config = dict(encoder_widths=list(encoder_widths), decoder_widths=list(decoder_widths),
                           hidden_widths_omega=hidden, num_complex_pairs=self.num_complex_pairs,
                           num_real=self.num_real, memory=self.memory,
                           parameterization=parameterization, delta_t=delta_t)

    # ----- construction -------------------------------------------------------------------------

    @classmethod
    def for_system(cls, system, mode, sdp=None):
        """Build the network used in the paper for `system` and training `mode`."""
        cfg = get_config(system)
        mode = canonical_mode(mode)
        if mode == "lusch":
            nc, nr = cfg.lusch_eigs
            memory = 1
        else:
            if sdp is None:
                raise ValueError(f"mode {mode!r} needs the SDP result for {system}")
            nc, nr, memory = sdp.num_complex_pairs, sdp.num_real, sdp.order
        return cls(cfg.encoder_widths, cfg.decoder_widths, cfg.hidden_widths_omega, nc, nr, memory)

    # ----- dynamics -----------------------------------------------------------------------------

    def omegas(self, z):
        """Eigenvalue parameters from the latent state z of shape (B, latent_dim).

        Returns (B, num_complex_pairs, 2) with [mu, omega] per pair and (B, num_real).
        """
        nc = self.num_complex_pairs
        cplx = [self.auxiliary[j](z[:, 2 * j:2 * j + 2].norm(dim=1, keepdim=True)) for j in range(nc)]
        real = [self.auxiliary[nc + j](z[:, 2 * nc + j:2 * nc + j + 1]).squeeze(1) for j in range(self.num_real)]
        cplx = torch.stack(cplx, dim=1) if cplx else z.new_zeros(z.shape[0], 0, 2)
        real = torch.stack(real, dim=1) if real else z.new_zeros(z.shape[0], 0)
        return cplx, real

    def koopman_matrix(self, z):
        """Block-diagonal one-step Koopman matrices, shape (B, latent_dim, latent_dim)."""
        cplx, real = self.omegas(z)
        dt = self.delta_t
        K = z.new_zeros(z.shape[0], self.latent_dim, self.latent_dim)
        if self.num_complex_pairs:
            mu, om = cplx[..., 0], cplx[..., 1]
            scale = torch.exp(mu * dt) if self.parameterization == "exp" else mu ** dt
            c, s = scale * torch.cos(om * dt), scale * torch.sin(om * dt)
            for j in range(self.num_complex_pairs):
                i = 2 * j
                K[:, i, i], K[:, i, i + 1] = c[:, j], -s[:, j]
                K[:, i + 1, i], K[:, i + 1, i + 1] = s[:, j], c[:, j]
        if self.num_real:
            lam = torch.exp(real * dt) if self.parameterization == "exp" else real ** dt
            idx = torch.arange(2 * self.num_complex_pairs, self.latent_dim)
            K[:, idx, idx] = lam
        return K

    def advance(self, z):
        """One step in the latent space."""
        return torch.bmm(self.koopman_matrix(z), z.unsqueeze(-1)).squeeze(-1)

    def step(self, x):
        """One step in state space. x has shape (B, state_dim * memory)."""
        return self.decoder(self.advance(self.encoder(x)))

    forward = step

    # ----- saving and loading -------------------------------------------------------------------

    def save(self, path, **extra):
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ck["config"])
        model.load_state_dict(ck["state_dict"])
        return model, {k: v for k, v in ck.items() if k not in ("config", "state_dict")}

    def load_legacy_state(self, ck):
        """Load a checkpoint written by the original research code (training_new.py)."""
        self.encoder.load_state_dict(ck["encoder_state_dict"])
        self.decoder.load_state_dict(ck["decoder_state_dict"])
        aux = ck["auxillary_state_dict"]
        if len(aux) != len(self.auxiliary):
            raise ValueError(f"checkpoint has {len(aux)} auxiliary networks, model has {len(self.auxiliary)}")
        for net, sd in zip(self.auxiliary, aux):
            net.load_state_dict(sd)
        return self
