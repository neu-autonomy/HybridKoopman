"""Per-system hyperparameters used for the experiments in the paper.

Network sizes and loss weights follow the best parameters reported by
Lusch et al. (2018) for each system. The training schedule (batch size,
number of epochs, length of the reconstruction-only phase) was recovered
from the optimizer state stored in the released checkpoints.
"""

from dataclasses import dataclass, field, replace

SYSTEMS = ("DiscreteSpectrumExample", "FluidFlowOnAttractor", "Pendulum", "Lorenz")

# Training modes compared in the paper.
#   lusch               : eigenvalue count taken from Lusch et al., auxiliary networks randomly initialized
#   hybrid_no_pretrain  : latent size, memory and eigenvalue structure taken from the SDP
#   hybrid              : as above, with the auxiliary networks pretrained on the SDP eigenvalues
MODES = ("lusch", "hybrid_no_pretrain", "hybrid")
MODE_LABELS = {
    "lusch": "Lusch et al.",
    "hybrid_no_pretrain": "Hybrid, without pretraining",
    "hybrid": "Hybrid, with pretraining",
}
# Names used by the original research code and in the original checkpoint names.
LEGACY_MODE_NAMES = {"lusch_params": "lusch", "no_hot_start": "hybrid_no_pretrain", "hot_start": "hybrid"}


@dataclass(frozen=True)
class SystemConfig:
    name: str
    dim: int                      # state dimension
    len_time: int                 # samples per trajectory
    delta_t: float                # sampling interval of the data
    widths: tuple                 # encoder and decoder widths, as in Lusch et al.
    hidden_widths_omega: tuple    # hidden widths of each auxiliary network
    lusch_eigs: tuple             # (complex pairs, real eigenvalues) for the Lusch baseline
    recon_lam: float = 0.1
    mid_shift_lam: float = 1.0
    Linf_lam: float = 1e-7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-13
    batch_size: int = 128         # trajectories per batch
    num_shifts: int = 10          # prediction horizon k used in the loss
    epochs: int = 1000
    recon_epochs: int = 100       # reconstruction-only phase
    freeze_epochs: int = 100      # hybrid mode: epochs the pretrained auxiliary networks stay frozen
    pretrain_epochs: int = 200    # hybrid mode: epochs of auxiliary pretraining on the SDP eigenvalues
    checkpoint_every: int = 200

    @property
    def encoder_widths(self):
        depth = (len(self.widths) - 4) // 2
        return list(self.widths[: depth + 2])

    @property
    def decoder_widths(self):
        depth = (len(self.widths) - 4) // 2
        return list(self.widths[depth + 2:])

    def with_overrides(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **kwargs)


CONFIGS = {
    "DiscreteSpectrumExample": SystemConfig(
        name="DiscreteSpectrumExample", dim=2, len_time=51, delta_t=0.02,
        widths=(2, 30, 30, 2, 2, 30, 30, 2), hidden_widths_omega=(10, 10, 10),
        lusch_eigs=(0, 2), recon_lam=0.1, Linf_lam=1e-7,
    ),
    "FluidFlowOnAttractor": SystemConfig(
        name="FluidFlowOnAttractor", dim=3, len_time=121, delta_t=0.05,
        widths=(3, 105, 2, 2, 105, 3), hidden_widths_omega=(300,),
        lusch_eigs=(1, 0), recon_lam=0.1, Linf_lam=1e-7,
    ),
    "Pendulum": SystemConfig(
        name="Pendulum", dim=2, len_time=51, delta_t=0.02,
        widths=(2, 80, 80, 2, 2, 80, 80, 2), hidden_widths_omega=(170,),
        lusch_eigs=(1, 0), recon_lam=0.001, Linf_lam=1e-9,
    ),
    "Lorenz": SystemConfig(
        name="Lorenz", dim=3, len_time=121, delta_t=0.05,
        widths=(3, 105, 2, 2, 105, 3), hidden_widths_omega=(300,),
        lusch_eigs=(2, 2), recon_lam=0.6, mid_shift_lam=0.9, Linf_lam=1e-7,
        batch_size=64, epochs=3000, recon_epochs=500, freeze_epochs=500,
    ),
}


def get_config(system, **overrides):
    if system not in CONFIGS:
        raise KeyError(f"unknown system {system!r}, expected one of {SYSTEMS}")
    return CONFIGS[system].with_overrides(**overrides)


def canonical_mode(mode):
    mode = LEGACY_MODE_NAMES.get(mode, mode)
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}, expected one of {MODES}")
    return mode
