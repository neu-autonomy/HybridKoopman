"""Load the output of the SDP stage (matlab/run_sdp.m).

The SDP returns a low-dimensional embedding of one training trajectory and
the system order (memory). A Koopman matrix is fit to the time-delay
embedding by least squares, and its eigenvalues set the latent size and the
eigenvalue structure of the network. Each eigenvalue is stored as its
logarithm: log(e) for a real eigenvalue and log|e| + i*angle(e) for a complex one.
"""

from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat


@dataclass
class SDPResult:
    koopman: np.ndarray      # (n, n) Koopman matrix on the time-delay embedding
    order: int               # system order, used as the memory of the network
    log_eigs: np.ndarray     # (n,) complex logarithms of the Koopman eigenvalues
    embedding: np.ndarray    # (r, T) embedded trajectory

    @property
    def complex_log_eigs(self):
        """One representative per conjugate pair, in file order (the SDP stores conjugates adjacently)."""
        cplx = [e for e in self.log_eigs if e.imag != 0]
        return np.array(cplx[0::2])

    @property
    def real_log_eigs(self):
        return np.array([e.real for e in self.log_eigs if e.imag == 0])

    @property
    def num_complex_pairs(self):
        return len(self.complex_log_eigs)

    @property
    def num_real(self):
        return len(self.real_log_eigs)

    @property
    def latent_dim(self):
        return self.koopman.shape[0]


def load_sdp(path):
    d = loadmat(path)
    res = SDPResult(
        koopman=np.asarray(d["koop"], dtype=np.float64),
        order=int(np.asarray(d["gramRank"]).ravel()[0]),
        log_eigs=np.asarray(d["E"]).ravel().astype(np.complex128),
        embedding=np.asarray(d["Yn"], dtype=np.float64),
    )
    if 2 * res.num_complex_pairs + res.num_real != res.latent_dim:
        raise ValueError(f"{path}: eigenvalue count does not match the Koopman matrix size")
    return res
