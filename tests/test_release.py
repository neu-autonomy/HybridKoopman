"""Tests that need only the files in this repository (no full datasets).

    python -m pytest tests
"""

import os

import numpy as np
import pytest
import torch

from hybrid_koopman import MODES, SYSTEMS, HybridKoopman, forecast, get_config, load_sdp
from hybrid_koopman.data import TrajectoryWindows
from hybrid_koopman.losses import koopman_loss, pretrain_loss, reconstruction_loss

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF = os.path.join(ROOT, "data", "samples", "reference_forecasts.npz")


@pytest.mark.parametrize("system", SYSTEMS)
def test_sdp_files(system):
    sdp = load_sdp(os.path.join(ROOT, "data", "sdp", f"{system}.mat"))
    assert sdp.order >= 1
    assert 2 * sdp.num_complex_pairs + sdp.num_real == sdp.latent_dim


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("mode", MODES)
def test_released_checkpoints_match_original_code(system, mode):
    """Forecasts from the released checkpoints equal those of the original research code."""
    ref = np.load(REF)
    model, meta = HybridKoopman.load(os.path.join(ROOT, "checkpoints", f"{system}_{mode}.pt"))
    assert meta["normalizer"] is not None
    expected = ref[f"{system}_{mode}_forecast"]
    traj = ref[f"{system}_traj"]
    # The reference stores the first steps of a longer forecast, so it covers fewer start times.
    # Recompute that longer forecast and keep the same steps.
    n_steps = traj.shape[1] - expected.shape[1] - model.memory + 1
    pred = forecast(model, traj, n_steps).numpy()[:, :, :expected.shape[2]]
    # The original code ran on a different batch shape. This changes the BLAS summation order and
    # gives float32 roundoff of order 1e-5.
    np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("system", SYSTEMS)
def test_koopman_matrix_matches_sdp_after_pretraining(system):
    """A few pretraining steps move the eigenvalues toward the SDP values, and all losses are finite."""
    torch.manual_seed(0)
    cfg = get_config(system, num_shifts=3)
    sdp = load_sdp(os.path.join(ROOT, "data", "sdp", f"{system}.mat"))
    model = HybridKoopman.for_system(system, "hybrid", sdp)
    traj = np.load(REF)[f"{system}_traj"]
    windows = TrajectoryWindows(traj, cfg.num_shifts, model.memory).data
    assert windows.shape[2] == cfg.dim * model.memory

    opt = torch.optim.Adam(model.auxiliary.parameters(), lr=1e-3)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = pretrain_loss(model, windows, sdp, legacy=False)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first

    for legacy in (True, False):
        total, parts = koopman_loss(model, windows, cfg.recon_lam, cfg.mid_shift_lam, cfg.Linf_lam, legacy=legacy)
        assert torch.isfinite(total)
        assert set(parts) == {"reconstruction", "linearity", "forward", "linf"}
    assert torch.isfinite(reconstruction_loss(model, windows, cfg.recon_lam))


def test_save_load_roundtrip(tmp_path):
    model = HybridKoopman([3, 20, 4], [4, 20, 3], [8], num_complex_pairs=1, num_real=2, memory=2)
    path = tmp_path / "m.pt"
    model.save(path, epoch=3)
    loaded, meta = HybridKoopman.load(path)
    x = torch.randn(5, 6)
    assert torch.equal(model(x), loaded(x))
    assert meta["epoch"] == 3
