"""Multi-step prediction with a trained network."""

import torch

from .data import stack_memory


@torch.no_grad()
def forecast(model, traj, n_steps):
    """Predict n_steps ahead from every admissible start time.

    traj: (N, T, dim) normalized trajectories.
    Returns (N, P, n_steps, dim) with P = T - n_steps - memory + 1. Entry [:, t, h] is the
    prediction of state t + memory + h made from the true states t, ..., t + memory - 1.
    """
    model.eval()
    traj = torch.as_tensor(traj, dtype=torch.float32)
    N, T, dim = traj.shape
    mem = model.memory
    P = T - n_steps - mem + 1
    if P < 1:
        raise ValueError("trajectory is too short for this horizon")
    x = stack_memory(traj, mem)[:, :P].reshape(-1, dim * mem)
    out = []
    for _ in range(n_steps):
        x = model.step(x)
        out.append(x[:, -dim:].reshape(N, P, dim))
    return torch.stack(out, dim=2)


def horizon_mse(model, traj, n_steps):
    """Mean squared error at each horizon 1..n_steps over all start times."""
    traj = torch.as_tensor(traj, dtype=torch.float32)
    pred = forecast(model, traj, n_steps)
    N, P, H, dim = pred.shape
    mem = model.memory
    truth = torch.stack([traj[:, mem + h:mem + h + P] for h in range(H)], dim=2)
    return ((pred - truth) ** 2).mean(dim=(0, 1, 3))


def one_step_mse(model, traj):
    return horizon_mse(model, traj, 1)[0].item()
