"""Trajectory data: loading, normalization and the shifted windows used for training.

Each CSV file stores one state per row. Trajectories are consecutive blocks of
`len_time` rows. All splits are scaled to [-1, 1] per state component using the
minimum and maximum of the training split.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def csv_path(data_dir, system, split):
    """split is one of 'train1', 'val', 'test'."""
    return os.path.join(data_dir, f"{system}_{split}_x.csv")


def load_trajectories(path, len_time):
    """Return an array of shape (num_traj, len_time, dim)."""
    x = pd.read_csv(path, header=None, dtype=np.float64).values
    n = (x.shape[0] // len_time) * len_time
    return x[:n].reshape(-1, len_time, x.shape[1])


@dataclass
class Normalizer:
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, traj):
        flat = traj.reshape(-1, traj.shape[-1])
        return cls(flat.min(0), flat.max(0))

    def __call__(self, x):
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

    def inverse(self, x):
        return (x + 1.0) / 2.0 * (self.hi - self.lo) + self.lo

    def to_dict(self):
        return {"lo": np.asarray(self.lo).tolist(), "hi": np.asarray(self.hi).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["lo"]), np.asarray(d["hi"]))


def stack_memory(traj, memory):
    """(N, T, dim) -> (N, T - memory + 1, dim * memory). Row t holds states t, ..., t + memory - 1."""
    traj = torch.as_tensor(traj, dtype=torch.float32)
    T = traj.shape[1]
    return torch.cat([traj[:, i:T - memory + 1 + i] for i in range(memory)], dim=-1)


def shift_windows(x, k):
    """(N, T, D) -> (N, T - k, D, k + 1). Entry [..., j] is the state j steps after the window start."""
    T = x.shape[1]
    return torch.stack([x[:, j:T - k + j] for j in range(k + 1)], dim=-1)


class TrajectoryWindows(Dataset):
    """Training windows of k + 1 consecutive memory-stacked states, grouped by trajectory.

    One item is one trajectory, of shape (T - memory + 1 - k, dim * memory, k + 1),
    so a batch of size B contains every window of B trajectories.
    """

    def __init__(self, traj, k, memory):
        self.traj = torch.as_tensor(traj, dtype=torch.float32)
        self.data = shift_windows(stack_memory(self.traj, memory), k)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def load_split(data_dir, system, split, len_time, normalizer=None):
    traj = load_trajectories(csv_path(data_dir, system, split), len_time)
    if normalizer is None:
        normalizer = Normalizer.fit(traj)
    return normalizer(traj), normalizer
