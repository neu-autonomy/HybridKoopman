# HybridKoopman

Code for "A Hybrid Framework for Efficient Koopman Operator Learning" (L. Jung, A. Spiro,
A. Estornell, M. Everett, M. Sznaier, IEEE CDC 2025). [arXiv:2504.18676](https://arxiv.org/abs/2504.18676)

The method first solves a semidefinite program (SDP) on trajectory data to find a Koopman
operator and the dimension of its lifted space. A deep Koopman network in the style of Lusch,
Kutz and Brunton (2018) is then built with that dimension and eigenvalue structure. Its
auxiliary networks are pretrained on the SDP eigenvalues before the full network is trained.

<p align="center"><img src="docs/forecasts.png" width="800" alt="3-step predictions of both models on five test trajectories of each of the four systems"></p>

## Contents

| Path | Contents |
|---|---|
| `hybrid_koopman/` | Model, losses, training schedule, data loading and prediction |
| `scripts/` | `train.py`, `evaluate.py`, `make_video.py`, `convert_legacy.py` |
| `checkpoints/` | Trained models for the four systems and three training modes |
| `data/sdp/` | SDP results used to build and pretrain the hybrid models |
| `data/samples/` | Reference forecasts of the original research code, used by the tests |
| `matlab/` | SDP solver (`run_sdp.m`) and data generators (`data/`) |
| `tests/` | Unit tests and a check that the checkpoints reproduce the original forecasts |

## Installation

```bash
git clone https://github.com/neu-autonomy/HybridKoopman.git
cd HybridKoopman
pip install -e ".[video,test]"      # or: pip install -r requirements.txt
```

The code runs on CPU. It requires Python 3.9 or later and PyTorch 2.0 or later.

## Data

The trajectory files (about 136 MB) are not stored in the repository. Place them in
`data/csv/` as `{System}_train1_x.csv` and `{System}_test_x.csv`, or regenerate them in MATLAB
with the scripts in `matlab/data/` (see the README there). Each file holds one state per row,
with trajectories stored as consecutive blocks.

The SDP results in `data/sdp/` are included. To recompute them, install
[CVX](http://cvxr.com/cvx/) with the SDPT3 solver and run `matlab/run_sdp.m` from the
`matlab/` directory.

## Usage

Evaluate the released checkpoints on the test split:

```bash
python scripts/evaluate.py --horizons 1 10 30
```

Train a model. The defaults follow the paper's settings.

```bash
python scripts/train.py --system Pendulum --mode hybrid
```

The modes are `lusch` (eigenvalue count from Lusch et al., auxiliary networks randomly
initialized), `hybrid_no_pretrain` (latent size, memory and eigenvalue structure from the SDP)
and `hybrid` (as `hybrid_no_pretrain`, with the auxiliary networks pretrained on the SDP
eigenvalues). The systems are `DiscreteSpectrumExample`, `FluidFlowOnAttractor`, `Pendulum` and
`Lorenz`.

Render the forecast video:

```bash
python scripts/make_video.py --out koopman.mp4 --poster koopman.png
```

Load a checkpoint in Python:

```python
from hybrid_koopman import HybridKoopman, Normalizer, forecast, load_trajectories

model, meta = HybridKoopman.load("checkpoints/Pendulum_hybrid.pt")
x = Normalizer.from_dict(meta["normalizer"])(load_trajectories("data/csv/Pendulum_test_x.csv", 51))
pred = forecast(model, x, n_steps=10)     # (trajectories, start times, steps, state dimension)
```

Run the tests with `pytest`. The checkpoint test needs only the files in the repository.

## Results

1-step mean squared error on test trajectories unseen during training (Table II of the paper):

| System | Lusch et al. | Hybrid, without pretraining | Hybrid, with pretraining |
|---|---|---|---|
| Discrete spectrum | 7.06e-6 | 7.06e-6 | 3.73e-6 |
| Fluid flow on attractor | 6.9e-4 | 3.11e-5 | 6.14e-6 |
| Pendulum | 9.93e-4 | 9.93e-4 | 3.82e-4 |
| Lorenz | 1.77e-4 | 7.42e-5 | 5.35e-5 |

## Citation

```bibtex
@inproceedings{jung2025koopman,
  title     = {A Hybrid Framework for Efficient Koopman Operator Learning},
  author    = {Jung, Leonard and Spiro, Alenna and Estornell, Alexander and Everett, Michael and Sznaier, Mario},
  booktitle = {2025 IEEE 64th Conference on Decision and Control (CDC)},
  pages     = {4214--4220},
  year      = {2025}
}
```

## Acknowledgments

The network, the loss and the data generators follow
[DeepKoopman](https://github.com/BethanyL/DeepKoopman) by Bethany Lusch (MIT License,
copyright 2018 Bethany Lusch): B. Lusch, J. N. Kutz and S. L. Brunton, "Deep learning for
universal linear embeddings of nonlinear dynamics," *Nature Communications* 9, 4950 (2018).
