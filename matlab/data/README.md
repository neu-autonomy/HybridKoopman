# Data generators

These scripts are the generators of Lusch, Kutz and Brunton (2018), with Lorenz added.
Each writes `{System}_train1_x.csv`, `{System}_val_x.csv` and `{System}_test_x.csv`,
one state per row with trajectories stored consecutively.

| System | Script | Trajectory length | State dimension |
|---|---|---|---|
| Discrete spectrum | `DiscreteSpectrumExample.m` | 51 | 2 |
| Fluid flow on attractor | `FluidFlowOnAttractor.m` | 121 | 3 |
| Pendulum | `Pendulum.m` | 51 | 2 |
| Lorenz | `lorenz.m` | 121 | 3 |

Each script fixes its random seeds.

## Lorenz sampling

`LorenzFn.m` passes `T = [0 50]` to `ode45` and stores the returned time vector in the same
variable `T`. The first initial condition is therefore integrated with adaptive steps, and each
later one is evaluated at the time points of the previous solve. The script keeps every fifth
solver output after the first 100, so the time between samples is not constant within a
trajectory and differs across trajectories. The released models were trained on this data, so
the script is kept unchanged.
