"""Render a video of short-horizon forecasts on test trajectories of each system.

    python scripts/make_video.py --out koopman.mp4 --poster koopman.png

Each panel shows several test trajectories of one system. At each frame the current
time moves forward along the trajectories. From the true state(s) at that time,
each model predicts the next `--horizon` states, which are drawn beside the true
continuation. Each panel states the mean squared error at that horizon over the
whole test split. Coordinates are normalized to [-1, 1] per state component.

The trajectories are chosen by a fixed rule, not by hand: the test trajectories are
ranked by forecast error (mean over start times, horizons and the models shown), and
the trajectories at the quantiles given by `--quantiles` are drawn.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybrid_koopman import MODE_LABELS, SYSTEMS, HybridKoopman, Normalizer, forecast, get_config  # noqa: E402
from hybrid_koopman.data import csv_path, load_trajectories  # noqa: E402

NAMES = {"DiscreteSpectrumExample": "Discrete spectrum", "FluidFlowOnAttractor": "Fluid flow on attractor",
         "Pendulum": "Pendulum", "Lorenz": "Lorenz"}
# Light theme of the project website. The forecast colors pass a color-vision-deficiency check
# on the background. The proposed method is drawn widest and on top. The baseline is thinner and dashed.
BG, BAND, INK, MUTED, LINE = "#f2eef4", "#e8e0ee", "#2a2131", "#5a5163", "#cbbdd6"
STYLE = {"hybrid": dict(color="#8a3fd1", ls="-", lw=3.0, ms=6, z=6),
         "hybrid_no_pretrain": dict(color="#3b7fa8", ls=(0, (1, 1.5)), lw=1.8, ms=4, z=5),
         "lusch": dict(color="#1f8a70", ls=(0, (3, 2)), lw=1.6, ms=4, z=4)}


def load_system(system, modes, checkpoints, data, quantiles, horizon):
    """Forecasts of each model on the selected test trajectories, and the test MSE at the last step."""
    cfg = get_config(system)
    raw = load_trajectories(csv_path(data, system, "test"), cfg.len_time)
    out = {"pred": {}, "mse": {}, "memory": {}}
    per_traj = []
    for mode in modes:
        model, meta = HybridKoopman.load(os.path.join(checkpoints, f"{system}_{mode}.pt"))
        x = torch.as_tensor(Normalizer.from_dict(meta["normalizer"])(raw), dtype=torch.float32)
        pred = forecast(model, x, horizon)                                    # (N, P, H, dim)
        P, mem = pred.shape[1], model.memory
        truth = torch.stack([x[:, mem + h:mem + h + P] for h in range(horizon)], dim=2)
        err = (pred - truth) ** 2
        out["mse"][mode] = err[:, :, -1].mean().item()
        per_traj.append(err.mean(dim=(1, 2, 3)))
        out["pred"][mode] = pred
        out["memory"][mode] = mem
    order = torch.argsort(torch.stack(per_traj).mean(0)).numpy()
    idx = [int(order[min(int(q * len(order)), len(order) - 1)]) for q in quantiles]
    out["index"] = idx
    out["truth"] = x[idx].double().numpy()                                    # (n, T, dim)
    out["pred"] = {m: p[idx].numpy() for m, p in out["pred"].items()}        # (n, P, H, dim)
    return out


def style_axes(ax, three, lim):
    ax.set_facecolor(BG)
    if three:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(matplotlib.colors.to_rgba(BAND))
            axis._axinfo["grid"]["color"] = LINE
            axis._axinfo["grid"]["linewidth"] = 0.6
            axis._axinfo["tick"]["inward_factor"] = 0
            axis._axinfo["tick"]["outward_factor"] = 0
            axis.set_ticklabels([])
        ax.set_box_aspect((1, 1, 0.9), zoom=1.2)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
        ax.set_xlabel("$x_1$", labelpad=-12); ax.set_ylabel("$x_2$", labelpad=-12)
        ax.set_zlabel("$x_3$", labelpad=-12)
    else:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.grid(color=BAND, lw=1)
        ax.set_xlabel("$x_1$", labelpad=2); ax.set_ylabel("$x_2$", labelpad=2)
        for sp in ax.spines.values():
            sp.set_color(LINE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="koopman.mp4")
    ap.add_argument("--poster", help="also save one frame as an image")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--data", default="data/csv")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--modes", nargs="+", default=["hybrid", "lusch"])
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    H = a.horizon
    systems = {s: load_system(s, a.modes, a.checkpoints, a.data, a.quantiles, H) for s in a.systems}
    for s, d in systems.items():
        print(f"{s}: test trajectories {d['index']}, "
              + ", ".join(f"{m} MSE at step {H} {v:.2e}" for m, v in d["mse"].items()))

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": LINE})
    dpi = 100
    fig = plt.figure(figsize=(a.width / dpi, a.height / dpi), dpi=dpi, facecolor=BG)
    cols = 2 if len(systems) > 1 else 1
    rows = int(np.ceil(len(systems) / cols))
    fig.text(0.5, 0.965, f"{H}-step forecasts from the true state on test trajectories",
             ha="center", va="center", fontsize=13, color=INK)
    handles = [Line2D([], [], color=INK, lw=2, label=f"True trajectory (next {H} steps)")]
    handles += [Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"],
                       label=f"Forecast, {MODE_LABELS[m]}") for m in a.modes]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))

    panels = []
    for i, (s, d) in enumerate(systems.items()):
        truth = d["truth"]
        n_traj, T, dim = truth.shape
        three = dim == 3
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d" if three else None)
        # The data fill [-1, 1]. Limits also cover every forecast drawn, so nothing leaves the axes.
        pts = np.vstack([truth.reshape(-1, dim)] + [p.reshape(-1, dim) for p in d["pred"].values()])
        lim = [(min(-1.08, lo - 0.03), max(1.08, hi + 0.03)) for lo, hi in zip(pts.min(0), pts.max(0))]
        style_axes(ax, three, lim)
        ax.set_title(NAMES.get(s, s), fontsize=12, color=INK, pad=4)
        mse_txt = f"Test MSE at step {H}:   " + "    ".join(
            f"{MODE_LABELS[m]} {d['mse'][m]:.1e}" for m in a.modes)
        place = ax.text2D if three else ax.text
        place(0.5, -0.02 if three else -0.1, mse_txt, transform=ax.transAxes, ha="center", va="top",
              fontsize=9, color=MUTED)

        def line(**kw):
            return ax.plot(*([[]] * dim), **kw)[0]

        art = []
        for j in range(n_traj):
            ax.plot(*truth[j].T, color=LINE, lw=1.1)
            art.append({"future": line(color=INK, lw=2.0, zorder=3),
                        "now": line(color=INK, marker="o", ms=5.5, ls="none", zorder=7),
                        "pred": {m: line(color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"],
                                         zorder=STYLE[m]["z"]) for m in a.modes},
                        "end": {m: line(marker="o", ms=STYLE[m]["ms"], ls="none", color=STYLE[m]["color"],
                                        zorder=STYLE[m]["z"]) for m in a.modes}})
        # Current time c is the index of the last true state given to the models.
        c_min, c_max = max(d["memory"].values()) - 1, T - H - 1
        panels.append((ax, three, d, art, c_min, c_max))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.91, bottom=0.1, wspace=0.08, hspace=0.3)

    def set_line(artist, pts, three):
        pts = np.atleast_2d(pts)
        artist.set_data(pts[:, 0], pts[:, 1])
        if three:
            artist.set_3d_properties(pts[:, 2])

    def draw(frac):
        for ax, three, d, art, c_min, c_max in panels:
            c = c_min + int(round(frac * (c_max - c_min)))
            for j, aj in enumerate(art):
                truth = d["truth"][j]
                set_line(aj["future"], truth[c:c + H + 1], three)
                set_line(aj["now"], truth[c], three)
                for m in aj["pred"]:
                    # A model with memory k predicts from the true states c-k+1..c: row c-k+1 of the forecast.
                    p = d["pred"][m][j, c - d["memory"][m] + 1]
                    set_line(aj["pred"][m], np.vstack([truth[c], p]), three)
                    set_line(aj["end"][m], p[-1], three)
            if three:
                ax.view_init(elev=22, azim=-60 + 40 * frac)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    if a.poster:
        draw(0.4)
        fig.savefig(a.poster, dpi=dpi, facecolor=BG)
        print("saved", a.poster)

    import imageio.v2 as imageio
    n = int(a.seconds * a.fps)
    with imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, pixelformat="yuv420p",
                            macro_block_size=8, ffmpeg_params=["-movflags", "+faststart"]) as w:
        for f in range(n):
            w.append_data(draw(f / (n - 1)))
    print("saved", a.out)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
