"""Static PNG figures for results/figures (E4 adaptation, E3b replica counts)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ============= Constants =============

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# categorical slots 1 and 2 of the validated reference palette, assigned in fixed order
SERIES = ("#2a78d6", "#eb6834")
EVENT_MARK = "#52514e"


# ============= Helpers =============


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def kyoto_per_year(per_year: dict[str, dict[str, dict]], path: Path) -> Path:
    """
    PR-AUC per streamed year for each mode (static first, adaptive second).

    Parameters
    ----------
    per_year : dict
        Mode name to `{year: {"pr_auc": float | None}}`.
    path : Path
        Output PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for color, (mode, years) in zip(SERIES, per_year.items(), strict=False):
        xs = sorted(years)
        ys = [years[x]["pr_auc"] for x in xs]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            label=mode,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        last = next((i for i in range(len(ys) - 1, -1, -1) if ys[i] is not None), None)
        if last is not None:
            ax.annotate(
                mode,
                (xs[last], ys[last]),
                xytext=(8, 0),
                textcoords="offset points",
                color=TEXT_PRIMARY,
                va="center",
            )
    _style(ax, "Kyoto 2006+: PR-AUC per streamed year", "PR-AUC")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def cicids_timeline(
    rolling: list[dict], drift_times: list[float], gate: list[tuple[float, str]], path: Path
) -> Path:
    """Rolling PR-AUC over the CICIDS2017 week with drift events and gate decisions marked."""
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    xs = [(r["end_event_time"] - rolling[0]["start_event_time"]) / 3.6e6 for r in rolling]
    ax.plot(
        xs, [r["pr_auc"] for r in rolling], color=SERIES[0], linewidth=2, label="rolling PR-AUC"
    )
    origin = rolling[0]["start_event_time"] if rolling else 0
    for i, t in enumerate(drift_times):
        ax.axvline(
            (t - origin) / 3.6e6,
            color=EVENT_MARK,
            linewidth=1,
            linestyle=":",
            label="drift event" if i == 0 else None,
        )
    for i, (t, action) in enumerate(gate):
        ax.axvline(
            (t - origin) / 3.6e6,
            color=SERIES[1],
            linewidth=1.5,
            label="gate decision" if i == 0 else None,
        )
        ax.annotate(
            action,
            ((t - origin) / 3.6e6, 0.02),
            rotation=90,
            color=TEXT_SECONDARY,
            fontsize=8,
            va="bottom",
        )
    _style(ax, "CICIDS2017 week: rolling PR-AUC with drift and gate events", "PR-AUC")
    ax.set_xlabel("hours of event time since stream start", color=TEXT_SECONDARY)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def replica_timeline(samples: list[dict], path: Path) -> Path:
    """Consumer replica count over time for E3b, with the load steps shaded."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    xs = [s["t_s"] / 60 for s in samples]
    ax.step(
        xs,
        [s["replicas"] for s in samples],
        where="post",
        color=SERIES[0],
        linewidth=2,
        label="consumer replicas",
    )
    _style(ax, "Consumer replicas under a 250 -> 2,000 -> 250 events/s load", "replicas")
    ax.set_xlabel("minutes since start", color=TEXT_SECONDARY)
    ax.set_ylim(0, max([s["replicas"] for s in samples] + [6]) + 0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
