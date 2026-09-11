#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot the closed-loop and open-loop sweeps from `vllm bench serve` result JSONs.

    uv run scripts/plot.py                                   # results/<model>/{closed,open} -> plots/
    uv run scripts/plot.py --results results/Qwen/Qwen2.5-7B-Instruct --out plots

Each JSON is one sweep point and holds aggregate percentiles only, so every curve here is
built from one number per point. Closed points are keyed by max_concurrency (C), open points
by request_rate (R). Neither sweep is evenly spaced, so x-axes use the true values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

# SLO from slo.md. failed == 0 is the completeness row.
SLO_TTFT_MS = 500
SLO_TPOT_MS = 50
MAX_NUM_SEQS = 1024  # vLLM default running-batch cap on an H100
FIGSIZE = (7.2, 6.4)  # one size for every figure so README pairs line up

# Light-surface reference palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.labelcolor": INK2,
    "axes.titlesize": 12.5,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK2,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ---------- data ----------

def load(folder: Path, key: str) -> list[dict]:
    points = []
    for f in sorted(folder.glob("*.json")):
        if "warmup" in f.name:
            continue
        d = json.loads(f.read_text())
        d["_file"] = f.name
        points.append(d)
    points.sort(key=lambda d: float(d[key]))
    return points


def col(points: list[dict], key: str) -> list[float]:
    return [float(d[key]) for d in points]


def meets_slo(d: dict) -> bool:
    return (
        d["p95_ttft_ms"] <= SLO_TTFT_MS
        and d["p95_tpot_ms"] <= SLO_TPOT_MS
        and int(d.get("failed") or 0) == 0
    )


def request_shape(points: list[dict]) -> tuple[str, int, int]:
    d = points[0]
    n = max(int(d["completed"]), 1)
    model = str(d["model_id"]).split("/")[-1]
    return model, round(d["total_input_tokens"] / n), round(d["total_output_tokens"] / n)


# ---------- drawing helpers ----------

def ms_fmt(v, _=None) -> str:
    return f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"


def log_ms_axis(ax, lo: float, hi: float) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(FuncFormatter(ms_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=0)


def series(ax, x, y, color, ok=None, label=None) -> None:
    """One line. Filled markers meet the SLO, hollow markers miss it."""
    ax.plot(x, y, color=color, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=2, label=label)
    ok = ok if ok is not None else [True] * len(x)
    hit = [(a, b) for a, b, o in zip(x, y, ok) if o]
    miss = [(a, b) for a, b, o in zip(x, y, ok) if not o]
    if hit:
        ax.plot(*zip(*hit), "o", color=color, ms=5.5, mec=SURFACE, mew=1.0, zorder=3)
    if miss:
        ax.plot(*zip(*miss), "o", mfc=SURFACE, mec=color, mew=1.4, ms=5.5, zorder=3)


def hline(ax, y: float, text: str, dashed: bool = False) -> None:
    """Reference line with its label at the left edge, where every curve here is still low."""
    ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)) if dashed else "-", zorder=1)
    tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate(text, (0.008, y), xycoords=tr, xytext=(0, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=INK2)


def vline(ax, x: float, text: str) -> None:
    ax.axvline(x, color=MUTED, lw=0.9, zorder=1)
    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(text, (x, 0.97), xycoords=tr, xytext=(-5, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8.5, color=INK2)


def label_at(ax, x: float, y: float, text: str, side: str = "left") -> None:
    """Direct label beside one point. side: left, right, above, below."""
    pos = {
        "left": dict(xytext=(-7, 0), ha="right", va="center"),
        "right": dict(xytext=(7, 0), ha="left", va="center"),
        "above": dict(xytext=(0, 7), ha="center", va="bottom"),
        "below": dict(xytext=(0, -7), ha="center", va="top"),
    }[side]
    ax.annotate(text, (x, y), textcoords="offset points", fontsize=8.5, color=INK2, **pos)


def nearest(points: list[dict], key: str, target: float) -> dict:
    return min(points, key=lambda d: abs(float(d[key]) - target))


def slo_handles(color: str) -> list:
    return [
        Line2D([], [], color=color, marker="o", ms=5.5, mec=SURFACE, mew=1.0, lw=1.6, label="meets SLO"),
        Line2D([], [], color=color, marker="o", ms=5.5, mfc=SURFACE, mec=color, mew=1.4, lw=1.6, label="misses SLO"),
    ]


def title(ax, text: str, subtitle: str) -> None:
    ax.set_title(text, pad=20)
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, color=INK2)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ---------- figures ----------

def fig_closed_throughput(closed: list[dict], out: Path, sub: str) -> None:
    c = col(closed, "max_concurrency")
    rps = col(closed, "request_throughput")
    ok = [meets_slo(d) for d in closed]
    _, _, out_len = request_shape(closed)
    ceiling = max(rps)

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    series(ax, c, rps, BLUE, ok)
    hline(ax, ceiling, f"ceiling {ceiling:.1f} req/s")
    for d in closed:
        if int(d["max_concurrency"]) in (32, 64, 128):
            label_at(ax, d["max_concurrency"], d["request_throughput"], f"{d['request_throughput']:.0f}", "above")

    # C doubles for most of the sweep but not all of it, so plot every point at its true x.
    ax.set_xscale("log", base=2)
    pow2 = [v for v in c if math.isclose(v, 2 ** round(math.log2(v)))]
    ax.xaxis.set_major_locator(FixedLocator(pow2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(c[0] / 1.4, c[-1] * 1.4)
    ax.set_ylim(0, ceiling * 1.18)
    ax.set_xlabel("concurrent requests C (log scale)")
    ax.set_ylabel("requests / s")

    # Same series in the other unit: output length is fixed, so tokens/s is a pure rescale.
    sec = ax.secondary_yaxis("right", functions=(lambda r: r * out_len, lambda t: t / out_len))
    sec.spines["right"].set_visible(True)
    sec.spines["right"].set_color(AXIS)
    sec.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=9)
    sec.set_ylabel(f"output tokens / s  (req/s x {out_len})", color=MUTED)

    ax.legend(handles=slo_handles(BLUE), loc="lower right")
    title(ax, "Closed loop: throughput vs concurrency", sub)
    save(fig, out / "closed-throughput.png")


def fig_closed_latency(closed: list[dict], out: Path, sub: str) -> None:
    x = col(closed, "request_throughput")
    ok = [meets_slo(d) for d in closed]
    ttft = col(closed, "p95_ttft_ms")
    tpot = col(closed, "p95_tpot_ms")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    series(a1, x, ttft, BLUE, ok)
    log_ms_axis(a1, 10, max(ttft) * 2.2)
    hline(a1, SLO_TTFT_MS, f"SLO: p95 TTFT <= {SLO_TTFT_MS} ms", dashed=True)
    a1.set_ylabel("p95 time to first token (log scale)")

    series(a2, x, tpot, BLUE, ok)
    hline(a2, SLO_TPOT_MS, f"SLO: p95 TPOT <= {SLO_TPOT_MS} ms", dashed=True)
    a2.set_ylim(0, max(tpot) * 1.2)
    a2.set_ylabel("p95 time per output token (ms)")
    a2.set_xlabel("achieved throughput (requests / s)")
    a2.set_xlim(0, max(x) * 1.08)

    for d in closed:
        c = int(d["max_concurrency"])
        if c in (1, 32, 64, 128, 256, 512, 1024):
            side = "right" if c == 1 else "left"
            label_at(a1, d["request_throughput"], d["p95_ttft_ms"], f"C={c}", side)
        if c in (64, 256, 512, 1024):
            label_at(a2, d["request_throughput"], d["p95_tpot_ms"], f"C={c}", "left")

    a1.legend(handles=slo_handles(BLUE), loc="upper left")
    title(a1, "Closed loop: latency vs throughput", sub)
    save(fig, out / "closed-latency-vs-throughput.png")


def fig_open_latency(open_: list[dict], closed: list[dict], out: Path, sub: str) -> None:
    r = col(open_, "request_rate")
    ok = [meets_slo(d) for d in open_]
    ceiling = max(col(closed, "request_throughput")) if closed else None

    fig, (a1, a2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    panels = ((a1, "ttft", SLO_TTFT_MS), (a2, "tpot", SLO_TPOT_MS))
    for ax, m, slo in panels:
        ax.fill_between(r, col(open_, f"p50_{m}_ms"), col(open_, f"p99_{m}_ms"), color=BLUE, alpha=0.12, lw=0, zorder=1)
        series(ax, r, col(open_, f"p95_{m}_ms"), BLUE, ok)
        hline(ax, slo, f"SLO: p95 {m.upper()} <= {slo} ms", dashed=True)
        if ceiling:
            vline(ax, ceiling, f"closed-loop ceiling {ceiling:.1f} req/s")

    ttft99 = col(open_, "p99_ttft_ms")
    log_ms_axis(a1, 10, max(ttft99) * 2.2)
    a1.set_ylabel("p95 time to first token (log scale)")
    a2.set_ylim(0, max(col(open_, "p99_tpot_ms")) * 1.2)
    a2.set_ylabel("p95 time per output token (ms)")
    a2.set_xlabel("offered arrival rate R (requests / s, Poisson)")
    a2.set_xticks(r)
    a2.set_xlim(0, max(r) * 1.06)

    handles = slo_handles(BLUE) + [Patch(color=BLUE, alpha=0.12, label="p50 to p99")]
    a1.legend(handles=handles, loc="upper left")
    title(a1, "Open loop: latency vs arrival rate", sub)
    save(fig, out / "open-latency-vs-rate.png")


def fig_open_saturation(open_: list[dict], closed: list[dict], out: Path, sub: str) -> None:
    r = col(open_, "request_rate")
    got = col(open_, "request_throughput")
    inflight = col(open_, "max_concurrent_requests")
    ok = [meets_slo(d) for d in open_]
    ceiling = max(col(closed, "request_throughput")) if closed else None
    xmax = max(r) * 1.06

    fig, (a1, a2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    a1.plot([0, xmax], [0, xmax], color=MUTED, lw=0.9, zorder=1)
    ref = nearest(open_, "request_rate", max(r) * 0.3)
    a1.annotate("achieved = offered", (ref["request_rate"], ref["request_rate"]), xytext=(-4, 10),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK2)
    if ceiling:
        hline(a1, ceiling, f"closed-loop ceiling {ceiling:.1f} req/s")
    series(a1, r, got, BLUE, ok)
    last = open_[-1]
    label_at(a1, last["request_rate"], last["request_throughput"], f"{last['request_throughput']:.1f}", "below")
    a1.set_ylim(0, xmax)
    a1.set_ylabel("achieved throughput (requests / s)")

    # In-flight requests grow slowly, then explode past capacity: log scale keeps both regimes readable.
    series(a2, r, inflight, BLUE, ok)
    hline(a2, MAX_NUM_SEQS, f"max_num_seqs = {MAX_NUM_SEQS} (running-batch cap)")
    a2.set_yscale("log")
    a2.set_ylim(10, max(inflight) * 3)
    a2.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a2.yaxis.set_minor_formatter(NullFormatter())
    a2.tick_params(axis="y", which="minor", length=0)
    for d in open_[-3:]:
        label_at(a2, d["request_rate"], d["max_concurrent_requests"], f"{d['max_concurrent_requests']:,.0f}", "left")
    a2.set_ylabel("peak in-flight requests, client side (log scale)")
    a2.set_xlabel("offered arrival rate R (requests / s, Poisson)")
    a2.set_xticks(r)
    a2.set_xlim(0, xmax)

    a1.legend(handles=slo_handles(BLUE), loc="lower right")
    title(a1, "Open loop: what the server keeps up with", sub)
    save(fig, out / "open-saturation.png")


def fig_closed_vs_open(closed: list[dict], open_: list[dict], out: Path, sub: str) -> None:
    xc, xo = col(closed, "request_throughput"), col(open_, "request_throughput")
    okc, oko = [meets_slo(d) for d in closed], [meets_slo(d) for d in open_]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    for ax, m, slo in ((a1, "ttft", SLO_TTFT_MS), (a2, "tpot", SLO_TPOT_MS)):
        series(ax, xc, col(closed, f"p95_{m}_ms"), BLUE, okc, label="closed loop, fixed C in flight")
        series(ax, xo, col(open_, f"p95_{m}_ms"), ORANGE, oko, label="open loop, Poisson arrivals at R")
        hline(ax, slo, f"SLO: p95 {m.upper()} <= {slo} ms", dashed=True)
        # The curves meet at the ceiling, so label them where they are far apart.
        target = max(xc) * 0.63
        dc, do = nearest(closed, "request_throughput", target), nearest(open_, "request_throughput", target)
        label_at(ax, dc["request_throughput"], dc[f"p95_{m}_ms"], f"closed, C={int(dc['max_concurrency'])}", "above")
        label_at(ax, do["request_throughput"], do[f"p95_{m}_ms"], f"open, R={do['request_rate']:g}", "below")

    log_ms_axis(a1, 10, max(max(col(closed, "p95_ttft_ms")), max(col(open_, "p95_ttft_ms"))) * 2.2)
    a1.set_ylabel("p95 time to first token (log scale)")
    a2.set_ylim(0, max(max(col(closed, "p95_tpot_ms")), max(col(open_, "p95_tpot_ms"))) * 1.2)
    a2.set_ylabel("p95 time per output token (ms)")
    a2.set_xlabel("achieved throughput (requests / s)")
    a2.set_xlim(0, max(max(xc), max(xo)) * 1.08)

    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=5.5, mec=SURFACE, lw=1.6, label="closed loop, fixed C in flight"),
        Line2D([], [], color=ORANGE, marker="o", ms=5.5, mec=SURFACE, lw=1.6, label="open loop, Poisson arrivals at R"),
        Line2D([], [], ls="none", marker="o", ms=5.5, mfc=SURFACE, mec=INK2, mew=1.4, label="hollow marker: misses SLO"),
    ]
    a1.legend(handles=handles, loc="upper left")
    title(a1, "Closed vs open loop at the same throughput", sub)
    save(fig, out / "closed-vs-open.png")


# ---------- main ----------

def find_results(root: Path) -> Path:
    found = {p.parent.parent for p in root.rglob("closed-*.json")} | {p.parent.parent for p in root.rglob("open-*.json")}
    if len(found) == 1:
        return found.pop()
    if not found:
        sys.exit(f"no closed-*.json or open-*.json under {root}")
    sys.exit(f"{len(found)} result sets under {root}, pass --results:\n  " + "\n  ".join(map(str, sorted(found))))


def print_table(kind: str, points: list[dict], key: str, label: str) -> None:
    for d in points:
        verdict = "meets SLO" if meets_slo(d) else "misses SLO"
        failed = f"  failed={d['failed']}" if int(d.get("failed") or 0) else ""
        print(f"{kind:6} {label}={float(d[key]):>6g}  {d['request_throughput']:5.1f} req/s  "
              f"p95 TTFT {d['p95_ttft_ms']:7.0f} ms  p95 TPOT {d['p95_tpot_ms']:6.1f} ms  {verdict}{failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="model result dir holding closed/ and open/ (default: the one under results/)")
    ap.add_argument("--out", type=Path, default=Path("plots"), help="output folder (default: plots/)")
    args = ap.parse_args()

    results = args.results or find_results(Path("results"))
    closed = load(results / "closed", "max_concurrency") if (results / "closed").is_dir() else []
    open_ = load(results / "open", "request_rate") if (results / "open").is_dir() else []
    if not closed and not open_:
        sys.exit(f"no results in {results}")

    model, in_len, out_len = request_shape(closed or open_)
    shape = f"{model}, {in_len} in / {out_len} out tokens per request"
    print(f"results: {results}  ({len(closed)} closed points, {len(open_)} open points)")
    print_table("closed", closed, "max_concurrency", "C")
    print_table("open", open_, "request_rate", "R")

    if closed:
        fig_closed_throughput(closed, args.out, f"{shape}. Each client waits for its reply before sending the next.")
        fig_closed_latency(closed, args.out, f"{shape}. One point per concurrency level, C = 1 to {int(closed[-1]['max_concurrency'])}.")
    if open_:
        fig_open_latency(open_, closed, args.out, f"{shape}. No concurrency cap, {round(open_[0]['duration'])} s per point.")
        fig_open_saturation(open_, closed, args.out, f"{shape}. Requests arrive whether or not the server keeps up.")
    if closed and open_:
        fig_closed_vs_open(closed, open_, args.out, f"{shape}. Same server, same request shape, different load pattern.")


if __name__ == "__main__":
    main()
