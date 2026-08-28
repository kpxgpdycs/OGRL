from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assortment import (
    _assortment_at_lambda,
    _critical_points,
    _d_value,
    assortment,
    rmnl_expected_revenue,
)

ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def demo_products() -> np.ndarray:
    return np.array(
        [
            [1.60, 7.0],
            [1.30, 6.0],
            [1.00, 5.0],
            [0.80, 8.0],
            [0.60, 4.0],
            [0.40, 9.0],
            [0.72, 6.8],
            [0.52, 7.7],
        ],
        dtype=np.float64,
    )


def trace_binary_search(products: np.ndarray, capacity: int):
    v, w = products[:, 0], products[:, 1]
    points = _critical_points(v, w)
    trace = []

    if len(points) == 0:
        return points, trace, 0.0

    d_first = _d_value(v, w, capacity, float(points[0]))
    d_last = _d_value(v, w, capacity, float(points[-1]))
    trace.append({
        "phase": "bounds",
        "lo": 0,
        "hi": len(points) - 1,
        "mid": None,
        "lambda": None,
        "d": None,
        "decision": "Check D at the two extreme critical points",
    })

    if abs(d_first) <= 1e-11:
        return points, trace, float(points[0])
    if d_first < 0.0:
        return points, trace, float(points[0] - 1.0)
    if abs(d_last) <= 1e-11:
        return points, trace, float(points[-1])
    if d_last > 0.0:
        return points, trace, float(points[-1] + 1.0)

    lo, hi = 0, len(points) - 1
    step = 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lam = float(points[mid])
        d_mid = _d_value(v, w, capacity, lam)
        if abs(d_mid) <= 1e-11:
            trace.append({
                "phase": "search",
                "step": step,
                "lo": lo,
                "hi": hi,
                "mid": mid,
                "lambda": lam,
                "d": d_mid,
                "decision": "D(λ_mid) ≈ 0 → root found",
            })
            lo = hi = mid
            break

        decision = "D(λ_mid) > 0 → lo = mid" if d_mid > 0.0 else "D(λ_mid) < 0 → hi = mid"
        trace.append({
            "phase": "search",
            "step": step,
            "lo": lo,
            "hi": hi,
            "mid": mid,
            "lambda": lam,
            "d": d_mid,
            "decision": decision,
        })
        if d_mid > 0.0:
            lo = mid
        else:
            hi = mid
        trace.append({
            "phase": "updated",
            "step": step,
            "lo": lo,
            "hi": hi,
            "mid": None,
            "lambda": None,
            "d": None,
            "decision": f"Bracket shrinks to [{lo}, {hi}]",
        })
        step += 1

    if lo == hi:
        lam_test = float(points[lo])
    else:
        d_lo = _d_value(v, w, capacity, float(points[lo]))
        d_hi = _d_value(v, w, capacity, float(points[hi]))
        if abs(d_lo) <= 1e-11:
            lam_test = float(points[lo])
        elif abs(d_hi) <= 1e-11:
            lam_test = float(points[hi])
        else:
            lam_test = float((points[lo] + points[hi]) * 0.5)

    trace.append({
        "phase": "final",
        "step": step,
        "lo": lo,
        "hi": hi,
        "mid": None,
        "lambda": lam_test,
        "d": _d_value(v, w, capacity, lam_test),
        "decision": "Use an interior λ from the final bracket",
    })
    return points, trace, lam_test


def generate_binary_search_gif() -> Path:
    products = demo_products()
    capacity = 3
    v, w = products[:, 0], products[:, 1]
    selected_opt, optimum, _ = assortment(products, capacity)
    points, trace, lam_final = trace_binary_search(products, capacity)

    q = 1.0 + float(v.sum())
    beta = 1.0 - 1.0 / q
    x_min = float(points.min()) - 0.7
    x_max = float(points.max()) + 0.7
    lambda_grid = np.linspace(x_min, x_max, 500)
    e_lines = np.vstack([v[i] * (beta * w[i] - lambda_grid) for i in range(len(v))])
    d_grid = np.array([_d_value(v, w, capacity, float(x)) for x in lambda_grid])

    expanded = []
    for state in trace:
        repeats = 7 if state["phase"] in {"bounds", "final"} else 5
        expanded.extend([state] * repeats)

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], width_ratios=[1.22, 1.0], hspace=0.32, wspace=0.24)
    ax_scores = fig.add_subplot(gs[:, 0])
    ax_points = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("RMNL Assortment: Binary Search over Critical Points", fontsize=16, fontweight="bold", y=0.98)

    def draw(frame: int):
        state = expanded[frame]
        ax_scores.clear(); ax_points.clear(); ax_d.clear()

        lo = int(state["lo"]) if state["lo"] is not None else 0
        hi = int(state["hi"]) if state["hi"] is not None else len(points) - 1
        mid = state["mid"]
        lam = state["lambda"]

        if lam is None:
            lam_display = float((points[lo] + points[hi]) * 0.5)
        else:
            lam_display = float(lam)
        selected = list(_assortment_at_lambda(v, w, capacity, lam_display))
        selected_set = set(selected)

        # Panel A: parametric scores
        for i in range(len(v)):
            if i in selected_set:
                ax_scores.plot(lambda_grid, e_lines[i], linewidth=2.4, alpha=0.95, label=f"Item {i}")
            else:
                ax_scores.plot(lambda_grid, e_lines[i], linewidth=1.05, alpha=0.28)
        ax_scores.axhline(0, color="black", linewidth=0.9, alpha=0.6)
        ax_scores.axvline(lam_display, color="black", linestyle="--", linewidth=1.8)
        ax_scores.set_xlim(x_min, x_max)
        ax_scores.set_xlabel("λ")
        ax_scores.set_ylabel("eᵢ(λ)")
        ax_scores.set_title("(a) Parametric item scores", loc="left", fontweight="bold")
        ax_scores.text(
            0.025, 0.025,
            f"Current λ = {lam_display:.4f}\nTop-{capacity} positive items = {selected}",
            transform=ax_scores.transAxes, fontsize=10, va="bottom",
            bbox={"boxstyle":"round,pad=0.35", "facecolor":"white", "alpha":0.92}
        )

        # Panel B: actual binary-search bracket
        xs = np.arange(len(points))
        ax_points.scatter(xs, np.zeros_like(xs), s=22, alpha=0.55)
        ax_points.axhline(0, color="black", linewidth=0.9)
        ax_points.axvspan(lo, hi, alpha=0.10)
        ax_points.scatter([lo], [0], s=90, marker="o", zorder=4)
        ax_points.scatter([hi], [0], s=90, marker="o", zorder=4)
        ax_points.text(lo, 0.09, "lo", ha="center", fontsize=10, fontweight="bold")
        ax_points.text(hi, 0.09, "hi", ha="center", fontsize=10, fontweight="bold")
        if mid is not None:
            mid_i = int(mid)
            ax_points.scatter([mid_i], [0], s=120, marker="D", zorder=5)
            ax_points.text(mid_i, 0.18, "mid", ha="center", fontsize=10, fontweight="bold")
        ax_points.set_xlim(-1, len(points))
        ax_points.set_ylim(-0.22, 0.34)
        ax_points.set_yticks([])
        tick_step = max(1, len(points) // 12)
        ticks = np.arange(0, len(points), tick_step)
        ax_points.set_xticks(ticks, [str(i) for i in ticks], fontsize=8)
        ax_points.set_xlabel("index in sorted critical-point array")
        ax_points.set_title("(b) Binary-search interval", loc="left", fontweight="bold")

        step_label = "Initialization" if state["phase"] == "bounds" else (
            "Final bracket" if state["phase"] == "final" else f"Step {state.get('step', '')}"
        )
        detail = state["decision"]
        if state["d"] is not None and state["phase"] == "search":
            detail += f"\nλ_mid = {float(state['lambda']):.4f},   D(λ_mid) = {float(state['d']):.4f}"
        ax_points.text(
            0.02, 0.96, f"{step_label}\n{detail}", transform=ax_points.transAxes,
            va="top", fontsize=10,
            bbox={"boxstyle":"round,pad=0.35", "facecolor":"white", "alpha":0.94}
        )

        # Panel C: monotone D(lambda)
        ax_d.plot(lambda_grid, d_grid, linewidth=2.0)
        ax_d.axhline(0, color="black", linewidth=0.9)
        ax_d.axvline(lam_display, color="black", linestyle="--", linewidth=1.4)
        ax_d.scatter([lam_display], [_d_value(v, w, capacity, lam_display)], s=55, zorder=4)
        if lo != hi:
            ax_d.axvspan(float(points[lo]), float(points[hi]), alpha=0.10)
        ax_d.set_xlim(x_min, x_max)
        ax_d.set_xlabel("λ")
        ax_d.set_ylabel("D(λ)")
        ax_d.set_title("(c) Monotone feasibility oracle", loc="left", fontweight="bold")

        if state["phase"] == "final":
            final_s = list(_assortment_at_lambda(v, w, capacity, lam_final))
            final_r = rmnl_expected_revenue(products, final_s)
            fig.text(
                0.5, 0.015,
                f"Converged: final λ = {lam_final:.4f}   |   assortment = {final_s}   |   expected revenue = {final_r:.4f}   |   optimum = {optimum:.4f}",
                ha="center", fontsize=11, fontweight="bold"
            )
        else:
            fig.text(
                0.5, 0.015,
                "Each frame follows the same lo / mid / hi update used by the public assortment() implementation.",
                ha="center", fontsize=10
            )
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(expanded), interval=180, blit=False)
    out = ASSET_DIR / "assortment_search.gif"
    ani.save(out, writer=animation.PillowWriter(fps=6), dpi=105)
    plt.close(fig)
    return out


if __name__ == "__main__":
    output = generate_binary_search_gif()
    print(output)
