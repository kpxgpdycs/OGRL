# -*- coding: utf-8 -*-
"""Exact assortment optimization for the RMNL model."""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Sequence

import numpy as np

EPS = 1e-12


def _validate_products(products: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray, int]:
    arr = np.asarray(products, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"products must have shape (I, 2), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("products must contain at least one item")
    if not np.all(np.isfinite(arr)):
        raise ValueError("products contains NaN or Inf")
    if np.any(arr[:, 0] < -EPS) or np.any(arr[:, 1] < -EPS):
        raise ValueError("preference values and effective revenues must be non-negative")
    if int(capacity) != capacity or capacity < 0:
        raise ValueError(f"capacity must be a non-negative integer, got {capacity}")

    v = np.maximum(arr[:, 0], 0.0)
    w = np.maximum(arr[:, 1], 0.0)
    k = min(int(capacity), arr.shape[0])
    return v, w, k


def rmnl_choice_probabilities(products: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    """Return item probabilities followed by the no-purchase probability."""
    v, _w, _ = _validate_products(products, len(selected))
    selected_arr = np.asarray(sorted(set(int(i) for i in selected)), dtype=np.int64)
    if selected_arr.size and (selected_arr.min() < 0 or selected_arr.max() >= len(v)):
        raise IndexError("selected contains an invalid item index")

    q = 1.0 + float(v.sum())
    v_s = float(v[selected_arr].sum()) if selected_arr.size else 0.0
    denom_s = 1.0 + v_s

    p = v / (denom_s * q)
    if selected_arr.size:
        p[selected_arr] = v[selected_arr] / denom_s

    p0 = 1.0 - float(p.sum())
    if p0 < -1e-9:
        raise FloatingPointError(f"invalid RMNL probability mass: outside probability={p0}")
    return np.concatenate([p, np.array([max(0.0, p0)], dtype=np.float64)])


def rmnl_expected_revenue(products: np.ndarray, selected: Sequence[int]) -> float:
    """Return the RMNL expected revenue for an assortment."""
    v, w, _ = _validate_products(products, len(selected))
    selected_arr = np.asarray(sorted(set(int(i) for i in selected)), dtype=np.int64)
    if selected_arr.size and (selected_arr.min() < 0 or selected_arr.max() >= len(v)):
        raise IndexError("selected contains an invalid item index")

    q = 1.0 + float(v.sum())
    p_total = float(np.dot(v, w))
    if selected_arr.size:
        v_s = float(v[selected_arr].sum())
        vw_s = float(np.dot(v[selected_arr], w[selected_arr]))
    else:
        v_s = 0.0
        vw_s = 0.0

    return vw_s / (1.0 + v_s) + (p_total - vw_s) / ((1.0 + v_s) * q)


def _critical_points(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    q = 1.0 + float(v.sum())
    beta = 1.0 - 1.0 / q
    points: list[float] = []

    positive_v = np.flatnonzero(v > EPS)
    points.extend((beta * w[positive_v]).tolist())

    for i, j in itertools.combinations(range(len(v)), 2):
        denom = v[i] - v[j]
        if abs(denom) <= EPS:
            continue
        lam = beta * (v[i] * w[i] - v[j] * w[j]) / denom
        if math.isfinite(lam):
            points.append(float(lam))

    if not points:
        return np.empty(0, dtype=np.float64)
    return np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=12))


def _test_lambda_for_interval(points: np.ndarray, interval_idx: int) -> float:
    n = len(points)
    if not 0 <= interval_idx <= n:
        raise IndexError(f"interval_idx={interval_idx} outside [0,{n}]")
    if n == 0:
        return 0.0
    if interval_idx == 0:
        margin = max(1.0, abs(float(points[0])) + 1.0)
        return float(points[0] - margin)
    if interval_idx == n:
        margin = max(1.0, abs(float(points[-1])) + 1.0)
        return float(points[-1] + margin)
    return float((points[interval_idx - 1] + points[interval_idx]) * 0.5)


def _assortment_at_lambda(v: np.ndarray, w: np.ndarray, capacity: int, lam: float) -> tuple[int, ...]:
    if capacity <= 0:
        return tuple()
    q = 1.0 + float(v.sum())
    beta = 1.0 - 1.0 / q
    e = v * (beta * w - lam)
    positive = np.flatnonzero(e > EPS)
    if positive.size == 0:
        return tuple()
    order_local = np.lexsort((positive, -e[positive]))
    chosen = positive[order_local[:capacity]]
    return tuple(sorted(int(i) for i in chosen))


def _d_value(v: np.ndarray, w: np.ndarray, capacity: int, lam: float) -> float:
    q = 1.0 + float(v.sum())
    beta = 1.0 - 1.0 / q
    e0 = float(np.dot(v, w)) / q - lam
    if capacity <= 0:
        return e0

    positive = (v * (beta * w - lam))
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return e0
    if positive.size > capacity:
        positive = np.partition(positive, positive.size - capacity)[-capacity:]
    return float(e0 + positive.sum())


def assortment(products: np.ndarray, capacity: int) -> tuple[list[int], float, list[float]]:
    """Solve the capacity-constrained RMNL assortment problem."""
    v, w, k = _validate_products(products, capacity)
    canonical_products = np.column_stack([v, w])

    if k == 0 or float(v.sum()) <= EPS:
        selected: list[int] = []
        revenue = rmnl_expected_revenue(canonical_products, selected)
        probs = rmnl_choice_probabilities(canonical_products, selected).tolist()
        return selected, revenue, probs

    points = _critical_points(v, w)
    d_cache: dict[int, float] = {}

    def d_at_point(idx: int) -> float:
        if idx not in d_cache:
            d_cache[idx] = _d_value(v, w, k, float(points[idx]))
        return d_cache[idx]

    if len(points) == 0:
        lam_test = 0.0
    else:
        d_first = d_at_point(0)
        d_last = d_at_point(len(points) - 1)

        if abs(d_first) <= 1e-11:
            lam_test = float(points[0])
        elif d_first < 0.0:
            lam_test = _test_lambda_for_interval(points, 0)
        elif abs(d_last) <= 1e-11:
            lam_test = float(points[-1])
        elif d_last > 0.0:
            lam_test = _test_lambda_for_interval(points, len(points))
        else:
            lo, hi = 0, len(points) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                d_mid = d_at_point(mid)
                if abs(d_mid) <= 1e-11:
                    lo = hi = mid
                    break
                if d_mid > 0.0:
                    lo = mid
                else:
                    hi = mid

            if lo == hi:
                lam_test = float(points[lo])
            else:
                d_lo = d_at_point(lo)
                d_hi = d_at_point(hi)
                if abs(d_lo) <= 1e-11:
                    lam_test = float(points[lo])
                elif abs(d_hi) <= 1e-11:
                    lam_test = float(points[hi])
                else:
                    lam_test = float((points[lo] + points[hi]) * 0.5)

    selected = list(_assortment_at_lambda(v, w, k, lam_test))
    revenue = rmnl_expected_revenue(canonical_products, selected)
    probs = rmnl_choice_probabilities(canonical_products, selected).tolist()
    return selected, float(revenue), probs


def _exhaustive_assortment(products: np.ndarray, capacity: int) -> tuple[list[int], float]:
    v, w, k = _validate_products(products, capacity)
    canonical_products = np.column_stack([v, w])
    best_s: tuple[int, ...] = tuple()
    best_r = rmnl_expected_revenue(canonical_products, best_s)

    for size in range(1, k + 1):
        for selected in itertools.combinations(range(len(v)), size):
            revenue = rmnl_expected_revenue(canonical_products, selected)
            if revenue > best_r + 1e-12 or (
                abs(revenue - best_r) <= 1e-12 and selected < best_s
            ):
                best_s, best_r = selected, revenue
    return list(best_s), float(best_r)


def _run_test_case() -> None:
    products = np.array(
        [
            [2.0, 8.0],
            [1.2, 6.0],
            [0.8, 4.0],
            [0.4, 9.0],
        ],
        dtype=np.float64,
    )
    capacity = 3

    selected, revenue, probabilities = assortment(products, capacity)
    expected_selected, expected_revenue = _exhaustive_assortment(products, capacity)

    print("Test input:")
    print(products)
    print(f"capacity={capacity}")
    print("Test output:")
    print(f"selected={selected}")
    print(f"expected_revenue={revenue:.12f}")
    print(f"choice_probabilities={np.round(probabilities, 12).tolist()}")

    if selected != expected_selected:
        raise AssertionError(f"selected mismatch: {selected} != {expected_selected}")
    if not np.isclose(revenue, expected_revenue, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"revenue mismatch: {revenue} != {expected_revenue}")
    if not np.isclose(sum(probabilities), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("choice probabilities do not sum to 1")
    print("[PASS] deterministic assortment test")


def _run_random_tests(num_cases: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for case_idx in range(num_cases):
        num_items = int(rng.integers(2, 9))
        capacity = int(rng.integers(1, num_items + 1))
        products = np.column_stack(
            [
                rng.uniform(0.01, 3.0, size=num_items),
                rng.uniform(0.05, 10.0, size=num_items),
            ]
        )
        selected, revenue, probabilities = assortment(products, capacity)
        _expected_selected, expected_revenue = _exhaustive_assortment(products, capacity)
        if not np.isclose(revenue, expected_revenue, rtol=1e-10, atol=1e-10):
            raise AssertionError(
                f"random case {case_idx} failed: selected={selected}, "
                f"revenue={revenue}, expected={expected_revenue}"
            )
        if not np.isclose(sum(probabilities), 1.0, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"random case {case_idx} has invalid probability mass")
    print(f"[PASS] {num_cases} random assortment tests")


def main() -> None:
    parser = argparse.ArgumentParser(description="RMNL assortment self-test")
    parser.add_argument("--random-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    _run_test_case()
    _run_random_tests(args.random_cases, args.seed)


if __name__ == "__main__":
    main()
