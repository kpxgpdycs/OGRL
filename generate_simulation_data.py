"""Generate request, preference, and social-graph files for simulation."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def generate_initial_prefs(num_users: int, num_items: int, seed: int) -> np.ndarray:
    """Generate normalized user preferences."""
    rng = np.random.default_rng(seed)
    prefs = rng.random((num_users, num_items))
    row_sums = prefs.sum(axis=1, keepdims=True)
    return prefs / np.where(row_sums == 0.0, 1.0, row_sums)


def generate_social_graph(user_prefs: np.ndarray, knn_k: int) -> np.ndarray:
    """Build an undirected K-NN graph from cosine similarity."""
    prefs = np.asarray(user_prefs, dtype=np.float64)
    if prefs.ndim != 2:
        raise ValueError("user_prefs must be a 2D array")

    num_users = prefs.shape[0]
    if num_users == 0:
        raise ValueError("user_prefs must contain at least one user")

    k = min(max(int(knn_k), 0), max(num_users - 1, 0))
    norms = np.linalg.norm(prefs, axis=1, keepdims=True)
    normalized = prefs / np.where(norms == 0.0, 1.0, norms)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    graph = np.zeros((num_users, num_users), dtype=np.float32)
    if k == 0:
        return graph

    for user in range(num_users):
        neighbors = np.argpartition(similarity[user], -k)[-k:]
        graph[user, neighbors] = 1.0
        graph[neighbors, user] = 1.0
    return graph


def _generate_item_profiles(
    num_items: int,
    time_slots: int,
    py_rng: random.Random,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for item_id in range(num_items):
        release_t = py_rng.randint(0, time_slots // 4)
        peak_popularity = py_rng.uniform(0.5, 1.0)
        active_lifecycle_len = time_slots - release_t - py_rng.randint(0, max(1, time_slots // 5))
        active_lifecycle_len = max(active_lifecycle_len, 10)
        growth_len = int(active_lifecycle_len * py_rng.uniform(0.1, 0.2))
        peak_len = int(active_lifecycle_len * py_rng.uniform(0.3, 0.5))
        fluctuation_period = py_rng.uniform(4.0, 8.0)
        fluctuation_amplitude = peak_popularity * py_rng.uniform(0.15, 0.4)
        profiles.append(
            {
                "id": item_id,
                "release_t": release_t,
                "peak_popularity": peak_popularity,
                "growth_end_t": release_t + growth_len,
                "peak_end_t": release_t + growth_len + peak_len,
                "fluctuation_period": fluctuation_period,
                "fluctuation_amplitude": fluctuation_amplitude,
            }
        )
    return profiles


def _calculate_popularity_trends(
    item_profiles: list[dict[str, Any]],
    time_slots: int,
    np_rng: np.random.Generator,
    decay_rate: float,
) -> np.ndarray:
    pop_trends = np.zeros((len(item_profiles), time_slots), dtype=np.float64)

    for item_idx, profile in enumerate(item_profiles):
        for t in range(time_slots):
            if t < profile["release_t"]:
                pop = 0.0
            elif t <= profile["growth_end_t"]:
                progress = (t - profile["release_t"]) / (
                    profile["growth_end_t"] - profile["release_t"] + 1
                )
                pop = profile["peak_popularity"] * progress
            elif t <= profile["peak_end_t"]:
                pop = profile["peak_popularity"]
            else:
                time_past_peak = t - profile["peak_end_t"]
                pop = profile["peak_popularity"] * np.exp(-decay_rate * time_past_peak)

            if pop > 0.0:
                fluctuation = profile["fluctuation_amplitude"] * np.sin(
                    (2.0 * np.pi / profile["fluctuation_period"]) * t
                )
                pop += fluctuation + np_rng.normal(0.0, pop * 0.05)
            pop_trends[item_idx, t] = max(0.0, pop)

    column_sums = pop_trends.sum(axis=0, keepdims=True)
    valid = column_sums[0] > 0.0
    pop_trends[:, valid] /= column_sums[:, valid]
    return pop_trends


def _visualize_trends(pop_trends: np.ndarray, output_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_samples = min(10, pop_trends.shape[0])
    sample_indices = rng.choice(pop_trends.shape[0], size=num_samples, replace=False)

    plt.figure(figsize=(12, 7))
    for item_idx in sample_indices:
        curve = pop_trends[item_idx]
        max_value = float(np.max(curve))
        normalized = curve / max_value if max_value > 0.0 else curve
        plt.plot(range(pop_trends.shape[1]), normalized, label=f"Item {item_idx}")

    plt.xlabel("Time Slot")
    plt.ylabel("Normalized Popularity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_simulation_data(
    num_users: int,
    num_items: int,
    time_slots: int,
    total_requests: int,
    seed: int,
    decay_rate: float = 0.05,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Generate request tuples in (user_id, time_slot, item_id) order."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    profiles = _generate_item_profiles(num_items, time_slots, py_rng)
    pop_trends = _calculate_popularity_trends(profiles, time_slots, np_rng, decay_rate)

    valid_time_slots = np.flatnonzero(pop_trends.sum(axis=0) > 0.0).tolist()
    if not valid_time_slots:
        raise RuntimeError("no valid time slots with positive item popularity")

    requests: list[tuple[int, int, int]] = []
    item_ids = list(range(num_items))
    for _ in range(total_requests):
        t = py_rng.choice(valid_time_slots)
        uid = py_rng.randrange(num_users)
        item_id = py_rng.choices(item_ids, weights=pop_trends[:, t], k=1)[0]
        requests.append((uid, t, item_id))

    requests.sort(key=lambda record: record[1])
    return requests, pop_trends


def save_simulation_dataset(
    output_dir: Path,
    requests: list[tuple[int, int, int]],
    user_prefs: np.ndarray,
    social_graph: np.ndarray,
) -> None:
    """Save files consumed by the caching experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "records.pkl", "wb") as f:
        pickle.dump(requests, f)
    np.save(output_dir / "user_prefs.npy", user_prefs)
    np.save(output_dir / "social_graph.npy", social_graph)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simulation data")
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--num-items", type=int, default=100)
    parser.add_argument("--time-slots", type=int, default=100)
    parser.add_argument("--total-requests", type=int, default=5000)
    parser.add_argument("--social-knn-k", type=int, default=5)
    parser.add_argument("--decay-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--visualize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requests, pop_trends = generate_simulation_data(
        num_users=args.num_users,
        num_items=args.num_items,
        time_slots=args.time_slots,
        total_requests=args.total_requests,
        seed=args.seed,
        decay_rate=args.decay_rate,
    )
    user_prefs = generate_initial_prefs(args.num_users, args.num_items, args.seed)
    social_graph = generate_social_graph(user_prefs, args.social_knn_k)
    save_simulation_dataset(args.output_dir, requests, user_prefs, social_graph)

    if args.visualize:
        _visualize_trends(pop_trends, args.output_dir / "popularity_trends.png", args.seed)

    print(f"Saved {len(requests)} requests to {args.output_dir / 'records.pkl'}")
    print(f"Saved preferences to {args.output_dir / 'user_prefs.npy'}")
    print(f"Saved social graph to {args.output_dir / 'social_graph.npy'}")
    print("Request format: (user_id, time_slot, item_id)")
    print("Sample requests:")
    for record in requests[:20]:
        print(record)


if __name__ == "__main__":
    main()
