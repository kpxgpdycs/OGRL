
"""PPO-based dynamic caching with RMNL assortment recommendations."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import math
import os
import pickle
import random
import secrets
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict, MultiDiscrete
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import ActorProb

from assortment import assortment

LOG_DIR_BASE = Path("./logs")
MODEL_DIR_BASE = Path("./models")
LOG_DIR_BASE.mkdir(parents=True, exist_ok=True)
MODEL_DIR_BASE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("DynamicCacheWithSocialFixed")

@dataclass
class SystemConfig:
    profit_omega: float
    profit_discount: float
    cache_update_cost_per_bit: float
    thrashing_base_cost_omega_tilde: float

    U: int = 20
    I: int = 100
    T: int = 100
    K: int = 6
    C_cs: int = 1000

    pre_delta: Optional[float] = None
    social_influence_alpha: float = 0.1

    thrashing_max_interval_vm: int = 4

    request_history_window: int = 10
    num_request_features: int = 5

    gamma: float = 0.8
    lr: float = 1e-5
    epoch: int = 600
    batch_size: int = 256
    buffer_size: int = 30000
    num_train_envs: int = 1
    num_test_envs: int = 1
    step_per_epoch: int = 2000
    step_per_collect: int = 1000
    repeat_per_collect: int = 10
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    advantage_normalization: bool = True
    reward_normalization: bool = False
    deterministic_eval: bool = True

    lr_end_factor: float = 0.1

    episode_per_test: int = 3

    seed: Optional[int] = None
    item_size_seed: Optional[int] = None
    item_size_min: int = 10
    item_size_max: int = 100

    data_path: Path = Path("./data/records.pkl")
    prefs_path: Path = Path("./data/user_prefs.npy")
    social_graph_path: Path = Path("./data/social_graph.npy")

    run_name: str = "default_run"
    log_dir: Path = Path("./logs/default_run")
    model_dir: Path = Path("./models/default_run")
    checkpoint_path: Path = Path("./models/default_run/checkpoint.pth")

    s_i: np.ndarray = field(init=False, repr=False)
    device: str = field(init=False)

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = secrets.randbits(32)
        if self.item_size_seed is None:
            self.item_size_seed = self.seed
        if self.pre_delta is None:
            self.pre_delta = 1.0 / self.I
        rng = np.random.default_rng(self.item_size_seed)
        mean = (self.item_size_min + self.item_size_max) / 2.0
        std = (self.item_size_max - self.item_size_min) / 6.0
        self.s_i = np.clip(
            rng.normal(mean, std, self.I), self.item_size_min, self.item_size_max
        ).astype(np.int64)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thrashing_max_interval_vm = max(1, int(self.thrashing_max_interval_vm))
        self._validate()

    def _validate(self) -> None:
        runtime_values = {
            "profit_omega": self.profit_omega,
            "profit_discount": self.profit_discount,
            "cache_update_cost_per_bit": self.cache_update_cost_per_bit,
            "thrashing_base_cost_omega_tilde": self.thrashing_base_cost_omega_tilde,
        }
        for name, value in runtime_values.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative runtime value")
        if not (0.0 <= self.profit_discount <= 1.0):
            raise ValueError("profit_discount must be in [0,1]")
        if not (0.0 <= self.social_influence_alpha <= 1.0):
            raise ValueError("social_influence_alpha must be in [0,1]")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0,1]")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0,1]")
        if self.seed < 0 or self.item_size_seed < 0:
            raise ValueError("seed and item_size_seed must be non-negative")
        if self.U <= 0 or self.I <= 0 or self.T <= 0:
            raise ValueError("U, I, and T must be positive")
        if self.K < 1 or self.K > self.I:
            raise ValueError("K must satisfy 1 <= K <= I")
        if self.C_cs <= 0:
            raise ValueError("C_cs must be positive")
        if self.item_size_min <= 0 or self.item_size_max < self.item_size_min:
            raise ValueError("item_size_min/item_size_max must satisfy 0 < min <= max")
        if self.request_history_window <= 0:
            raise ValueError("request_history_window must be positive")
        if self.batch_size <= 0 or self.buffer_size <= 0 or self.epoch <= 0:
            raise ValueError("batch_size, buffer_size, and epoch must be positive")
        if self.num_train_envs <= 0 or self.num_test_envs <= 0 or self.episode_per_test <= 0:
            raise ValueError("environment counts and episode_per_test must be positive")
        if self.step_per_collect <= 0 or self.step_per_epoch <= 0:
            raise ValueError("step_per_collect and step_per_epoch must be positive")
        if self.step_per_collect > self.step_per_epoch:
            logger.warning(
                "step_per_collect=%d > step_per_epoch=%d; this causes each trainer epoch to overshoot. "
                "It is allowed, but step_per_collect <= step_per_epoch is recommended.",
                self.step_per_collect,
                self.step_per_epoch,
            )

    def policy_updates_per_training(self) -> int:

        updates_per_epoch = max(1, math.ceil(self.step_per_epoch / self.step_per_collect))
        return max(1, self.epoch * updates_per_epoch)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        sensitive = {
            "profit_omega",
            "profit_discount",
            "cache_update_cost_per_bit",
            "thrashing_base_cost_omega_tilde",
        }
        for key, value in list(d.items()):
            if key in sensitive:
                d[key] = "<runtime>"
            elif isinstance(value, Path):
                d[key] = str(value)
            elif isinstance(value, np.ndarray):
                d[key] = value.tolist()
        return d

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_static_data(cfg: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    prefs = np.load(cfg.prefs_path).astype(np.float64)
    graph = np.load(cfg.social_graph_path).astype(np.float64)
    if prefs.shape != (cfg.U, cfg.I):
        raise ValueError(f"user_prefs shape mismatch: expected {(cfg.U, cfg.I)}, got {prefs.shape}")
    if graph.shape != (cfg.U, cfg.U):
        raise ValueError(f"social_graph shape mismatch: expected {(cfg.U, cfg.U)}, got {graph.shape}")
    if not np.all(np.isfinite(prefs)) or not np.all(np.isfinite(graph)):
        raise ValueError("preprocessed preference/social data contains NaN or Inf")
    if np.any(prefs < 0) or np.any(graph < 0):
        raise ValueError("preference/social graph must be non-negative for the current RMNL/GCN model")
    return prefs, graph

class RecommendationGenerator:
    def __init__(self, cfg: SystemConfig, initial_user_prefs: np.ndarray, social_graph: np.ndarray):
        self.cfg = cfg
        self.initial_user_prefs = np.asarray(initial_user_prefs, dtype=np.float64).copy()
        self.social_graph = np.asarray(social_graph, dtype=np.float64).copy()
        self.user_prefs = self.initial_user_prefs.copy()
        self.profit = np.zeros(cfg.I, dtype=np.float64)
        self.p_gcn = self._precompute_gcn_propagation_matrix()

    def _precompute_gcn_propagation_matrix(self) -> np.ndarray:
        a_hat = self.social_graph + np.eye(self.cfg.U, dtype=np.float64)
        degree = a_hat.sum(axis=1)
        inv_sqrt = np.zeros_like(degree)
        mask = degree > 0
        inv_sqrt[mask] = degree[mask] ** -0.5
        return (inv_sqrt[:, None] * a_hat * inv_sqrt[None, :]).astype(np.float64)

    def reset(self) -> None:
        self.user_prefs = self.initial_user_prefs.copy()

    def update_user_prefs(self, u: int, i: int) -> None:
        if 0 <= u < self.cfg.U and 0 <= i < self.cfg.I:
            self.user_prefs[u, i] += float(self.cfg.pre_delta)
            total = float(self.user_prefs[u].sum())
            if total > 0:
                self.user_prefs[u] /= total
            else:
                self.user_prefs[u].fill(1.0 / self.cfg.I)

    def update_profit(self, cached: np.ndarray) -> None:
        base_profit = self.cfg.profit_omega * self.cfg.s_i.astype(np.float64)
        multiplier = np.where(np.asarray(cached, dtype=bool), 1.0, self.cfg.profit_discount)
        self.profit = base_profit * multiplier

    def socially_aware_preferences(self, base_prefs: np.ndarray) -> np.ndarray:
        base = np.asarray(base_prefs, dtype=np.float64)
        if base.shape != (self.cfg.U, self.cfg.I):
            raise ValueError(f"base_prefs shape mismatch: expected {(self.cfg.U, self.cfg.I)}, got {base.shape}")
        social = self.p_gcn @ base
        a = self.cfg.social_influence_alpha
        return np.maximum((1.0 - a) * base + a * social, 0.0)

    def generate_with_assortment(self, cached: np.ndarray, base_prefs: np.ndarray) -> np.ndarray:
        """Generate RMNL recommendations for all users."""
        final_prefs = self.socially_aware_preferences(base_prefs)
        self.update_profit(cached)
        w = np.maximum(self.profit, 0.0)
        sentinel = self.cfg.I
        all_selected: list[int] = []
        for u in range(self.cfg.U):
            products = np.column_stack([final_prefs[u], w])
            selected, _opt, _p = assortment(products, self.cfg.K)
            selected = list(selected[: self.cfg.K])
            selected.extend([sentinel] * (self.cfg.K - len(selected)))
            all_selected.extend(selected)
        return np.asarray(all_selected, dtype=np.int64)

class DynamicDataController:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        prefs, graph = load_static_data(cfg)
        self.recommender = RecommendationGenerator(cfg, prefs, graph)
        self.requests = self._load_requests()
        self.cached_recommendations: Optional[np.ndarray] = None
        self.cache_state = np.zeros(cfg.I, dtype=np.float32)
        self.item_request_history: deque[np.ndarray] = deque(maxlen=cfg.request_history_window)
        self.current_t = 0
        self.reset()

    def _load_requests(self) -> list[list[tuple[int, int]]]:
        with open(self.cfg.data_path, "rb") as f:
            raw = pickle.load(f)

        if isinstance(raw, list) and len(raw) == self.cfg.T:
            valid = True
            for t_list in raw:
                if not isinstance(t_list, list):
                    valid = False
                    break
                for item in t_list:
                    if not (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and 0 <= item[0] < self.cfg.U
                        and 0 <= item[1] < self.cfg.I
                    ):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return raw

        requests_per_t: list[list[tuple[int, int]]] = [[] for _ in range(self.cfg.T)]
        if isinstance(raw, list) and raw and isinstance(raw[0], tuple) and len(raw[0]) == 3:
            flat = raw
        else:
            try:
                flat = list(it for sub in raw for it in sub)
            except TypeError:
                flat = raw

        for record in flat:
            if isinstance(record, (tuple, list)) and len(record) == 3:
                uid, t, item_id = record
                if (
                    isinstance(t, (int, np.integer))
                    and 0 <= int(t) < self.cfg.T
                    and 0 <= int(item_id) < self.cfg.I
                    and 0 <= int(uid) < self.cfg.U
                ):
                    requests_per_t[int(t)].append((int(uid), int(item_id)))
        return requests_per_t

    def reset(self) -> None:
        self.recommender.reset()
        self.current_t = 0
        self.cache_state.fill(0.0)
        self.item_request_history.clear()
        for _ in range(self.cfg.request_history_window):
            self.item_request_history.append(np.zeros(self.cfg.I, dtype=np.float32))

        self.cached_recommendations = self.recommender.generate_with_assortment(
            self.cache_state, self.recommender.user_prefs
        )

    def advance_time_and_update_prefs(self) -> None:
        time_idx = self.current_t
        current_step_requests = np.zeros(self.cfg.I, dtype=np.float32)
        if 0 <= time_idx < len(self.requests):
            for uid, item_id in self.requests[time_idx]:
                current_step_requests[item_id] += 1.0
                self.recommender.update_user_prefs(uid, item_id)
        self.item_request_history.append(current_step_requests)

        self.current_t += 1

    def get_request_history_features(self) -> np.ndarray:
        if not self.item_request_history:
            return np.zeros((self.cfg.I, self.cfg.num_request_features), dtype=np.float32)
        history = np.asarray(self.item_request_history, dtype=np.float32)
        features = np.stack(
            [
                np.mean(history, axis=0),
                np.std(history, axis=0),
                np.max(history, axis=0),
                np.min(history, axis=0),
                history[-1],
            ],
            axis=1,
        )
        for j in range(features.shape[1]):
            col_sum = float(features[:, j].sum())
            if col_sum > 0:
                features[:, j] /= col_sum
        return features.astype(np.float32)

    def update_cache(self, new_cache: np.ndarray) -> None:
        self.cache_state = np.asarray(new_cache, dtype=np.float32).copy()

    def update_recommendations(self, new_recs: np.ndarray) -> None:
        self.cached_recommendations = np.asarray(new_recs, dtype=np.int64).copy()

    def get_current_recommendations(self) -> np.ndarray:
        assert self.cached_recommendations is not None
        return self.cached_recommendations

    def get_current_requests(self) -> list[tuple[int, int]]:
        if 0 <= self.current_t < len(self.requests):
            return self.requests[self.current_t]
        return []

    @property
    def current_cache_state(self) -> np.ndarray:
        return self.cache_state

class DynamicCacheEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.dc = DynamicDataController(cfg)
        self.episode_step = 0
        self.total_steps = 0
        self.current_recommendations: Optional[np.ndarray] = None
        self.cache_history: deque[np.ndarray] = deque(maxlen=cfg.thrashing_max_interval_vm + 2)
        self.last_reward_info: dict[str, float] = {}

        self.observation_space = GymDict(
            {
                "cache": Box(0.0, 1.0, shape=(cfg.I,), dtype=np.float32),
                "recommendations": MultiDiscrete([cfg.I + 1] * (cfg.U * cfg.K)),
                "request_history": Box(
                    0.0,
                    1.0,
                    shape=(cfg.I, cfg.num_request_features),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(0.0, 1.0, shape=(cfg.I,), dtype=np.float32)

    @property
    def cache_state(self) -> np.ndarray:
        return self.dc.current_cache_state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.dc.reset()
        self.episode_step = 0
        self.current_recommendations = self.dc.get_current_recommendations().copy()
        self.last_reward_info = {}
        self.cache_history.clear()
        zero = np.zeros(self.cfg.I, dtype=np.float32)
        for _ in range(self.cache_history.maxlen):
            self.cache_history.append(zero.copy())
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        action = np.clip(action, self.action_space.low, self.action_space.high)

        prev_cache = self.dc.current_cache_state.copy()
        new_cache = self._update_cache_from_action(action)
        current_requests = self.dc.get_current_requests()
        reward, reward_info = self._calculate_reward(
            current_cache=new_cache,
            previous_cache=prev_cache,
            requests=current_requests,
            cache_history=self.cache_history,
        )
        self.last_reward_info = reward_info

        self.dc.update_cache(new_cache)
        self.cache_history.appendleft(new_cache.copy())

        self.dc.advance_time_and_update_prefs()
        self.episode_step += 1
        self.total_steps += 1
        terminated = self.episode_step >= self.cfg.T
        truncated = False

        if not terminated:
            new_recs = self.dc.recommender.generate_with_assortment(
                cached=self.dc.current_cache_state,
                base_prefs=self.dc.recommender.user_prefs,
            )
            self.dc.update_recommendations(new_recs)
            self.current_recommendations = new_recs.copy()

        next_obs = self._get_obs()
        return next_obs, float(reward), terminated, truncated, reward_info

    def _get_obs(self) -> dict[str, np.ndarray]:
        cache_obs = self.dc.current_cache_state.astype(np.float32)
        if self.current_recommendations is None:
            self.current_recommendations = self.dc.get_current_recommendations().copy()
        rec_obs = self.current_recommendations.astype(np.int64)
        req_hist = self.dc.get_request_history_features()

        assert cache_obs.shape == (self.cfg.I,)
        assert rec_obs.shape == (self.cfg.U * self.cfg.K,)
        assert req_hist.shape == (self.cfg.I, self.cfg.num_request_features)
        return {
            "cache": cache_obs,
            "recommendations": rec_obs,
            "request_history": req_hist,
        }

    def _update_cache_from_action(self, action: np.ndarray) -> np.ndarray:
        priority_indices = np.argsort(-action, kind="stable")
        new_cache = np.zeros(self.cfg.I, dtype=np.float32)
        current_size = 0
        for idx in priority_indices:
            item_size = int(self.cfg.s_i[idx])
            if current_size + item_size <= self.cfg.C_cs:
                new_cache[idx] = 1.0
                current_size += item_size
        return new_cache

    def _calculate_reward(
        self,
        current_cache: np.ndarray,
        previous_cache: np.ndarray,
        requests: Sequence[tuple[int, int]],
        cache_history: deque[np.ndarray],
    ) -> tuple[float, dict[str, float]]:
        sizes = self.cfg.s_i
        valid_requests = [req for req in requests if 0 <= req[1] < self.cfg.I]
        num_requests = len(valid_requests)
        hit_reward = 0.0
        num_hits = 0

        if valid_requests:
            requested = np.asarray([req[1] for req in valid_requests], dtype=np.int64)
            is_hit = current_cache[requested].astype(bool)
            num_hits = int(np.sum(is_hit))
            hit_reward = float(
                np.sum(self.cfg.profit_omega * sizes[requested[is_hit]])
                + np.sum(
                    self.cfg.profit_discount
                    * self.cfg.profit_omega
                    * sizes[requested[~is_hit]]
                )
            )

        cache_change = current_cache - previous_cache
        items_added = cache_change > 0
        update_cost = float(
            np.sum(items_added * sizes) * self.cfg.cache_update_cost_per_bit
        )

        thrashing_cost = 0.0
        for item_idx in np.flatnonzero(items_added):
            for psi in range(1, self.cfg.thrashing_max_interval_vm + 1):
                if len(cache_history) >= psi + 1:
                    was_cached_before = cache_history[psi][item_idx] == 1
                    intermediate_absent = (
                        sum(float(cache_history[k][item_idx]) for k in range(psi)) == 0.0
                    )
                    if was_cached_before and intermediate_absent:
                        thrashing_cost += (
                            self.cfg.thrashing_base_cost_omega_tilde / psi
                        ) * float(sizes[item_idx])
                        break

        total = float(hit_reward - update_cost - thrashing_cost)
        info = {
            "system_reward": total,
            "reward_hit": float(hit_reward),
            "cost_update": float(update_cost),
            "cost_thrashing": float(thrashing_cost),
            "hit_rate": float(num_hits / num_requests) if num_requests else 0.0,
            "num_hits": float(num_hits),
            "num_requests": float(num_requests),
        }
        return total, info

def make_env(cfg: SystemConfig) -> DynamicCacheEnv:
    return DynamicCacheEnv(cfg)

class CustomNet(nn.Module):
    def __init__(self, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        self.cache_net = nn.Sequential(
            nn.Linear(cfg.I, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )
        self.embed_dim_rec = 8
        self.rec_embed = nn.Embedding(cfg.I + 1, self.embed_dim_rec, padding_idx=cfg.I)
        self.user_rec_aggregator = nn.Linear(cfg.K * self.embed_dim_rec, 32)

        req_hist_input_dim = cfg.I * cfg.num_request_features
        self.req_hist_net = nn.Sequential(
            nn.Linear(req_hist_input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        self.joint_feature_dim = 64 + 32 + 128
        self.output_dim = 256
        self.joint_net = nn.Sequential(
            nn.Linear(self.joint_feature_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, self.output_dim),
            nn.LeakyReLU(),
        )
        self.to(cfg.device)

    def _process_input(self, obs: Any):
        if isinstance(obs, Batch):
            obs_data = obs
        elif isinstance(obs, dict):
            obs_data = obs
        else:
            raise TypeError(f"Unsupported obs type: {type(obs)}")

        cache_in = obs_data.get("cache")
        recs_in = obs_data.get("recommendations")
        req_hist_in = obs_data.get("request_history")
        if cache_in is None or recs_in is None or req_hist_in is None:
            raise ValueError("Observation missing cache/recommendations/request_history")

        cache = torch.as_tensor(cache_in, dtype=torch.float32, device=self.device)
        recs = torch.as_tensor(recs_in, dtype=torch.long, device=self.device)
        req_hist = torch.as_tensor(req_hist_in, dtype=torch.float32, device=self.device)

        if cache.ndim == 1:
            cache = cache.unsqueeze(0)
        if recs.ndim == 1:
            recs = recs.unsqueeze(0)
        if req_hist.ndim == 2:
            req_hist = req_hist.unsqueeze(0)
        return cache, recs, req_hist, cache.shape[0]

    def forward(self, obs: Any, state: Any = None, info: Optional[dict] = None):
        cache, recs, req_hist, batch_size = self._process_input(obs)
        cache_feat = self.cache_net(cache)
        rec_emb = self.rec_embed(recs)
        rec_per_user = rec_emb.reshape(batch_size, self.cfg.U, self.cfg.K * self.embed_dim_rec)
        user_rec_feat = self.user_rec_aggregator(rec_per_user)
        global_rec_feat = user_rec_feat.mean(dim=1)
        req_feat = self.req_hist_net(req_hist.reshape(batch_size, -1))
        combined = torch.cat([cache_feat, global_rec_feat, req_feat], dim=1)
        return self.joint_net(combined), state

class CustomCriticModule(nn.Module):
    def __init__(self, preprocess_net: CustomNet, value_head: nn.Module):
        super().__init__()
        self.preprocess_net = preprocess_net
        self.value_head = value_head

    def forward(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        features, _ = self.preprocess_net(obs, state=kwargs.get("state"))
        return self.value_head(features).squeeze(-1)

class PolicyFactory:
    @staticmethod
    def build(cfg: SystemConfig, with_scheduler: bool = True):
        act_space = Box(0.0, 1.0, shape=(cfg.I,), dtype=np.float32)
        actor_net = CustomNet(cfg)
        critic_net = CustomNet(cfg)

        actor = ActorProb(
            preprocess_net=actor_net,
            action_shape=act_space.shape,
            hidden_sizes=[],
            max_action=1.0,
            device=cfg.device,
            unbounded=True,
            conditioned_sigma=True,
        ).to(cfg.device)

        critic_head = MLP(
            input_dim=critic_net.output_dim,
            output_dim=1,
            hidden_sizes=[256, 128],
            device=cfg.device,
        ).to(cfg.device)
        critic = CustomCriticModule(critic_net, critic_head).to(cfg.device)

        optim = torch.optim.Adam(
            [
                {"params": actor.parameters(), "lr": cfg.lr},
                {"params": critic.parameters(), "lr": cfg.lr},
            ]
        )

        scheduler = None
        if with_scheduler:
            scheduler = lr_scheduler.LinearLR(
                optim,
                start_factor=1.0,
                end_factor=cfg.lr_end_factor,
                total_iters=cfg.policy_updates_per_training(),
            )

        def dist_fn(actor_output):
            mu, sigma = actor_output
            return torch.distributions.Independent(
                torch.distributions.Normal(mu, sigma), 1
            )

        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=act_space,
            eps_clip=cfg.clip_eps,
            advantage_normalization=cfg.advantage_normalization,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            max_grad_norm=cfg.max_grad_norm,
            gae_lambda=cfg.gae_lambda,
            discount_factor=cfg.gamma,
            reward_normalization=cfg.reward_normalization,
            deterministic_eval=cfg.deterministic_eval,
            action_scaling=True,
            action_bound_method="tanh",
            lr_scheduler=scheduler,
        )
        return policy, scheduler

class TrainingManager:
    def __init__(self, cfg: SystemConfig, resume_mode: Optional[str] = None):
        self.cfg = cfg
        self.resume_mode = resume_mode
        self.resumed_from_checkpoint = False
        self.start_epoch = 0
        self.start_env_step = 0

        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        (cfg.log_dir / "resolved_config.json").write_text(
            json.dumps(cfg.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.policy, self.lr_scheduler = PolicyFactory.build(cfg, with_scheduler=True)
        if resume_mode == "checkpoint":
            self._load_checkpoint()
        elif resume_mode == "best_weights":
            self._load_best_weights()

        train_factories = [functools.partial(make_env, cfg) for _ in range(cfg.num_train_envs)]
        test_factories = [functools.partial(make_env, cfg) for _ in range(cfg.num_test_envs)]
        self.train_envs = SubprocVectorEnv(train_factories, context="spawn")
        self.test_envs = SubprocVectorEnv(test_factories, context="spawn")

        self.buffer = VectorReplayBuffer(cfg.buffer_size, cfg.num_train_envs)
        self.collector = Collector(self.policy, self.train_envs, self.buffer, exploration_noise=True)
        self.test_collector = Collector(self.policy, self.test_envs)

        self.writer = SummaryWriter(cfg.log_dir)
        self.tb_logger = TensorboardLogger(
            self.writer,
            train_interval=1,
            test_interval=1,
            update_interval=100,
        )
        self._last_callback_epoch = -1

    def _load_checkpoint(self) -> None:
        path = self.cfg.checkpoint_path
        if not path.exists():
            logger.warning("Checkpoint not found at %s; starting from scratch.", path)
            return
        checkpoint = torch.load(path, map_location=self.cfg.device)
        self.policy.load_state_dict(checkpoint["model"])
        if checkpoint.get("optim") is not None:
            self.policy.optim.load_state_dict(checkpoint["optim"])
        if self.lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.start_env_step = int(checkpoint.get("env_step", 0))
        self.resumed_from_checkpoint = True
        logger.info("Resumed from checkpoint epoch=%d env_step=%d", self.start_epoch, self.start_env_step)

    def _load_best_weights(self) -> None:
        path = self.cfg.model_dir / "best_policy.pth"
        if path.exists():
            self.policy.load_state_dict(torch.load(path, map_location=self.cfg.device))
            logger.info("Loaded best weights from %s", path)
        else:
            logger.warning("Best weights not found at %s; starting from scratch.", path)

    def _save_checkpoint_fn(self, epoch: int, env_step: int, gradient_step: int) -> str:
        state = {
            "model": self.policy.state_dict(),
            "optim": self.policy.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": int(epoch),
            "env_step": int(env_step),
            "gradient_step": int(gradient_step),
        }
        tmp = self.cfg.checkpoint_path.with_suffix(".pth.tmp")
        torch.save(state, tmp)
        tmp.replace(self.cfg.checkpoint_path)
        return str(self.cfg.checkpoint_path)

    def _epoch_callback(self, epoch: int, env_step: Optional[int] = None) -> None:

        if epoch == self._last_callback_epoch:
            return
        self._last_callback_epoch = epoch

        if self.lr_scheduler is not None:
            self.writer.add_scalar("train/learning_rate", self.lr_scheduler.get_last_lr()[0], epoch)

        try:
            info_list = [
                d for d in self.train_envs.get_env_attr("last_reward_info") if isinstance(d, dict) and d
            ]
            for metric in ("system_reward", "reward_hit", "cost_update", "cost_thrashing", "hit_rate"):
                vals = [float(d[metric]) for d in info_list if metric in d]
                if vals:
                    self.writer.add_scalar(f"env_metrics/train/{metric}", float(np.mean(vals)), epoch)
        except Exception as exc:
            logger.debug("epoch metric callback skipped: %s", exc)

    def _create_trainer(self) -> OnpolicyTrainer:
        def save_best_fn(policy) -> None:
            torch.save(policy.state_dict(), self.cfg.model_dir / "best_policy.pth")

        return OnpolicyTrainer(
            policy=self.policy,
            train_collector=self.collector,
            test_collector=self.test_collector,
            max_epoch=self.cfg.epoch,
            step_per_epoch=self.cfg.step_per_epoch,
            step_per_collect=self.cfg.step_per_collect,
            repeat_per_collect=self.cfg.repeat_per_collect,
            episode_per_test=self.cfg.episode_per_test,
            batch_size=self.cfg.batch_size,
            train_fn=self._epoch_callback,
            save_checkpoint_fn=self._save_checkpoint_fn,
            save_best_fn=save_best_fn,
            stop_fn=lambda mean_rewards: False,
            logger=self.tb_logger,
            resume_from_log=self.resumed_from_checkpoint,
            show_progress=True,
        )

    @staticmethod
    def _batch_single_obs(obs: dict[str, np.ndarray]) -> Batch:
        obs_batch = Batch({key: np.expand_dims(value, axis=0) for key, value in obs.items()})
        return Batch(obs=obs_batch, info=Batch())

    def run_final_evaluation(self) -> dict[str, float]:
        best_path = self.cfg.model_dir / "best_policy.pth"
        if not best_path.exists():
            logger.warning("Best policy not found at %s; evaluating final in-memory policy.", best_path)
            eval_policy, _ = PolicyFactory.build(self.cfg, with_scheduler=False)
            eval_policy.load_state_dict(self.policy.state_dict())
        else:
            eval_policy, _ = PolicyFactory.build(self.cfg, with_scheduler=False)
            eval_policy.load_state_dict(torch.load(best_path, map_location=self.cfg.device))

        eval_policy.eval()
        eval_policy.deterministic_eval = True
        eval_env = DynamicCacheEnv(self.cfg)
        obs, _ = eval_env.reset(seed=self.cfg.seed + 10_000)

        totals = {
            "system_reward": 0.0,
            "hit_reward": 0.0,
            "update_cost": 0.0,
            "thrashing_cost": 0.0,
            "num_hits": 0.0,
            "num_requests": 0.0,
        }
        step_rows: list[dict[str, float]] = []

        for t in range(self.cfg.T):
            with torch.no_grad():
                result = eval_policy(self._batch_single_obs(obs))
            raw_action = result.act.detach().cpu().numpy()

            action = eval_policy.map_action(raw_action)[0]

            obs, reward, terminated, truncated, info = eval_env.step(action)
            totals["system_reward"] += float(reward)
            totals["hit_reward"] += float(info["reward_hit"])
            totals["update_cost"] += float(info["cost_update"])
            totals["thrashing_cost"] += float(info["cost_thrashing"])
            totals["num_hits"] += float(info["num_hits"])
            totals["num_requests"] += float(info["num_requests"])

            row = {"step": float(t + 1), "reward": float(reward)}
            row.update({k: float(v) for k, v in info.items()})
            step_rows.append(row)
            logger.info(
                "Eval Step %d/%d: Reward=%.2f HitReward=%.2f UpdateCost=%.2f ThrashCost=%.2f HitRate=%.2f%%",
                t + 1,
                self.cfg.T,
                reward,
                info["reward_hit"],
                info["cost_update"],
                info["cost_thrashing"],
                100.0 * info["hit_rate"],
            )
            if terminated or truncated:
                break

        eval_env.close()
        aggregate_hit_rate = (
            totals["num_hits"] / totals["num_requests"] if totals["num_requests"] > 0 else 0.0
        )

        step_hit_rates = [r["hit_rate"] for r in step_rows if r["num_requests"] > 0]
        average_step_hit_rate = float(np.mean(step_hit_rates)) if step_hit_rates else 0.0

        summary = {
            "total_system_reward": totals["system_reward"],
            "total_hit_reward": totals["hit_reward"],
            "total_update_cost": totals["update_cost"],
            "total_thrashing_cost": totals["thrashing_cost"],
            "average_hit_rate": average_step_hit_rate,
            "aggregate_hit_rate": aggregate_hit_rate,
            "total_num_hits": totals["num_hits"],
            "total_num_requests": totals["num_requests"],
        }

        (self.cfg.log_dir / "final_evaluation.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with open(self.cfg.log_dir / "final_evaluation_steps.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()) if step_rows else ["step"])
            writer.writeheader()
            writer.writerows(step_rows)

        logger.info("=" * 80)
        logger.info("--- Final Evaluation Summary (T=%d) ---", self.cfg.T)
        logger.info("Total System Reward: %.2f", summary["total_system_reward"])
        logger.info("Total Hit Reward: %.2f", summary["total_hit_reward"])
        logger.info("Total Update Cost: %.2f", summary["total_update_cost"])
        logger.info("Total Thrashing Cost: %.2f", summary["total_thrashing_cost"])
        logger.info("Average Hit Rate: %.2f%%", 100.0 * summary["average_hit_rate"])
        logger.info("Aggregate Hit Rate: %.2f%%", 100.0 * summary["aggregate_hit_rate"])
        logger.info("=" * 80)
        return summary

    def run(self) -> dict[str, float]:
        trainer = self._create_trainer()
        try:
            result = trainer.run()
            logger.info("Training finished: %s", result)
            return self.run_final_evaluation()
        finally:
            torch.save(self.policy.state_dict(), self.cfg.model_dir / "final_policy.pth")
            self.train_envs.close()
            self.test_envs.close()
            self.writer.close()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO dynamic caching with RMNL recommendations")
    p.add_argument("--runtime-config", type=Path, required=True)
    p.add_argument("--run-name", type=str, default=f"run_{time.strftime('%Y%m%d-%H%M%S')}")

    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_const", dest="resume_mode", const="checkpoint")
    resume.add_argument("--resume-best", action="store_const", dest="resume_mode", const="best_weights")
    p.set_defaults(resume_mode=None)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--clip-eps", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--vf-coef", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--step-per-collect", type=int, default=None)
    p.add_argument("--repeat-per-collect", type=int, default=None)
    p.add_argument("--social-alpha", type=float, default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--step-per-epoch", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--num-train-envs", type=int, default=None)
    p.add_argument("--num-test-envs", type=int, default=None)
    p.add_argument("--episode-per-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-users", type=int, default=None)
    p.add_argument("--num-items", type=int, default=None)
    p.add_argument("--time-slots", type=int, default=None)
    p.add_argument("--assortment-capacity", type=int, default=None)
    p.add_argument("--cache-capacity", type=int, default=None)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--prefs-path", type=Path, default=None)
    p.add_argument("--social-graph-path", type=Path, default=None)
    return p


CONFIGURABLE_KEYS = {
    "profit_omega",
    "profit_discount",
    "cache_update_cost_per_bit",
    "thrashing_base_cost_omega_tilde",
    "U",
    "I",
    "T",
    "K",
    "C_cs",
    "pre_delta",
    "social_influence_alpha",
    "thrashing_max_interval_vm",
    "request_history_window",
    "gamma",
    "lr",
    "epoch",
    "batch_size",
    "buffer_size",
    "num_train_envs",
    "num_test_envs",
    "step_per_epoch",
    "step_per_collect",
    "repeat_per_collect",
    "gae_lambda",
    "max_grad_norm",
    "clip_eps",
    "vf_coef",
    "ent_coef",
    "advantage_normalization",
    "reward_normalization",
    "deterministic_eval",
    "lr_end_factor",
    "episode_per_test",
    "seed",
    "item_size_seed",
    "item_size_min",
    "item_size_max",
    "data_path",
    "prefs_path",
    "social_graph_path",
}

REQUIRED_RUNTIME_KEYS = {
    "profit_omega",
}

PUBLIC_RANDOM_RANGES = {
    "profit_discount": (0.2, 0.8),
    "cache_update_cost_per_bit": (0.01, 0.2),
    "thrashing_base_cost_omega_tilde": (0.2, 1.0),
}

INT_CONFIG_KEYS = {
    "U", "I", "T", "K", "C_cs", "thrashing_max_interval_vm",
    "request_history_window", "epoch", "batch_size", "buffer_size",
    "num_train_envs", "num_test_envs", "step_per_epoch", "step_per_collect",
    "repeat_per_collect", "episode_per_test", "seed", "item_size_seed",
    "item_size_min", "item_size_max",
}

FLOAT_CONFIG_KEYS = {
    "profit_omega", "profit_discount", "cache_update_cost_per_bit",
    "thrashing_base_cost_omega_tilde", "pre_delta", "social_influence_alpha",
    "gamma", "lr", "gae_lambda", "max_grad_norm", "clip_eps", "vf_coef",
    "ent_coef", "lr_end_factor",
}

BOOL_CONFIG_KEYS = {
    "advantage_normalization", "reward_normalization", "deterministic_eval",
}

PATH_CONFIG_KEYS = {"data_path", "prefs_path", "social_graph_path"}


def load_runtime_parameters(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime config must contain a JSON object")

    unknown = sorted(set(data) - CONFIGURABLE_KEYS)
    if unknown:
        raise ValueError(f"unknown runtime config keys: {', '.join(unknown)}")

    missing = sorted(key for key in REQUIRED_RUNTIME_KEYS if data.get(key) is None)
    if missing:
        raise ValueError(f"runtime config is missing values for: {', '.join(missing)}")

    nullable_keys = {
        "pre_delta",
        "seed",
        "item_size_seed",
        *PUBLIC_RANDOM_RANGES.keys(),
    }

    parsed: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            if key in nullable_keys:
                parsed[key] = None
            continue
        if key in INT_CONFIG_KEYS:
            parsed[key] = int(value)
        elif key in FLOAT_CONFIG_KEYS:
            parsed[key] = float(value)
        elif key in BOOL_CONFIG_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
            parsed[key] = value
        elif key in PATH_CONFIG_KEYS:
            parsed[key] = Path(value)
        else:
            parsed[key] = value
    return parsed


def config_from_args(args: argparse.Namespace) -> SystemConfig:
    runtime = load_runtime_parameters(args.runtime_config)

    overrides = {
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.batch_size,
        "step_per_collect": args.step_per_collect,
        "repeat_per_collect": args.repeat_per_collect,
        "social_influence_alpha": args.social_alpha,
        "epoch": args.epoch,
        "step_per_epoch": args.step_per_epoch,
        "buffer_size": args.buffer_size,
        "num_train_envs": args.num_train_envs,
        "num_test_envs": args.num_test_envs,
        "episode_per_test": args.episode_per_test,
        "seed": args.seed,
        "U": args.num_users,
        "I": args.num_items,
        "T": args.time_slots,
        "K": args.assortment_capacity,
        "C_cs": args.cache_capacity,
        "data_path": args.data_path,
        "prefs_path": args.prefs_path,
        "social_graph_path": args.social_graph_path,
    }
    for key, value in overrides.items():
        if value is not None:
            runtime[key] = value

    if runtime.get("seed") is None:
        runtime["seed"] = secrets.randbits(32)

    rng = np.random.default_rng(runtime["seed"])
    for key, (low, high) in PUBLIC_RANDOM_RANGES.items():
        if runtime.get(key) is None:
            runtime[key] = float(rng.uniform(low, high))
            logger.info(
                "%s was not specified; sampled %.6f from public range [%.3f, %.3f]",
                key,
                runtime[key],
                low,
                high,
            )

    if runtime.get("item_size_seed") is None:
        runtime["item_size_seed"] = runtime["seed"]

    run_log_dir = LOG_DIR_BASE / args.run_name
    run_model_dir = MODEL_DIR_BASE / args.run_name
    runtime.update(
        run_name=args.run_name,
        log_dir=run_log_dir,
        model_dir=run_model_dir,
        checkpoint_path=run_model_dir / "checkpoint.pth",
    )
    return SystemConfig(**runtime)


def preflight(cfg: SystemConfig) -> None:
    for path in (cfg.data_path, cfg.prefs_path, cfg.social_graph_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"required data file not found: {path}")
    prefs, graph = load_static_data(cfg)

    rec = RecommendationGenerator(cfg, prefs, graph)
    test_recs = rec.generate_with_assortment(np.zeros(cfg.I, dtype=np.float32), rec.user_prefs)
    if test_recs.shape != (cfg.U * cfg.K,):
        raise RuntimeError(f"assortment smoke-test shape mismatch: {test_recs.shape}")
    logger.info("Preflight passed; exact assortment output shape=%s", test_recs.shape)

def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_dir / f"{cfg.run_name}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    try:
        source = Path(sys.argv[0]).resolve()
        if source.exists():
            shutil.copy2(source, cfg.log_dir / f"{cfg.run_name}_{source.name}")
        assortment_source = Path(__file__).resolve().with_name("assortment.py")
        if assortment_source.exists():
            shutil.copy2(assortment_source, cfg.log_dir / f"{cfg.run_name}_assortment.py")
    except Exception as exc:
        logger.warning("Could not copy source snapshot: %s", exc)

    set_global_seed(cfg.seed)
    logger.info("Run=%s device=%s", cfg.run_name, cfg.device)
    logger.info("Seed=%d item_size_seed=%d", cfg.seed, cfg.item_size_seed)
    logger.info("Log dir=%s Model dir=%s", cfg.log_dir, cfg.model_dir)
    logger.info("Expected scheduler policy updates=%d", cfg.policy_updates_per_training())

    preflight(cfg)
    manager = TrainingManager(cfg, resume_mode=args.resume_mode)
    summary = manager.run()
    print("\nTraining completed.")
    print(json.dumps(summary, indent=2))
    print(f"Logs: {cfg.log_dir}")
    print(f"Models: {cfg.model_dir}")

if __name__ == "__main__":
    main()
