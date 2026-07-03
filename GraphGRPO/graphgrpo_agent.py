"""GraphGRPOAgent - GNN-enhanced GRPO Agent for FJSP (ported to SkyEngine-FJSP).

This module is a self-contained port of the SkyEngine `GraphGRPOAgent` (see
`GraphGRPOAgent移植文档.md`). All SkyEngine-internal dependencies that the
original file imported (`executor.packet_factory.*`, the component registry,
the project logger) have been replaced with local, dependency-free equivalents
defined at the top of this file:

- `BaseAgent` / `DEFAULT_STEP_TIME` / `FRONTEND|BACKEND|TRAINING|INFERENCE`
  → minimal inlined base class (doc §3.2).
- `resolve_device` / `log_device` → inlined device resolver with CPU fallback
  (doc §3.3), using stdlib `logging` instead of SkyEngine's `LOGGER`.
- `OperationStatus` / `MachineStatus` / `AGVStatus` → inlined enums (doc §3.4).
- `register_component` → no-op decorator (this project has no component registry).
- `LOGGER` → `logging.getLogger("GraphGRPOAgent")`.

The algorithm bodies (`FactoryGraphBuilder`, `GNNStateEncoder`, `GRPORolloutBuffer`,
the three Actors, `GraphGRPOAgent`) are reproduced verbatim from the doc. The agent
operates on environment objects that satisfy the interface contract in doc §3.5;
those adapters live in `env_adapter.py`.

Annotations are postponed (`from __future__ import annotations`) so the
`Operation/Machine/AGV/Job` type hints need not be imported here.
"""
from __future__ import annotations

import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# ============================================================
# Inlined internal dependencies (replacements for SkyEngine)
# ============================================================

LOGGER = logging.getLogger("GraphGRPOAgent")

# Default minimum step time (seconds)
DEFAULT_STEP_TIME = 1

# Agent run modes - 2x2 classification
FRONTEND = "frontend"  # UI enabled (visualization)
BACKEND = "backend"    # no UI
TRAINING = "training"      # learn + update
INFERENCE = "inference"    # greedy argmax


def register_component(component_id):
    """No-op decorator. The original project uses a component registry; this
    standalone port has none, so the decorator simply returns the class unchanged."""
    def decorator(cls):
        return cls
    return decorator


class OperationStatus:
    """Status of an operation (doc §3.4)."""
    WAITING = 0      # predecessors not satisfied
    READY = 1        # can be scheduled this round
    MOVING = 2       # being transported on an AGV
    WORKING = 3       # being processed
    FINISHED = 4       # done
    EXCEPTION = 99


class AGVStatus:
    """Status of an AGV (doc §3.4)."""
    READY = 0        # can transport an operation
    ASSIGNED = 1     # moving (without operation)
    LOADED = 2       # moving (with operation)
    EXCEPTION = 99


class MachineStatus:
    """Status of a machine (doc §3.4)."""
    READY = 0        # idle
    WORKING = 1      # processing an operation
    FAILED = 3
    EXCEPTION = 99


class JobStatus:
    B4START = 0
    STARTED = 1
    FINISHED = 2
    EXCEPTION = 99


def _cuda_works(device_index: int = 0) -> None:
    """Probe whether CUDA is actually usable: trigger init + a tiny allocation.

    Any exception means CUDA is unavailable (bad card / driver / env), and the
    caller falls back to CPU.
    """
    torch.cuda.get_device_name(device_index)        # trigger _cuda_init
    torch.zeros(1, device=f'cuda:{device_index}')   # verify memory allocation


def resolve_device(device: Optional[str] = None,
                   tag: str = "Agent") -> torch.device:
    """Resolve a torch compute device, falling back to CPU if CUDA init fails."""
    if device is not None and device != 'auto':
        try:
            dev = torch.device(device)
            if dev.type == 'cuda':
                idx = dev.index if dev.index is not None else 0
                _cuda_works(idx)
            return dev
        except Exception as e:
            LOGGER.warning(f"[{tag}] requested device '{device}' unavailable, "
                            f"falling back to CPU: {e}")
            return torch.device('cpu')

    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _cuda_works(0)
            return torch.device('cuda')
    except Exception as e:
        LOGGER.warning(f"[{tag}] CUDA unavailable, falling back to CPU: {e}")
    return torch.device('cpu')


def log_device(device: torch.device, tag: str = "Agent"):
    """Log the compute device in use."""
    if device.type == 'cuda':
        try:
            name = torch.cuda.get_device_name(device if device.index is not None else 0)
        except Exception:
            name = "CUDA"
        LOGGER.info(f"[{tag}] Using CUDA: {name}")
    else:
        LOGGER.info(f"[{tag}] Using CPU")


@register_component("packet_factory.BaseAgent")
class BaseAgent(ABC):
    """Generic agent base class (inlined minimal version of doc §3.2)."""

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.agent_id = agent_id
        self.context = context
        self.alive = True
        self.turns = 0

        # decision-time stats
        self.total_decision_time = 0.0
        self.decision_count = 0

        # RL config - 2x2 modes
        self.ui_mode = ui_mode
        self.task_mode = task_mode
        self.model_path = model_path
        self.mode = f"{ui_mode}_{task_mode}"

    def is_alive(self):
        return self.alive

    @abstractmethod
    def reward(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> Tuple[List[Any], float]:
        pass

    def before_sample(self, *args, **kwargs):
        pass

    def after_sample(self, *args, **kwargs):
        pass

    def decision(self, *args, **kwargs) -> Tuple[List[Any], float]:
        """Unified decision entry: wraps before/sample/after + timing."""
        start_time = time.time()
        self.before_sample(*args, **kwargs)
        result = self.sample(*args, **kwargs)
        self.after_sample(*args, **kwargs)
        end_time = time.time()
        decision_time = end_time - start_time
        self.total_decision_time += decision_time
        self.decision_count += 1

        if isinstance(result, tuple) and len(result) == 2:
            decision_list, step_time = result
            step_time = max(step_time, DEFAULT_STEP_TIME)
            return decision_list, step_time
        return [], DEFAULT_STEP_TIME

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def get_decision_stats(self) -> Dict[str, float]:
        avg_time = (self.total_decision_time / self.decision_count
                    if self.decision_count > 0 else 0)
        return {
            'total_decision_time': self.total_decision_time,
            'decision_count': self.decision_count,
            'average_decision_time': avg_time,
        }

    def get_training_metrics(self) -> Dict[str, Any]:
        metrics = {}
        if hasattr(self, '_episode_reward'):
            metrics['episode_reward'] = self._episode_reward
        elif hasattr(self, 'current_episode_reward'):
            metrics['episode_reward'] = self.current_episode_reward
        if hasattr(self, 'epsilon'):
            metrics['epsilon'] = self.epsilon
        if hasattr(self, 'training_history') and isinstance(self.training_history, dict):
            metrics['training_history'] = self.training_history
        return metrics

    def reset_decision_stats(self):
        self.total_decision_time = 0.0
        self.decision_count = 0

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"


# ============================================================
# Data structures
# ============================================================

@dataclass
class GraphBuildResult:
    """Result of building a factory graph, with metadata for action encoding."""
    data: Data
    node_type_offsets: Dict[str, Tuple[int, int]]
    id_to_node_idx: Dict[Tuple[str, int], int]
    num_nodes: int


class RunningMeanStd:
    """Running mean/std for reward normalization (Welford's online algorithm)."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = max(new_var, 1e-6)
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ============================================================
# Factory Graph Builder (reused from GraphDualAgent / GraphPPOAgent)
# ============================================================

class FactoryGraphBuilder:
    """Converts factory state into a PyG homogeneous Data graph with typed nodes.

    Node layout: [machines | operations | agvs | jobs]
    Node features are padded to a uniform max_dim so a single GNN input projection works.
    """

    MACHINE_DIM = 9
    OPERATION_DIM = 10
    AGV_DIM = 7
    JOB_DIM = 5
    MAX_DIM = 10  # pad all to this size

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def build(self, agvs: List, machines: List, jobs: List,
              factory_graph=None, current_time: float = 0.0) -> GraphBuildResult:
        """Build graph from factory state."""
        machine_feats, machine_id_map = self._machine_features(machines, current_time)
        op_feats, op_id_map = self._operation_features(jobs)
        agv_feats, agv_id_map = self._agv_features(agvs)
        job_feats, job_id_map = self._job_features(jobs)

        n_m = len(machine_feats)
        n_o = len(op_feats)
        n_a = len(agv_feats)
        n_j = len(job_feats)

        all_feats = np.concatenate([
            self._pad(machine_feats),
            self._pad(op_feats),
            self._pad(agv_feats),
            self._pad(job_feats),
        ], axis=0)

        node_type = np.concatenate([
            np.zeros(n_m, dtype=np.int64),
            np.ones(n_o, dtype=np.int64),
            np.full(n_a, 2, dtype=np.int64),
            np.full(n_j, 3, dtype=np.int64),
        ])

        data = Data(
            x=torch.tensor(all_feats, dtype=torch.float),
            node_type=torch.tensor(node_type, dtype=torch.long),
        )

        # Build edges
        src_list, dst_list = [], []

        # Operation <-> Machine (capability)
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.id not in op_id_map:
                    continue
                op_idx = n_m + op_id_map[op.id]
                for m_id, _ in op.durations:
                    if m_id in machine_id_map:
                        m_idx = machine_id_map[m_id]
                        src_list.extend([op_idx, m_idx])
                        dst_list.extend([m_idx, op_idx])

        # Job <-> Operation (contains)
        for job in jobs:
            if job.id not in job_id_map:
                continue
            job_idx = n_m + n_o + n_a + job_id_map[job.id]
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.id in op_id_map:
                    op_idx = n_m + op_id_map[op.id]
                    src_list.extend([job_idx, op_idx])
                    dst_list.extend([op_idx, job_idx])

        # Operation -> Operation (sequence, directed)
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                next_op = op.get_next_operation()
                if op.id in op_id_map and next_op and next_op.id in op_id_map:
                    src_list.append(n_m + op_id_map[op.id])
                    dst_list.append(n_m + op_id_map[next_op.id])

        # AGV <-> Machine (reachable)
        for agv in agvs:
            if agv.id not in agv_id_map:
                continue
            agv_idx = n_m + n_o + agv_id_map[agv.id]
            for m in machines:
                if m.id not in machine_id_map:
                    continue
                m_idx = machine_id_map[m.id]
                if factory_graph and hasattr(factory_graph, 'get_path'):
                    if not factory_graph.get_path(agv.point_id, m.point_id):
                        continue
                src_list.extend([agv_idx, m_idx])
                dst_list.extend([m_idx, agv_idx])

        if src_list:
            data.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)

        id_to_node_idx: Dict[Tuple[str, int], int] = {}
        for mid, idx in machine_id_map.items():
            id_to_node_idx[("machine", mid)] = idx
        for oid, idx in op_id_map.items():
            id_to_node_idx[("operation", oid)] = n_m + idx
        for aid, idx in agv_id_map.items():
            id_to_node_idx[("agv", aid)] = n_m + n_o + idx
        for jid, idx in job_id_map.items():
            id_to_node_idx[("job", jid)] = n_m + n_o + n_a + idx

        offsets = {
            'machine': (0, n_m),
            'operation': (n_m, n_o),
            'agv': (n_m + n_o, n_a),
            'job': (n_m + n_o + n_a, n_j),
        }

        data = data.to(self.device)
        return GraphBuildResult(
            data=data,
            node_type_offsets=offsets,
            id_to_node_idx=id_to_node_idx,
            num_nodes=n_m + n_o + n_a + n_j,
        )

    def _pad(self, feats: list) -> np.ndarray:
        if not feats:
            return np.zeros((1, self.MAX_DIM), dtype=np.float32)
        arr = np.array(feats, dtype=np.float32)
        if arr.shape[1] < self.MAX_DIM:
            arr = np.pad(arr, ((0, 0), (0, self.MAX_DIM - arr.shape[1])))
        return arr

    def _machine_features(self, machines, current_time):
        feats, id_map = [], {}
        for idx, m in enumerate(machines):
            id_map[m.id] = idx
            timer = m.timer / 1000.0 if m.status == MachineStatus.WORKING else current_time / 1000.0
            load = len(m.input_queue) / 20.0
            soh = [0.0] * 4
            soh[min(m.status, 3)] = 1.0
            avail = 1.0 if m.is_available() else 0.0
            feats.append([timer, load] + soh + [m.x / 100.0, m.y / 100.0, avail])
        return feats or [[0.0] * self.MACHINE_DIM], id_map

    def _operation_features(self, jobs):
        feats, id_map = [], {}
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                id_map[op.id] = len(feats)
                durs = [d for _, d in op.durations] if op.durations else [0.0]
                dur_mean = float(np.mean(durs)) / 100.0
                dur_std = float(np.std(durs)) / 100.0 if len(durs) > 1 else 0.0
                dur_min = min(durs) / 100.0
                soh = [0.0] * 5
                soh[min(op.get_status(), 4)] = 1.0
                remaining = 0
                nxt = op.get_next_operation()
                while nxt is not None:
                    remaining += 1
                    nxt = nxt.get_next_operation()
                progress = min(op.process_time / (dur_mean * 100.0 + 1e-5), 1.0)
                feats.append([dur_mean, dur_std, dur_min] + soh + [remaining / 10.0, progress])
        return feats or [[0.0] * self.OPERATION_DIM], id_map

    def _agv_features(self, agvs):
        feats, id_map = [], {}
        for idx, agv in enumerate(agvs):
            id_map[agv.id] = idx
            soh = [0.0] * 3
            soh[min(agv.get_status(), 2)] = 1.0
            feats.append([agv.timer / 1000.0, agv.velocity / 10.0] + soh
                         + [agv.x / 100.0, agv.y / 100.0])
        return feats or [[0.0] * self.AGV_DIM], id_map

    def _job_features(self, jobs):
        feats, id_map = [], {}
        for idx, job in enumerate(jobs):
            id_map[job.id] = idx
            n_ops = job.get_operation_count()
            n_fin = sum(1 for i in range(n_ops)
                        if job.get_operation(i).get_status() == OperationStatus.FINISHED)
            progress = n_fin / n_ops if n_ops > 0 else 0.0
            rem_time = 0.0
            for i in range(n_ops):
                op = job.get_operation(i)
                if op.get_status() != OperationStatus.FINISHED and op.durations:
                    rem_time += op.get_duration(op.durations[0][0])
            rem_time /= 1000.0
            feats.append([progress, rem_time, n_ops / 20.0,
                          1.0 if job.is_finished() else 0.0, len(id_map) / 100.0])
        return feats or [[0.0] * self.JOB_DIM], id_map


# ============================================================
# GNN State Encoder (reused from GraphDualAgent / GraphPPOAgent)
# ============================================================

class GNNStateEncoder(nn.Module):
    """Heterogeneous-aware GNN that encodes a factory graph into a global state vector.

    Type embeddings + type-specific projections, then SAGEConv message passing,
    then per-type mean pooling to produce a fixed-size global state vector.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2,
                 machine_dim: int = 10, operation_dim: int = 10,
                 agv_dim: int = 10, job_dim: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.type_embedding = nn.Embedding(4, hidden_dim)

        self.machine_proj = nn.Linear(machine_dim, hidden_dim)
        self.operation_proj = nn.Linear(operation_dim, hidden_dim)
        self.agv_proj = nn.Linear(agv_dim, hidden_dim)
        self.job_proj = nn.Linear(job_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, data: Data,
                node_type_offsets: Dict[str, Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        node_type = data.node_type

        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        for type_idx, proj in enumerate([self.machine_proj, self.operation_proj,
                                          self.agv_proj, self.job_proj]):
            mask = (node_type == type_idx)
            if mask.any():
                h[mask] = F.leaky_relu(proj(x[mask]))

        type_embs = self.type_embedding(node_type)
        h = h + type_embs
        h = F.leaky_relu(h)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index)
            h = norm(h + h_new)
            h = F.leaky_relu(h)

        parts = []
        for type_name in ['machine', 'operation', 'agv', 'job']:
            start, count = node_type_offsets[type_name]
            if count > 0:
                pooled = h[start:start + count].mean(dim=0)
            else:
                pooled = torch.zeros(self.hidden_dim, device=h.device)
            parts.append(pooled)

        global_state = torch.cat(parts, dim=-1)
        return global_state, h


# ============================================================
# GRPO Rollout Buffer (group-relative advantage, no critic/GAE)
# ============================================================

class GRPORolloutBuffer:
    """On-policy rollout buffer for GRPO with group-relative advantage normalization.

    Unlike PPO's RolloutBuffer:
    - No value/GAE tracking (GRPO is critic-free)
    - Stores group_id per transition for group-relative normalization
    - Stores full candidate feature arrays for route/AGV decisions so training can
      reconstruct the same categorical distribution over candidates.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        """Clear all stored transitions."""
        # Shared
        self.states = []
        self.seq_features = []
        self.dones = []
        self.rewards = []
        self.group_ids = []

        # Sequencing
        self.seq_actions = []
        self.seq_log_probs = []

        # Routing — store ALL candidate features per step
        self.route_actions = []
        self.route_log_probs = []
        self.route_all_candidate_features = []  # list of np.ndarray [n_candidates, feat_dim]

        # AGV — store ALL candidate features per step
        self.agv_actions = []
        self.agv_log_probs = []
        self.agv_all_candidate_features = []  # list of np.ndarray [n_candidates, feat_dim]

        # Computed after rollout
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.rewards)

    def push(self, state: np.ndarray, seq_features: np.ndarray,
             reward: float, done: bool, group_id: int,
             # Sequencing
             seq_action: int = -1, seq_log_prob: float = 0.0,
             # Routing
             route_action: int = -1, route_log_prob: float = 0.0,
             route_all_candidate_feats: Optional[np.ndarray] = None,
             # AGV
             agv_action: int = -1, agv_log_prob: float = 0.0,
             agv_all_candidate_feats: Optional[np.ndarray] = None):
        """Store one transition step."""
        self.states.append(state)
        self.seq_features.append(seq_features)
        self.dones.append(done)
        self.rewards.append(reward)
        self.group_ids.append(group_id)

        self.seq_actions.append(seq_action)
        self.seq_log_probs.append(seq_log_prob)

        self.route_actions.append(route_action)
        self.route_log_probs.append(route_log_prob)
        self.route_all_candidate_features.append(
            route_all_candidate_feats if route_all_candidate_feats is not None
            else np.zeros((1, 1), dtype=np.float32)
        )

        self.agv_actions.append(agv_action)
        self.agv_log_probs.append(agv_log_prob)
        self.agv_all_candidate_features.append(
            agv_all_candidate_feats if agv_all_candidate_feats is not None
            else np.zeros((1, 1), dtype=np.float32)
        )

    def compute_group_relative_advantages(self, gamma: float, min_group_size: int = 2):
        """Compute discounted returns and group-relative advantages (no critic)."""
        n = len(self.rewards)
        if n == 0:
            return

        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute discounted returns
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            if dones[t]:
                running = 0.0
            running = rewards[t] + gamma * running
            returns[t] = running

        # Compute group-relative advantages
        advantages = np.zeros(n, dtype=np.float32)
        global_mean = returns.mean()
        global_std = returns.std() + 1e-8

        group_ids_arr = np.array(self.group_ids)
        unique_groups = np.unique(group_ids_arr)

        for gid in unique_groups:
            idx = np.where(group_ids_arr == gid)[0]
            group_returns = returns[idx]

            if len(idx) >= min_group_size:
                mean = group_returns.mean()
                std = group_returns.std() + 1e-8
            else:
                mean = global_mean
                std = global_std

            advantages[idx] = (group_returns - mean) / std

        self.returns = returns
        self.advantages = advantages

    def get_tensors(self, sub_agent: str) -> Dict[str, Any]:
        """Convert stored data to tensors for GRPO update."""
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        seq_feats = torch.tensor(np.array(self.seq_features), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)

        if sub_agent == 'seq':
            actions = torch.tensor(np.array(self.seq_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.seq_log_probs), dtype=torch.float32, device=self.device)
            return {
                'states': states, 'seq_features': seq_feats,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
            }
        elif sub_agent == 'route':
            actions = torch.tensor(np.array(self.route_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.route_log_probs), dtype=torch.float32, device=self.device)
            return {
                'states': states,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
                'route_all_candidate_features': self.route_all_candidate_features,
            }
        elif sub_agent == 'agv':
            actions = torch.tensor(np.array(self.agv_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.agv_log_probs), dtype=torch.float32, device=self.device)
            return {
                'states': states,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
                'agv_all_candidate_features': self.agv_all_candidate_features,
            }
        else:
            raise ValueError(f"Unknown sub_agent: {sub_agent}")


# ============================================================
# Actor Networks (Policy Networks — same as GraphPPOAgent)
# ============================================================

class SequencingActor(nn.Module):
    """Sequencing Actor: outputs action logits for operation prioritization.

    Input: global_state [state_dim] + seq_features [seq_feature_dim]
    Output: logits [max_seq_output_dim] for categorical distribution over operation slots
    """

    def __init__(self, state_dim: int, seq_feature_dim: int, max_output_dim: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = state_dim + seq_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 64)
        self.fc_out = nn.Linear(64, max_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                seq_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        sf = self.feat_norm(seq_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, sf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc_out(x)


class RoutingActor(nn.Module):
    """Routing Actor: outputs action logits for machine assignment.

    Input: global_state [state_dim] + route_features [route_feature_dim]
    Output: logits [max_route_output_dim] for categorical distribution over machine slots
    """

    def __init__(self, state_dim: int, route_feature_dim: int, max_output_dim: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = state_dim + route_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc_out = nn.Linear(64, max_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                route_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        rf = self.feat_norm(route_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, rf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x)


class AGVSelectionActor(nn.Module):
    """AGV Selection Actor: outputs action logits for AGV selection.

    Input: global_state [state_dim] + agv_features [agv_feature_dim]
    Output: logits [max_agv_output_dim] for categorical distribution over AGV slots
    """

    def __init__(self, state_dim: int, agv_feature_dim: int, max_output_dim: int,
                 hidden: int = 64):
        super().__init__()
        input_dim = state_dim + agv_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 32)
        self.fc_out = nn.Linear(32, max_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                agv_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        af = self.feat_norm(agv_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, af], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x)


# ============================================================
# GraphGRPOAgent
# ============================================================

@register_component("packet_factory.GraphGRPOAgent")
class GraphGRPOAgent(BaseAgent):
    """GNN-enhanced GRPO Agent for FJSP.

    Combines GNN state encoding with GRPO (Group Relative Policy Optimization):
    1. GNN encodes factory graph -> global_state [256] + node_embs
    2. SequencingActor ranks READY operations by priority
    3. For each op in priority order:
       a. RoutingActor selects the best capable machine
       b. AGVSelectionActor selects the best available AGV
    4. Training: critic-free GRPO with group-relative advantage normalization,
       clipped surrogate, optional KL penalty, entropy bonus

    Reward design references DualDRLAgent (load balance, makespan, utilization,
    queue reduction, bottleneck relief) combined with graph-agent dense shaping.
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None,
                 # Graph encoder
                 hidden_dim: int = 64,
                 # GRPO common
                 gamma: float = 0.99,
                 # Learning rates
                 lr_gnn: float = 1e-4,
                 lr_actor: float = 3e-4,
                 # GRPO hyperparameters
                 clip_epsilon: float = 0.2,
                 grpo_epochs: int = 4,
                 grpo_batch_size: int = 64,
                 grpo_group_size: int = 8,
                 min_group_size: int = 2,
                 entropy_coeff: float = 0.01,
                 kl_coeff: float = 0.0,
                 max_grad_norm: float = 0.5,
                 rollout_steps: int = 128,
                 # Output dimensions (max action space sizes)
                 max_seq_output_dim: int = 100,
                 max_route_output_dim: int = 50,
                 max_agv_output_dim: int = 20,
                 # Feature dimensions
                 seq_feature_dim: int = 24,
                 route_feature_dim: int = 16,
                 agv_feature_dim: int = 12,
                 # Misc
                 allow_agv_reassignment: bool = False,
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)

        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim * 4
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.grpo_epochs = grpo_epochs
        self.grpo_batch_size = grpo_batch_size
        self.grpo_group_size = grpo_group_size
        self.min_group_size = min_group_size
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.max_seq_output_dim = max_seq_output_dim
        self.max_route_output_dim = max_route_output_dim
        self.max_agv_output_dim = max_agv_output_dim
        self.seq_feature_dim = seq_feature_dim
        self.route_feature_dim = route_feature_dim
        self.agv_feature_dim = agv_feature_dim
        self.allow_agv_reassignment = allow_agv_reassignment

        # Device
        self.device = resolve_device(device, tag="GraphGRPOAgent")
        log_device(self.device, tag="GraphGRPOAgent")

        # Graph builder
        self.graph_builder = FactoryGraphBuilder(device=str(self.device))

        # GNN encoder (shared across all Actors)
        self.gnn_encoder = GNNStateEncoder(
            hidden_dim=hidden_dim, num_layers=2,
        ).to(self.device)

        # Initialize Actors only (no critics — GRPO is critic-free)
        self._initialize_networks(lr_gnn, lr_actor)

        # GRPO rollout buffer
        self.rollout_buffer = GRPORolloutBuffer(self.device)

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # Rollout step counter
        self._rollout_step_count = 0

        # GRPO group tracking
        self._grpo_group_id = 0
        self._samples_in_current_group = 0

        # State tracking for transition collection
        self._prev_global_state: Optional[np.ndarray] = None
        self._prev_seq_features: Optional[np.ndarray] = None
        self._prev_decisions_info: List[Dict] = []

        # DualDRLAgent-style queue tracking for reward
        self._prev_queue_len = 0

        # Training statistics
        self.training_history: Dict[str, list] = {
            'episodes': [], 'policy_loss': [],
            'entropy': [], 'total_loss': [], 'makespans': [],
            'kl_approx': [],
        }
        self._episode_reward = 0.0
        self._train_step = 0

        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[GraphGRPOAgent] Loaded model from {model_path}")
        else:
            LOGGER.info(f"[GraphGRPOAgent] Initialized on {self.device}, "
                        f"hidden_dim={hidden_dim}, state_dim={self.state_dim}, "
                        f"clip_epsilon={clip_epsilon}, grpo_group_size={grpo_group_size}")

    def _initialize_networks(self, lr_gnn, lr_actor):
        """Initialize all Actors and optimizers (no critics)."""
        # Sequencing Actor
        self.seq_actor = SequencingActor(
            self.state_dim, self.seq_feature_dim, self.max_seq_output_dim,
        ).to(self.device)

        # Routing Actor
        self.route_actor = RoutingActor(
            self.state_dim, self.route_feature_dim, self.max_route_output_dim,
        ).to(self.device)

        # AGV Selection Actor
        self.agv_actor = AGVSelectionActor(
            self.state_dim, self.agv_feature_dim, self.max_agv_output_dim,
        ).to(self.device)

        # Optimizers: actor + GNN for each sub-agent
        self.optimizer_seq = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.seq_actor.parameters()),
            lr=lr_actor,
        )
        self.optimizer_route = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.route_actor.parameters()),
            lr=lr_actor,
        )
        self.optimizer_agv = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.agv_actor.parameters()),
            lr=lr_actor,
        )

        # Separate GNN optimizer for save/load state dict only
        self.optimizer_gnn = torch.optim.Adam(
            self.gnn_encoder.parameters(), lr=lr_gnn,
        )

    # ----------------------------------------------------------
    # Feature Extraction (unchanged from GraphPPOAgent)
    # ----------------------------------------------------------

    def _extract_seq_features(self, machines: List, jobs: List,
                              current_time: float) -> np.ndarray:
        """Extract sequencing-specific features (24-dim)."""
        features = []

        # [0-1] READY / WAITING op counts
        total_ops = 0
        ready_count = 0
        waiting_count = 0
        ready_durations = []
        jobs_with_ready = 0
        remaining_ops_per_job = []

        for job in jobs:
            if job.is_finished():
                continue
            job_has_ready = False
            job_remaining = 0
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                total_ops += 1
                status = op.get_status()
                if status == OperationStatus.READY:
                    ready_count += 1
                    job_has_ready = True
                    if op.durations:
                        ready_durations.append(sum(d for _, d in op.durations) / len(op.durations))
                elif status == OperationStatus.WAITING:
                    waiting_count += 1
                if status != OperationStatus.FINISHED:
                    job_remaining += 1
            if job_has_ready:
                jobs_with_ready += 1
                remaining_ops_per_job.append(job_remaining)

        features.append(ready_count / 50.0)
        features.append(waiting_count / max(total_ops, 1))

        # [2-5] READY op duration stats
        if ready_durations:
            features.append(float(np.mean(ready_durations)) / 100.0)
            features.append(float(np.std(ready_durations)) / 100.0 if len(ready_durations) > 1 else 0.0)
            features.append(min(ready_durations) / 100.0)
            features.append(max(ready_durations) / 100.0)
        else:
            features.extend([0.0] * 4)

        # [6-7] Jobs with READY ops
        total_jobs = len(jobs) if jobs else 1
        features.append(jobs_with_ready / total_jobs)
        features.append(float(np.mean(remaining_ops_per_job)) / 10.0 if remaining_ops_per_job else 0.0)

        # [8-9] System urgency
        remaining_time_sum = 0.0
        total_remaining = 0
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() != OperationStatus.FINISHED and op.durations:
                    remaining_time_sum += sum(d for _, d in op.durations) / len(op.durations)
                    total_remaining += 1
        features.append(remaining_time_sum / 1000.0)
        features.append(total_remaining / max(total_ops, 1))

        # [10-11] Machine queue stats
        if machines:
            queue_lens = [len(m.input_queue) for m in machines]
            features.append(float(np.mean(queue_lens)) / 20.0)
            features.append(float(max(queue_lens)) / 20.0)
        else:
            features.extend([0.0, 0.0])

        # [12-13] Machine utilization & availability
        if machines:
            working = sum(1 for m in machines if m.status == MachineStatus.WORKING)
            features.append(working / len(machines))
            available = sum(1 for m in machines if m.is_available())
            features.append(available / len(machines))
        else:
            features.extend([0.0, 0.0])

        # [14-15] Imminent completion times
        imminent = []
        for m in machines:
            if m.input_queue:
                head_op = m.input_queue[0]
                time_left = head_op.get_duration(m.id) - head_op.process_time
                imminent.append(max(0.0, time_left))
        if imminent:
            features.append(min(imminent) / 100.0)
            features.append(float(np.mean(imminent)) / 100.0)
        else:
            features.extend([0.0, 0.0])

        # [16-17] Load balance & job completion
        if machines:
            loads = [len(m.input_queue) for m in machines]
            if len(loads) > 1 and np.mean(loads) > 0:
                cv = float(np.std(loads)) / (float(np.mean(loads)) + 1e-5)
                features.append(cv)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        completed = sum(1 for j in jobs if j.is_finished())
        features.append(completed / total_jobs)

        # [18-23] Extended features
        waiting_durs = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.WAITING and op.durations:
                    waiting_durs.append(sum(d for _, d in op.durations) / len(op.durations))
        features.append(float(np.mean(waiting_durs)) / 100.0 if waiting_durs else 0.0)
        features.append(len(waiting_durs) / 50.0)

        # Urgent jobs ratio (jobs with >50% ops still pending)
        urgent_jobs = 0
        for job in jobs:
            if job.is_finished():
                continue
            n_ops = job.get_operation_count()
            n_fin = sum(1 for i in range(n_ops)
                        if job.get_operation(i).get_status() == OperationStatus.FINISHED)
            if n_fin / max(n_ops, 1) < 0.5:
                urgent_jobs += 1
        features.append(urgent_jobs / total_jobs)

        # Longest remaining chain
        max_chain = 0
        for job in jobs:
            if job.is_finished():
                continue
            chain = 0
            for i in range(job.get_operation_count()):
                if job.get_operation(i).get_status() != OperationStatus.FINISHED:
                    chain += 1
            max_chain = max(max_chain, chain)
        features.append(max_chain / 20.0)

        # Shortest imminent (repeated for padding)
        features.append(min(imminent) / 100.0 if imminent else 0.0)
        features.append(float(np.std(imminent)) / 100.0 if len(imminent) > 1 else 0.0)

        # Pad/truncate to seq_feature_dim
        features = features[:self.seq_feature_dim]
        while len(features) < self.seq_feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _precompute_context(self, machines: List, jobs: List):
        """Pre-compute shared data structures once per sample() call (avoids O(n^2))."""
        op_to_job = {}
        job_progress = {}
        for job in jobs:
            if job.is_finished():
                continue
            n_ops = job.get_operation_count()
            n_fin = 0
            for i in range(n_ops):
                op = job.get_operation(i)
                op_to_job[op.id] = job
                if op.get_status() == OperationStatus.FINISHED:
                    n_fin += 1
            job_progress[job.id] = n_fin / max(n_ops, 1)

        # Competing READY ops per machine (computed once, O(n*m))
        competing_ops = {}
        ready_ops_set = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    ready_ops_set.append(op)

        for m in machines:
            cnt = 0
            for op in ready_ops_set:
                if op.is_machine_capable(m.id):
                    cnt += 1
            competing_ops[m.id] = cnt

        avg_queue = float(np.mean([len(m.input_queue) for m in machines])) if machines else 0.0

        return op_to_job, competing_ops, avg_queue, job_progress

    def _extract_route_features(self, op, machine,
                                machines: List, jobs: List,
                                current_time: float, op_to_job, competing_ops,
                                avg_queue, job_progress) -> np.ndarray:
        """Extract routing features (16-dim) for a specific (op, machine) pair."""
        features = []

        # [0] Processing time on this machine / max_dur
        dur = op.get_duration(machine.id) if op.is_machine_capable(machine.id) else 0.0
        all_durs = [d for _, d in op.durations] if op.durations else [0.0]
        max_dur = max(all_durs) if all_durs else 1.0
        features.append(dur / max(max_dur, 1e-5))

        # [1] Machine timer
        timer = machine.timer / max(current_time + 1e-5, 1.0) if machine.status == MachineStatus.WORKING else 0.0
        features.append(timer)

        # [2] Queue length
        features.append(len(machine.input_queue) / 20.0)

        # [3] Machine availability
        features.append(1.0 if machine.is_available() else 0.0)

        # [4-5] Machine position
        features.append(machine.x / 100.0)
        features.append(machine.y / 100.0)

        # [6] Duration std across capable machines
        dur_std = float(np.std(all_durs)) if len(all_durs) > 1 else 0.0
        features.append(dur_std / 100.0)

        # [7] Remaining successor ops
        remaining = 0
        nxt = op.get_next_operation()
        while nxt is not None:
            remaining += 1
            nxt = nxt.get_next_operation()
        features.append(remaining / 10.0)

        # [8] Job completion progress (from precomputed map)
        features.append(job_progress.get(op_to_job[op.id].id, 0.0) if op.id in op_to_job else 0.0)

        # [9] Queue deviation from average
        features.append((len(machine.input_queue) - avg_queue) / 20.0)

        # [10] Efficiency ratio: this dur / min capable dur
        min_dur = min(all_durs) if all_durs else 1.0
        features.append(dur / max(min_dur, 1e-5))

        # [11] Estimated completion time
        est_completion = (machine.timer if machine.status == MachineStatus.WORKING else 0.0) + dur
        features.append(est_completion / 1000.0)

        # [12] Is first op in job
        is_first = 0.0
        job = op_to_job.get(op.id)
        if job:
            first_op = job.get_operation(0)
            if first_op and first_op.id == op.id:
                is_first = 1.0
        features.append(is_first)

        # [13] Is last op in job
        is_last = 0.0
        if job:
            last_op = job.get_operation(job.get_operation_count() - 1)
            if last_op and last_op.id == op.id:
                is_last = 1.0
        features.append(is_last)

        # [14] Machine working status
        features.append(1.0 if machine.status == MachineStatus.WORKING else 0.0)

        # [15] Competing READY ops count for this machine (from precomputed map)
        features.append(competing_ops.get(machine.id, 0) / 20.0)

        features = features[:self.route_feature_dim]
        while len(features) < self.route_feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _extract_agv_features(self, agv, machine,
                              current_time: float, factory_graph,
                              n_available_agvs: int, n_total_agvs: int) -> np.ndarray:
        """Extract AGV-specific features (12-dim) for a candidate AGV."""
        features = []

        # [0] AGV timer
        features.append(agv.timer / 1000.0)

        # [1] Velocity
        features.append(agv.velocity / 10.0)

        # [2] Status (READY=1)
        features.append(1.0 if agv.get_status() == AGVStatus.READY else 0.0)

        # [3-4] Position
        features.append(agv.x / 100.0)
        features.append(agv.y / 100.0)

        # [5] Estimated travel distance via graph
        travel_dist = 0.0
        if factory_graph and hasattr(factory_graph, 'get_path'):
            path = factory_graph.get_path(agv.point_id, machine.point_id)
            if path and hasattr(factory_graph, 'get_path_weight'):
                travel_dist = factory_graph.get_path_weight(path)
        features.append(travel_dist / 100.0)

        # [6] Estimated travel time
        travel_time = travel_dist / max(agv.velocity, 0.1)
        features.append(travel_time / 100.0)

        # [7] Todo queue length
        features.append(len(agv.todo_queue) / 10.0)

        # [8] AGV already at machine location
        at_location = 1.0 if abs(agv.x - machine.x) < 1.0 and abs(agv.y - machine.y) < 1.0 else 0.0
        features.append(at_location)

        # [9] Idle time
        idle_time = max(0.0, current_time - agv.timer)
        features.append(idle_time / 1000.0)

        # [10] Euclidean distance
        eucl_dist = math.sqrt((agv.x - machine.x) ** 2 + (agv.y - machine.y) ** 2)
        features.append(eucl_dist / 100.0)

        # [11] Available AGV ratio
        features.append(n_available_agvs / max(n_total_agvs, 1))

        features = features[:self.agv_feature_dim]
        while len(features) < self.agv_feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    # ----------------------------------------------------------
    # Core: Decision Flow (GRPO policy sampling)
    # ----------------------------------------------------------

    def sample(self, agvs: List, machines: List,
               jobs: List) -> Tuple[List, float]:
        """Three-phase decision: Sequence -> Route -> AGV Select.

        In training mode, actions are sampled from the policy distribution (stochastic).
        In inference mode, actions are selected greedily (argmax).
        """
        decisions = []

        # 1. Build and encode current state
        factory_graph = self._get_factory_graph(agvs)
        current_time = self._get_current_time()

        graph_result = self.graph_builder.build(
            agvs, machines, jobs, factory_graph, current_time,
        )

        self.gnn_encoder.eval()
        self.seq_actor.eval()
        self.route_actor.eval()
        self.agv_actor.eval()

        with torch.no_grad():
            global_state, node_embs = self.gnn_encoder(
                graph_result.data, graph_result.node_type_offsets,
            )

        global_state_np = global_state.cpu().numpy()

        # 2. Collect READY operations
        ready_ops = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    ready_ops.append(op)

        if not ready_ops:
            # Check completion
            if all(j.is_finished() for j in jobs):
                self.alive = False
                return [], 0
            # Store transition with no decisions
            self._store_transition(global_state_np, None, None, None, None, None)
            return [], DEFAULT_STEP_TIME

        # 3. Sequencing: rank operations by policy
        seq_features = self._extract_seq_features(machines, jobs, current_time)

        with torch.no_grad():
            gs_t = torch.tensor(global_state_np, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            sf_t = torch.tensor(seq_features, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

            # Get action logits (no critic in GRPO)
            seq_logits = self.seq_actor(gs_t, sf_t).squeeze(0)

            # Mask invalid actions (only first len(ready_ops) are valid)
            valid_logits = seq_logits[:len(ready_ops)]

            if self.task_mode == TRAINING:
                # Sample from categorical distribution (stochastic policy)
                dist = Categorical(logits=valid_logits)
                seq_action = dist.sample()
                seq_log_prob = dist.log_prob(seq_action).item()
            else:
                # Greedy selection (argmax)
                seq_action = torch.argmax(valid_logits)
                seq_log_prob = 0.0

        # Build priority order from the sequencing action
        with torch.no_grad():
            all_logits = seq_logits[:len(ready_ops)]
            if self.task_mode == TRAINING:
                # Stochastic priority: sample permutation weighted by softmax probabilities
                probs = F.softmax(all_logits, dim=-1)
                priority_order = torch.multinomial(probs, len(ready_ops), replacement=False).tolist()
            else:
                # Greedy: rank by logit value (descending)
                priority_order = torch.argsort(all_logits, descending=True).tolist()

        # 4. Pre-compute shared data structures once (avoids O(n^2))
        op_to_job, competing_ops, avg_queue, job_progress = self._precompute_context(machines, jobs)

        # 5. For each op in priority order: Route + AGV Select
        assigned_agvs = set()
        current_decisions_info = []

        for rank_idx in priority_order:
            op = ready_ops[rank_idx]

            # Find capable machines
            capable_machines = [m for m in machines
                                if op.is_machine_capable(m.id) and m.is_available()]
            if not capable_machines:
                continue

            # Find available AGVs
            available_agvs = [a for a in agvs
                              if a.get_status() == AGVStatus.READY
                              and (self.allow_agv_reassignment or a.id not in assigned_agvs)]
            if not available_agvs:
                continue

            # === ROUTING DECISION ===
            route_feats_all = []
            for m in capable_machines:
                rf = self._extract_route_features(
                    op, m, machines, jobs, current_time,
                    op_to_job, competing_ops, avg_queue, job_progress,
                )
                route_feats_all.append(rf)

            route_feats_np = np.array(route_feats_all, dtype=np.float32)
            with torch.no_grad():
                rf_t = torch.tensor(route_feats_np, dtype=torch.float32,
                                    device=self.device)
                gs_batch = gs_t.expand(len(capable_machines), -1)
                route_logits = self.route_actor(gs_batch, rf_t)

                valid_route_logits = route_logits[:, 0]  # first logit per candidate

                if self.task_mode == TRAINING:
                    dist = Categorical(logits=valid_route_logits)
                    route_action = dist.sample()
                    route_log_prob = dist.log_prob(route_action).item()
                else:
                    route_action = torch.argmax(valid_route_logits)
                    route_log_prob = 0.0

            selected_machine = capable_machines[route_action.item()]

            # === AGV SELECTION DECISION ===
            n_avail = len(available_agvs)
            agv_feats_all = []
            for a in available_agvs:
                af = self._extract_agv_features(
                    a, selected_machine, current_time, factory_graph,
                    n_avail, len(agvs),
                )
                agv_feats_all.append(af)

            agv_feats_np = np.array(agv_feats_all, dtype=np.float32)
            with torch.no_grad():
                af_t = torch.tensor(agv_feats_np, dtype=torch.float32,
                                    device=self.device)
                gs_batch = gs_t.expand(len(available_agvs), -1)
                agv_logits = self.agv_actor(gs_batch, af_t)

                valid_agv_logits = agv_logits[:, 0]  # first logit per candidate

                if self.task_mode == TRAINING:
                    dist = Categorical(logits=valid_agv_logits)
                    agv_action = dist.sample()
                    agv_log_prob = dist.log_prob(agv_action).item()
                else:
                    agv_action = torch.argmax(valid_agv_logits)
                    agv_log_prob = 0.0

            selected_agv = available_agvs[agv_action.item()]

            decisions.append((op, selected_agv, selected_machine))
            assigned_agvs.add(selected_agv.id)

            current_decisions_info.append({
                'seq_action': rank_idx,
                'route_action': route_action.item(),
                'agv_action': agv_action.item(),
                'seq_features': seq_features,
                'route_features': route_feats_np[route_action.item()],
                'route_all_candidate_features': route_feats_np,  # full candidate set
                'agv_features': agv_feats_np[agv_action.item()],
                'agv_all_candidate_features': agv_feats_np,      # full candidate set
                'seq_log_prob': seq_log_prob,
                'route_log_prob': route_log_prob,
                'agv_log_prob': agv_log_prob,
            })

        # Store transition data for training
        self._store_transition(
            global_state_np, seq_features, current_decisions_info,
            None, None, None,
        )

        return decisions, DEFAULT_STEP_TIME

    # ----------------------------------------------------------
    # Reward (references DualDRLAgent design + graph-agent dense shaping)
    # ----------------------------------------------------------

    def reward(self, *args, **kwargs) -> float:
        """Compute composite reward referencing DualDRLAgent design.

        Compatible with both calling patterns:
        - agent.reward({}) or agent.reward() — uses self.context
        - agent.reward(env_info) — uses env_info with makespan/machines/jobs/agvs
        """
        env_info = None
        if args and isinstance(args[0], dict):
            env_info = args[0]
        elif 'env_info' in kwargs:
            env_info = kwargs['env_info']
        elif 'env' in kwargs:
            env_info = kwargs['env']

        r = self._compute_reward(env_info)
        self._episode_reward += r
        return r

    def _compute_reward(self, env_info: Optional[Dict] = None) -> float:
        """Pure reward computation without side effects."""
        reward = 0.0

        makespan = 0.0
        machines = None
        jobs = None
        agvs = None

        if env_info and isinstance(env_info, dict):
            makespan = env_info.get('makespan', 0.0)
            machines = env_info.get('machines', None)
            jobs = env_info.get('jobs', None)
            agvs = env_info.get('agvs', None)

        # Fallback to self.context if env_info is insufficient
        if machines is None and self.context and hasattr(self.context, 'machines'):
            machines = self.context.machines
        if jobs is None and self.context and hasattr(self.context, 'jobs'):
            jobs = self.context.jobs
        if agvs is None and self.context and hasattr(self.context, 'agvs'):
            agvs = self.context.agvs
        if makespan == 0.0 and self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline

        # ---- DualDRLAgent-style routing reward ----
        # Makespan penalty
        reward -= 0.005 * makespan

        # Machine utilization
        if machines:
            total_m = len(machines) if machines else 1
            working = sum(1 for m in machines if m.status == MachineStatus.WORKING)
            reward += 0.5 * (working / total_m)

            # Load balance — penalize high CV of machine queues
            loads = [len(m.input_queue) for m in machines]
            if len(loads) > 1 and np.mean(loads) > 0:
                cv = float(np.std(loads)) / (float(np.mean(loads)) + 1e-5)
                reward -= 3.0 * cv
        else:
            total_m = 1

        # ---- DualDRLAgent-style sequencing reward ----
        total_queue_len = sum(len(m.input_queue) for m in machines) if machines else 0
        if hasattr(self, '_prev_queue_len'):
            queue_reduction = -(total_queue_len - self._prev_queue_len)
            reward += 0.5 * queue_reduction
            if queue_reduction > 0:
                reward += 0.1
        self._prev_queue_len = total_queue_len

        # ---- Graph-agent dense shaping ----
        if jobs:
            total = len(jobs) if jobs else 1
            completed = sum(1 for j in jobs if j.is_finished())
            reward += 50.0 * (completed / total)

            waiting = sum(
                1 for j in jobs
                for i in range(j.get_operation_count())
                if j.get_operation(i).get_status() == OperationStatus.WAITING
            )
            reward -= 0.5 * waiting

        # Active AGV ratio
        if agvs:
            total_a = len(agvs) if agvs else 1
            active = sum(1 for a in agvs if a.get_status() != AGVStatus.READY)
            reward += 1.5 * (active / total_a)

        return reward

    # ----------------------------------------------------------
    # Transition Collection
    # ----------------------------------------------------------

    def _store_transition(self, global_state_np, seq_features, decisions_info,
                          next_global_state_np, next_seq_features, reward):
        """Store current step data and collect transition from previous step into rollout buffer."""
        if self.task_mode != TRAINING:
            self._prev_global_state = global_state_np
            self._prev_seq_features = seq_features
            self._prev_decisions_info = []
            return

        # Collect transitions from previous step into rollout buffer
        if self._prev_global_state is not None and self._prev_decisions_info:
            step_reward = self._compute_reward()

            # Normalize reward
            self.reward_normalizer.update(np.array([step_reward]))
            norm_reward = self.reward_normalizer.normalize(np.array([step_reward]))[0]

            done = False
            if self.context and hasattr(self.context, 'jobs'):
                done = all(j.is_finished() for j in self.context.jobs)

            n_decisions = len(self._prev_decisions_info)
            shared_reward = norm_reward / max(n_decisions, 1)

            # Push every decision (GRPO benefits from more samples)
            for info in self._prev_decisions_info:
                route_candidate_feats = info.get('route_all_candidate_features')
                agv_candidate_feats = info.get('agv_all_candidate_features')

                self.rollout_buffer.push(
                    state=self._prev_global_state,
                    seq_features=self._prev_seq_features,
                    reward=shared_reward,
                    done=float(done),
                    group_id=self._grpo_group_id,
                    # Sequencing
                    seq_action=info.get('seq_action', -1),
                    seq_log_prob=info.get('seq_log_prob', 0.0),
                    # Routing
                    route_action=info.get('route_action', -1),
                    route_log_prob=info.get('route_log_prob', 0.0),
                    route_all_candidate_feats=route_candidate_feats,
                    # AGV
                    agv_action=info.get('agv_action', -1),
                    agv_log_prob=info.get('agv_log_prob', 0.0),
                    agv_all_candidate_feats=agv_candidate_feats,
                )

                # Update group tracking
                self._samples_in_current_group += 1
                if self._samples_in_current_group >= self.grpo_group_size:
                    self._grpo_group_id += 1
                    self._samples_in_current_group = 0

            self._rollout_step_count += 1

        # Update previous state
        self._prev_global_state = global_state_np
        self._prev_seq_features = seq_features
        self._prev_decisions_info = decisions_info if decisions_info else []

    def after_sample(self, *args, **kwargs):
        """Post-sample hook: no-op (transition collection is done inside _store_transition)."""
        pass

    def before_sample(self, *args, **kwargs):
        """Pre-sample hook: no-op."""
        pass

    # ----------------------------------------------------------
    # GRPO Training
    # ----------------------------------------------------------

    def train(self, *args, **kwargs):
        """Train all three Actors using GRPO (critic-free, group-relative advantage).

        Triggered every rollout_steps environment steps.
        """
        if self.task_mode != TRAINING:
            return

        # Only update when rollout buffer is full enough
        if len(self.rollout_buffer) < self.rollout_steps:
            return

        # Compute group-relative advantages (no critic needed)
        self.rollout_buffer.compute_group_relative_advantages(
            gamma=self.gamma, min_group_size=self.min_group_size,
        )

        # GRPO update for all three sub-agents
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for epoch in range(self.grpo_epochs):
            seq_stats = self._grpo_update_sub_agent('seq')
            route_stats = self._grpo_update_sub_agent('route')
            agv_stats = self._grpo_update_sub_agent('agv')

            for stats in [seq_stats, route_stats, agv_stats]:
                if stats is not None:
                    total_policy_loss += stats['policy_loss']
                    total_entropy += stats['entropy']
                    total_kl += stats['kl_approx']
                    n_updates += 1

            # 主动释放 GIL，防止 CPU 密集的 GRPO 训练阻塞 asyncio 事件循环
            time.sleep(0)

        self._train_step += 1

        # Record training stats
        if n_updates > 0:
            avg_policy_loss = total_policy_loss / n_updates
            avg_entropy = total_entropy / n_updates
            avg_kl = total_kl / n_updates
            avg_total = avg_policy_loss + self.kl_coeff * avg_kl - self.entropy_coeff * avg_entropy

            self.training_history['policy_loss'].append(avg_policy_loss)
            self.training_history['entropy'].append(avg_entropy)
            self.training_history['kl_approx'].append(avg_kl)
            self.training_history['total_loss'].append(avg_total)

        # Clear rollout buffer after update
        self.rollout_buffer.reset()
        self._rollout_step_count = 0

    def _grpo_update_sub_agent(self, sub_agent: str) -> Optional[Dict[str, float]]:
        """Perform one GRPO update epoch for a sub-agent."""
        data = self.rollout_buffer.get_tensors(sub_agent)
        n_samples = data['states'].shape[0]

        if n_samples < self.grpo_batch_size:
            # Use all data if not enough for a mini-batch
            mini_batches = [None]
        else:
            n_batches = max(1, n_samples // self.grpo_batch_size)
            indices = np.random.permutation(n_samples)
            mini_batches = [indices[i * self.grpo_batch_size:(i + 1) * self.grpo_batch_size]
                           for i in range(n_batches)]

        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_batches_actual = 0

        # Select the correct networks and optimizer
        if sub_agent == 'seq':
            actor = self.seq_actor
            optimizer = self.optimizer_seq
            feat_key = 'seq_features'
            max_output_dim = self.max_seq_output_dim
        elif sub_agent == 'route':
            actor = self.route_actor
            optimizer = self.optimizer_route
            max_output_dim = self.max_route_output_dim
        elif sub_agent == 'agv':
            actor = self.agv_actor
            optimizer = self.optimizer_agv
            max_output_dim = self.max_agv_output_dim

        # Set networks to training mode
        self.gnn_encoder.train()
        actor.train()

        for batch_indices in mini_batches:
            if sub_agent == 'seq':
                # Sequencing: standard categorical distribution over fixed output dims
                if batch_indices is None:
                    states = data['states']
                    features = data[feat_key]
                    actions = data['actions']
                    old_log_probs = data['old_log_probs']
                    advantages = data['advantages']
                else:
                    states = data['states'][batch_indices]
                    features = data[feat_key][batch_indices]
                    actions = data['actions'][batch_indices]
                    old_log_probs = data['old_log_probs'][batch_indices]
                    advantages = data['advantages'][batch_indices]

                # Filter valid actions
                valid_mask = actions >= 0
                if valid_mask.sum() == 0:
                    continue
                states = states[valid_mask]
                features = features[valid_mask]
                actions = actions[valid_mask]
                old_log_probs = old_log_probs[valid_mask]
                advantages = advantages[valid_mask]

                # Clamp actions to valid range
                actions = actions.clamp(0, max_output_dim - 1)

                # Evaluate current policy
                logits = actor(states, features)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

            else:
                # Route / AGV: reconstruct distribution over stored candidate sets
                all_candidate_features = data[
                    f'{sub_agent}_all_candidate_features'
                ]

                if batch_indices is None:
                    batch_idx_list = list(range(n_samples))
                else:
                    batch_idx_list = batch_indices.tolist()

                new_log_probs_list = []
                old_log_probs_list = []
                advantages_list = []
                entropy_sum = 0.0
                n_valid = 0

                for i in batch_idx_list:
                    action_i = data['actions'][i].item()
                    if action_i < 0:
                        continue

                    candidate_feats = all_candidate_features[i]
                    if candidate_feats.shape[0] <= 1 and candidate_feats.shape[1] <= 1:
                        continue

                    state_i = data['states'][i]
                    advantage_i = data['advantages'][i]
                    old_lp_i = data['old_log_probs'][i].item()

                    # Reconstruct categorical distribution over candidates
                    n_cand = candidate_feats.shape[0]
                    cand_t = torch.tensor(candidate_feats, dtype=torch.float32,
                                          device=self.device)
                    gs_batch = state_i.unsqueeze(0).expand(n_cand, -1)

                    logits = actor(gs_batch, cand_t)
                    valid_logits = logits[:, 0]  # first logit per candidate
                    dist = Categorical(logits=valid_logits)

                    action_clamped = min(action_i, n_cand - 1)
                    new_lp = dist.log_prob(torch.tensor(action_clamped, device=self.device))
                    ent = dist.entropy()

                    new_log_probs_list.append(new_lp.unsqueeze(0))
                    old_log_probs_list.append(old_lp_i)
                    advantages_list.append(advantage_i.item())
                    entropy_sum += ent.item()
                    n_valid += 1

                if n_valid == 0:
                    continue

                new_log_probs = torch.cat(new_log_probs_list, dim=0)
                old_log_probs = torch.tensor(old_log_probs_list, dtype=torch.float32,
                                            device=self.device)
                advantages = torch.tensor(advantages_list, dtype=torch.float32,
                                         device=self.device)
                entropy = torch.tensor(entropy_sum / n_valid, dtype=torch.float32,
                                      device=self.device)

            # GRPO clipped surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Approximate KL divergence (for monitoring / optional penalty)
            approx_kl = (old_log_probs - new_log_probs).mean()

            # Total loss: policy + optional KL penalty - entropy bonus
            loss = policy_loss + self.kl_coeff * approx_kl - self.entropy_coeff * entropy

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) + list(actor.parameters()),
                self.max_grad_norm,
            )
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item() if isinstance(entropy, torch.Tensor) else entropy
            total_kl += approx_kl.item() if isinstance(approx_kl, torch.Tensor) else approx_kl
            n_batches_actual += 1

        if n_batches_actual == 0:
            return None

        return {
            'policy_loss': total_policy_loss / n_batches_actual,
            'entropy': total_entropy / n_batches_actual,
            'kl_approx': total_kl / n_batches_actual,
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _get_factory_graph(self, agvs):
        """Get the factory floor graph from AGVs."""
        if agvs and hasattr(agvs[0], 'graph'):
            return agvs[0].graph
        if self.context and hasattr(self.context, 'graph'):
            return self.context.graph
        return None

    def _get_current_time(self) -> float:
        """Get current simulation time."""
        if self.context and hasattr(self.context, 'env_timeline'):
            return self.context.env_timeline
        return 0.0

    def _remaining_processing_time(self, op) -> float:
        """Compute total remaining processing time for an operation's job."""
        remaining = 0.0
        nxt = op
        while nxt is not None:
            if nxt.durations:
                avg_dur = sum(d for _, d in nxt.durations) / len(nxt.durations)
                remaining += avg_dur
            nxt = nxt.get_next_operation()
        return remaining

    # ----------------------------------------------------------
    # Model save / load
    # ----------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> bool:
        """Save all network state_dicts and hyperparameters to a .pt file."""
        try:
            if path is None:
                agent_name = self.name or "GraphGRPOAgent"
                agent_dir = f"training_logs/models/{agent_name}"
                os.makedirs(agent_dir, exist_ok=True)
                path = f"{agent_dir}/agent_model.pt"

            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            checkpoint = {
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'seq_actor': self.seq_actor.state_dict(),
                'route_actor': self.route_actor.state_dict(),
                'agv_actor': self.agv_actor.state_dict(),
                'optimizer_gnn': self.optimizer_gnn.state_dict(),
                'optimizer_seq': self.optimizer_seq.state_dict(),
                'optimizer_route': self.optimizer_route.state_dict(),
                'optimizer_agv': self.optimizer_agv.state_dict(),
                'hyperparams': {
                    'hidden_dim': self.hidden_dim,
                    'gamma': self.gamma,
                    'clip_epsilon': self.clip_epsilon,
                    'grpo_epochs': self.grpo_epochs,
                    'grpo_batch_size': self.grpo_batch_size,
                    'grpo_group_size': self.grpo_group_size,
                    'min_group_size': self.min_group_size,
                    'entropy_coeff': self.entropy_coeff,
                    'kl_coeff': self.kl_coeff,
                    'max_grad_norm': self.max_grad_norm,
                    'rollout_steps': self.rollout_steps,
                    'max_seq_output_dim': self.max_seq_output_dim,
                    'max_route_output_dim': self.max_route_output_dim,
                    'max_agv_output_dim': self.max_agv_output_dim,
                    'seq_feature_dim': self.seq_feature_dim,
                    'route_feature_dim': self.route_feature_dim,
                    'agv_feature_dim': self.agv_feature_dim,
                    'allow_agv_reassignment': self.allow_agv_reassignment,
                },
                'reward_normalizer': {
                    'mean': self.reward_normalizer.mean,
                    'var': self.reward_normalizer.var,
                    'count': self.reward_normalizer.count,
                },
                'training_history': self.training_history,
                'train_step': self._train_step,
                'mode': self.mode,
            }
            torch.save(checkpoint, path)
            LOGGER.info(f"[GraphGRPOAgent] Model saved to {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphGRPOAgent] Save failed: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load all network state_dicts and hyperparameters from a .pt file.

        Tolerant of missing critic keys (for loading from PPO checkpoints,
        only GNN/actor weights are loaded).
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.seq_actor.load_state_dict(checkpoint['seq_actor'])
            self.route_actor.load_state_dict(checkpoint['route_actor'])
            self.agv_actor.load_state_dict(checkpoint['agv_actor'])

            for opt_key, optimizer in [
                ('optimizer_gnn', self.optimizer_gnn),
                ('optimizer_seq', self.optimizer_seq),
                ('optimizer_route', self.optimizer_route),
                ('optimizer_agv', self.optimizer_agv),
            ]:
                if opt_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[opt_key])

            hp = checkpoint.get('hyperparams', {})
            self.gamma = hp.get('gamma', self.gamma)
            self.clip_epsilon = hp.get('clip_epsilon', self.clip_epsilon)
            self.grpo_epochs = hp.get('grpo_epochs', self.grpo_epochs)
            self.grpo_batch_size = hp.get('grpo_batch_size', self.grpo_batch_size)
            self.grpo_group_size = hp.get('grpo_group_size', self.grpo_group_size)
            self.min_group_size = hp.get('min_group_size', self.min_group_size)
            self.entropy_coeff = hp.get('entropy_coeff', self.entropy_coeff)
            self.kl_coeff = hp.get('kl_coeff', self.kl_coeff)

            if 'reward_normalizer' in checkpoint:
                rn = checkpoint['reward_normalizer']
                self.reward_normalizer.mean = rn['mean']
                self.reward_normalizer.var = rn['var']
                self.reward_normalizer.count = rn['count']

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            self._train_step = checkpoint.get('train_step', 0)

            LOGGER.info(f"[GraphGRPOAgent] Model loaded from {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphGRPOAgent] Load failed: {e}")
            return False

    # ----------------------------------------------------------
    # Training Metrics
    # ----------------------------------------------------------

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics for convergence detection."""
        metrics = {
            'episode_reward': self._episode_reward,
            'training_history': self.training_history,
        }
        if self.training_history.get('policy_loss'):
            metrics['policy_loss'] = self.training_history['policy_loss'][-1]
        if self.training_history.get('entropy'):
            metrics['entropy'] = self.training_history['entropy'][-1]
        if self.training_history.get('total_loss'):
            metrics['total_loss'] = self.training_history['total_loss'][-1]
        if self.training_history.get('kl_approx'):
            metrics['kl_approx'] = self.training_history['kl_approx'][-1]
        return metrics

    def new_episode(self):
        """Reset episode-level state. Call at the start of each episode."""
        self._episode_reward = 0.0
        self._prev_global_state = None
        self._prev_seq_features = None
        self._prev_decisions_info = []
        self._prev_queue_len = 0

        # Record episode stats
        if self.training_history['episodes']:
            last_ep = self.training_history['episodes'][-1] + 1
        else:
            last_ep = 1
        self.training_history['episodes'].append(last_ep)

        makespan = 0.0
        if self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline
        self.training_history['makespans'].append(makespan)

    def __repr__(self):
        return (f"<GraphGRPOAgent id={self.agent_id} name={self.name} "
                f"device={self.device} group_size={self.grpo_group_size}>")
