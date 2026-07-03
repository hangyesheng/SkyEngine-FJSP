"""Environment adapter for GraphGRPOAgent (doc §3.5 contract).

The agent operates on `Operation / Machine / AGV / Job / Graph` objects with a
specific attribute/method interface. This module builds lightweight objects that
satisfy that contract directly from the SkyEngine-FJSP JSON `obs` format:

    obs = {
      "jobs":     [{"job_id": int, "ops": [{"op_id": int,
                                           "machine_options": [int, ...],
                                           "proc_times": {"<mid>": float, ...}}]}],
      "machines": [{"id": int, "location": [x, y]}]
    }

The objects are plain data holders; all simulation/state transitions are driven by
`OnlineGraphGRPOSolver` (see `online_graphgrpo_solver.py`), which mutates their
attributes (`status`, `timer`, `start_time`, ...) between agent calls.

Status constants are imported from `graphgrpo_agent` so the adapter and the agent
share one source of truth (int-valued: WAITING=0, READY=1, ...).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from graphgrpo_agent import (
    AGVStatus,
    MachineStatus,
    OperationStatus,
)

# Depot point id (machines use point_id == their machine id, so the depot needs a
# distinct sentinel). AGVs start at the depot.
DEPOT_POINT = -1


class Operation:
    """An operation within a job. Satisfies the Operation contract (doc §3.5)."""

    __slots__ = (
        "id", "durations", "process_time", "status", "job",
        "prev_op", "next_op", "assigned_machine",
        "start_time", "end_time",
    )

    def __init__(self, op_id: int, durations: List[Tuple[int, float]], job=None):
        self.id = op_id
        # [(machine_id, processing_time), ...] — capable machines for this op.
        self.durations = durations
        self.process_time = 0.0        # accumulated processing (for feature progress)
        self.status = OperationStatus.WAITING
        self.job = job
        self.prev_op: Optional["Operation"] = None
        self.next_op: Optional["Operation"] = None
        self.assigned_machine: Optional["Machine"] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def get_status(self) -> int:
        return self.status

    def get_next_operation(self) -> Optional["Operation"]:
        return self.next_op

    def is_machine_capable(self, machine_id: int) -> bool:
        return any(m == machine_id for m, _ in self.durations)

    def get_duration(self, machine_id: int) -> float:
        for m, d in self.durations:
            if m == machine_id:
                return float(d)
        return 0.0


class Machine:
    """A processing machine. Satisfies the Machine contract (doc §3.5)."""

    __slots__ = ("id", "status", "timer", "input_queue", "x", "y", "point_id")

    def __init__(self, machine_id: int, x: float, y: float):
        self.id = machine_id
        self.status = MachineStatus.READY
        self.timer = 0.0          # when WORKING: op end_time; when idle: current_time
        self.input_queue: List[Operation] = []
        self.x = float(x)
        self.y = float(y)
        # point_id coincides with machine id; the factory graph treats it as a node.
        self.point_id = machine_id

    def is_available(self) -> bool:
        return self.status == MachineStatus.READY


class AGV:
    """An automated guided vehicle. Satisfies the AGV contract (doc §3.5)."""

    __slots__ = (
        "id", "timer", "velocity", "x", "y", "point_id",
        "todo_queue", "graph", "status",
    )

    def __init__(self, agv_id: int, velocity: float, graph=None):
        self.id = agv_id
        self.timer = 0.0          # time the AGV becomes free (after current delivery)
        self.velocity = float(velocity)
        self.x = 0.0
        self.y = 0.0
        self.point_id = DEPOT_POINT
        self.todo_queue: List = []
        self.graph = graph        # factory floor graph (used by the agent)
        self.status = AGVStatus.READY

    def get_status(self) -> int:
        return self.status


class Job:
    """A job (ordered sequence of operations). Satisfies the Job contract (doc §3.5)."""

    __slots__ = ("id", "ops", "status")

    def __init__(self, job_id: int, ops: List[Operation]):
        self.id = job_id
        self.ops = ops
        self.status = 0   # not used by the agent directly; is_finished() drives logic

    def get_operation_count(self) -> int:
        return len(self.ops)

    def get_operation(self, i: int) -> Optional[Operation]:
        if 0 <= i < len(self.ops):
            return self.ops[i]
        return None

    def is_finished(self) -> bool:
        return all(op.status == OperationStatus.FINISHED for op in self.ops)


class FactoryGraph:
    """Minimal factory floor graph: a fully-connected set of points with Euclidean
    edge weights. Points = depot + machines. Satisfies the Graph contract (doc §3.5).

    `get_path` returns the trivial `[src, dst]` for any known pair (everything is
    mutually reachable); `get_path_weight` returns the Euclidean distance along it.
    """

    def __init__(self, coords: Dict[int, Tuple[float, float]]):
        self.coords = coords

    def get_path(self, src_point_id: int, dst_point_id: int) -> List[int]:
        if src_point_id in self.coords and dst_point_id in self.coords:
            return [src_point_id, dst_point_id]
        return []

    def get_path_weight(self, path: List[int]) -> float:
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ax, ay = self.coords.get(a, (0.0, 0.0))
            bx, by = self.coords.get(b, (0.0, 0.0))
            total += math.hypot(ax - bx, ay - by)
        return total


class Context:
    """Environment context exposed to the agent via `__init__(context=...)`.

    `env_timeline` is mutated by the solver before each `sample()` call so that
    `agent._get_current_time()` and `_compute_reward()` see the current sim time.
    """

    __slots__ = ("machines", "jobs", "agvs", "env_timeline", "graph")

    def __init__(self, machines, jobs, agvs, graph, env_timeline=0.0):
        self.machines = machines
        self.jobs = jobs
        self.agvs = agvs
        self.graph = graph
        self.env_timeline = env_timeline


# ============================================================
# Build env from JSON obs
# ============================================================

def _locations_are_degenerate(machines_json: List[dict]) -> bool:
    """True if all machine locations are missing or [0, 0] (the bundled test data)."""
    for m in machines_json:
        loc = m.get("location")
        if not loc:
            continue
        if list(loc) != [0, 0]:
            return False
    return True


def _spread_coordinates(n_machines: int) -> List[Tuple[float, float]]:
    """Place machines on a square grid (10-unit spacing, offset from depot)."""
    cols = max(1, math.ceil(math.sqrt(n_machines)))
    coords = []
    for i in range(n_machines):
        x = float((i % cols + 1) * 10)
        y = float((i // cols + 1) * 10)
        coords.append((x, y))
    return coords


def build_env(obs: dict, n_agvs: Optional[int] = None,
              spread_machines: bool = True,
              agv_velocity: float = 1.0):
    """Build adapter env objects from a SkyEngine-FJSP `obs` dict.

    Args:
        obs: {"jobs": [...], "machines": [...]} (list or compact dict format).
        n_agvs: number of AGVs; default = n_machines (abundant — never blocks).
        spread_machines: if True and all machine locations are [0,0], auto-place
            machines on a grid so positional/AGV-travel features are non-degenerate.
        agv_velocity: AGV speed in distance units per time unit.

    Returns:
        (context, agvs, machines, jobs, factory_graph)
    """
    jobs_json = obs.get("jobs", [])
    machines_json = obs.get("machines", [])

    # --- machines ---
    n_machines = len(machines_json)
    if spread_machines and _locations_are_degenerate(machines_json):
        spread = _spread_coordinates(max(n_machines, 1))
    else:
        spread = None

    machines: List[Machine] = []
    for idx, m in enumerate(machines_json):
        mid = int(m.get("id", idx))
        if spread is not None:
            x, y = spread[idx]
        else:
            loc = m.get("location", [0, 0])
            x, y = float(loc[0]), float(loc[1])
        machines.append(Machine(mid, x, y))
    machine_by_id = {m.id: m for m in machines}

    # --- jobs + operations ---
    jobs: List[Job] = []
    for j in jobs_json:
        jid = int(j.get("job_id", len(jobs)))
        ops_json = j.get("ops", [])
        ops: List[Operation] = []
        for o in ops_json:
            oid = int(o.get("op_id", len(ops)))
            machine_options = [int(mid) for mid in o.get("machine_options", [])]
            proc_times = o.get("proc_times", {})
            durations: List[Tuple[int, float]] = []
            for mid in machine_options:
                # proc_times keys may be int or str (JSON); look up both.
                pt = proc_times.get(mid, proc_times.get(str(mid), 0.0))
                durations.append((mid, float(pt)))
            ops.append(Operation(oid, durations))
        # link the job's op chain + set initial statuses
        for i, op in enumerate(ops):
            op.job = None  # set after Job creation below
            if i + 1 < len(ops):
                op.next_op = ops[i + 1]
            if i - 1 >= 0:
                op.prev_op = ops[i - 1]
            # first op of each job starts READY; the rest wait for their predecessor.
            op.status = OperationStatus.READY if i == 0 else OperationStatus.WAITING
        job = Job(jid, ops)
        for op in ops:
            op.job = job
        jobs.append(job)

    # --- factory graph (depot + machines) ---
    coords: Dict[int, Tuple[float, float]] = {DEPOT_POINT: (0.0, 0.0)}
    for m in machines:
        coords[m.point_id] = (m.x, m.y)
    factory_graph = FactoryGraph(coords)

    # --- AGVs (start at depot, abundant by default) ---
    if n_agvs is None:
        n_agvs = max(1, n_machines)
    agvs: List[AGV] = []
    for i in range(n_agvs):
        agv = AGV(i, velocity=agv_velocity, graph=factory_graph)
        agvs.append(agv)

    context = Context(machines, jobs, agvs, factory_graph, env_timeline=0.0)
    return context, agvs, machines, jobs, factory_graph
