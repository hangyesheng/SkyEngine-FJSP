"""Online GraphGRPO Solver — SkyEngine-FJSP JobSolver interface.

Mirrors `End-to-end-DRL-for-FJSP-main/online_drl_solver.py`: the same HTTP-facing
`plan(obs)` contract (offline solve on first call, then release per time step).

The GraphGRPO agent is an *online* decision-maker
(`sample(agvs, machines, jobs) -> [(op, agv, machine), ...]`). To fit this project's
offline-then-release API, `_solve_offline` runs an event-driven simulation: it builds
adapter env objects from `obs` (see `env_adapter.py`), repeatedly calls the agent to
assign ready operations to machines + AGVs, commits each decision to a concrete
(start_time, end_time), advances the sim clock to the next completion event, and
records a full schedule. That schedule is then split into `machine_actions` and
`transfer_requests` exactly like the DRL solver, and released step by step.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_adapter import DEPOT_POINT, build_env
from graphgrpo_agent import (
    AGVStatus,
    MachineStatus,
    OperationStatus,
    GraphGRPOAgent,
)

LOGGER = logging.getLogger("OnlineGraphGRPOSolver")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


class OnlineGraphGRPOSolver:
    """Online FJSP scheduler backed by the GraphGRPO agent.

    Args:
        device: 'cpu' or 'cuda' (auto-falls-back to CPU on CUDA failure).
        model_path: path to agent_model.pt; default <here>/models/agent_model.pt.
        n_agvs: number of AGVs in the sim; default = n_machines (abundant).
        spread_machines: auto-grid machine positions when obs locations are all [0,0].
        consider_transport: if True, AGV travel time delays op start times (and the
            makespan). Default False — matches the sibling solvers (DE/PSO/CP-SAT/DRL),
            which report a pure-processing makespan; the AGV selection sub-agent still
            runs and sees real graph distances in its features.
        seed: RNG seed (inference is greedy, so this only affects incidental randomness).
    """

    def __init__(self, device: str = "cpu", model_path: Optional[str] = None,
                 n_agvs: Optional[int] = None, spread_machines: bool = True,
                 consider_transport: bool = False, seed: int = 42):
        self.device_str = device
        self.n_agvs = n_agvs
        self.spread_machines = spread_machines
        self.consider_transport = consider_transport
        self.seed = seed

        here = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(here, "models", "agent_model.pt")

        # internal state
        self.initialized = False
        self.time_stamp = 0
        self.task_idx = 0
        self._machine_actions = []
        self._transfer_requests = []

        # env objects (set on solve)
        self._context = None
        self._agvs = []
        self._machines = []
        self._jobs = []
        self._factory_graph = None

        # determinism (inference is greedy; seed is for reproducibility only)
        random.seed(seed)
        np.random.seed(seed)

        self._agent = self._build_agent()

    # ------------------------------------------------------------------ #
    #                       Agent construction                            #
    # ------------------------------------------------------------------ #

    def _build_agent(self) -> GraphGRPOAgent:
        model_arg = self.model_path if os.path.exists(self.model_path) else None
        if model_arg is None:
            LOGGER.warning(f"model not found at {self.model_path!r}; "
                           "running with randomly initialized weights "
                           "(schedule quality will be poor).")

        return GraphGRPOAgent(
            name="GraphGRPOAgent",
            agent_id=1,
            context=None,
            ui_mode="backend",
            task_mode="inference",
            model_path=model_arg,
            device=self.device_str,
        )

    # ------------------------------------------------------------------ #
    #                       From obs → env objects                        #
    # ------------------------------------------------------------------ #

    def _obs_to_env(self, obs: dict):
        (self._context, self._agvs, self._machines, self._jobs,
         self._factory_graph) = build_env(
            obs, n_agvs=self.n_agvs, spread_machines=self.spread_machines)
        # wire context into the agent (used by _get_current_time / _compute_reward)
        self._agent.context = self._context

    # ------------------------------------------------------------------ #
    #                       Event-driven simulation                       #
    # ------------------------------------------------------------------ #

    def _update_statuses(self, current_time: float):
        """Complete working ops that have finished by `current_time`, promote their
        successors to READY, free machines whose queued work is done, and free AGVs.

        Machines are freed by `machine.timer` (the latest committed end time on
        that machine), NOT when an individual op finishes — a machine may have
        several ops serialized via `timer`, and freeing it early would let a new
        op overlap the ones still queued/running on it.
        """
        eps = 1e-9
        # finish ops whose end time has been reached (unlocks their successor)
        for job in self._jobs:
            for op in job.ops:
                if (op.status == OperationStatus.WORKING
                        and op.end_time is not None
                        and op.end_time <= current_time + eps):
                    op.status = OperationStatus.FINISHED
                    if op.assigned_machine is not None:
                        op.process_time = op.get_duration(op.assigned_machine.id)
                        # pop the finished head of this machine's queue (FIFO order)
                        m = op.assigned_machine
                        if m.input_queue and m.input_queue[0] is op:
                            m.input_queue.pop(0)
                    nxt = op.next_op
                    if nxt is not None and nxt.status == OperationStatus.WAITING:
                        nxt.status = OperationStatus.READY
        # free machines whose queued work is all done
        for m in self._machines:
            if m.status == MachineStatus.WORKING and m.timer <= current_time + eps:
                m.status = MachineStatus.READY
                m.timer = current_time
                m.input_queue = []
        # free AGVs whose delivery completed
        for agv in self._agvs:
            if agv.status != AGVStatus.READY and agv.timer <= current_time + eps:
                agv.status = AGVStatus.READY

    def _next_event_time(self, current_time: float) -> Optional[float]:
        """Earliest future sim event: a working op finishing or a busy AGV freeing."""
        eps = 1e-9
        best = None
        for job in self._jobs:
            for op in job.ops:
                if (op.status == OperationStatus.WORKING
                        and op.end_time is not None
                        and op.end_time > current_time + eps):
                    if best is None or op.end_time < best:
                        best = op.end_time
        for agv in self._agvs:
            if agv.status != AGVStatus.READY and agv.timer > current_time + eps:
                if best is None or agv.timer < best:
                    best = agv.timer
        return best

    def _commit_decision(self, op, agv, machine, current_time: float):
        """Commit one (op, agv, machine) decision to concrete times.

        Machines may receive several ops per decision round (the agent scores each
        op against the same pre-decision machine state); they are serialized via
        `machine.timer`, so no two ops overlap on the same machine.
        """
        prev = op.prev_op
        ready_at = prev.end_time if (prev is not None and prev.end_time is not None) else 0.0

        agv_free = max(agv.timer, current_time)
        pickup = max(ready_at, agv_free)

        from_point = (prev.assigned_machine.point_id
                      if (prev is not None and prev.assigned_machine is not None)
                      else DEPOT_POINT)
        to_point = machine.point_id
        path = self._factory_graph.get_path(from_point, to_point)
        travel_dist = self._factory_graph.get_path_weight(path)
        if self.consider_transport:
            travel_time = travel_dist / max(agv.velocity, 1e-6)
        else:
            # Pure-processing makespan (matches sibling solvers); AGV selection
            # still runs and sees `travel_dist` in its features.
            travel_time = 0.0
        arrival = pickup + travel_time

        machine_free = max(machine.timer, current_time)
        start = max(arrival, machine_free)
        dur = op.get_duration(machine.id)
        end = start + dur

        # commit
        op.status = OperationStatus.WORKING
        op.assigned_machine = machine
        op.start_time = start
        op.end_time = end
        op.process_time = 0.0

        machine.status = MachineStatus.WORKING
        machine.timer = end
        machine.input_queue.append(op)

        agv.status = AGVStatus.LOADED
        agv.timer = arrival  # AGV is free again once the delivery completes
        agv.point_id = to_point
        agv.x, agv.y = self._factory_graph.coords.get(to_point, (0.0, 0.0))

    def _run_simulation(self, max_steps: int = 20000) -> bool:
        """Drive the agent to a complete schedule. Returns True if all jobs finished."""
        current_time = 0.0
        steps = 0
        while not all(j.is_finished() for j in self._jobs):
            if steps >= max_steps:
                LOGGER.warning(f"step cap ({max_steps}) reached at t={current_time}; "
                               "finishing with a partial schedule.")
                return False
            self._update_statuses(current_time)
            self._context.env_timeline = current_time
            decisions, _ = self._agent.decision(self._agvs, self._machines, self._jobs)
            for (op, agv, machine) in decisions:
                self._commit_decision(op, agv, machine, current_time)
            if all(j.is_finished() for j in self._jobs):
                return True
            nxt = self._next_event_time(current_time)
            if nxt is None or nxt <= current_time + 1e-9:
                # No pending event but not all finished — avoid an infinite loop.
                stuck = [op for job in self._jobs for op in job.ops
                         if op.status != OperationStatus.FINISHED]
                LOGGER.warning(f"no progressing event at t={current_time}; "
                               f"{len(stuck)} ops unscheduled. Breaking.")
                return False
            current_time = nxt
            steps += 1
        return True

    def _collect_schedule(self) -> list:
        schedule = []
        for job in self._jobs:
            for op in job.ops:
                if (op.assigned_machine is not None
                        and op.start_time is not None
                        and op.end_time is not None):
                    schedule.append({
                        "job_id": job.id,
                        "op_id": op.id,
                        "machine": op.assigned_machine.id,
                        "start_time": float(op.start_time),
                        "end_time": float(op.end_time),
                    })
        schedule.sort(key=lambda x: (x["start_time"], x["job_id"], x["op_id"]))
        return schedule

    # ------------------------------------------------------------------ #
    #                       Schedule → actions / transfers               #
    # ------------------------------------------------------------------ #

    def _build_actions_and_transfers(self, schedule: list):
        self._machine_actions = [
            {
                "machine_id": s["machine"],
                "job_id": s["job_id"],
                "op_id": s["op_id"],
                "start_time": s["start_time"],
                "expected_end": s["end_time"],
            }
            for s in schedule
        ]

        sched_by_job = {(s["job_id"], s["op_id"]): s for s in schedule}
        self._transfer_requests = []
        for job in self._jobs:
            jid = job.id
            n_ops = job.get_operation_count()
            first = sched_by_job.get((jid, 0))
            if first:
                self._transfer_requests.append({
                    "job_id": jid, "op_id": 0,
                    "from_machine": -1, "to_machine": first["machine"],
                    "ready_time": 0.0,
                })
            for o in range(1, n_ops):
                prev = sched_by_job.get((jid, o - 1))
                curr = sched_by_job.get((jid, o))
                if prev and curr:
                    self._transfer_requests.append({
                        "job_id": jid, "op_id": o,
                        "from_machine": prev["machine"], "to_machine": curr["machine"],
                        "ready_time": prev["end_time"],
                    })

        self._machine_actions.sort(key=lambda x: x["start_time"])
        self._transfer_requests.sort(key=lambda x: x["ready_time"])

    # ------------------------------------------------------------------ #
    #                       Offline solve                                 #
    # ------------------------------------------------------------------ #

    def _solve_offline(self, obs: dict) -> float:
        self._obs_to_env(obs)
        finished = self._run_simulation()
        schedule = self._collect_schedule()
        makespan = max((s["end_time"] for s in schedule), default=0.0)
        self._build_actions_and_transfers(schedule)
        LOGGER.info(f"[OnlineGraphGRPO] solve complete: finished={finished}, "
                    f"makespan={makespan:.1f}, actions={len(self._machine_actions)}, "
                    f"transfers={len(self._transfer_requests)}")
        return makespan

    # ------------------------------------------------------------------ #
    #                       RoutingTask (SkyEngine-compatible)            #
    # ------------------------------------------------------------------ #

    def _create_routing_task(self, task_dict: dict) -> dict:
        task = {
            "task_id": self.task_idx,
            "job_id": task_dict["job_id"],
            "op_id": task_dict["op_id"],
            "source": (task_dict["from_machine"], 0),
            "destination": (task_dict["to_machine"], 0),
            "candidate_machines": [task_dict["to_machine"]],
            "ready_time": task_dict["ready_time"],
        }
        self.task_idx += 1
        return task

    # ------------------------------------------------------------------ #
    #                       plan — main interface                         #
    # ------------------------------------------------------------------ #

    def plan(self, obs: dict) -> dict:
        """Online scheduling entry, called once per time step.

        First call triggers the offline solve; subsequent calls release the
        precomputed schedule by `time_stamp`.
        """
        self.time_stamp += 1

        if not self.initialized:
            self._solve_offline(obs)
            self.initialized = True

        current_time = float(self.time_stamp)

        ready_actions = []
        while (self._machine_actions
               and self._machine_actions[0]["start_time"] <= current_time + 1e-6):
            ready_actions.append(self._machine_actions.pop(0))

        ready_transfers = []
        while (self._transfer_requests
               and self._transfer_requests[0]["ready_time"] <= current_time + 1e-6):
            tr = self._transfer_requests.pop(0)
            ready_transfers.append(self._create_routing_task(tr))

        return {
            "machine_actions": ready_actions,
            "transfer_requests": ready_transfers,
        }
