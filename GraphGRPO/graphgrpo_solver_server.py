"""GraphGRPO Solver HTTP Server — remote interface for SkyEngine via HTTP.

Mirrors `End-to-end-DRL-for-FJSP-main/drl_solver_server.py`. Same lifecycle:

  1. POST /init  → take obs JSON, build the GraphGRPO agent, run the offline
                  event-driven simulation, return makespan + action/transfer counts.
  2. POST /plan  → per time step, return the machine_actions + transfer_requests
                  due now (first call auto-triggers init if needed).
  3. POST /reset → drop the current instance.

Listens on port 8002 by default, like every other FJSP solver in this project.
"""
import os
import sys

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_graphgrpo_solver import OnlineGraphGRPOSolver

app = Flask(__name__)

# ── global solver instance ──────────────────────────────────────────────
solver: OnlineGraphGRPOSolver = None


# ── serialization / deserialization ──────────────────────────────────────

def deserialize_obs(obs_json: dict) -> dict:
    """Normalize an obs payload to the list format the solver expects.

    Accepts two formats (same as the DRL server):
      1) compact:   jobs {jid: [[(duration, machine), ...], ...]},
                    machines {mid: {"loc": [x, y]}}
      2) list:      jobs [{"job_id", "ops": [{"op_id", "machine_options", "proc_times"}]}],
                    machines [{"id", "location"}]
    """
    raw_jobs = obs_json.get("jobs", {})
    raw_machines = obs_json.get("machines", {})

    is_compact = (
        isinstance(raw_jobs, dict)
        and isinstance(raw_machines, dict)
        and len(raw_jobs) > 0
        and all(isinstance(k, (int, str)) for k in raw_jobs)
    )

    if is_compact:
        jobs = []
        for jid_str, ops_list in sorted(raw_jobs.items(), key=lambda x: int(x[0])):
            jid = int(jid_str)
            ops = []
            for oid, alternatives in enumerate(ops_list):
                machine_options = []
                proc_times = {}
                for pt, mid in alternatives:
                    machine_options.append(mid)
                    proc_times[str(mid)] = pt
                ops.append({
                    "op_id": oid,
                    "machine_options": machine_options,
                    "proc_times": proc_times,
                })
            jobs.append({"job_id": jid, "ops": ops})

        machines = []
        for mid_str, m_info in sorted(raw_machines.items(), key=lambda x: int(x[0])):
            mid = int(mid_str)
            machines.append({
                "id": mid,
                "location": m_info.get("loc", [0, 0]),
            })
        return {"jobs": jobs, "machines": machines}

    return obs_json


def _build_config(data: dict) -> dict:
    cfg = data.get("config", {})
    n_agvs = cfg.get("n_agvs", None)
    if n_agvs is not None:
        n_agvs = int(n_agvs)
    return {
        "device": cfg.get("device", "cpu"),
        "model_dir": cfg.get("model_dir", None),  # accepted for parity; unused
        "model_path": cfg.get("model_path", None),
        "n_agvs": n_agvs,
        "spread_machines": bool(cfg.get("spread_machines", True)),
        "consider_transport": bool(cfg.get("consider_transport", False)),
        "seed": int(os.getenv("SEED", cfg.get("seed", 42))),
    }


def _make_solver(config: dict) -> OnlineGraphGRPOSolver:
    return OnlineGraphGRPOSolver(
        device=config["device"],
        model_path=config["model_path"],
        n_agvs=config["n_agvs"],
        spread_machines=config["spread_machines"],
        consider_transport=config["consider_transport"],
        seed=config["seed"],
    )


def _serialize_result(result: dict, sv: OnlineGraphGRPOSolver) -> dict:
    tr = []
    for t in result["transfer_requests"]:
        tr.append({
            "task_id": t["task_id"],
            "job_id": t["job_id"],
            "op_id": t["op_id"],
            "source": list(t["source"]),
            "destination": list(t["destination"]),
            "candidate_machines": t["candidate_machines"],
            "ready_time": t["ready_time"],
        })
    return {
        "machine_actions": result["machine_actions"],
        "transfer_requests": tr,
        "time_stamp": sv.time_stamp,
        "remaining_actions": len(sv._machine_actions),
        "remaining_transfers": len(sv._transfer_requests),
    }


# ── API routes ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "solver": "GraphGRPO",
        "initialized": solver is not None and solver.initialized,
    })


@app.route("/init", methods=["POST"])
def init_solver():
    """Explicit init + offline solve."""
    global solver

    data = request.get_json(force=True)
    obs_json = data.get("obs", {})
    config = _build_config(data)

    obs = deserialize_obs(obs_json)
    solver = _make_solver(config)
    solver.plan(obs)

    makespan = max(
        (a["expected_end"] for a in solver._machine_actions),
        default=0,
    )
    return jsonify({
        "status": "initialized",
        "makespan": makespan,
        "total_actions": len(solver._machine_actions),
        "total_transfers": len(solver._transfer_requests),
    })


@app.route("/plan", methods=["POST"])
def plan():
    """Per-time-step release. Auto-initializes on first call if obs is supplied."""
    global solver

    data = request.get_json(force=True)

    if solver is None:
        obs_json = data.get("obs", {})
        config = _build_config(data)
        obs = deserialize_obs(obs_json)
        solver = _make_solver(config)
        result = solver.plan(obs)
    else:
        obs_json = data.get("obs", {})
        if obs_json:
            obs = deserialize_obs(obs_json)
        else:
            obs = {"jobs": [], "machines": []}
        result = solver.plan(obs)

    return jsonify(_serialize_result(result, solver))


@app.route("/reset", methods=["POST"])
def reset():
    global solver
    solver = None
    return jsonify({"status": "reset"})


# ── entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GraphGRPO Solver HTTP Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    print(f"[GraphGRPO Solver Server] Starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
