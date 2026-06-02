"""
DRL Solver HTTP Server — 远程调用接口，供 SkyEngine 通过 HTTP 调度

基于 End-to-end DRL (Multi-PPO) 求解器。

生命周期:
  1. POST /init   → 传入 obs JSON，初始化 solver 并完成离线推理
  2. POST /plan   → 逐时间步调用，返回当前时刻的 actions / transfers
  3. POST /reset  → 重置 solver，准备下一个实例

也可直接从 POST /plan 开始（首次调用自动触发 init）。
"""

import sys
import os

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_drl_solver import OnlineDRLSolver

app = Flask(__name__)

# ── 全局 solver 实例 ──────────────────────────────────────────────
solver: OnlineDRLSolver = None


# ── 序列化 / 反序列化 ────────────────────────────────────────────

def deserialize_obs(obs_json: dict) -> dict:
    """将 JSON 格式的 obs 还原为含 Python 对象的 obs dict

    接受两种格式:
    1) 紧凑格式 (新):
       jobs: {job_id: [[(duration, machine), ...], ...]}
       machines: {machine_id: {"loc": [x,y], "current_op": ...}}
    2) 列表格式 (旧):
       jobs: [{"job_id", "ops": [{"op_id", "machine_options", "proc_times"}]}]
       machines: [{"id", "location"}]

    DRL solver 内部直接操作 dict（不做对象化），所以统一转为列表 dict 格式。
    """
    raw_jobs = obs_json.get("jobs", {})
    raw_machines = obs_json.get("machines", {})

    # 自动检测格式：dict 且 key 为 int/str 数字 → 紧凑格式
    is_compact = (
        isinstance(raw_jobs, dict)
        and isinstance(raw_machines, dict)
        and len(raw_jobs) > 0
        and all(isinstance(k, (int, str)) for k in raw_jobs)
    )

    if is_compact:
        # 紧凑格式 → 转为 DRL solver 期望的列表 dict 格式
        jobs = []
        for jid_str, ops_list in sorted(raw_jobs.items(), key=lambda x: int(x[0])):
            jid = int(jid_str)
            ops = []
            for oid, alternatives in enumerate(ops_list):
                machine_options = []
                proc_times = {}
                for pt, mid in alternatives:
                    machine_options.append(mid)
                    proc_times[str(mid)] = pt  # DRL _obs_to_matrix 用 str key
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

    # 旧格式（列表）直接透传
    return obs_json


def _build_config(data: dict) -> dict:
    """从请求中提取 solver 配置"""
    cfg = data.get("config", {})
    return {
        "device": cfg.get("device", "cuda"),
        "model_dir": cfg.get("model_dir", None),
        "seed": int(os.getenv("SEED", cfg.get("seed", 42))),
    }


def _make_solver(config: dict) -> OnlineDRLSolver:
    return OnlineDRLSolver(**config)


def _serialize_result(result: dict, sv: OnlineDRLSolver) -> dict:
    """将 plan 返回值序列化为 JSON-safe 结构"""
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


# ── API 路由 ──────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "solver": "DRL",
        "initialized": solver is not None and solver.initialized,
    })


@app.route("/init", methods=["POST"])
def init_solver():
    """显式初始化 solver 并完成离线推理"""
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
    """每时间步调用一次，返回当前时刻的 machine_actions + transfer_requests

    首次调用时自动初始化（需在请求体中附带 obs 和 config）。
    """
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
    """重置 solver，释放当前实例"""
    global solver
    solver = None
    return jsonify({"status": "reset"})


# ── 启动入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DRL Solver (Multi-PPO) HTTP Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    print(f"[DRL Solver Server] Starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
