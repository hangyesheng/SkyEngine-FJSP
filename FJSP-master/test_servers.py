"""
测试 DE/PSO HTTP Server — 用 mock obs JSON 验证完整生命周期
"""
import json
import requests
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DE_solver"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "PSO_solver"))


def load_obs_json(txt_path, n_jobs, n_ops, n_machines):
    """从 txt 数据文件构造 obs JSON"""
    contents = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                contents.append(line.split())

    jobs = []
    for j in range(n_jobs):
        ops = []
        for o in range(n_ops):
            row = contents[j * n_ops + o]
            mch_opts = []
            pt = 1.0
            for m in range(n_machines):
                if row[m] != "-":
                    mch_opts.append(m)
                    pt = float(row[m])
            ops.append({
                "op_id": o,
                "machine_options": mch_opts,
                "proc_time": pt,
            })
        jobs.append({"job_id": j, "ops": ops})

    machines = [{"id": m, "location": [0, 0]} for m in range(n_machines)]
    return {"jobs": jobs, "machines": machines}


def test_server(base_url, solver_name, obs_json, config=None):
    if config is None:
        config = {"init_strategy": "extreme", "popsize": 30, "maxgen": 100, "seed": 42}

    print(f"\n{'='*55}")
    print(f"  Testing: {solver_name} @ {base_url}")
    print(f"{'='*55}")

    # 1) Health check
    r = requests.get(f"{base_url}/health")
    print(f"  [health] {r.json()}")

    # 2) Init
    payload = {"obs": obs_json, "config": config}
    r = requests.post(f"{base_url}/init", json=payload)
    info = r.json()
    print(f"  [init]   {info}")
    assert info["status"] == "initialized"

    # 3) 逐时间步调用 plan
    total_actions = 0
    total_transfers = 0
    for step in range(1, 300):
        r = requests.post(f"{base_url}/plan", json={})
        result = r.json()
        ma = result["machine_actions"]
        tr = result["transfer_requests"]

        if ma or tr:
            for a in ma:
                print(f"  t={step:3d} | M{a['machine_id']} 加工 J{a['job_id']}-O{a['op_id']} "
                      f"[{a['start_time']:.0f} ~ {a['expected_end']:.0f}]")
            for t in tr:
                print(f"  t={step:3d} | 搬运 J{t['job_id']}-O{t['op_id']} "
                      f"M{t['source'][0]} -> M{t['destination'][0]} (ready={t['ready_time']:.0f})")

        total_actions += len(ma)
        total_transfers += len(tr)

        if result["remaining_actions"] == 0 and result["remaining_transfers"] == 0:
            print(f"  t={step}: 所有调度已输出完毕")
            break

    print(f"\n  统计: {total_actions} actions, {total_transfers} transfers, {step} 时间步")

    # 4) Reset
    r = requests.post(f"{base_url}/reset")
    print(f"  [reset]  {r.json()}")

    # 5) Health after reset
    r = requests.get(f"{base_url}/health")
    print(f"  [health] {r.json()}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    txt = os.path.join(base, "data", "data_first.txt")
    obs_json = load_obs_json(txt, 10, 5, 6)

    de_url = "http://localhost:5001"
    pso_url = "http://localhost:5002"

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "de":
            test_server(de_url, "DE Server", obs_json)
        elif target == "pso":
            test_server(pso_url, "PSO Server", obs_json)
    else:
        print("Usage: python test_servers.py [de|pso]")
        print("  请先启动对应 server:")
        print("    python DE_solver/de_solver_server.py --port 5001")
        print("    python PSO_solver/pso_solver_server.py --port 5002")
