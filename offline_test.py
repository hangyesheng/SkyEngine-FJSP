# 用于黑盒测试 server 是否能正常工作，利用 requests 库，发送测试数据，保证单独的镜像功能完善。

"""
FJSP Docker 服务黑盒测试脚本

直接从 data/ 目录读取标准 benchmark JSON 数据，转为 solver 接口需要的 obs 格式。

用法:
  # 先启动某个 solver 服务:
  docker run -d --name fjsp-test -p 8002:8002 skyengine-fjsp-de:latest

  # 测试全部服务 + 全部数据:
  python offline_test.py de

  # 指定数据集:
  python offline_test.py de mk01
  python offline_test.py de k1

  # 使用自定义服务地址:
  SOLVER_URL=http://host:8002 python offline_test.py de
"""
import json
import os
import sys
import time
import requests

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# fjsp-master 实例（自定义数据）
FJSP_MASTER_DIR = os.path.join(DATA_DIR, "fjsp-master")
FJSP_MASTER_INSTANCES = {
    "J10P5M6":   {"file": "J10P5M6.json",   "jobs": 10, "ops": 5,  "machines": 6},
    "J20P10M10": {"file": "J20P10M10.json",  "jobs": 20, "ops": 10, "machines": 10},
    "J20P20M15": {"file": "J20P20M15.json",  "jobs": 20, "ops": 20, "machines": 15},
}

# fjsp-instances-main 标准 benchmarks
INSTANCES_JSON = os.path.join(DATA_DIR, "fjsp-instances-main", "instances.json")

SERVICES = {
    "de": {
        "url": os.environ.get("DE_URL", "http://localhost:8002"),
        "name": "DE",
    },
    "pso": {
        "url": os.environ.get("PSO_URL", "http://localhost:8002"),
        "name": "PSO",
    },
    "best": {
        "url": os.environ.get("BEST_URL", "http://localhost:8002"),
        "name": "CP-SAT",
    },
    "drl": {
        "url": os.environ.get("DRL_URL", "http://localhost:8002"),
        "name": "DRL",
    },
    "graphgrpo": {
        "url": os.environ.get("GRAPHGRPO_URL", "http://localhost:8002"),
        "name": "GraphGRPO",
    },
}

# 每种服务对应的默认 solver config
DEFAULT_CONFIGS = {
    "de": {
        "init_strategy": "extreme",
        "popsize": 30,
        "maxgen": 100,
        "seed": 42,
        "F": 0.1,
        "Cr": 0.1,
    },
    "pso": {
        "init_strategy": "extreme",
        "popsize": 30,
        "maxgen": 100,
        "seed": 42,
        "w": 0.9,
        "lr": [2, 2],
    },
    "best": {
        "time_limit": 60,
        "num_workers": 4,
        "seed": 42,
    },
    "drl": {
        "device": "cuda",
        "seed": 42,
    },
    "graphgrpo": {
        "device": "cpu",
        "seed": 42,
        "n_agvs": None,             # None → n_machines (abundant, non-blocking)
        "spread_machines": True,     # auto-grid machine positions when obs locs are [0,0]
        "consider_transport": False, # False → pure-processing makespan (matches siblings)
    },
}


# ---------------------------------------------------------------------------
# 数据加载: benchmark JSON → solver obs JSON
# ---------------------------------------------------------------------------

def benchmark_to_obs(benchmark_data: dict) -> dict:
    """将标准 benchmark JSON 转为 solver 接口需要的 obs 格式.

    benchmark JSON: {machines: int, jobs: [[[{machine, processing}]]]}
    obs JSON:       {jobs: [{job_id, ops}], machines: [{id, location}]}
    """
    n_machines = benchmark_data["machines"]

    jobs = []
    for job_id, job in enumerate(benchmark_data["jobs"]):
        ops = []
        for op_id, task in enumerate(job):
            machine_options = [alt["machine"] for alt in task]
            # 每台可选机器的加工时间 {machine_id: processing_time}
            proc_times = {alt["machine"]: float(alt["processing"]) for alt in task}
            ops.append({
                "op_id": op_id,
                "machine_options": machine_options,
                "proc_times": proc_times,
            })
        jobs.append({"job_id": job_id, "ops": ops})

    machines = [{"id": m, "location": [0, 0]} for m in range(n_machines)]
    return {"jobs": jobs, "machines": machines}


def load_all_datasets() -> dict:
    """加载所有可用数据集，返回 {name: obs_json}"""
    datasets = {}

    # 1) fjsp-master 自定义实例
    for name, meta in FJSP_MASTER_INSTANCES.items():
        path = os.path.join(FJSP_MASTER_DIR, meta["file"])
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            datasets[name] = benchmark_to_obs(data)

    # 2) fjsp-instances-main 标准 benchmarks
    if os.path.exists(INSTANCES_JSON):
        with open(INSTANCES_JSON) as f:
            instances = json.load(f)
        base_dir = os.path.dirname(INSTANCES_JSON)
        for inst in instances:
            name = inst["name"]
            path = os.path.join(base_dir, inst["path"].replace(".txt", ".json"))
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                datasets[name] = benchmark_to_obs(data)

    return datasets

# ---------------------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------------------


def test_health(base_url, solver_name):
    """1. 健康检查"""
    r = requests.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200, f"health 状态码异常: {r.status_code}"
    data = r.json()
    assert data["status"] == "ok", f"health 状态异常: {data}"
    assert (
        data["solver"] == solver_name
    ), f"solver 名称不匹配: {data['solver']} != {solver_name}"
    return data


def test_init(base_url, obs_json, config):
    """2. 初始化并离线求解"""
    payload = {"obs": obs_json, "config": config}
    r = requests.post(f"{base_url}/init", json=payload, timeout=300)
    assert r.status_code == 200, f"init 状态码异常: {r.status_code}"
    data = r.json()
    assert data["status"] == "initialized", f"init 状态异常: {data}"
    assert data["makespan"] > 0, f"makespan 应大于 0: {data['makespan']}"
    assert data["total_actions"] > 0, f"total_actions 应大于 0: {data['total_actions']}"
    return data


def test_plan_all(base_url, solver_instance_name):
    """3. 逐步调用 /plan 直到调度输出完毕，返回汇总信息"""
    total_actions = 0
    total_transfers = 0
    steps = 0
    max_steps = 500

    for step in range(1, max_steps + 1):
        r = requests.post(f"{base_url}/plan", json={}, timeout=60)
        assert r.status_code == 200, f"plan step={step} 状态码异常: {r.status_code}"
        result = r.json()

        ma = result["machine_actions"]
        tr = result["transfer_requests"]
        total_actions += len(ma)
        total_transfers += len(tr)
        steps = step

        if result["remaining_actions"] == 0 and result["remaining_transfers"] == 0:
            break
    else:
        print(f"    警告: {max_steps} 步后调度仍未完成")

    return {
        "steps": steps,
        "total_actions": total_actions,
        "total_transfers": total_transfers,
    }


def test_reset(base_url):
    """4. 重置"""
    r = requests.post(f"{base_url}/reset", timeout=5)
    assert r.status_code == 200, f"reset 状态码异常: {r.status_code}"
    data = r.json()
    assert data["status"] == "reset", f"reset 状态异常: {data}"
    return data


def test_health_after_reset(base_url, solver_name):
    """5. 重置后再检查 health，确认 initialized=False"""
    data = test_health(base_url, solver_name)
    assert data["initialized"] == False, f"重置后 initialized 应为 False: {data}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_test(service_key, dataset_name=None):
    """对单个服务运行完整测试"""
    svc = SERVICES[service_key]
    base_url = svc["url"]
    solver_name = svc["name"]
    config = DEFAULT_CONFIGS[service_key]

    print(f"\n{'='*60}")
    print(f"  测试服务: {solver_name} @ {base_url}")
    print(f"{'='*60}")

    # 加载所有数据集
    all_datasets = load_all_datasets()

    # 确定测试数据集
    if dataset_name:
        if dataset_name not in all_datasets:
            print(f"\n  未知数据集: {dataset_name}")
            print(f"  可选: {', '.join(sorted(all_datasets.keys())[:20])} ...")
            return []
        datasets = {dataset_name: all_datasets[dataset_name]}
    else:
        datasets = all_datasets

    results = []

    for ds_name, obs_json in sorted(datasets.items()):

        n_jobs = len(obs_json["jobs"])
        n_machines = len(obs_json["machines"])
        total_ops = sum(len(j["ops"]) for j in obs_json["jobs"])

        print(
            f"\n  --- 数据集: {ds_name} ({n_jobs} jobs, {total_ops} ops, {n_machines} machines) ---"
        )

        t_start = time.time()

        # 1) Health
        h = test_health(base_url, solver_name)
        print(f"  [health] {h}")

        # 2) Init (离线求解)
        init_info = test_init(base_url, obs_json, config)
        print(
            f"  [init]   makespan={init_info['makespan']:.1f}  "
            f"actions={init_info['total_actions']}  transfers={init_info['total_transfers']}"
        )

        # 3) Plan (逐步取回)
        plan_info = test_plan_all(base_url, solver_name)
        print(
            f"  [plan]   {plan_info['steps']} steps, "
            f"{plan_info['total_actions']} actions, {plan_info['total_transfers']} transfers"
        )

        # 验证 plan 输出与 init 预告一致
        assert (
            plan_info["total_actions"] == init_info["total_actions"]
        ), f"actions 数量不一致: plan={plan_info['total_actions']} vs init={init_info['total_actions']}"
        assert (
            plan_info["total_transfers"] == init_info["total_transfers"]
        ), f"transfers 数量不一致: plan={plan_info['total_transfers']} vs init={init_info['total_transfers']}"

        # 4) Reset
        test_reset(base_url)
        print(f"  [reset]  ok")

        # 5) Health after reset
        test_health_after_reset(base_url, solver_name)
        print(f"  [health] initialized=False ✓")

        elapsed = time.time() - t_start
        results.append(
            {
                "dataset": ds_name,
                "makespan": init_info["makespan"],
                "actions": plan_info["total_actions"],
                "transfers": plan_info["total_transfers"],
                "steps": plan_info["steps"],
                "time_sec": round(elapsed, 2),
            }
        )
        print(f"  [done]   {elapsed:.2f}s ✓")

    return results


def main():
    # 解析参数
    target_service = sys.argv[1] if len(sys.argv) > 1 else None
    target_dataset = sys.argv[2] if len(sys.argv) > 2 else None

    if target_service and target_service not in SERVICES:
        print(f"未知服务: {target_service}")
        print(f"可选: {', '.join(SERVICES.keys())}")
        sys.exit(1)

    services_to_test = [target_service] if target_service else list(SERVICES.keys())

    all_results = {}
    has_failure = False

    for svc_key in services_to_test:
        try:
            results = run_test(svc_key, target_dataset)
            all_results[svc_key] = results
        except AssertionError as e:
            print(f"\n  ✗ 断言失败: {e}")
            has_failure = True
        except requests.ConnectionError:
            print(f"\n  ✗ 无法连接 {SERVICES[svc_key]['url']}，请确认服务已启动")
            has_failure = True
        except Exception as e:
            print(f"\n  ✗ 异常: {type(e).__name__}: {e}")
            has_failure = True

    # 汇总报告
    print(f"\n{'='*60}")
    print("  汇总报告")
    print(f"{'='*60}")
    for svc_key, results in all_results.items():
        print(f"\n  {SERVICES[svc_key]['name']}:")
        for r in results:
            print(
                f"    {r['dataset']:15s}  makespan={r['makespan']:7.1f}  "
                f"actions={r['actions']:4d}  transfers={r['transfers']:4d}  "
                f"steps={r['steps']:4d}  time={r['time_sec']}s"
            )

    if has_failure:
        print("\n✗ 存在测试失败")
        sys.exit(1)
    else:
        print("\n✓ 全部通过")


if __name__ == "__main__":
    main()
