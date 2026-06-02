#!/usr/bin/env python3
"""基于 OR-Tools CP-SAT 的 FJSP 精确求解器.

读取 data/fjsp-master/ 下的 JSON 实例，用约束规划求最优 makespan.
"""

import collections
import json
import os
import sys

from ortools.sat.python import cp_model


# ────────────────────────── 数据加载 ──────────────────────────

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "fjsp-master",
)

INSTANCES = {
    "J10P5M6":   {"file": "J10P5M6.json",   "jobs": 10, "ops": 5,  "machines": 6},
    "J20P10M10": {"file": "J20P10M10.json",  "jobs": 20, "ops": 10, "machines": 10},
    "J20P20M15": {"file": "J20P20M15.json",  "jobs": 20, "ops": 20, "machines": 15},
}


def load_instance(name: str) -> tuple[list, int]:
    """从 JSON 加载实例，返回 (jobs, num_machines).

    jobs[job_id][task_id] = [(processing_time, machine_id), ...]
    """
    meta = INSTANCES[name]
    path = os.path.join(DATA_DIR, meta["file"])
    with open(path, "r") as f:
        data = json.load(f)

    num_machines = data["machines"]
    jobs = []
    for job in data["jobs"]:
        job_tasks = []
        for op in job:
            alternatives = [(alt["processing"], alt["machine"]) for alt in op]
            job_tasks.append(alternatives)
        jobs.append(job_tasks)

    return jobs, num_machines


# ────────────────────────── CP-SAT 求解 ──────────────────────────

def solve_fjsp(jobs: list, num_machines: int, time_limit: int = 300) -> dict | None:
    """用 CP-SAT 求解 FJSP，返回调度结果字典.

    Args:
        jobs: jobs[job_id][task_id] = [(duration, machine), ...]
        num_machines: 机器总数
        time_limit: 求解时间上限（秒）

    Returns:
        {
            "makespan": int,
            "status": str,
            "schedule": [(job_id, task_id, machine, start, duration), ...],
            "wall_time": float,
        }
        或 None（无可行解）
    """
    num_jobs = len(jobs)
    all_machines = range(num_machines)

    model = cp_model.CpModel()

    # ── 计算 horizon ──
    horizon = 0
    for job in jobs:
        for task in job:
            horizon += max(alt[0] for alt in task)

    # ── 全局变量 ──
    intervals_per_machine = collections.defaultdict(list)
    starts = {}       # (job_id, task_id)
    presences = {}    # (job_id, task_id, alt_id)
    job_ends = []

    # ── 逐 Job / Task 建模 ──
    for job_id, job in enumerate(jobs):
        previous_end = None
        for task_id, task in enumerate(job):
            num_alts = len(task)
            min_dur = min(a[0] for a in task)
            max_dur = max(a[0] for a in task)

            suffix = f"_j{job_id}_t{task_id}"
            start = model.new_int_var(0, horizon, "start" + suffix)
            duration = model.new_int_var(min_dur, max_dur, "duration" + suffix)
            end = model.new_int_var(0, horizon, "end" + suffix)
            interval = model.new_interval_var(start, duration, end, "interval" + suffix)

            starts[(job_id, task_id)] = start

            # 工序顺序约束：同一 job 内按序执行
            if previous_end is not None:
                model.add(start >= previous_end)
            previous_end = end

            # ── 多机器选择（可选区间变量）──
            if num_alts > 1:
                alt_presences = []
                for alt_id, (alt_dur, alt_mach) in enumerate(task):
                    a_suffix = f"_j{job_id}_t{task_id}_a{alt_id}"
                    p = model.new_bool_var("presence" + a_suffix)
                    s = model.new_int_var(0, horizon, "start" + a_suffix)
                    e = model.new_int_var(0, horizon, "end" + a_suffix)
                    opt_interval = model.new_optional_interval_var(
                        s, alt_dur, e, p, "interval" + a_suffix
                    )
                    alt_presences.append(p)

                    # 关联全局变量与局部变量
                    model.add(start == s).only_enforce_if(p)
                    model.add(duration == alt_dur).only_enforce_if(p)
                    model.add(end == e).only_enforce_if(p)

                    intervals_per_machine[alt_mach].append(opt_interval)
                    presences[(job_id, task_id, alt_id)] = p

                model.add_exactly_one(alt_presences)
            else:
                # 只有一台可选机器，无需 optional interval
                mach = task[0][1]
                intervals_per_machine[mach].append(interval)
                presences[(job_id, task_id, 0)] = model.new_constant(1)

        if previous_end is not None:
            job_ends.append(previous_end)

    # ── 机器不重叠约束 ──
    for mach_id in all_machines:
        intervals = intervals_per_machine[mach_id]
        if len(intervals) > 1:
            model.add_no_overlap(intervals)

    # ── 目标：最小化 makespan ──
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, job_ends)
    model.minimize(makespan)

    # ── 求解 ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8

    status = solver.Solve(model)

    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  求解状态: {status_name}，无可行解")
        return None

    # ── 提取结果 ──
    schedule = []
    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            start_val = solver.value(starts[(job_id, task_id)])
            for alt_id, (alt_dur, alt_mach) in enumerate(task):
                if solver.boolean_value(presences[(job_id, task_id, alt_id)]):
                    schedule.append((job_id, task_id, alt_mach, start_val, alt_dur))
                    break

    result = {
        "makespan": int(solver.objective_value),
        "status": status_name,
        "schedule": schedule,
        "wall_time": solver.wall_time,
        "num_branches": solver.num_branches,
        "num_booleans": solver.num_booleans,
    }
    return result


# ────────────────────────── 结果输出 ──────────────────────────

def print_result(name: str, result: dict, num_machines: int) -> None:
    """格式化打印求解结果."""
    print(f"\n{'='*60}")
    print(f"  实例: {name}")
    print(f"  Makespan: {result['makespan']}")
    print(f"  状态: {result['status']}")
    print(f"  求解时间: {result['wall_time']:.2f}s")
    print(f"  分支数: {result['num_branches']}")
    print(f"{'='*60}")

    # 按机器分组展示
    by_machine = collections.defaultdict(list)
    for job_id, task_id, mach, start, dur in result["schedule"]:
        by_machine[mach].append((start, job_id, task_id, dur))

    for mach in range(num_machines):
        tasks = sorted(by_machine[mach])
        line = f"  M{mach}: "
        parts = []
        for start, job_id, task_id, dur in tasks:
            parts.append(f"[J{job_id}-T{task_id} @{start}×{dur}]")
        line += " ".join(parts)
        print(line)

    # 按时间轴可视化（简易 Gantt）
    makespan = result["makespan"]
    print(f"\n  时间轴 (makespan={makespan}):")
    for mach in range(num_machines):
        tasks = sorted(by_machine[mach])
        bar = [" "] * makespan
        for start, job_id, task_id, dur in tasks:
            for t in range(start, min(start + dur, makespan)):
                bar[t] = str(job_id) if t == start else "-"
        print(f"  M{mach:2d} |{''.join(bar)}|")


# ────────────────────────── 主入口 ──────────────────────────

def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else list(INSTANCES.keys())

    for name in names:
        if name not in INSTANCES:
            print(f"未知实例: {name}，可选: {list(INSTANCES.keys())}")
            continue

        meta = INSTANCES[name]
        print(f"\n>>> 加载实例 {name} ({meta['jobs']} jobs × {meta['ops']} ops × {meta['machines']} machines)...")

        jobs, num_machines = load_instance(name)

        # 根据规模设置时间限制
        total_tasks = meta["jobs"] * meta["ops"]
        time_limit = 60 if total_tasks <= 50 else (120 if total_tasks <= 200 else 300)

        result = solve_fjsp(jobs, num_machines, time_limit=time_limit)

        if result is not None:
            print_result(name, result, num_machines)


if __name__ == "__main__":
    main()
