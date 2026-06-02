#!/usr/bin/env python3
"""best_solver.py 的测试套件.

读取 instances.json 中的基准实例，调用 solve_fjsp 求解，
与已知最优解 (optimum) 或上下界 (bounds) 对比，验证求解器正确性。

分类:
  - 快速测试 (smoke): 极小实例，秒级完成，验证基本正确性
  - 最优解验证: 有已知 optimum 的中小实例，断言求解结果 == optimum
  - 边界验证: 无已知 optimum 但有 bounds 的实例，断言 lb <= makespan <= ub
  - 可行性验证: 对每个结果做约束检查 (工序顺序、机器不重叠)

用法:
  pytest test_best_solver.py -v                 # 全部运行
  pytest test_best_solver.py -v -m smoke        # 仅快速测试
  pytest test_best_solver.py -v -k "mk01"       # 仅 mk01
"""

import collections
import json
import os

import pytest

# ────────────────────────── 路径配置 ──────────────────────────

SOLVER_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_JSON = os.path.join(
    SOLVER_DIR, "..", "data", "fjsp-instances-main", "instances.json"
)
INSTANCES_DIR = os.path.join(SOLVER_DIR, "..", "data", "fjsp-instances-main")

from best_solver import solve_fjsp


# ────────────────────────── 工具函数 ──────────────────────────

def _load_instances_meta() -> list[dict]:
    """加载 instances.json 中的所有实例元数据."""
    with open(INSTANCES_JSON, "r") as f:
        return json.load(f)


def _load_instance_json(path_relative: str) -> tuple[list, int]:
    """从 JSON 文件加载实例数据，返回 (jobs, num_machines).

    与 best_solver.load_instance 返回相同格式:
      jobs[job_id][task_id] = [(processing_time, machine_id), ...]
    """
    json_path = path_relative.replace(".txt", ".json")
    full_path = os.path.join(INSTANCES_DIR, json_path)
    with open(full_path, "r") as f:
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


def _validate_schedule(jobs: list, num_machines: int, result: dict) -> list[str]:
    """验证调度结果的可行性，返回错误信息列表（空 = 合法）."""
    errors = []
    schedule = result["schedule"]
    makespan = result["makespan"]

    # 1. 检查任务数量一致
    expected_tasks = sum(len(job) for job in jobs)
    if len(schedule) != expected_tasks:
        errors.append(
            f"任务数量不一致: 期望 {expected_tasks}, 实际 {len(schedule)}"
        )

    # 2. 检查每个 (job_id, task_id) 恰好出现一次
    seen = set()
    for job_id, task_id, mach, start, dur in schedule:
        key = (job_id, task_id)
        if key in seen:
            errors.append(f"重复任务: J{job_id}-T{task_id}")
        seen.add(key)

    # 3. 检查工序顺序约束 & 机器选择合法性
    job_task_schedule = collections.defaultdict(dict)
    for job_id, task_id, mach, start, dur in schedule:
        job_task_schedule[job_id][task_id] = (mach, start, dur)

    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            if (job_id, task_id) not in seen:
                errors.append(f"缺失任务: J{job_id}-T{task_id}")
                continue

            mach, start, dur = job_task_schedule[job_id][task_id]

            # 检查 (machine, duration) 是否在候选列表中
            valid_alts = [(alt_dur, alt_mach) for alt_dur, alt_mach in task]
            if (dur, mach) not in valid_alts:
                errors.append(
                    f"J{job_id}-T{task_id}: 非法机器选择 "
                    f"(machine={mach}, dur={dur}), "
                    f"候选: {valid_alts}"
                )

            # 检查同一 job 内工序顺序
            if task_id > 0:
                if (job_id, task_id - 1) in job_task_schedule[job_id]:
                    _, prev_start, prev_dur = job_task_schedule[job_id][task_id - 1]
                    if start < prev_start + prev_dur:
                        errors.append(
                            f"J{job_id}: 工序顺序违反 "
                            f"T{task_id - 1} 结束于 {prev_start + prev_dur}, "
                            f"T{task_id} 开始于 {start}"
                        )

            # 检查不越界
            if start + dur > makespan:
                errors.append(
                    f"J{job_id}-T{task_id}: 越界 "
                    f"(start={start}, dur={dur}, makespan={makespan})"
                )
            if start < 0:
                errors.append(f"J{job_id}-T{task_id}: start < 0")

    # 4. 检查机器不重叠约束
    by_machine = collections.defaultdict(list)
    for job_id, task_id, mach, start, dur in schedule:
        by_machine[mach].append((start, start + dur, job_id, task_id))

    for mach_id, intervals in by_machine.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            s1, e1, j1, t1 = intervals[i]
            s2, e2, j2, t2 = intervals[i + 1]
            if s2 < e1:
                errors.append(
                    f"机器 M{mach_id} 重叠: "
                    f"J{j1}-T{t1} [{s1},{e1}) 与 J{j2}-T{t2} [{s2},{e2})"
                )

    # 5. 检查机器编号范围
    for job_id, task_id, mach, start, dur in schedule:
        if mach < 0 or mach >= num_machines:
            errors.append(
                f"J{job_id}-T{task_id}: 机器编号 {mach} 超出范围 [0, {num_machines})"
            )

    return errors


# ────────────────────────── 测试数据 ──────────────────────────

ALL_INSTANCES = _load_instances_meta()
INST_BY_NAME = {inst["name"]: inst for inst in ALL_INSTANCES}

# ── 快速测试实例 (极小规模，求解 < 10s) ──
SMOKE_TESTS = [
    # kacem 小实例
    ("k1", 11),
    ("k2", 11),
    ("k3", 7),
    ("k4", 12),
    # fattahi 小实例
    ("sfjs01", 66),
    ("sfjs02", 107),
    ("sfjs07", 397),
    ("sfjs09", 210),
]

# ── brandimarte 有最优解的实例 ──
BRANDIMARTE_OPT = [
    ("mk01", 40),
    ("mk03", 204),
    ("mk04", 60),
    ("mk08", 523),
    ("mk09", 307),
    ("mk12", 508),
    ("mk14", 694),
]

# ── brandimarte 仅有 bounds 的实例 ──
BRANDIMARTE_BOUNDS = [
    ("mk02", 24, 26),
    ("mk05", 168, 172),
]

# ── hurink/vdata 有最优解的小实例 (<= 10 jobs × 10 machines) ──
HURINK_V_OPT = [
    ("v-mt06", 47),
    ("v-la01", 570),
    ("v-la02", 529),
    ("v-la03", 477),
    ("v-la04", 502),
    ("v-la05", 457),
    ("v-la16", 717),
    ("v-la17", 646),
    ("v-abz6", 742),
    ("v-orb1", 695),
    ("v-orb2", 620),
    ("v-orb3", 648),
    ("v-orb4", 753),
    ("v-orb5", 584),
    ("v-orb7", 275),
    ("v-orb8", 573),
    ("v-orb9", 659),
    ("v-orb10", 681),
]

# ── barnes 有最优解的实例 ──
BARNES_OPT = [
    ("mt10x", 918),
    ("mt10xy", 905),
    ("mt10xyz", 847),
]


# ────────────────────────── 辅助 ──────────────────────────

def _solve_instance(name: str, time_limit: int = 120) -> dict:
    meta = INST_BY_NAME[name]
    jobs, num_machines = _load_instance_json(meta["path"])
    return solve_fjsp(jobs, num_machines, time_limit=time_limit)


# ────────────────────────── 测试用例 ──────────────────────────


class TestSmoke:
    """极小实例快速验证."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("name,expected", SMOKE_TESTS,
                             ids=[f"{n}={e}" for n, e in SMOKE_TESTS])
    def test_smoke_optimum(self, name, expected):
        result = _solve_instance(name, time_limit=30)
        assert result is not None, f"{name}: 求解失败（无可行解）"
        assert result["makespan"] == expected, (
            f"{name}: makespan={result['makespan']}, 期望 optimum={expected}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name,expected", SMOKE_TESTS,
                             ids=[f"{n}=feas" for n, e in SMOKE_TESTS])
    def test_smoke_feasibility(self, name, expected):
        meta = INST_BY_NAME[name]
        jobs, num_machines = _load_instance_json(meta["path"])
        result = _solve_instance(name, time_limit=30)
        assert result is not None

        errors = _validate_schedule(jobs, num_machines, result)
        assert not errors, f"{name} 调度不可行:\n" + "\n".join(errors)


class TestBrandimarte:
    """Brandimarte 基准 (mk01-mk15)."""

    @pytest.mark.parametrize("name,expected", BRANDIMARTE_OPT,
                             ids=[f"{n}={e}" for n, e in BRANDIMARTE_OPT])
    def test_optimum(self, name, expected):
        result = _solve_instance(name, time_limit=120)
        assert result is not None, f"{name}: 求解失败"
        assert result["makespan"] == expected, (
            f"{name}: makespan={result['makespan']}, 期望 optimum={expected}"
        )

    @pytest.mark.parametrize("name,lb,ub", BRANDIMARTE_BOUNDS,
                             ids=[f"{n}[{lb},{ub}]" for n, lb, ub in BRANDIMARTE_BOUNDS])
    def test_bounds(self, name, lb, ub):
        result = _solve_instance(name, time_limit=120)
        assert result is not None, f"{name}: 求解失败"
        assert lb <= result["makespan"] <= ub, (
            f"{name}: makespan={result['makespan']} 超出已知边界 [{lb}, {ub}]"
        )

    @pytest.mark.parametrize("name,expected", BRANDIMARTE_OPT,
                             ids=[f"{n}=feas" for n, e in BRANDIMARTE_OPT])
    def test_feasibility(self, name, expected):
        meta = INST_BY_NAME[name]
        jobs, num_machines = _load_instance_json(meta["path"])
        result = _solve_instance(name, time_limit=120)
        assert result is not None

        errors = _validate_schedule(jobs, num_machines, result)
        assert not errors, f"{name} 调度不可行:\n" + "\n".join(errors)


class TestHurinkVdata:
    """Hurink vdata 小规模基准."""

    @pytest.mark.parametrize("name,expected", HURINK_V_OPT,
                             ids=[f"{n}={e}" for n, e in HURINK_V_OPT])
    def test_optimum(self, name, expected):
        result = _solve_instance(name, time_limit=120)
        assert result is not None, f"{name}: 求解失败"
        assert result["makespan"] == expected, (
            f"{name}: makespan={result['makespan']}, 期望 optimum={expected}"
        )

    @pytest.mark.parametrize("name,expected", HURINK_V_OPT,
                             ids=[f"{n}=feas" for n, e in HURINK_V_OPT])
    def test_feasibility(self, name, expected):
        meta = INST_BY_NAME[name]
        jobs, num_machines = _load_instance_json(meta["path"])
        result = _solve_instance(name, time_limit=120)
        assert result is not None

        errors = _validate_schedule(jobs, num_machines, result)
        assert not errors, f"{name} 调度不可行:\n" + "\n".join(errors)


class TestBarnes:
    """Barnes 基准."""

    @pytest.mark.parametrize("name,expected", BARNES_OPT,
                             ids=[f"{n}={e}" for n, e in BARNES_OPT])
    def test_optimum(self, name, expected):
        result = _solve_instance(name, time_limit=120)
        assert result is not None, f"{name}: 求解失败"
        assert result["makespan"] == expected, (
            f"{name}: makespan={result['makespan']}, 期望 optimum={expected}"
        )

    @pytest.mark.parametrize("name,expected", BARNES_OPT,
                             ids=[f"{n}=feas" for n, e in BARNES_OPT])
    def test_feasibility(self, name, expected):
        meta = INST_BY_NAME[name]
        jobs, num_machines = _load_instance_json(meta["path"])
        result = _solve_instance(name, time_limit=120)
        assert result is not None

        errors = _validate_schedule(jobs, num_machines, result)
        assert not errors, f"{name} 调度不可行:\n" + "\n".join(errors)


class TestLargeFeasibility:
    """对更大规模实例仅做可行性验证（不要求命中 optimum）."""

    @pytest.mark.parametrize("name", [
        "mk05", "mk06", "mk07", "mk10", "mk11", "mk13", "mk15",
    ])
    def test_brandimarte_large(self, name):
        """大 brandimarte 实例: 验证可行性 + bounds."""
        meta = INST_BY_NAME[name]
        jobs, num_machines = _load_instance_json(meta["path"])

        time_limit = 300 if meta["jobs"] * 10 > 200 else 180
        result = solve_fjsp(jobs, num_machines, time_limit=time_limit)
        assert result is not None, f"{name}: 求解失败"

        # 可行性检查
        errors = _validate_schedule(jobs, num_machines, result)
        assert not errors, f"{name} 调度不可行:\n" + "\n".join(errors)

        # bounds 检查
        if meta.get("bounds"):
            lb = meta["bounds"]["lower"]
            ub = meta["bounds"]["upper"]
            assert lb <= result["makespan"] <= ub, (
                f"{name}: makespan={result['makespan']} "
                f"超出已知边界 [{lb}, {ub}]"
            )
