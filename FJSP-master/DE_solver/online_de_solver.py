"""
Online DE Solver — 符合 SkyEngine JobSolver 接口的在线调度器

流程:
  1. 首次 plan(obs) 被调用时:
     - 从 obs 提取问题参数
     - 调用 DESolver 离线求解，生成完整调度方案
     - 将方案拆分为 machine_actions 和 transfer_requests
  2. 后续每次 plan(obs) 被调用时:
     - time_stamp 递增
     - 根据 time_stamp 输出当前时刻应开始的 machine_actions
     - 根据 time_stamp 输出当前时刻应触发的 transfer_requests
"""

import sys
import os
import numpy as np

# 确保能导入同目录的 de_solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from de_solver import DESolver


class OnlineDESolver:
    """基于差分进化的在线 FJSP 调度器

    Args:
        init_strategy: 初始化策略 "random" | "roulette" | "extreme"
        popsize: 种群规模
        maxgen: 最大迭代次数
        F: 变异率
        Cr: 交叉率
        seed: 随机种子
    """

    def __init__(self, init_strategy="extreme", popsize=50, maxgen=500,
                 F=0.1, Cr=0.1, seed=42):
        self.init_strategy = init_strategy
        self.popsize = popsize
        self.maxgen = maxgen
        self.F = F
        self.Cr = Cr
        self.seed = seed

        # 内部状态
        self.initialized = False
        self.time_stamp = 0
        self.task_idx = 0

        # 离线求解后的完整方案
        self._machine_actions = []       # 所有 machine_action，按 start_time 排序
        self._transfer_requests = []     # 所有 transfer_request，按 ready_time 排序

    # ------------------------------------------------------------------ #
    #                       从 obs 提取问题参数                             #
    # ------------------------------------------------------------------ #

    def _obs_to_data(self, obs):
        """从 SkyEngine obs 中提取加工时间矩阵和规模参数

        Returns:
            data: shape (n_jobs * n_ops, n_machines) 的 str 矩阵，"-" 表示不可用
            n_jobs, n_ops, n_machines
        """
        jobs = obs["jobs"]
        machines = obs["machines"]
        n_machines = len(machines)
        n_jobs = len(jobs)
        n_ops = max(len(job.ops) for job in jobs)

        # 构建 (total_process, n_machines) 矩阵
        data = np.full((n_jobs * n_ops, n_machines), "-", dtype=str)
        for job in jobs:
            jid = job.job_id
            for op in job.ops:
                oid = op.op_id
                row = jid * n_ops + oid
                # 确保 proc_times key 为 int（JSON 反序列化可能将 int key 变为 string）
                proc_times = {int(k): v for k, v in op.proc_times.items()}
                for mid in range(n_machines):
                    if mid in proc_times:
                        data[row, mid] = str(int(proc_times[mid]))

        return data, n_jobs, n_ops, n_machines

    # ------------------------------------------------------------------ #
    #                        离线求解 → 生成方案                            #
    # ------------------------------------------------------------------ #

    def _solve_offline(self, obs):
        """首次调用时执行离线优化，生成完整调度方案"""
        data, n_jobs, n_ops, n_machines = self._obs_to_data(obs)

        # 构建 DESolver 并求解
        solver = DESolver(data, n_jobs, n_ops, n_machines,
                          init_strategy=self.init_strategy,
                          popsize=self.popsize,
                          maxgen=self.maxgen,
                          F=self.F, Cr=self.Cr,
                          seed=self.seed)

        best_x, best_makespan, history = solver.solve()
        schedule = solver.get_schedule(best_x)

        # 拆分为 machine_actions 和 transfer_requests
        self._machine_actions = []
        self._transfer_requests = []

        for item in schedule:
            self._machine_actions.append({
                "machine_id": item["machine"],
                "job_id": item["job_id"],
                "op_id": item["op_id"],
                "start_time": item["start_time"],
                "expected_end": item["end_time"],
            })

        # 生成搬运请求: 所有工序都需要 transfer
        for j in range(n_jobs):
            # op_id=0: depot → 第一台机器
            first = schedule[j * n_ops]
            self._transfer_requests.append({
                "job_id": j,
                "op_id": 0,
                "from_machine": -1,
                "to_machine": first["machine"],
                "ready_time": 0,
            })

            # op_id>=1: prev machine → curr machine（含同机器）
            for o in range(1, n_ops):
                prev = schedule[j * n_ops + (o - 1)]
                curr = schedule[j * n_ops + o]
                self._transfer_requests.append({
                    "job_id": j,
                    "op_id": o,
                    "from_machine": prev["machine"],
                    "to_machine": curr["machine"],
                    "ready_time": prev["end_time"],
                })

        # 按时间排序
        self._machine_actions.sort(key=lambda x: x["start_time"])
        self._transfer_requests.sort(key=lambda x: x["ready_time"])

        print(f"[OnlineDE] 离线求解完成, makespan={best_makespan:.0f}, "
              f"actions={len(self._machine_actions)}, "
              f"transfers={len(self._transfer_requests)}")

    # ------------------------------------------------------------------ #
    #                       RoutingTask 构建（兼容 SkyEngine）              #
    # ------------------------------------------------------------------ #

    def _create_routing_task(self, task_dict):
        """将 transfer dict 转为 RoutingTask 兼容格式"""
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
    #                           plan 主接口                                #
    # ------------------------------------------------------------------ #

    def plan(self, obs: dict) -> dict:
        """在线调度接口，每个时间步被调用一次

        Args:
            obs: {"jobs": List[Job], "machines": List[Machine]}

        Returns:
            {
                "machine_actions": [...],       # 本时刻应启动的加工指令
                "transfer_requests": [...]      # 本时刻应触发的搬运请求
            }
        """
        self.time_stamp += 1

        # === 第一次调用: 离线求解 ===
        if not self.initialized:
            self._solve_offline(obs)
            self.initialized = True

        # === 逐时间步输出 ===
        current_time = float(self.time_stamp)

        # 取出当前时刻应启动的 machine_actions
        ready_actions = []
        while (self._machine_actions
               and self._machine_actions[0]["start_time"] <= current_time + 1e-6):
            ready_actions.append(self._machine_actions.pop(0))

        # 取出当前时刻应触发的 transfer_requests
        ready_transfers = []
        while (self._transfer_requests
               and self._transfer_requests[0]["ready_time"] <= current_time + 1e-6):
            tr = self._transfer_requests.pop(0)
            ready_transfers.append(self._create_routing_task(tr))

        return {
            "machine_actions": ready_actions,
            "transfer_requests": ready_transfers,
        }
