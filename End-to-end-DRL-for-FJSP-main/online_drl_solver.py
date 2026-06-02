"""
Online DRL Solver — 符合 SkyEngine JobSolver 接口的在线调度器

基于 End-to-end DRL (Multi-PPO) 求解 FJSP，流程:
  1. 首次 plan(obs) 被调用时:
     - 从 obs 提取问题参数，转换为 DRL env 的矩阵格式
     - 自动匹配最合适的预训练模型
     - 执行模型推理，逐步得到调度决策
     - 将决策转换为 machine_actions 和 transfer_requests
  2. 后续每次 plan(obs) 被调用时:
     - time_stamp 递增
     - 根据 time_stamp 输出当前时刻应开始的 machine_actions
     - 根据 time_stamp 输出当前时刻应触发的 transfer_requests
"""

import sys
import os
import glob
import numpy as np
import torch
from copy import deepcopy

# 确保能导入 FJSP_MultiPPO 下的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FJSP_MultiPPO'))


class OnlineDRLSolver:
    """基于 DRL (Multi-PPO) 的在线 FJSP 调度器

    Args:
        device: 推理设备 "cuda" 或 "cpu"
        model_dir: 预训练模型目录
        seed: 随机种子
    """

    def __init__(self, device='cuda', model_dir=None, seed=42):
        self.device_str = device
        self.seed = seed
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'FJSP_MultiPPO', 'saved_network'
        )

        # 内部状态
        self.initialized = False
        self.time_stamp = 0
        self.task_idx = 0

        # 离线求解后的完整方案
        self._machine_actions = []
        self._transfer_requests = []

        # 模型相关（延迟加载）
        self._policy_job = None
        self._policy_mch = None
        self._n_j = 0
        self._n_m = 0

    # ------------------------------------------------------------------ #
    #                       从 obs 提取问题参数                             #
    # ------------------------------------------------------------------ #

    def _obs_to_matrix(self, obs):
        """从 SkyEngine obs 转换为 DRL env 期望的矩阵格式

        obs 结构:
          jobs: [{job_id, ops: [{op_id, machine_options, proc_times}]}]
          machines: [{id, location}]

        DRL env 期望: shape (n_j, n_m, n_m)
          matrix[job][op][machine] = processing_time, 0 表示不可用

        关键处理:
          - 每个 job 的 ops 数可能不同，需要 pad 到 n_m
          - pad 出的 dummy ops 在所有真实机器上加工时间为 1

        Returns:
            matrix: np.ndarray of shape (n_j, n_m, n_m)
            n_jobs, n_machines, max_ops
        """
        jobs = obs["jobs"]
        machines = obs["machines"]
        n_machines = len(machines)
        n_jobs = len(jobs)
        max_ops = max(len(job["ops"]) for job in jobs)

        # env 要求 n_m >= max_ops (每个 job 至少有 n_m 个 ops)
        n_m = max(n_machines, max_ops)

        matrix = np.zeros((n_jobs, n_m, n_m), dtype=np.float32)

        for job in jobs:
            jid = job["job_id"]
            for op in job["ops"]:
                oid = op["op_id"]
                if oid >= n_m:
                    continue
                proc_times = op.get("proc_times", {})
                for mid_str, pt in proc_times.items():
                    mid = int(mid_str)
                    if mid < n_m:
                        matrix[jid][oid][mid] = float(pt)

            # pad dummy operations
            n_real_ops = len(job["ops"])
            for oid in range(n_real_ops, n_m):
                matrix[jid][oid][:n_machines] = 1.0

        return matrix, n_jobs, n_m, max_ops

    # ------------------------------------------------------------------ #
    #                          模型匹配与加载                               #
    # ------------------------------------------------------------------ #

    def _find_model(self, n_jobs, n_machines):
        """根据实例规模匹配预训练模型"""
        # 精确匹配
        exact = f"FJSP_J{n_jobs}M{n_machines}"
        model_dir = os.path.join(self.model_dir, exact)
        if os.path.isdir(model_dir):
            return model_dir, n_jobs, n_machines

        # 查找所有可用模型
        available = []
        if os.path.isdir(self.model_dir):
            for d in os.listdir(self.model_dir):
                full = os.path.join(self.model_dir, d)
                if d.startswith("FJSP_J") and os.path.isdir(full):
                    parts = d.replace("FJSP_J", "").split("M")
                    if len(parts) == 2:
                        j, m = int(parts[0]), int(parts[1])
                        available.append((j, m, d))

        if not available:
            return None, None, None

        # 选 jobs >= n_jobs 且 machines >= n_machines 的最小模型
        candidates = [(j, m, d) for j, m, d in available
                       if j >= n_jobs and m >= n_machines]
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            j, m, d = candidates[0]
            return os.path.join(self.model_dir, d), j, m

        # 兜底：选最大的
        available.sort(key=lambda x: (x[0], x[1]), reverse=True)
        j, m, d = available[0]
        return os.path.join(self.model_dir, d), j, m

    def _find_model_files(self, model_dir):
        """在模型目录中找权重文件"""
        candidates = []
        for entry in sorted(os.listdir(model_dir)):
            full = os.path.join(model_dir, entry)
            if not os.path.isdir(full):
                continue
            job_p = os.path.join(full, 'policy_job.pth')
            mch_p = os.path.join(full, 'policy_mch.pth')
            if os.path.exists(job_p) and os.path.exists(mch_p):
                is_best = entry.startswith('best_value')
                candidates.append((is_best, entry, job_p, mch_p))

        if candidates:
            best = [c for c in candidates if c[0]]
            if best:
                best.sort(key=lambda x: int(x[1].replace('best_value', '') or '0'),
                          reverse=True)
                return best[0][2], best[0][3]
            return candidates[-1][2], candidates[-1][3]

        # 直接在目录下找
        job_p = os.path.join(model_dir, 'policy_job.pth')
        mch_p = os.path.join(model_dir, 'policy_mch.pth')
        if os.path.exists(job_p) and os.path.exists(mch_p):
            return job_p, mch_p
        return None, None

    def _load_model(self, model_dir, model_n_j, model_n_m):
        """加载预训练模型"""
        from Params import configs
        from PPOwithValue import PPO

        device = torch.device(self.device_str if torch.cuda.is_available() else 'cpu')

        # PPO 内部用 configs.n_j/n_m 创建网络，必须先修改
        configs.n_j = model_n_j
        configs.n_m = model_n_m

        ppo = PPO(
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            n_j=model_n_j,
            n_m=model_n_m,
            num_layers=configs.num_layers,
            neighbor_pooling_type=configs.neighbor_pooling_type,
            input_dim=configs.input_dim,
            hidden_dim=configs.hidden_dim,
            num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=configs.num_mlp_layers_actor,
            hidden_dim_actor=configs.hidden_dim_actor,
            num_mlp_layers_critic=configs.num_mlp_layers_critic,
            hidden_dim_critic=configs.hidden_dim_critic,
        )

        job_pth, mch_pth = self._find_model_files(model_dir)
        ppo.policy_job.load_state_dict(
            torch.load(job_pth, map_location=device), strict=False)
        ppo.policy_mch.load_state_dict(
            torch.load(mch_pth, map_location=device), strict=False)
        ppo.policy_job.eval()
        ppo.policy_mch.eval()

        self._policy_job = ppo.policy_job
        self._policy_mch = ppo.policy_mch
        self._n_j = model_n_j
        self._n_m = model_n_m
        self._device = device
        self._configs = configs

    # ------------------------------------------------------------------ #
    #                        DRL 推理 → 生成调度                           #
    # ------------------------------------------------------------------ #

    def _solve_with_drl(self, matrix, n_jobs, n_machines):
        """用 DRL 模型推理，提取完整的调度方案

        Returns:
            schedule: list of {job_id, op_id, machine, start_time, end_time}
        """
        from FJSP_Env import FJSP
        from mb_agg import aggr_obs, g_pool_cal

        n_j = self._n_j
        n_m = self._n_m
        device = self._device

        # 确保 matrix shape 匹配模型
        if matrix.shape != (n_jobs, n_m, n_m):
            # 需要扩展到模型尺寸
            print(f"[DEBUG] matrix 需要扩展: ({n_jobs},{n_m},{n_m}) → ({n_j},{n_m},{n_m})")
            full_matrix = np.zeros((n_j, n_m, n_m), dtype=np.float32)
            for j in range(n_j):
                if j < n_jobs:
                    # 真实 job：复制真实工序到真实机器，虚拟机器保持0（不可用）
                    for o in range(matrix.shape[1]):
                        for m in range(n_machines):
                            full_matrix[j][o][m] = matrix[j][o][m]
                    # 虚拟工序（pad的）：在真实机器上 time=1，让 env 能快速调度
                    for o in range(matrix.shape[1], n_m):
                        full_matrix[j][o][:n_machines] = 1.0
                else:
                    # 虚拟 job（pad的）：所有工序所有机器 time=1
                    full_matrix[j, :, :] = 1.0
            matrix = full_matrix
            actual_n_jobs = n_jobs
            actual_n_machines = n_machines
        else:
            actual_n_jobs = n_jobs
            actual_n_machines = n_machines

        # 添加 batch 维度
        batch_data = np.expand_dims(matrix, axis=0)

        env = FJSP(n_j, n_m)
        g_pool_step = g_pool_cal(
            graph_pool_type=self._configs.graph_pool_type,
            batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
            n_nodes=n_j * n_m,
            device=device,
        )

        adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(batch_data)

        # 记录每个 operation 的调度结果
        schedule = []
        op_counter = {}  # job_id -> 当前 op 索引

        with torch.no_grad():
            pool = None
            step = 0
            while True:
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), n_j * n_m)
                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
                env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
                env_dur = torch.from_numpy(np.copy(dur)).float().to(device)

                action, _, _, action_node, _, mask_mch_action, hx = self._policy_job(
                    x=env_fea,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=env_adj,
                    candidate=env_candidate,
                    mask=env_mask,
                    mask_mch=env_mask_mch,
                    dur=env_dur,
                    a_index=0,
                    old_action=0,
                    mch_pool=pool,
                    old_policy=True,
                    T=1,
                    greedy=True,
                )

                pi_mch, pool = self._policy_mch(
                    action_node, hx, mask_mch_action, env_mch_time)
                _, mch_a = pi_mch.squeeze(-1).max(1)

                # 记录调度决策
                action_val = action.item()
                mch_val = mch_a.item()
                job_id = action_val // n_m
                op_id = action_val % n_m

                # 只记录真实 operations（跳过 dummy）
                if job_id < actual_n_jobs and op_id < actual_n_machines:
                    schedule.append({
                        "action_idx": action_val,
                        "job_id": job_id,
                        "op_id": op_id,
                        "machine": mch_val,
                    })

                adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(
                    action.cpu().numpy(), mch_a
                )

                step += 1
                if env.done():
                    print(f"[DEBUG] env.done() at step={step}, scheduled {len(schedule)} ops")
                    break

        # 从 env 的内部状态提取 start_time 和 end_time
        makespan = env.mchsEndTimes.max(-1).max(-1)[0]

        # 利用 env 的 opIDsOnMchs 和 mchsStartTimes/mchsEndTimes 重建完整调度
        full_schedule = self._extract_schedule_from_env(
            env, actual_n_jobs, actual_n_machines)

        return full_schedule, makespan

    def _extract_schedule_from_env(self, env, n_jobs, n_machines):
        """从 env 内部状态提取带时间的完整调度"""
        schedule = []

        # env.mchsStartTimes[0]: shape (n_m, n_tasks) - batch=0
        # env.mchsEndTimes[0]: shape (n_m, n_tasks)
        # env.opIDsOnMchs[0]: shape (n_m, n_tasks) - 每个 machine 上加工的 op ID

        start_times = env.mchsStartTimes[0]
        end_times = env.mchsEndTimes[0]
        op_ids = env.opIDsOnMchs[0]

        n_m = env.number_of_machines
        n_j = env.number_of_jobs

        for mch in range(min(n_machines, n_m)):
            for slot in range(n_j * n_m):
                op_id = op_ids[mch][slot]
                if op_id < 0:
                    break
                # op_id 是全局 index: job_id * n_m + op_idx
                job_id = op_id // n_m
                op_idx = op_id % n_m

                if job_id >= n_jobs or op_idx >= n_machines:
                    continue

                st = start_times[mch][slot]
                et = end_times[mch][slot]
                if st < 0:
                    continue

                schedule.append({
                    "job_id": int(job_id),
                    "op_id": int(op_idx),
                    "machine": int(mch),
                    "start_time": float(st),
                    "end_time": float(et),
                })

        schedule.sort(key=lambda x: (x["start_time"], x["job_id"], x["op_id"]))
        return schedule

    # ------------------------------------------------------------------ #
    #                        离线求解 → 生成方案                            #
    # ------------------------------------------------------------------ #

    def _solve_offline(self, obs):
        """首次调用时执行离线优化，生成完整调度方案"""
        matrix, n_jobs, n_machines, max_ops = self._obs_to_matrix(obs)

        print(f"[DEBUG] 实际问题规模: n_jobs={n_jobs}, n_machines={n_machines}, max_ops={max_ops}")
        print(f"[DEBUG] matrix shape: {matrix.shape}")
        print(f"[DEBUG] matrix 非零元素数: {np.count_nonzero(matrix)}")

        # 匹配模型
        model_dir, model_n_j, model_n_m = self._find_model(n_jobs, n_machines)
        if model_dir is None:
            raise RuntimeError(f"没有可用的预训练模型匹配 {n_jobs}x{n_machines}")

        print(f"[DEBUG] 匹配模型: {os.path.basename(model_dir)} "
              f"(trained on {model_n_j}x{model_n_m})")

        # 加载模型
        self._load_model(model_dir, model_n_j, model_n_m)

        # 推理
        schedule, makespan = self._solve_with_drl(matrix, n_jobs, n_machines)

        print(f"[DEBUG] DRL推理完成: makespan={makespan}, schedule条数={len(schedule)}")
        for s in schedule:
            print(f"[DEBUG]   job={s['job_id']} op={s['op_id']} mch={s['machine']} "
                  f"start={s['start_time']:.1f} end={s['end_time']:.1f}")

        # 拆分为 machine_actions 和 transfer_requests
        self._machine_actions = []
        for item in schedule:
            self._machine_actions.append({
                "machine_id": item["machine"],
                "job_id": item["job_id"],
                "op_id": item["op_id"],
                "start_time": item["start_time"],
                "expected_end": item["end_time"],
            })

        # 生成搬运请求: 所有工序都需要 transfer
        # op_id=0: 从 depot(-1) 送到第一台机器
        # op_id>=1: 从上一台机器送到当前机器（含同机器）
        self._transfer_requests = []
        sched_by_job = {}
        for item in schedule:
            key = (item["job_id"], item["op_id"])
            sched_by_job[key] = item

        for j in range(n_jobs):
            job_ops = obs["jobs"][j]["ops"]
            n_ops = len(job_ops)

            # op_id=0: depot → 第一台机器
            first = sched_by_job.get((j, 0))
            if first:
                self._transfer_requests.append({
                    "job_id": j,
                    "op_id": 0,
                    "from_machine": -1,
                    "to_machine": first["machine"],
                    "ready_time": 0,
                })

            # op_id>=1: prev machine → curr machine
            for o in range(1, n_ops):
                prev = sched_by_job.get((j, o - 1))
                curr = sched_by_job.get((j, o))
                if prev and curr:
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

        print(f"[OnlineDRL] 推理完成, makespan={makespan:.1f}, "
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
