"""
PSO Solver — 将 FJSP-master/PSO 中的三个脚本统一为一个可复用的类

支持三种初始化策略:
  - "random"  : 完全随机 (对应 PSO_first.py)
  - "roulette" : 轮盘赌 (对应 PSO_second.py)
  - "extreme"  : 极限完工时间最小化 (对应 PSO_third.py)

用法:
    solver = PSOSolver(data, n_jobs=10, n_ops=5, n_machines=6,
                       init_strategy="roulette", maxgen=500)
    best_x, best_makespan, history = solver.solve()
    schedule = solver.get_schedule(best_x)
"""

import numpy as np
import json
import os


class PSOSolver:
    """粒子群优化求解器 for FJSP"""

    def __init__(self, data, n_jobs, n_ops, n_machines,
                 init_strategy="random", popsize=50, maxgen=500,
                 w=0.9, lr=(2, 2), seed=None):
        """
        Args:
            data: 加工时间矩阵，shape (n_jobs * n_ops, n_machines)
                  元素为 str，数字表示加工时间，"-" 表示不可用
                  或 np.ndarray，-1 表示不可用
            n_jobs: 工件数
            n_ops: 每个工件的工序数
            n_machines: 机器数
            init_strategy: "random" | "roulette" | "extreme"
            popsize: 种群规模
            maxgen: 最大迭代次数
            w: 惯性权重
            lr: 加速因子 (c1, c2)
            seed: 随机种子
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.n_jobs = n_jobs
        self.n_ops = n_ops
        self.n_machines = n_machines
        self.total_process = n_jobs * n_ops
        self.popsize = popsize
        self.maxgen = maxgen
        self.w = w
        self.lr = lr
        self.init_strategy = init_strategy

        # 解析 data → proc_time: shape (total_process, n_machines)，-1 表示不可用
        self.proc_time = self._parse_data(data)

        # 预处理: 每道工序的可用机器信息（用于轮盘赌和兜底）
        self._build_clean_contents()

    # ------------------------------------------------------------------ #
    #                          数据预处理                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_data(data):
        """将 data 统一为 int ndarray，不可用机器标记为 -1"""
        if isinstance(data, np.ndarray):
            raw = data.astype(str)
        else:
            raw = np.array(data, dtype=str)

        result = np.zeros(raw.shape, dtype=int)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                val = raw[i, j].strip()
                if val == "-":
                    result[i, j] = -1
                else:
                    result[i, j] = int(val)
        return result

    def _build_clean_contents(self):
        """预处理: 为每道工序构建可用机器列表和轮盘赌累积概率

        clean_contents[i] = [[cum_prob, machine_id], ...]
        按累积概率升序排列，最后一个元素即加工时间最短的机器（兜底用）
        """
        self.clean_contents = []
        for i in range(self.total_process):
            avail = []
            for j in range(self.n_machines):
                t = self.proc_time[i, j]
                if t > 0:
                    avail.append([t, j + 1])  # [加工时间, 机器号(1-indexed)]

            # 以 1/加工时间 为权重计算累积概率
            inv_sum = sum(1.0 / a[0] for a in avail)
            for k in range(len(avail)):
                avail[k][0] = (1.0 / avail[k][0]) / inv_sum
            avail.sort()  # 按概率升序

            # 累积概率
            cum = 0.0
            for k in range(len(avail)):
                cum += avail[k][0]
                avail[k][0] = cum

            self.clean_contents.append(avail)

    # ------------------------------------------------------------------ #
    #                         编码 / 解码                                  #
    # ------------------------------------------------------------------ #

    def _decode_ops(self, x):
        """工序编码 → 有序工序列表 [(job_id, op_seq), ...]

        x[:total_process] 是 1-indexed 工件号序列，
        第 k 次出现表示该工件的第 k 道工序。
        """
        piece_mark = np.zeros(self.n_jobs, dtype=int)
        result = []
        for i in range(self.total_process):
            job = int(x[i]) - 1
            piece_mark[job] += 1
            result.append((int(x[i]), int(piece_mark[job])))
        return result

    # ------------------------------------------------------------------ #
    #                          适应度计算                                  #
    # ------------------------------------------------------------------ #

    def calculate(self, x):
        """计算个体 x 的适应度（最大完工时间 makespan）

        Args:
            x: shape (total_process * 2,), 前半段工序编码，后半段机器编码

        Returns:
            float: makespan
        """
        Tm = np.zeros(self.n_machines)                    # 每台机器的完工时间
        Te = np.zeros((self.n_jobs, self.n_ops))          # 每个工序的完成时间
        ops = self._decode_ops(x)

        for i in range(self.total_process):
            job_id, op_seq = ops[i]
            mch_idx = int(x[self.total_process + (job_id - 1) * self.n_ops + (op_seq - 1)]) - 1
            proc_row = (job_id - 1) * self.n_ops + (op_seq - 1)
            proc_time = self.proc_time[proc_row, mch_idx]

            if op_seq == 1:
                Tm[mch_idx] += proc_time
            else:
                Tm[mch_idx] = max(Te[job_id - 1, op_seq - 2], Tm[mch_idx]) + proc_time
            Te[job_id - 1, op_seq - 1] = Tm[mch_idx]

        return float(Tm.max())

    # ------------------------------------------------------------------ #
    #                          初始化策略                                  #
    # ------------------------------------------------------------------ #

    def _init_ops(self):
        """生成一个随机的工序编码（所有工件号打乱）"""
        ops = np.zeros(self.total_process)
        for j in range(self.n_jobs):
            for p in range(self.n_ops):
                ops[j * self.n_ops + p] = j + 1
        np.random.shuffle(ops[:self.total_process])
        return ops

    def _init_machine_random(self, pop_i):
        """随机初始化机器编码"""
        for j in range(self.total_process):
            idx = np.random.randint(0, self.n_machines)
            while self.proc_time[j, idx] < 0:
                idx = np.random.randint(0, self.n_machines)
            pop_i[j + self.total_process] = idx + 1

    def _init_machine_roulette(self, pop_i):
        """轮盘赌初始化机器编码（偏好短加工时间的机器）"""
        for j in range(self.total_process):
            rate = np.random.rand()
            for cum_prob, mch_id in self.clean_contents[j]:
                if rate <= cum_prob:
                    pop_i[j + self.total_process] = mch_id
                    break

    def _init_machine_extreme_global(self, x, time_and_mchindex):
        """全局极限完工时间最小化初始化"""
        machload = np.zeros(self.n_machines)
        job_list = list(range(self.n_jobs))
        cur_begin = 0

        while job_list:
            job_idx = np.random.randint(0, len(job_list))
            for i in range(self.n_ops):
                x[cur_begin + i] = job_list[job_idx] + 1

            proc_list = list(range(self.n_ops))
            while proc_list:
                proc_idx = np.random.randint(0, len(proc_list))
                cur = time_and_mchindex[job_list[job_idx] * self.n_ops + proc_list[proc_idx]]
                temp_machload = np.zeros(self.n_machines)
                for item in cur:
                    temp_machload[item[1] - 1] = item[0] + machload[item[1] - 1]

                # 选负载最小的机器
                min_load = np.inf
                for v in temp_machload:
                    if v > 0 and v < min_load:
                        min_load = v
                min_load_machs = [i + 1 for i, v in enumerate(temp_machload) if v == min_load]

                # 负载相同则选加工时间最短的
                min_time = np.inf
                for m in min_load_machs:
                    for item in cur:
                        if item[1] == m and item[0] < min_time:
                            min_time = item[0]
                            break
                min_time_machs = [item[1] for item in cur if item[0] == min_time]

                pos = self.total_process + job_list[job_idx] * self.n_ops + proc_list[proc_idx]
                x[pos] = min_time_machs[np.random.randint(0, len(min_time_machs))]
                machload[int(x[pos]) - 1] += min_time
                del proc_list[proc_idx]

            cur_begin += self.n_ops
            del job_list[job_idx]

    def _init_machine_extreme_local(self, x, time_and_mchindex):
        """局部极限完工时间最小化初始化"""
        machload = np.zeros(self.n_machines)
        cur_begin = 0

        for p in range(self.n_jobs):
            for i in range(self.n_ops):
                x[cur_begin + i] = p + 1

            proc_list = list(range(self.n_ops))
            while proc_list:
                proc_idx = np.random.randint(0, len(proc_list))
                cur = time_and_mchindex[p * self.n_ops + proc_list[proc_idx]]
                temp_machload = np.zeros(self.n_machines)
                for item in cur:
                    temp_machload[item[1] - 1] = item[0] + machload[item[1] - 1]

                min_load = np.inf
                for v in temp_machload:
                    if v > 0 and v < min_load:
                        min_load = v
                min_load_machs = [i + 1 for i, v in enumerate(temp_machload) if v == min_load]

                min_time = np.inf
                for m in min_load_machs:
                    for item in cur:
                        if item[1] == m and item[0] < min_time:
                            min_time = item[0]
                            break
                min_time_machs = [item[1] for item in cur if item[0] == min_time]

                pos = self.total_process + p * self.n_ops + proc_list[proc_idx]
                x[pos] = min_time_machs[np.random.randint(0, len(min_time_machs))]
                machload[int(x[pos]) - 1] += min_time
                del proc_list[proc_idx]

            cur_begin += self.n_ops
            machload[:] = 0  # 局部模式: 每个工件处理后重置负载

    def _init_population(self):
        """根据策略初始化种群，返回 (pop, v, fitness)"""
        pop = np.zeros((self.popsize, self.total_process * 2))
        v = np.zeros((self.popsize, self.total_process * 2))
        fitness = np.zeros(self.popsize)

        if self.init_strategy == "extreme":
            time_and_mchindex = []
            for i in range(self.total_process):
                time_and_mchindex.append(
                    [[self.proc_time[i, j], j + 1]
                     for j in range(self.n_machines) if self.proc_time[i, j] > 0]
                )
            global_size = self.popsize // 2
            local_size = self.popsize - global_size

            for i in range(global_size):
                self._init_machine_extreme_global(pop[i], time_and_mchindex)
                np.random.shuffle(pop[i, :self.total_process])
                fitness[i] = self.calculate(pop[i])

            for i in range(global_size, global_size + local_size):
                self._init_machine_extreme_local(pop[i], time_and_mchindex)
                np.random.shuffle(pop[i, :self.total_process])
                fitness[i] = self.calculate(pop[i])
        else:
            for i in range(self.popsize):
                pop[i, :self.total_process] = self._init_ops()
                if self.init_strategy == "roulette":
                    self._init_machine_roulette(pop[i])
                else:
                    self._init_machine_random(pop[i])
                fitness[i] = self.calculate(pop[i])

        return pop, v, fitness

    # ------------------------------------------------------------------ #
    #                       PSO 位置更新与修复                              #
    # ------------------------------------------------------------------ #

    def _update_positions(self, pop, v):
        """PSO 位置更新: 工序部分用基于排序的映射，机器部分做越界修复

        工序部分: 按速度分量对原位置排序，得到新的工序排列
        机器部分: 对越界或不可行的机器编码进行修复
        """
        for j in range(self.popsize):
            # ---- 工序部分: 基于排序的位置更新 ----
            store = []
            before = pop[j, :self.total_process].copy()
            pop[j] += v[j]
            reference = v[j, :self.total_process].copy()
            for p in range(self.total_process):
                store.append((reference[p], before[p]))
            store.sort()
            for p in range(self.total_process):
                pop[j, p] = store[p][1]

        # 向上取整
        np.ceil(pop, out=pop)

        # ---- 机器部分: 越界与可行性修复 ----
        for j in range(self.popsize):
            ops = self._decode_ops(pop[j])
            for job_id, op_seq in ops:
                row = (job_id - 1) * self.n_ops + (op_seq - 1)
                mch_idx = int(pop[j, self.total_process + row]) - 1

                # 越界或机器不可用 → 用最短加工时间的机器兜底
                if (mch_idx < 0 or mch_idx >= self.n_machines
                        or self.proc_time[row, mch_idx] < 0):
                    pop[j, self.total_process + row] = \
                        self.clean_contents[row][-1][1]

    # ------------------------------------------------------------------ #
    #                          求解主流程                                  #
    # ------------------------------------------------------------------ #

    def solve(self):
        """运行 PSO 优化

        Returns:
            gbestpop: 最优个体编码, shape (total_process * 2,)
            gbestfitness: 最优适应度 (makespan)
            history: 每代全局最优 makespan 的列表
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        pop, v, fitness = self._init_population()

        # 初始化全局最优和个体最优
        gbestpop = pop[fitness.argmin()].copy()
        gbestfitness = fitness.min()
        pbestpop = pop.copy()
        pbestfitness = fitness.copy()

        history = []

        for gen in range(self.maxgen):
            # 速度更新
            for j in range(self.popsize):
                v[j] = (self.w * v[j]
                        + self.lr[0] * np.random.rand() * (pbestpop[j] - pop[j])
                        + self.lr[1] * np.random.rand() * (gbestpop - pop[j]))

            # 位置更新（含修复）
            self._update_positions(pop, v)

            # 适应度更新
            for j in range(self.popsize):
                fitness[j] = self.calculate(pop[j])

            # 更新个体最优
            for j in range(self.popsize):
                if fitness[j] < pbestfitness[j]:
                    pbestfitness[j] = fitness[j]
                    pbestpop[j] = pop[j].copy()

            # 更新全局最优
            if pbestfitness.min() < gbestfitness:
                gbestfitness = pbestfitness.min()
                gbestpop = pop[pbestfitness.argmin()].copy()

            history.append(gbestfitness)

        return gbestpop, gbestfitness, history

    # ------------------------------------------------------------------ #
    #                          调度结果输出                                #
    # ------------------------------------------------------------------ #

    def get_schedule(self, x):
        """从个体编码生成完整调度方案

        Returns:
            list of dict, 每项:
                job_id, op_id, machine, start_time, end_time
        """
        Tm = np.zeros(self.n_machines)
        Te = np.zeros((self.n_jobs, self.n_ops))
        Ts = np.zeros((self.n_jobs, self.n_ops))  # 开始时间
        ops = self._decode_ops(x)

        for i in range(self.total_process):
            job_id, op_seq = ops[i]
            mch_idx = int(x[self.total_process + (job_id - 1) * self.n_ops + (op_seq - 1)]) - 1
            proc_row = (job_id - 1) * self.n_ops + (op_seq - 1)
            proc_time = self.proc_time[proc_row, mch_idx]

            if op_seq == 1:
                Ts[job_id - 1, op_seq - 1] = Tm[mch_idx]
                Tm[mch_idx] += proc_time
            else:
                start = max(Te[job_id - 1, op_seq - 2], Tm[mch_idx])
                Ts[job_id - 1, op_seq - 1] = start
                Tm[mch_idx] = start + proc_time
            Te[job_id - 1, op_seq - 1] = Tm[mch_idx]

        # 组装结果
        schedule = []
        for j in range(self.n_jobs):
            for o in range(self.n_ops):
                mch = int(x[self.total_process + j * self.n_ops + o]) - 1
                schedule.append({
                    "job_id": j,
                    "op_id": o,
                    "machine": mch,
                    "start_time": float(Ts[j, o]),
                    "end_time": float(Te[j, o]),
                })
        return schedule

    # ------------------------------------------------------------------ #
    #                      从 txt / json 文件加载                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_txt(cls, filepath, n_jobs, n_ops, n_machines, **kwargs):
        """从原始 txt 文件构造 solver"""
        contents = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    contents.append(line.split())
        data = np.array(contents)
        return cls(data, n_jobs, n_ops, n_machines, **kwargs)

    @classmethod
    def from_json(cls, filepath, **kwargs):
        """从标准 JSON 文件构造 solver (需要额外指定规模参数)"""
        with open(filepath) as f:
            obj = json.load(f)
        n_machines = obj["machines"]
        n_jobs = len(obj["jobs"])
        n_ops = len(obj["jobs"][0])

        # JSON → txt 兼容的 (total_process, n_machines) 矩阵
        data = np.full((n_jobs * n_ops, n_machines), "-", dtype=str)
        for j, job in enumerate(obj["jobs"]):
            for o, alternatives in enumerate(job):
                for alt in alternatives:
                    data[j * n_ops + o, alt["machine"]] = str(alt["processing"])

        return cls(data, n_jobs, n_ops, n_machines, **kwargs)


# ====================================================================== #
#                               主程序                                    #
# ====================================================================== #

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # ---- 测试配置 ----
    test_cases = [
        {
            "file": os.path.join(parent_dir, "data", "data_first.txt"),
            "n_jobs": 10, "n_ops": 5, "n_machines": 6,
            "name": "J10P5M6",
        },
        {
            "file": os.path.join(parent_dir, "data", "data_second.txt"),
            "n_jobs": 20, "n_ops": 10, "n_machines": 10,
            "name": "J20P10M10",
        },
        {
            "file": os.path.join(parent_dir, "data", "data_third.txt"),
            "n_jobs": 20, "n_ops": 20, "n_machines": 15,
            "name": "J20P20M15",
        },
    ]

    strategies = ["random", "roulette", "extreme"]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"  问题规模: {tc['name']}")
        print(f"{'='*60}")

        for strategy in strategies:
            solver = PSOSolver.from_txt(
                tc["file"], tc["n_jobs"], tc["n_ops"], tc["n_machines"],
                init_strategy=strategy, popsize=50, maxgen=500,
                w=0.9, lr=(2, 2), seed=42,
            )

            t0 = time.time()
            best_x, best_fit, history = solver.solve()
            elapsed = time.time() - t0

            print(f"  [{strategy:>8s}] makespan = {best_fit:.0f}  ({elapsed:.2f}s)")

    print("\n完成。")
