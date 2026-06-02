"""
DE Solver — 将 FJSP-master/DE 中的三个脚本统一为一个可复用的类

支持三种初始化策略:
  - "random"  : 完全随机 (对应 DE_first.py)
  - "roulette" : 轮盘赌 (对应 DE_second.py)
  - "extreme"  : 极限完工时间最小化 (对应 DE_third.py)

用法:
    solver = DESolver(data, n_jobs=10, n_ops=5, n_machines=6,
                      init_strategy="roulette", maxgen=500)
    best_x, best_makespan = solver.solve()
    schedule = solver.get_schedule(best_x)
"""

import numpy as np
import json
import os


class DESolver:
    """差分进化求解器 for FJSP"""

    def __init__(self, data, n_jobs, n_ops, n_machines,
                 init_strategy="random", popsize=50, maxgen=500,
                 F=0.1, Cr=0.1, seed=None):
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
            F: 变异率
            Cr: 交叉率
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
        self.F = F
        self.Cr = Cr
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
            # 机器编码中的位置: (工件索引 * 工序数 + 工序索引)
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
                if not cur:
                    print(f"[DEBUG] EMPTY cur! job={job_list[job_idx]}, proc={proc_list[proc_idx]}, "
                          f"idx={job_list[job_idx] * self.n_ops + proc_list[proc_idx]}, "
                          f"proc_time_row={self.proc_time[job_list[job_idx] * self.n_ops + proc_list[proc_idx]]}")
                temp_machload = np.zeros(self.n_machines)
                for item in cur:
                    temp_machload[item[1] - 1] = item[0] + machload[item[1] - 1]

                # 选负载最小的机器
                min_load = np.inf
                for v in temp_machload:
                    if v > 0 and v < min_load:
                        min_load = v
                min_load_machs = [i + 1 for i, v in enumerate(temp_machload) if v == min_load]

                if not min_load_machs:
                    print(f"[DEBUG] EMPTY min_load_machs! job={job_list[job_idx]}, proc={proc_list[proc_idx]}, "
                          f"cur={cur}, temp_machload={temp_machload}, machload={machload}, min_load={min_load}")

                # 负载相同则选加工时间最短的
                min_time = np.inf
                for m in min_load_machs:
                    for item in cur:
                        if item[1] == m and item[0] < min_time:
                            min_time = item[0]
                            break
                min_time_machs = [item[1] for item in cur if item[0] == min_time]

                if not min_time_machs:
                    print(f"[DEBUG] EMPTY min_time_machs! min_load_machs={min_load_machs}, "
                          f"min_time={min_time}, cur={cur}")

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
        """根据策略初始化种群"""
        pop = np.zeros((self.popsize, self.total_process * 2))
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

        return pop, fitness

    # ------------------------------------------------------------------ #
    #                          交叉算子                                    #
    # ------------------------------------------------------------------ #

    def _pox_crossover(self, parent1, parent2):
        """POX (Precedence Operation Crossover)

        随机将工件集分为两组，父代1保留组1的工件位置，
        其余位置按顺序填入父代2中组2的工件。
        返回两个子代中适应度更优的那个。
        """
        seq = list(range(1, self.n_jobs + 1))
        n_select = np.random.randint(2, len(seq))
        set1 = set()
        for _ in range(n_select):
            idx = np.random.randint(0, len(seq))
            set1.add(seq.pop(idx))
        set2 = set(seq)

        child1 = parent1.copy()
        child2 = parent2.copy()

        remain1 = [i for i in range(self.total_process) if parent1[i] in set2]
        remain2 = [i for i in range(self.total_process) if parent2[i] in set2]

        c1, c2 = 0, 0
        for i in range(self.total_process):
            if parent2[i] in set2:
                child1[remain1[c1]] = parent2[i]
                c1 += 1
            if parent1[i] in set2:
                child2[remain2[c2]] = parent1[i]
                c2 += 1

        return child1 if self.calculate(child1) < self.calculate(child2) else child2

    # ------------------------------------------------------------------ #
    #                          变异 + 交叉选择                             #
    # ------------------------------------------------------------------ #

    def _mutation(self, pop, gbestpop, pbestpop):
        """DE 变异: 每个个体以概率 F 与 gbest/pbest 交叉"""
        mid = np.zeros_like(pop)
        for i in range(self.popsize):
            # 内层交叉: 与全局最优
            if np.random.rand() <= self.F:
                inside = self._pox_crossover(pop[i], gbestpop)
            else:
                inside = pop[i].copy()

            # 外层交叉: 与随机个体交叉（保持与原始代码一致的广播比较）
            idx = np.random.randint(0, self.popsize)
            while idx == i or not (pop[idx] - pbestpop).any():
                idx = np.random.randint(0, self.popsize)

            if np.random.rand() <= self.F:
                outside = self._pox_crossover(inside, pop[idx])
            else:
                outside = inside

            mid[i] = outside
        return mid

    def _cross_and_select(self, pop, mid):
        """机器编码交叉 + 不可行修复 + 贪心选择"""
        for i in range(self.popsize):
            candidate = pop[i].copy()
            for j in range(self.total_process):
                if np.random.rand() <= self.Cr:
                    candidate[j + self.total_process] = mid[i, j + self.total_process]

            # 修复不可行机器分配
            ops = self._decode_ops(candidate)
            for job_id, op_seq in ops:
                row = (job_id - 1) * self.n_ops + (op_seq - 1)
                mch = int(candidate[self.total_process + row]) - 1
                if mch < 0 or mch >= self.n_machines or self.proc_time[row, mch] < 0:
                    # 兜底: 选可用机器中加工时间最短的
                    candidate[self.total_process + row] = \
                        self.clean_contents[row][-1][1]

            if self.calculate(candidate) < self.calculate(pop[i]):
                pop[i] = candidate

    # ------------------------------------------------------------------ #
    #                          求解主流程                                  #
    # ------------------------------------------------------------------ #

    def solve(self):
        """运行 DE 优化

        Returns:
            gbestpop: 最优个体编码, shape (total_process * 2,)
            gbestfitness: 最优适应度 (makespan)
            history: 每代全局最优 makespan 的列表
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        pop, fitness = self._init_population()

        gbestpop = pop[fitness.argmin()].copy()
        gbestfitness = fitness.min()
        pbestpop = pop.copy()
        pbestfitness = fitness.copy()

        history = []

        for gen in range(self.maxgen):
            mid = self._mutation(pop, gbestpop, pbestpop)
            self._cross_and_select(pop, mid)

            # 更新适应度
            for j in range(self.popsize):
                fitness[j] = self.calculate(pop[j])
                if fitness[j] < pbestfitness[j]:
                    pbestfitness[j] = fitness[j]
                    pbestpop[j] = pop[j].copy()

            if fitness.min() < gbestfitness:
                gbestfitness = fitness.min()
                gbestpop = pop[fitness.argmin()].copy()

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
            solver = DESolver.from_txt(
                tc["file"], tc["n_jobs"], tc["n_ops"], tc["n_machines"],
                init_strategy=strategy, popsize=50, maxgen=500,
                F=0.1, Cr=0.1, seed=42,
            )

            t0 = time.time()
            best_x, best_fit, history = solver.solve()
            elapsed = time.time() - t0

            print(f"  [{strategy:>8s}] makespan = {best_fit:.0f}  ({elapsed:.2f}s)")

        # 绘制最后一个策略的收敛曲线（作为代表）
    print("\n完成。")
