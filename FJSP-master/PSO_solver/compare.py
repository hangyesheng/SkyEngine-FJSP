"""
对比脚本：原始 PSO 代码 vs 重构后 PSOSolver

同一种子、同一数据集、同一次运行，对比 makespan 是否一致。
只在 data_first (J10P5M6) 上做验证，降低等待时间。
"""

import numpy as np
import sys
import os
import time

# 把 PSO_solver 加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pso_solver import PSOSolver

SEED = 42
POPSIZE = 50
MAXGEN = 200  # 用 200 代做快速验证
W = 0.9
LR = (2, 2)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def run_original(strategy):
    """直接运行原始 PSO 代码的核心逻辑"""
    np.random.seed(SEED)

    workpiece = 10
    process = 5
    total_process = workpiece * process
    machine = 6
    rangepop = (1, machine)

    # 读取数据
    contents = []
    with open(os.path.join(data_dir, "data_first.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                contents.append(line.split())

    # 预处理 clean_contents
    clean_contents = []
    for i in range(total_process):
        clean_contents.append(
            [[int(contents[i][j]), j + 1] for j in range(machine) if contents[i][j] != "-"]
        )
        temp_sum = sum(1.0 / c[0] for c in clean_contents[i])
        for j in range(len(clean_contents[i])):
            clean_contents[i][j][0] = (1.0 / clean_contents[i][j][0]) / temp_sum
        clean_contents[i].sort()
        cumulation = 0
        for j in range(len(clean_contents[i])):
            cumulation += clean_contents[i][j][0]
            clean_contents[i][j][0] = cumulation

    # ---- 通用函数 ----
    def handle(x):
        piece_mark = np.zeros(workpiece, dtype=int)
        array = []
        for i in range(total_process):
            piece_mark[int(x[i] - 1)] += 1
            array.append((int(x[i]), int(piece_mark[int(x[i] - 1)])))
        return array

    def calculate(x):
        Tm = np.zeros(machine)
        Te = np.zeros((workpiece, process))
        array = handle(x)
        for i in range(total_process):
            machine_index = int(x[total_process + (array[i][0] - 1) * process + (array[i][1] - 1)]) - 1
            process_index = (array[i][0] - 1) * process + (array[i][1] - 1)
            process_time = int(contents[process_index][machine_index])
            if array[i][1] == 1:
                Tm[machine_index] += process_time
                Te[array[i][0] - 1][array[i][1] - 1] = Tm[machine_index]
            else:
                Tm[machine_index] = max(Te[array[i][0] - 1][array[i][1] - 2], Tm[machine_index]) + process_time
                Te[array[i][0] - 1][array[i][1] - 1] = Tm[machine_index]
        return max(Tm)

    # ---- 初始化 ----
    pop = np.zeros((POPSIZE, total_process * 2))
    v = np.zeros((POPSIZE, total_process * 2))
    fitness = np.zeros(POPSIZE)

    if strategy == "extreme":
        time_and_mchindex = []
        for i in range(total_process):
            time_and_mchindex.append(
                [[int(contents[i][j]), j + 1] for j in range(machine) if contents[i][j] != "-"]
            )
        global_size = POPSIZE // 2
        local_size = POPSIZE - global_size

        for i in range(global_size):
            machload = np.zeros(machine)
            job_list = list(range(workpiece))
            cur_begin = 0
            while job_list:
                job_index = np.random.randint(0, len(job_list))
                for ii in range(process):
                    pop[i][cur_begin + ii] = job_list[job_index] + 1
                proc_list = list(range(process))
                while proc_list:
                    proc_index = np.random.randint(0, len(proc_list))
                    cur = time_and_mchindex[job_list[job_index] * process + proc_list[proc_index]]
                    temp_ml = np.zeros(machine)
                    for item in cur:
                        temp_ml[item[1] - 1] = item[0] + machload[item[1] - 1]
                    min_load = np.inf
                    for val in temp_ml:
                        if val > 0 and val < min_load:
                            min_load = val
                    min_load_list = [ii + 1 for ii, val in enumerate(temp_ml) if val == min_load]
                    min_time = np.inf
                    for m in min_load_list:
                        for item in cur:
                            if item[1] == m and item[0] < min_time:
                                min_time = item[0]
                                break
                    min_time_list = [item[1] for item in cur if item[0] == min_time]
                    pos = total_process + job_list[job_index] * process + proc_list[proc_index]
                    pop[i][pos] = min_time_list[np.random.randint(0, len(min_time_list))]
                    machload[int(pop[i][pos]) - 1] += min_time
                    del proc_list[proc_index]
                cur_begin += process
                del job_list[job_index]
            np.random.shuffle(pop[i][:total_process])
            fitness[i] = calculate(pop[i])

        for i in range(global_size, global_size + local_size):
            machload = np.zeros(machine)
            cur_begin = 0
            for p in range(workpiece):
                for ii in range(process):
                    pop[i][cur_begin + ii] = p + 1
                proc_list = list(range(process))
                while proc_list:
                    proc_index = np.random.randint(0, len(proc_list))
                    cur = time_and_mchindex[p * process + proc_list[proc_index]]
                    temp_ml = np.zeros(machine)
                    for item in cur:
                        temp_ml[item[1] - 1] = item[0] + machload[item[1] - 1]
                    min_load = np.inf
                    for val in temp_ml:
                        if val > 0 and val < min_load:
                            min_load = val
                    min_load_list = [ii + 1 for ii, val in enumerate(temp_ml) if val == min_load]
                    min_time = np.inf
                    for m in min_load_list:
                        for item in cur:
                            if item[1] == m and item[0] < min_time:
                                min_time = item[0]
                                break
                    min_time_list = [item[1] for item in cur if item[0] == min_time]
                    pos = total_process + p * process + proc_list[proc_index]
                    pop[i][pos] = min_time_list[np.random.randint(0, len(min_time_list))]
                    machload[int(pop[i][pos]) - 1] += min_time
                    del proc_list[proc_index]
                cur_begin += process
                machload[:] = 0
            np.random.shuffle(pop[i][:total_process])
            fitness[i] = calculate(pop[i])
    else:
        for i in range(POPSIZE):
            for j in range(workpiece):
                for p in range(process):
                    pop[i][j * process + p] = j + 1
            np.random.shuffle(pop[i][:total_process])

            for j in range(total_process):
                if strategy == "roulette":
                    rate = np.random.rand()
                    for p in range(len(clean_contents[j])):
                        if rate <= clean_contents[j][p][0]:
                            pop[i][j + total_process] = clean_contents[j][p][1]
                            break
                else:
                    index = np.random.randint(0, machine)
                    while contents[j][index] == "-":
                        index = np.random.randint(0, machine)
                    pop[i][j + total_process] = index + 1
            fitness[i] = calculate(pop[i])

    gbestpop = pop[fitness.argmin()].copy()
    gbestfitness = fitness.min()
    pbestpop = pop.copy()
    pbestfitness = fitness.copy()

    # ---- 迭代 ----
    for gen in range(MAXGEN):
        # 速度更新
        for j in range(POPSIZE):
            v[j] = W * v[j] + LR[0] * np.random.rand() * (pbestpop[j] - pop[j]) \
                   + LR[1] * np.random.rand() * (gbestpop - pop[j])

        # 位置更新 - 工序部分
        for j in range(POPSIZE):
            store = []
            before = pop[j][:total_process].copy()
            pop[j] += v[j]
            reference = v[j][:total_process].copy()
            for p in range(total_process):
                store.append((reference[p], before[p]))
            store.sort()
            for p in range(total_process):
                pop[j][p] = store[p][1]

        pop = np.ceil(pop)

        # 位置更新 - 机器部分修复
        for j in range(POPSIZE):
            array = handle(pop[j])
            for p in range(total_process):
                row = (array[p][0] - 1) * process + (array[p][1] - 1)
                mch_val = pop[j][total_process + row]
                if (mch_val < rangepop[0] or mch_val > rangepop[1]) \
                        or (contents[row][int(mch_val) - 1] == "-"):
                    pop[j][total_process + row] = clean_contents[row][len(clean_contents[row]) - 1][1]

        # 适应度更新
        for j in range(POPSIZE):
            fitness[j] = calculate(pop[j])

        # 更新个体最优
        for j in range(POPSIZE):
            if fitness[j] < pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()

        # 更新全局最优
        if pbestfitness.min() < gbestfitness:
            gbestfitness = pbestfitness.min()
            gbestpop = pop[pbestfitness.argmin()].copy()

    return gbestfitness


def run_refactored(strategy):
    """运行重构后的 PSOSolver"""
    np.random.seed(SEED)
    solver = PSOSolver.from_txt(
        os.path.join(data_dir, "data_first.txt"),
        n_jobs=10, n_ops=5, n_machines=6,
        init_strategy=strategy, popsize=POPSIZE, maxgen=MAXGEN,
        w=W, lr=LR, seed=SEED,
    )
    _, best_fit, _ = solver.solve()
    return best_fit


if __name__ == "__main__":
    print(f"对比: 原始 PSO 代码 vs 重构 PSOSolver")
    print(f"数据集: data_first.txt (J10P5M6), 种子={SEED}, {MAXGEN}代, 种群{POPSIZE}")
    print(f"{'='*55}")

    for strategy in ["random", "roulette", "extreme"]:
        print(f"\n策略: {strategy}")
        print(f"{'-'*40}")

        # 原始代码
        t0 = time.time()
        np.random.seed(SEED)
        orig_result = run_original(strategy)
        orig_time = time.time() - t0

        # 重构代码
        t0 = time.time()
        new_result = run_refactored(strategy)
        new_time = time.time() - t0

        match = "Y 一致" if orig_result == new_result else "X 不一致"

        print(f"  原始代码 makespan: {orig_result:<8.0f} ({orig_time:.2f}s)")
        print(f"  重构代码 makespan: {new_result:<8.0f} ({new_time:.2f}s)")
        print(f"  结果: {match}")
