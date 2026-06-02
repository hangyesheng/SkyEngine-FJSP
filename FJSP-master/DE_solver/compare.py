"""
对比脚本：原始 DE 代码 vs 重构后 DESolver

同一种子、同一数据集、同一次运行，对比 makespan 是否一致。
只在 data_first (J10P5M6) 上做验证，降低等待时间。
"""

import numpy as np
import sys
import os
import time

# 把 DE_solver 加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from de_solver import DESolver

SEED = 42
POPSIZE = 50
MAXGEN = 200  # 用 200 代做快速验证
F = 0.1
Cr = 0.1

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def run_original(strategy):
    """直接运行原始 DE 代码的核心逻辑"""
    np.random.seed(SEED)

    workpiece = 10
    process = 5
    total_process = workpiece * process
    machine = 6

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

    def func(x1, x2):
        seq = [i + 1 for i in range(workpiece)]
        random_length1 = np.random.randint(2, len(seq) - 1)
        set1 = set()
        for i in range(random_length1):
            index = np.random.randint(0, len(seq))
            set1.add(seq[index])
            seq.remove(seq[index])
        set2 = set(seq)
        child1 = np.copy(x1)
        child2 = np.copy(x2)
        remain1 = [i for i in range(total_process) if x1[i] in set2]
        remain2 = [i for i in range(total_process) if x2[i] in set2]
        cursor1, cursor2 = 0, 0
        for i in range(total_process):
            if x2[i] in set2:
                child1[remain1[cursor1]] = x2[i]
                cursor1 += 1
            if x1[i] in set2:
                child2[remain2[cursor2]] = x1[i]
                cursor2 += 1
        return child1 if calculate(child1) < calculate(child2) else child2

    # ---- 初始化 ----
    pop = np.zeros((POPSIZE, total_process * 2))
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
                    for v in temp_ml:
                        if v > 0 and v < min_load:
                            min_load = v
                    min_load_list = [ii + 1 for ii, v in enumerate(temp_ml) if v == min_load]
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
                    for v in temp_ml:
                        if v > 0 and v < min_load:
                            min_load = v
                    min_load_list = [ii + 1 for ii, v in enumerate(temp_ml) if v == min_load]
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
    def mutation():
        mid = np.zeros((POPSIZE, total_process * 2))
        for i in range(POPSIZE):
            if np.random.rand() > F:
                inside = np.copy(pop[i])
            else:
                inside = func(pop[i], gbestpop)
            index = np.random.randint(0, POPSIZE)
            while index == i or not (pop[index] - pbestpop).any():
                index = np.random.randint(0, POPSIZE)
            if np.random.rand() > F:
                outside = inside
            else:
                outside = func(inside, pop[index])
            mid[i] = outside
        return mid

    def cross_and_select(mid):
        for i in range(POPSIZE):
            individual = np.copy(pop[i])
            for j in range(total_process):
                if np.random.rand() <= Cr:
                    individual[j + total_process] = mid[i][j + total_process]
                else:
                    individual[j + total_process] = pop[i][j + total_process]
            array = handle(individual)
            for j in range(total_process):
                row = (array[j][0] - 1) * process + array[j][1] - 1
                if contents[row][int(individual[total_process + row]) - 1] == "-":
                    individual[total_process + row] = clean_contents[row][len(clean_contents[row]) - 1][1]
            if calculate(individual) < calculate(pop[i]):
                pop[i] = individual

    for gen in range(MAXGEN):
        mid = mutation()
        cross_and_select(mid)
        for j in range(POPSIZE):
            fitness[j] = calculate(pop[j])
            if fitness[j] < pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()
        if fitness.min() < gbestfitness:
            gbestfitness = fitness.min()
            gbestpop = pop[fitness.argmin()].copy()

    return gbestfitness


def run_refactored(strategy):
    """运行重构后的 DESolver"""
    np.random.seed(SEED)
    solver = DESolver.from_txt(
        os.path.join(data_dir, "data_first.txt"),
        n_jobs=10, n_ops=5, n_machines=6,
        init_strategy=strategy, popsize=POPSIZE, maxgen=MAXGEN,
        F=F, Cr=Cr, seed=SEED,
    )
    _, best_fit, _ = solver.solve()
    return best_fit


if __name__ == "__main__":
    print(f"对比: 原始 DE 代码 vs 重构 DESolver")
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

        match = "✓ 一致" if orig_result == new_result else "✗ 不一致"

        print(f"  原始代码 makespan: {orig_result:<8.0f} ({orig_time:.2f}s)")
        print(f"  重构代码 makespan: {new_result:<8.0f} ({new_time:.2f}s)")
        print(f"  结果: {match}")
