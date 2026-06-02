# FJSP 算法复现与 JobSolver 规范化步骤

## 一、项目目标

将 `FJSP-master/` 中的 6 个脚本式算法（DE × 3 + PSO × 3）以及 `End-to-end-DRL-for-FJSP-main/` 中的深度强化学习算法，重构为符合 SkyEngine `JobSolver` 接口的统一求解器。

---

## 二、现状分析

### 2.1 现有代码问题

| 问题 | 说明 |
|------|------|
| 脚本式编写 | 所有逻辑在全局作用域 / `__main__` 中，无法被外部调用 |
| 硬编码参数 | `workpiece`, `process`, `machine` 等写死在代码中 |
| 文件路径硬编码 | `with open("data_first.txt")` 无法适配不同数据源 |
| 大量重复代码 | 6 个文件共享 `handle()`, `calculate()`, 编码/解码逻辑 |
| 无在线调度能力 | 离线跑完整个优化，不支持逐步 `plan()` 调用 |
| 无统一接口 | 无基类、无工厂注册，无法融入 SkyEngine |

### 2.2 DRL 项目代码问题（End-to-end-DRL-for-FJSP-main）

| 问题 | 说明 |
|------|------|
| 参数全局 argparse | `Params.py` 用 `parse_args()` 在 import 时解析，外部调用无法覆盖 |
| 硬编码 n_j × n_m | 网络层维度依赖固定的问题规模（如 30×20），`reset()` 中 `number_of_tasks = n_j * n_m` |
| 数据格式不一致 | MultiPPO 用随机生成的 `(n_j, n_m, n_m)` 矩阵，RealWorld 用 `.fjs` 文件解析 |
| 非均匀工序数 | 原始环境假设每个工件恰好 `n_m` 道工序（`number_of_tasks = n_j * n_m`），SkyEngine 中每个工件工序数不同 |
| 不可用机器表示 | 随机实例用负数（`< 0`）表示不可用机器，真实数据用 0 表示，需统一 |
| 批量推理设计 | 环境和模型都按 batch 维度设计，单实例推理需 batch_size=1 包装 |
| 重复代码 | `FJSP_MultiPPO/` 和 `FJSP_RealWorld/` 之间大量文件重复（FJSP_Env、models、utils 等） |

### 2.3 三种初始化策略（PSO/DE）

| 策略 | 文件后缀 | 原理 |
|------|---------|------|
| 随机初始化 | `*_first.py` | 机器部分完全随机选择（排除不可用机器） |
| 轮盘赌初始化 | `*_second.py` | 以 `1/加工时间` 为概率，偏好选短耗时机器 |
| 极限完工时间最小化 | `*_third.py` | 50% 全局初始化 + 50% 局部初始化，综合考虑机器负载和最短加工时间 |

### 2.4 两种元启发式算法核心差异（PSO/DE）

| | DE（差分进化） | PSO（粒子群） |
|---|---|---|
| **搜索机制** | POX 交叉变异 + 贪心选择 | 速度更新 + 位置映射 |
| **工序部分更新** | POX 交叉交换工件块 | 排序映射保持合法排列 |
| **机器部分更新** | 交叉率 Cr 控制继承 | 越界时用 `clean_contents` 兜底 |
| **特有参数** | F（变异率）, Cr（交叉率） | w（惯性权重）, lr（加速因子） |

---

## 三、目标架构

```
FJSP/
├── algorithms/
│   ├── __init__.py                  # 导出所有 Solver
│   ├── base_solver.py               # 基类，定义 plan() 接口
│   ├── data_adapter.py              # 统一数据适配（txt / obs / .fjs）
│   ├── encoding.py                  # 统一编码/解码（handle, encode, decode）
│   ├── fitness.py                   # 统一适应度计算（calculate）
│   ├── initialization.py            # 三种初始化策略（PSO/DE 用）
│   ├── crossover.py                 # POX 交叉算子
│   ├── output_builder.py            # plan() 输出转换
│   ├── de_solver.py                 # DE 求解器
│   ├── pso_solver.py                # PSO 求解器
│   ├── drl_solver.py                # DRL 求解器（Multi-PPO）
│   └── drl/                         # DRL 子模块
│       ├── __init__.py
│       ├── env.py                   # FJSP 析取图环境（适配非均匀工序）
│       ├── models/
│       │   ├── gin.py               # GraphCNN (GIN 图神经网络)
│       │   ├── actor.py             # Job_Actor + Mch_Actor
│       │   ├── mlp.py               # MLPActor / MLPCritic
│       │   └── pointer.py           # Pointer 网络
│       ├── agent_utils.py           # 动作采样/贪心选择
│       ├── ppo.py                   # Multi-PPO 训练逻辑
│       ├── permissible_ls.py        # Permissible Left Shift 调度优化
│       ├── update_adj.py            # 邻接矩阵更新
│       ├── update_endtime_lb.py     # 下界时间计算
│       ├── mb_agg.py                # 批量图聚合
│       └── data_utils.py            # 数据生成与加载
├── PSO/                             # 原始代码（保留作参考）
├── DE/                              # 原始代码（保留作参考）
├── End-to-end-DRL-for-FJSP-main/   # 原始 DRL 代码（保留作参考）
└── data/                            # 数据文件
```

---

## 四、实施步骤

### Step 1：创建统一数据适配层

**目标**：将 `obs: dict`（SkyEngine 格式）转换为算法内部可用的数据结构。

```python
# algorithms/data_adapter.py

class FJSPData:
    """从 SkyEngine obs 或原始 txt 文件构建的统一内部数据表示"""
    def __init__(self):
        self.workpiece: int          # 工件数
        self.process: int            # 每工件工序数
        self.machine: int            # 机器数
        self.total_process: int      # 总工序数
        self.proc_time: np.ndarray   # shape (total_process, machine)，-1 表示不可用
        self.machines: list          # Machine 对象列表

    @classmethod
    def from_txt(cls, filepath: str, workpiece: int, process: int, machine: int) -> "FJSPData":
        """从原始 txt 数据文件构建（复现阶段用）"""
        ...

    @classmethod
    def from_obs(cls, obs: dict) -> "FJSPData":
        """从 SkyEngine obs 构建（对接阶段用）"""
        ...
```

**要点**：
- `proc_time[row][col]` 存储第 `row` 道工序在机器 `col` 上的加工时间，不可用为 `-1`
- 预计算 `clean_contents`（轮盘赌用的累积概率分布），避免每次初始化重复计算
- `from_obs()` 需遍历 `obs["jobs"]` 中每个 `Operation.machine_options` 和 `proc_time` 构建 `proc_time` 矩阵

---

### Step 2：抽取统一编码与解码模块

**目标**：将 6 个文件中重复的 `handle()` 和编码逻辑统一。

```python
# algorithms/encoding.py
import numpy as np

def decode_operation_sequence(x: np.ndarray, workpiece: int, process: int) -> list:
    """工序编码 → 有序工序列表 [(job_id, op_seq), ...]

    原始代码中的 handle() 函数。
    输入: x[:total_process] 段（1-indexed 工件号序列）
    输出: [(工件号, 第几道工序), ...] 按加工顺序排列
    """
    piece_mark = np.zeros(workpiece, dtype=int)
    array = []
    total_process = workpiece * process
    for i in range(total_process):
        job = int(x[i]) - 1
        piece_mark[job] += 1
        array.append((int(x[i]), int(piece_mark[job])))
    return array


def encode_individual(op_seq: list, machine_assign: list) -> np.ndarray:
    """将工序排列和机器分配合并为一条完整编码

    输出: shape (total_process * 2,)
      前半段: 工序编码 (工件号序列)
      后半段: 机器编码 (每道工序对应的机器号)
    """
    return np.concatenate([op_seq, machine_assign])
```

---

### Step 3：抽取统一适应度计算

**目标**：将 6 个文件中完全相同的 `calculate()` 统一，同时支持输出甘特图所需的调度详情。

```python
# algorithms/fitness.py

def calculate_makespan(x: np.ndarray, data: FJSPData) -> float:
    """计算最大完工时间（适应度值）

    原始代码中的 calculate() 函数。
    遍历工序编码，模拟每台机器的加工过程，返回 max(Tm)。
    """
    ...

def calculate_schedule(x: np.ndarray, data: FJSPData) -> dict:
    """计算完整调度方案（用于生成 machine_actions）

    除了返回 makespan，还返回每个工序的:
    - assigned_machine
    - start_time
    - end_time
    用于后续生成 plan() 输出。
    """
    ...
```

**关键逻辑**（从原始代码提取，保持不变）：
```
对每个工序 i:
    取其机器索引 machine_index 和加工时间 process_time
    如果是该工件第 1 道工序:
        Tm[machine_index] += process_time
    否则:
        Tm[machine_index] = max(前一道工序完成时间, 机器当前时间) + process_time
    记录 Te[job][op] = Tm[machine_index]
返回 max(Tm)
```

---

### Step 4：抽取三种初始化策略

**目标**：统一接口，根据策略名选择不同初始化方式。

```python
# algorithms/initialization.py

def init_random(pop: np.ndarray, data: FJSPData, i: int):
    """完全随机初始化（对应 *_first.py）

    工序部分: 随机打乱工件号序列
    机器部分: 对每道工序，从可用机器中随机选一个
    """
    ...

def init_roulette(pop: np.ndarray, data: FJSPData, i: int):
    """轮盘赌初始化（对应 *_second.py）

    工序部分: 同上
    机器部分: 以 1/proc_time 为权重，轮盘赌选择（偏好短加工时间）
    使用 data.clean_contents 中的累积概率分布
    """
    ...

def init_extreme(pop: np.ndarray, data: FJSPData, i: int, mode: str = "global"):
    """极限完工时间最小化初始化（对应 *_third.py）

    mode="global": 全局视角，跨工件考虑机器负载
    mode="local":  局部视角，单工件内优化后重置负载

    核心思路: 对每道工序，在可选机器中选择「负载最小」的机器，
    若负载相同则选「加工时间最短」的。
    """
    ...

INIT_STRATEGIES = {
    "random": init_random,
    "roulette": init_roulette,
    "extreme": init_extreme,
}
```

---

### Step 5：抽取 POX 交叉算子

**目标**：DE 和 PSO 的 third 版本都使用了相同的 POX 交叉。

```python
# algorithms/crossover.py

def pox_crossover(parent1: np.ndarray, parent2: np.ndarray,
                  workpiece: int, total_process: int,
                  fitness_fn) -> np.ndarray:
    """POX (Precedence Operation Crossover)

    1. 随机将工件集合分为两个子集 set1, set2
    2. child1 保留 parent1 中属于 set1 的工件位置，从 parent2 填入 set2 的工件
    3. child2 反之
    4. 返回适应度更优的子代
    """
    ...
```

---

### Step 6：实现 DE 求解器

```python
# algorithms/de_solver.py

class DESolver(FJSPSolver):
    """差分进化求解器

    参数:
        init_strategy: "random" | "roulette" | "extreme"
        popsize: 种群规模 (默认 50)
        maxgen: 最大迭代次数 (默认 500)
        F: 变异率 (默认 0.1)
        Cr: 交叉率 (默认 0.1)
    """

    def __init__(self, init_strategy="random", popsize=50, maxgen=500,
                 F=0.1, Cr=0.1, **kwargs):
        ...

    def _solve(self, data: FJSPData) -> np.ndarray:
        """运行完整 DE 优化，返回最优编码

        核心流程（从原始代码提取）:
        1. 初始化种群 (调用 init_strategy)
        2. for gen in range(maxgen):
            a. mutation(): 以概率 F 对每个个体与 gbestpop 做 POX 交叉，
               再以概率 F 与随机异于自身的个体交叉
            b. cross_and_select(): 以概率 Cr 从中间种群继承机器编码，
               修复不可行解，贪心选择
            c. 更新 pbest, gbest
        3. 返回 gbestpop
        """
        ...

    def plan(self, obs: dict) -> dict:
        """JobSolver 接口实现

        1. data = FJSPData.from_obs(obs)
        2. best = self._solve(data)
        3. schedule = calculate_schedule(best, data)
        4. 转换为 machine_actions + transfer_requests 格式
        """
        ...
```

---

### Step 7：实现 PSO 求解器

```python
# algorithms/pso_solver.py

class PSOSolver(FJSPSolver):
    """粒子群优化求解器

    参数:
        init_strategy: "random" | "roulette" | "extreme"
        popsize: 种群规模 (默认 50)
        maxgen: 最大迭代次数 (默认 500)
        w: 惯性权重 (默认 0.9)
        lr: 加速因子 (默认 (2, 2))
    """

    def __init__(self, init_strategy="random", popsize=50, maxgen=500,
                 w=0.9, lr=(2, 2), **kwargs):
        ...

    def _solve(self, data: FJSPData) -> np.ndarray:
        """运行完整 PSO 优化，返回最优编码

        核心流程（从原始代码提取）:
        1. 初始化种群和速度
        2. for gen in range(maxgen):
            a. 速度更新: v = w*v + lr[0]*rand()*(pbest-pop) + lr[1]*rand()*(gbest-pop)
            b. 工序位置更新: pop += v 后排序映射恢复合法排列
            c. 机器位置更新: ceil(pop) 后修复越界和不可行解
            d. 更新 pbest, gbest
        3. 返回 gbestpop
        """
        ...

    def plan(self, obs: dict) -> dict:
        """JobSolver 接口实现（同 DESolver）"""
        ...
```

---

### Step 8：实现基类与工厂注册

```python
# algorithms/base_solver.py

class FJSPSolver:
    """所有 FJSP 求解器的基类"""

    def plan(self, obs: dict) -> dict:
        """
        在线调度接口

        输入 obs:
            obs["jobs"]:      List[Job]      所有工单
            obs["machines"]:  List[Machine]  所有机器

        输出:
            {
                "machine_actions": [
                    {"machine_id": int, "job_id": int, "op_id": int,
                     "start_time": float, "expected_end": float},
                    ...
                ],
                "transfer_requests": [
                    {"job_id": int, "op_id": int, "from_machine": int,
                     "to_machine": int, "ready_time": float},
                    ...
                ]
            }
        """
        raise NotImplementedError
```

```python
# algorithms/__init__.py

from .base_solver import FJSPSolver
from .de_solver import DESolver
from .pso_solver import PSOSolver
from .drl_solver import DRLSolver

# 工厂注册（后续对接 SkyEngine）
SOLVER_REGISTRY = {
    "de_random":    lambda **kw: DESolver(init_strategy="random", **kw),
    "de_roulette":  lambda **kw: DESolver(init_strategy="roulette", **kw),
    "de_extreme":   lambda **kw: DESolver(init_strategy="extreme", **kw),
    "pso_random":   lambda **kw: PSOSolver(init_strategy="random", **kw),
    "pso_roulette": lambda **kw: PSOSolver(init_strategy="roulette", **kw),
    "pso_extreme":  lambda **kw: PSOSolver(init_strategy="extreme", **kw),
    "drl_multippo": lambda **kw: DRLSolver(**kw),
}

def create_solver(name: str, **kwargs) -> FJSPSolver:
    if name not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver: {name}. Available: {list(SOLVER_REGISTRY.keys())}")
    return SOLVER_REGISTRY[name](**kwargs)
```

---

### Step 9：编写验证脚本

**目标**：用原始数据验证重构后算法的正确性（结果应与原始代码一致或更优）。

```python
# test_reproduce.py

"""验证重构后的求解器能复现原始代码的结果"""

from algorithms import create_solver
from algorithms.data_adapter import FJSPData

def test_de_first():
    data = FJSPData.from_txt("data/data_first.txt", workpiece=10, process=5, machine=6)
    solver = DESolver(init_strategy="random", popsize=50, maxgen=500, F=0.1, Cr=0.1)
    best = solver._solve(data)
    makespan = calculate_makespan(best, data)
    print(f"DE(random) makespan: {makespan}")

def test_all_combinations():
    """测试所有 6 种组合 × 3 个数据集"""
    datasets = [
        ("data/data_first.txt", 10, 5, 6),
        ("data/data_second.txt", 20, 10, 10),
        ("data/data_third.txt", 20, 20, 15),
    ]
    solvers = ["de_random", "de_roulette", "de_extreme",
               "pso_random", "pso_roulette", "pso_extreme"]

    for filepath, j, p, m in datasets:
        data = FJSPData.from_txt(filepath, j, p, m)
        for name in solvers:
            solver = create_solver(name, maxgen=500, popsize=50)
            best = solver._solve(data)
            makespan = calculate_makespan(best, data)
            print(f"[{name}] {filepath}: makespan = {makespan}")
```

---

### Step 10：plan() 输出转换

**目标**：将内部编码的调度结果转换为 SkyEngine 要求的 `machine_actions` 和 `transfer_requests` 格式。

```python
# algorithms/output_builder.py

def build_plan_output(best: np.ndarray, data: FJSPData) -> dict:
    """从最优编码生成 plan() 的返回值

    1. 用 calculate_schedule() 获取每个工序的 (machine, start, end)
    2. 遍历所有工序，构建 machine_actions:
       - machine_id: 分配的机器
       - job_id / op_id: 工序标识
       - start_time: 开始时间
       - expected_end: 结束时间
    3. 遍历所有工件，如果连续两道工序不在同一机器，
       构建 transfer_requests:
       - from_machine: 前一道工序的机器
       - to_machine: 当前工序的机器
       - ready_time: 前一道工序的完成时间
    """
    machine_actions = []
    transfer_requests = []

    schedule = calculate_schedule(best, data)
    for (job_id, op_id), info in schedule.items():
        machine_actions.append({
            "machine_id": info["machine"],
            "job_id": job_id,
            "op_id": op_id,
            "start_time": info["start"],
            "expected_end": info["end"],
        })

    # 构建搬运请求
    for job_id in range(data.workpiece):
        for op_id in range(1, data.process):
            prev = schedule[(job_id, op_id - 1)]
            curr = schedule[(job_id, op_id)]
            if prev["machine"] != curr["machine"]:
                transfer_requests.append({
                    "job_id": job_id,
                    "op_id": op_id,
                    "from_machine": prev["machine"],
                    "to_machine": curr["machine"],
                    "ready_time": prev["end"],
                })

    return {
        "machine_actions": machine_actions,
        "transfer_requests": transfer_requests,
    }
```

---

## 五、执行顺序总结

### A. PSO/DE 部分（Step 1-10）

```
Step 1  data_adapter.py       数据适配（txt / obs → 统一内部格式）
Step 2  encoding.py           编码/解码（handle, encode）
Step 3  fitness.py            适应度计算（calculate_makespan, calculate_schedule）
Step 4  initialization.py     三种初始化策略（random, roulette, extreme）
Step 5  crossover.py          POX 交叉算子
Step 6  de_solver.py          DE 求解器
Step 7  pso_solver.py         PSO 求解器
Step 8  base_solver.py + __init__.py   基类与工厂注册
Step 9  test_reproduce.py     验证复现结果
Step 10 output_builder.py     plan() 输出转换
```

**依赖关系**：Step 1-5 无相互依赖可并行编写 → Step 6-7 依赖 1-5 → Step 8 依赖 6-7 → Step 9-10 依赖全部。

### B. DRL 部分（Step 11-15）

```
Step 11 drl/env.py            重构 FJSP 环境（适配非均匀工序数 + SkyEngine obs）
Step 12 drl/models/           迁移 GIN + 双 Actor 网络
Step 13 drl_solver.py         DRL 求解器（推理 + plan 接口）
Step 14 test_drl.py           验证 DRL 复现结果
Step 15 drl/ppo.py            Multi-PPO 训练逻辑（可选，如需重训练）
```

**依赖关系**：Step 8（base_solver）先完成 → Step 11-12 可并行 → Step 13 依赖 11-12 → Step 14 依赖 13 → Step 15 独立。

---

## 六、DRL 详细实施步骤

### Step 11：重构 FJSP 析取图环境

**目标**：将 `FJSP_Env.py` 重构为支持非均匀工序数、可从 SkyEngine obs 初始化的环境。

```python
# algorithms/drl/env.py

class FJSPGraphEnv:
    """基于析取图的 FJSP 环境，支持非均匀工序数

    原始代码 FJSP_Env.py 的关键改造:
    1. number_of_tasks 不再固定为 n_j * n_m，而是根据实际工序数计算
    2. 邻接矩阵大小适配实际工序总数
    3. 支持 from_obs() 构建
    """

    def __init__(self, n_j: int, n_m: int, ops_per_job: list):
        """
        Args:
            n_j: 工件数
            n_m: 机器数
            ops_per_job: 每个工件的工序数列表 [op_count_j1, op_count_j2, ...]
        """
        self.n_j = n_j
        self.n_m = n_m
        self.ops_per_job = ops_per_job
        self.n_tasks = sum(ops_per_job)  # 实际总工序数（非 n_j * n_m）
        ...

    def reset(self, dur_matrix: np.ndarray):
        """初始化环境状态

        Args:
            dur_matrix: shape (n_j, max_ops, n_m)，0 或负数表示不可用

        核心逻辑（从原始 FJSP_Env.py reset() 提取）:
        1. 构建邻接矩阵: 对角线(自身) + 下对角线(合取弧，同一工件内前序)
           - first_col / last_col 需按每个工件的实际工序数计算
        2. 初始化特征: LBm = cumsum(min_proc_time) / et_normalize_coef
        3. 初始化候选集 omega: 每个工件的第一道工序
        4. 初始化 mask_mch: 标记不可用机器
        5. 不可用机器的 dur 替换为同工序可用机器的平均值
        """
        ...

    @classmethod
    def from_obs(cls, obs: dict) -> "FJSPGraphEnv":
        """从 SkyEngine obs 构建

        遍历 obs["jobs"] 构建:
        - ops_per_job: [len(job.ops) for job in jobs]
        - dur_matrix: 根据 Operation.machine_options 和 proc_time 填充
        """
        ...

    def step(self, action: int, mch_action: int):
        """执行一步调度

        核心逻辑（从原始 FJSP_Env.py step() 提取）:
        1. 记录: finished_mark, partial_sol_sequence, m[job][op] = mch_action
        2. Permissible Left Shift: 计算最早可插入位置 startTime_a
        3. 更新 omega / mask: 如果当前工序不是该工件最后一道，omega 前进；否则 mask 置 1
        4. 更新 LBm: calEndTimeLBm() 重新计算下界
        5. 更新邻接矩阵: 添加析取弧（与同机器上前后工序的连接）
        6. 计算奖励: reward = -(LBm.max() - max_endTime)，若为 0 则给正奖励
        """
        ...
```

**原始环境核心数据结构（需保持）**：

| 变量 | 形状 | 说明 |
|------|------|------|
| `adj` | (batch, n_tasks, n_tasks) | 邻接矩阵（稀疏） |
| `fea` | (batch, n_tasks, 2) | 节点特征 [LBm/norm, finished_mark] |
| `omega` | (batch, n_j) | 每个工件的当前候选工序 ID |
| `mask` | (batch, n_j) | 工件完成掩码 |
| `mask_mch` | (batch, n_tasks, n_m) | 不可用机器掩码 |
| `mchsStartTimes` | (batch, n_m, n_tasks) | 每台机器上各工序的开始时间 |
| `mchsEndTimes` | (batch, n_m, n_tasks) | 每台机器上各工序的结束时间 |
| `opIDsOnMchs` | (batch, n_m, n_tasks) | 每台机器上加工的工序 ID 序列 |
| `dur` | (batch, n_j, max_ops, n_m) | 加工时间矩阵 |

---

### Step 12：迁移 GIN + 双 Actor 网络

**目标**：将模型文件迁移到统一目录，解除对全局 `configs` 的依赖。

```python
# algorithms/drl/models/gin.py
# 从 graphcnn_congForSJSSP.py 迁移

class GraphCNN(nn.Module):
    """Graph Isomorphism Network

    原始代码不变，仅移除对全局 Params.configs 的依赖。
    所有参数通过 __init__() 传入。
    """
    def __init__(self, num_layers, num_mlp_layers, input_dim,
                 hidden_dim, learn_eps, neighbor_pooling_type, device):
        ...

    def forward(self, x, graph_pool, padded_nei, adj):
        """
        输入:
            x: (batch * n_tasks, input_dim) 节点特征
            graph_pool: (batch, n_tasks) 稀疏图池化矩阵
            adj: (batch * n_tasks, n_tasks) 稀疏邻接矩阵
        输出:
            h_pooled: (batch, hidden_dim) 全局图嵌入
            h_nodes: (batch * n_tasks, hidden_dim) 节点嵌入
        """
        ...
```

```python
# algorithms/drl/models/actor.py
# 从 PPO_Actor.py 迁移

class JobActor(nn.Module):
    """工序选择 Actor

    核心前向逻辑（从原始 PPO_Actor.py Job_Actor 提取）:
    1. encoder(x, graph_pool, padded_nei, adj) → h_pooled, h_nodes
    2. 从 h_nodes 中 gather 候选工序特征 → candidate_feature
    3. 拼接 [candidate_feature, h_pooled, mch_pooled] → concateFea
    4. MLPActor 打分 → candidate_scores * 10
    5. mask 屏蔽已完成工件 → Softmax → 采样/贪心
    6. gather 获取选中工序的机器加工时间 → action_node
    7. gather 获取选中工序的节点嵌入 → action_feature
    """
    ...

class MchActor(nn.Module):
    """机器选择 Actor

    核心前向逻辑（从原始 PPO_Actor.py Mch_Actor 提取）:
    1. 拼接 [mch_time/norm, action_node/norm] → feature
    2. Linear(2 → hidden) + BatchNorm → action_node_embedding
    3. mean pool → pool（机器全局表征）
    4. 拼接 [action_node_emb, pool, hx] → concateFea
    5. MLPActor 打分 → mch_scores * 10
    6. mask 屏蔽不可用机器 → Softmax → pi_mch
    """
    ...
```

**关键改造点**：
- `n_j`, `n_m` 不再硬编码在网络 `__init__` 中，改为运行时通过输入 tensor shape 推断
- `et_normalize_coef` 从全局 configs 改为构造参数传入
- `device` 从 configs 改为参数传入

---

### Step 13：实现 DRL 求解器

**目标**：包装推理逻辑，实现 `plan()` 接口。

```python
# algorithms/drl_solver.py

class DRLSolver(FJSPSolver):
    """Multi-PPO DRL 求解器

    与 PSO/DE 求解器不同，DRL 天然是在线逐步决策的:
    - 每次调用 plan()，Agent 根据当前 obs 逐步选择工序和机器
    - 不需要离线优化，直接用训练好的策略网络推理

    参数:
        model_dir: 预训练模型目录路径
        n_j: 训练时的工件数（用于确定模型维度）
        n_m: 训练时的机器数
        hidden_dim: 隐层维度（默认 128）
        num_layers: GIN 层数（默认 3）
        device: "cuda" 或 "cpu"
    """

    def __init__(self, model_dir: str, n_j: int = 30, n_m: int = 20,
                 hidden_dim: int = 128, num_layers: int = 3,
                 device: str = "cpu", **kwargs):
        self.job_actor = JobActor(n_j, n_m, num_layers, ...)
        self.mch_actor = MchActor(n_j, n_m, num_layers, ...)
        # 加载预训练权重
        self.job_actor.load_state_dict(torch.load(f"{model_dir}/policy_job.pth"))
        self.mch_actor.load_state_dict(torch.load(f"{model_dir}/policy_mch.pth"))
        self.job_actor.eval()
        self.mch_actor.eval()

    def plan(self, obs: dict) -> dict:
        """在线调度：逐步推理直到所有工序调度完毕

        核心流程（从原始 validation.py 的 eval_model_bat 提取）:
        1. env = FJSPGraphEnv.from_obs(obs)
        2. state = env.reset(dur_matrix)
        3. machine_actions = []
        4. while not env.done():
            a. 构建输入 tensor: adj(sparse), fea, candidate, mask, mask_mch, dur
            b. job_actor(x, graph_pool, adj, candidate, mask, ..., greedy=True)
               → action(工序), action_node(加工时间), mask_mch_action, hx
            c. mch_actor(action_node, hx, mask_mch_action, mch_time)
               → pi_mch → mch_a = argmax(pi_mch)
            d. state, reward, done = env.step(action, mch_a)
            e. 从 env.mchsStartTimes/mchsEndTimes 提取本次调度的 start/end
            f. 追加到 machine_actions
        5. 构建 transfer_requests（同 output_builder 逻辑）
        6. 返回 {"machine_actions": ..., "transfer_requests": ...}
        """
        ...
```

**DRL 求解器与 PSO/DE 的 plan() 差异**：

| | PSO/DE | DRL |
|---|---|---|
| **plan() 内部** | 调用 `_solve()` 离线优化 → 解码结果 → 输出 | 直接逐步推理 → 每步收集 action → 输出 |
| **耗时** | 较长（需迭代优化） | 极快（单次前向传播 × n_steps） |
| **动态插单** | 需重新优化 | 天然支持（reset 新状态继续推理） |
| **质量** | 依赖迭代次数和种群规模 | 依赖训练质量 |

---

### Step 14：编写 DRL 验证脚本

**目标**：验证迁移后的 DRL 推理能复现原始代码的结果。

```python
# test_drl.py

"""验证 DRL 求解器的正确性"""

def test_random_instances():
    """在随机生成的实例上测试"""
    # 与原始 validation.py 对比:
    # 1. 用相同种子生成随机实例
    # 2. 用迁移后的 DRLSolver 推理
    # 3. 对比 makespan 与原始代码的结果
    ...

def test_realworld_instances():
    """在标准 benchmark 上测试"""
    # 与原始 validation_realWorld.py 对比:
    # 1. 加载 .fjs 文件 (Brandimarte, Hurink 等)
    # 2. 用 DataRead.getdata() 解析
    # 3. 转换为 dur_matrix
    # 4. DRLSolver.plan() 推理
    # 5. 对比 makespan 与原始结果
    ...

def test_obs_interface():
    """测试 SkyEngine obs 接口"""
    # 模拟构建 obs dict (含 jobs, machines)
    # 调用 DRLSolver.plan(obs)
    # 验证输出格式符合规范
    ...
```

---

### Step 15：Multi-PPO 训练逻辑迁移（可选）

**目标**：如需重新训练或微调，迁移训练代码。

```python
# algorithms/drl/ppo.py
# 从 PPOwithValue.py 迁移

class MultiPPO:
    """Multi-PPO 训练器

    核心逻辑（从原始 PPOwithValue.py PPO 类提取）:

    双 Actor + 双 Critic 架构:
    - policy_job / old_policy_job: Job_Actor（当前策略 / 旧策略）
    - policy_mch / old_policy_mch: Mch_Actor（当前策略 / 旧策略）

    训练主循环:
    1. 采样阶段:
       for batch in dataloader:
           env.reset(batch)
           while not done:
               action, _, _, action_node, _, mask_mch_action, hx = policy_job(...)
               pi_mch, pool = policy_mch(action_node, hx, mask_mch_action, mch_time)
               mch_a = sample(pi_mch)
               env.step(action, mch_a)
               memory.store(adj, fea, reward, action, mch_a, log_prob, ...)
           memory.rewards = discount_rewards(memory.rewards, gamma)

    2. 更新阶段 (k_epochs 轮):
       for epoch in range(k_epochs):
           # 计算新 log_prob 和 entropy
           entropy_job, v_job, log_a_job, ... = policy_job(old_policy=False, ...)
           entropy_mch, ... = policy_mch(...)

           # PPO 裁剪损失
           ratio = exp(log_a_new - log_a_old)
           surr1 = ratio * advantage
           surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
           loss_job = -min(surr1, surr2).mean() + 0.5 * v_loss - 0.01 * entropy_job

           # 反向传播
           optimizer_job.zero_grad(); loss_job.backward(); optimizer_job.step()
           optimizer_mch.zero_grad(); loss_mch.backward(); optimizer_mch.step()

       # 同步旧策略
       old_policy_job.load_state_dict(policy_job.state_dict())
       old_policy_mch.load_state_dict(policy_mch.state_dict())

    3. 验证 & 保存:
       每 20 个 batch 用贪心策略验证，保存最优模型
    """
    ...
```

**数据生成**（从 `uniform_instance.py` 迁移）：

```python
# algorithms/drl/data_utils.py

def generate_random_instance(n_j: int, n_m: int, low: int = -99, high: int = 99):
    """生成随机 FJSP 实例

    原始逻辑（从 uniform_instance.py uni_instance_gen 提取）:
    1. 生成 shape (n_j, n_m, n_m-1) 的随机数 (low ~ high)
    2. 生成 shape (n_j, n_m, 1) 的正随机数 (1 ~ high)，保证每道工序至少 1 台可用机器
    3. 拼接 → shape (n_j, n_m, n_m)
    4. 每行 permute_rows 打乱（保证每行恰好 1 个正值）
    """
    ...
```

---

## 七、注意事项

1. **复现优先**：先确保重构后算法在相同数据集上能复现原始结果（或更优），再做 SkyEngine 对接
2. **随机种子**：验证时固定 `np.random.seed()` 以便对比
3. **在线适配**：PSO/DE 原始算法是离线的（一次性跑完所有优化），`plan()` 要求在线逐步调用。初期可先用离线优化结果作为 `plan()` 的返回，后续再考虑增量优化
4. **`from_obs` 适配**：`Operation.machine_options` 是可选机器列表，需要与 `Operation.proc_time` 配合构建 `proc_time` 矩阵
5. **transfer_requests**：原始代码没有考虑 AGV 搬运，此部分在 Step 10 根据调度结果推断生成
6. **DRL 非均匀工序**：原始环境假设所有工件恰好 `n_m` 道工序，SkyEngine 中工件工序数可不同。Step 11 需改造 `number_of_tasks`、`first_col`、`last_col`、`adj` 的构建逻辑
7. **DRL 模型维度**：原始网络在固定 n_j=30, n_m=20 上训练，加载到不同规模的实例时，GIN 部分可泛化（权重与节点数无关），但 `candidate` gather 操作依赖 `n_j`，需确认兼容性
8. **DRL 不可用机器**：原始代码在 `reset()` 中将 `dur <= 0` 的机器标记为不可用（`mask_mch`），并将该位置的时间替换为可用机器的平均值。SkyEngine 传入的 `proc_time` 需按同样逻辑处理
9. **DRL 设备管理**：原始代码硬编码 `.cuda()`，需统一为参数化的 `device`
10. **DRL 训练可选**：如果已有预训练模型且满足需求，Step 15 可跳过。仅在做 fine-tune 或适应新规模时需要
