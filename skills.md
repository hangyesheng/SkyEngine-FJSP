# FJSP 算法接入 Skills

将一个任意 FJSP 求解算法接入 SkyEngine 的完整流程。以 DE / PSO 为已验证的参考实现。

---

## 流程总览

```
原始算法代码
    │
    ▼
┌──────────────────────────┐
│ 1. 判断在线 / 离线        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 2. 重构为统一 Solver 类    │  ← 离线算法需要
│    (离线优化, get_schedule) │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 3. 编写 OnlineSolver       │  ← 核心桥接层
│    (plan() 逐时间步输出)    │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 4. 编写 HTTP Server        │  ← 远程调用
│    (Flask, obs 序列化)      │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 5. Dockerfile + Compose    │  ← 容器化部署
└──────────────────────────┘
```

---

## Step 1：判断算法类型

### 1.1 什么是在线 / 离线

| 类型 | 特征 | 典型算法 |
|------|------|---------|
| **离线** | 一次性完成所有优化，输出完整调度方案 | DE, PSO, GA, OR-Tools |
| **在线** | 逐步做出决策，天然支持逐步调用 | DRL (PPO, DQN), 贪心启发式 |

### 1.2 判断标准

看算法的核心循环：

```
离线算法典型结构:
  初始化种群 / 解
  for iter in range(maxgen):      ← 迭代优化
      交叉 / 变异 / 更新
      评估适应度
  输出最优解                       ← 一次性出结果

在线算法典型结构:
  state = env.reset()
  while not done:                  ← 逐步决策
      action = model(state)        ← 每步调用一次
      state, reward, done = env.step(action)
```

### 1.3 对接入方式的影响

- **离线算法** → 需要 Step 2（重构为类）+ Step 3（OnlineSolver 包装离线结果，逐时间步吐出）
- **在线算法** → 跳过 Step 2，直接在 Step 3 中逐步调用模型推理

---

## Step 2：重构为统一 Solver 类（离线算法）

### 2.1 目标

将脚本式代码重构为可被外部调用的类，提供三个核心方法：

```python
class XxxSolver:
    def __init__(self, data, n_jobs, n_ops, n_machines, **params):
        """data: shape (n_jobs*n_ops, n_machines) 的 str 矩阵，'-' 表示不可用"""

    def solve(self) -> Tuple[np.ndarray, float, list]:
        """运行优化，返回 (最优编码, makespan, 收敛历史)"""

    def get_schedule(self, x) -> List[dict]:
        """从编码解码为完整调度方案"""
```

### 2.2 数据格式

算法内部统一使用 `proc_time: np.ndarray, shape (total_process, n_machines), dtype=int`：
- 正数 = 加工时间
- `-1` = 不可用

从 txt 文件读取时：
```python
@staticmethod
def _parse_data(data):
    """str 矩阵 → int ndarray, '-' → -1"""
    raw = np.array(data, dtype=str)
    result = np.zeros(raw.shape, dtype=int)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = raw[i, j].strip()
            result[i, j] = -1 if val == "-" else int(val)
    return result
```

### 2.3 编码方案（两段式）

```
个体编码 x: shape (total_process * 2,)

前半段 x[:total_process]  — 工序编码
  值为 1~n_jobs 的工件号序列
  第 k 次出现表示该工件的第 k 道工序
  例: [2, 1, 1, 3, 2, 3] → J2-O1, J1-O1, J1-O2, J3-O1, J2-O2, J3-O2

后半段 x[total_process:]  — 机器编码
  按 (job_id-1)*n_ops + (op_seq-1) 索引
  值为 1~n_machines 的机器号
  例: x[2] = 3 → J1-O2 分配到 M3
```

### 2.4 适应度计算（所有元启发式共用）

```python
def calculate(self, x):
    Tm = np.zeros(self.n_machines)               # 每台机器完工时间
    Te = np.zeros((self.n_jobs, self.n_ops))     # 每个工序完成时间
    ops = self._decode_ops(x)

    for i in range(self.total_process):
        job_id, op_seq = ops[i]
        mch_idx = int(x[self.total_process + (job_id-1)*self.n_ops + (op_seq-1)]) - 1
        proc_row = (job_id-1) * self.n_ops + (op_seq-1)
        proc_time = self.proc_time[proc_row, mch_idx]

        if op_seq == 1:
            Tm[mch_idx] += proc_time
        else:
            Tm[mch_idx] = max(Te[job_id-1, op_seq-2], Tm[mch_idx]) + proc_time
        Te[job_id-1, op_seq-1] = Tm[mch_idx]

    return float(Tm.max())
```

### 2.5 get_schedule（输出完整调度详情）

```python
def get_schedule(self, x):
    Tm = np.zeros(self.n_machines)
    Te = np.zeros((self.n_jobs, self.n_ops))
    Ts = np.zeros((self.n_jobs, self.n_ops))  # 开始时间
    ops = self._decode_ops(x)

    for i in range(self.total_process):
        job_id, op_seq = ops[i]
        mch_idx = int(x[self.total_process + (job_id-1)*self.n_ops + (op_seq-1)]) - 1
        proc_row = (job_id-1) * self.n_ops + (op_seq-1)
        proc_time = self.proc_time[proc_row, mch_idx]

        if op_seq == 1:
            Ts[job_id-1, op_seq-1] = Tm[mch_idx]
            Tm[mch_idx] += proc_time
        else:
            start = max(Te[job_id-1, op_seq-2], Tm[mch_idx])
            Ts[job_id-1, op_seq-1] = start
            Tm[mch_idx] = start + proc_time
        Te[job_id-1, op_seq-1] = Tm[mch_idx]

    schedule = []
    for j in range(self.n_jobs):
        for o in range(self.n_ops):
            mch = int(x[self.total_process + j * self.n_ops + o]) - 1
            schedule.append({
                "job_id": j, "op_id": o, "machine": mch,
                "start_time": float(Ts[j, o]),
                "end_time": float(Te[j, o]),
            })
    return schedule
```

### 2.6 参考实现

- DE: `FJSP-master/DE_solver/de_solver.py`
- PSO: `FJSP-master/PSO_solver/pso_solver.py`

---

## Step 3：编写 OnlineSolver

### 3.1 核心思路

**离线算法**：首次 `plan()` 时执行完整离线优化，将结果缓存到两个列表中；后续每次 `plan()` 被调用时，根据 `time_stamp` 逐步吐出当前时刻应执行的 actions 和 transfers。

```
首次 plan() ──→ 离线求解 ──→ 缓存全部结果
                                │
后续 plan() ──→ time_stamp++ ──→ pop 当前时刻的 items ──→ 返回
```

**在线算法**：每次 `plan()` 直接执行一步推理，天然适配。

### 3.2 类模板

```python
class OnlineXxxSolver:
    def __init__(self, **params):
        self.initialized = False
        self.time_stamp = 0
        self.task_idx = 0
        self._machine_actions = []       # 按 start_time 排序
        self._transfer_requests = []     # 按 ready_time 排序

    def _obs_to_data(self, obs):
        """从 SkyEngine obs 提取加工时间矩阵

        obs["jobs"]: List[Job], 每个 Job 有 .ops: List[Operation]
            Operation 有 .op_id, .machine_options, .proc_time
        obs["machines"]: List[Machine]

        Returns: data, n_jobs, n_ops, n_machines
        """
        jobs = obs["jobs"]
        machines = obs["machines"]
        n_machines = len(machines)
        n_jobs = len(jobs)
        n_ops = max(len(job.ops) for job in jobs)

        data = np.full((n_jobs * n_ops, n_machines), "-", dtype=str)
        for job in jobs:
            jid = job.job_id
            for op in job.ops:
                row = jid * n_ops + op.op_id
                for mid in range(n_machines):
                    if mid in op.machine_options:
                        data[row, mid] = str(int(op.proc_time))
        return data, n_jobs, n_ops, n_machines

    def _solve_offline(self, obs):
        """首次调用: 离线优化 → 拆分为 actions + transfers"""
        data, n_jobs, n_ops, n_machines = self._obs_to_data(obs)

        solver = XxxSolver(data, n_jobs, n_ops, n_machines, ...)
        best_x, best_makespan, history = solver.solve()
        schedule = solver.get_schedule(best_x)

        # ── 拆分 machine_actions ──
        self._machine_actions = []
        for item in schedule:
            self._machine_actions.append({
                "machine_id": item["machine"],
                "job_id": item["job_id"],
                "op_id": item["op_id"],
                "start_time": item["start_time"],
                "expected_end": item["end_time"],
            })

        # ── 生成 transfer_requests ──
        self._transfer_requests = []
        for j in range(n_jobs):
            for o in range(1, n_ops):
                prev = schedule[j * n_ops + (o - 1)]
                curr = schedule[j * n_ops + o]
                if prev["machine"] != curr["machine"]:
                    self._transfer_requests.append({
                        "job_id": j, "op_id": o,
                        "from_machine": prev["machine"],
                        "to_machine": curr["machine"],
                        "ready_time": prev["end_time"],
                    })

        self._machine_actions.sort(key=lambda x: x["start_time"])
        self._transfer_requests.sort(key=lambda x: x["ready_time"])

    def _create_routing_task(self, task_dict):
        """转为 SkyEngine RoutingTask 兼容格式"""
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

    def plan(self, obs: dict) -> dict:
        """在线调度主接口，每时间步被调用一次"""
        self.time_stamp += 1

        if not self.initialized:
            self._solve_offline(obs)
            self.initialized = True

        current_time = float(self.time_stamp)

        # pop 当前时刻的 actions
        ready_actions = []
        while (self._machine_actions
               and self._machine_actions[0]["start_time"] <= current_time + 1e-6):
            ready_actions.append(self._machine_actions.pop(0))

        # pop 当前时刻的 transfers
        ready_transfers = []
        while (self._transfer_requests
               and self._transfer_requests[0]["ready_time"] <= current_time + 1e-6):
            tr = self._transfer_requests.pop(0)
            ready_transfers.append(self._create_routing_task(tr))

        return {
            "machine_actions": ready_actions,
            "transfer_requests": ready_transfers,
        }
```

### 3.3 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| time_stamp 起始值 | 0，plan() 中先 `+= 1` | 首次 plan() 时 time_stamp=1，给离线求解留出时间步 |
| 判断时机 | `start_time <= current_time + 1e-6` | 浮点容差，避免因精度丢失漏掉 t=0 的任务 |
| transfer 生成 | 同工件连续工序不同机器时生成 | 匹配 SkyEngine 的 AGV 搬运模型 |
| RoutingTask 格式 | `source=(machine_id, 0)` | SkyEngine 位置用 (machine, slot) 二元组 |
| pop(0) | 从有序列表头部弹出 | 已排序，保证时间顺序 |

### 3.4 参考实现

- DE: `FJSP-master/DE_solver/online_de_solver.py`
- PSO: `FJSP-master/PSO_solver/online_pso_solver.py`

---

## Step 4：编写 HTTP Server

### 4.1 目录结构

```
<算法名>_solver/
├── xxx_solver.py           # Step 2: 离线 Solver 类
├── online_xxx_solver.py    # Step 3: OnlineSolver 包装
├── xxx_solver_server.py    # Step 4: HTTP Server
└── Dockerfile              # Step 5: 容器化
```

### 4.2 Server 模板

```python
"""
Xxx Solver HTTP Server — 远程调用接口
生命周期: /init → /plan (×N) → /reset
"""
import sys, os
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_xxx_solver import OnlineXxxSolver

app = Flask(__name__)
solver: OnlineXxxSolver = None


# ── 序列化 / 反序列化 ────────────────────────────────────

class Op:
    def __init__(self, op_id, machine_options, proc_time):
        self.op_id = op_id
        self.machine_options = machine_options
        self.proc_time = proc_time
        self.assigned_machine = None
        self.status = "PENDING"

class Job:
    def __init__(self, job_id, ops, release=0.0, due=None):
        self.job_id = job_id
        self.ops = ops
        self.release = release
        self.due = due
        self.completion_time = 0.0

class Machine:
    def __init__(self, mid, loc=(0, 0)):
        self.id = mid
        self.location = loc
        self.current_op = None
        self.total_work_time = 0


def deserialize_obs(obs_json: dict) -> dict:
    """JSON → Python 对象 (供 OnlineSolver 消费)"""
    jobs = []
    for j in obs_json.get("jobs", []):
        ops = [Op(op_id=o["op_id"],
                  machine_options=o["machine_options"],
                  proc_time=o["proc_time"])
               for o in j.get("ops", [])]
        jobs.append(Job(job_id=j["job_id"], ops=ops,
                        release=j.get("release", 0.0), due=j.get("due")))
    machines = [Machine(mid=m["id"], loc=tuple(m.get("location", [0, 0])))
                for m in obs_json.get("machines", [])]
    return {"jobs": jobs, "machines": machines}


def _build_config(data: dict) -> dict:
    """从请求中提取 solver 参数，附默认值"""
    cfg = data.get("config", {})
    return {
        # 按具体算法填写，示例:
        "init_strategy": cfg.get("init_strategy", "extreme"),
        "popsize": cfg.get("popsize", 30),
        "maxgen": cfg.get("maxgen", 100),
        "seed": cfg.get("seed", 42),
        # ...
    }


def _serialize_result(result: dict, sv) -> dict:
    """plan() 返回值 → JSON-safe (tuple → list)"""
    tr = []
    for t in result["transfer_requests"]:
        tr.append({
            "task_id": t["task_id"],
            "job_id": t["job_id"],
            "op_id": t["op_id"],
            "source": list(t["source"]),          # tuple → list
            "destination": list(t["destination"]),
            "candidate_machines": t["candidate_machines"],
            "ready_time": t["ready_time"],
        })
    return {
        "machine_actions": result["machine_actions"],
        "transfer_requests": tr,
        "time_stamp": sv.time_stamp,
        "remaining_actions": len(sv._machine_actions),
        "remaining_transfers": len(sv._transfer_requests),
    }


# ── API 路由 ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "solver": "<算法名>",
        "initialized": solver is not None and solver.initialized,
    })

@app.route("/init", methods=["POST"])
def init_solver():
    """显式初始化: 离线求解 + 缓存结果"""
    global solver
    data = request.get_json(force=True)
    obs = deserialize_obs(data.get("obs", {}))
    config = _build_config(data)
    solver = OnlineXxxSolver(**config)
    solver.plan(obs)
    makespan = max((a["expected_end"] for a in solver._machine_actions), default=0)
    return jsonify({
        "status": "initialized",
        "makespan": makespan,
        "total_actions": len(solver._machine_actions),
        "total_transfers": len(solver._transfer_requests),
    })

@app.route("/plan", methods=["POST"])
def plan():
    """逐步输出 (首次调用自动初始化)"""
    global solver
    data = request.get_json(force=True)

    if solver is None:
        obs = deserialize_obs(data.get("obs", {}))
        config = _build_config(data)
        solver = OnlineXxxSolver(**config)
        result = solver.plan(obs)
    else:
        obs_json = data.get("obs", {})
        obs = deserialize_obs(obs_json) if obs_json else {"jobs": [], "machines": []}
        result = solver.plan(obs)

    return jsonify(_serialize_result(result, solver))

@app.route("/reset", methods=["POST"])
def reset():
    global solver
    solver = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
```

### 4.3 API 接口汇总

| 端点 | 方法 | 请求体 | 返回 |
|------|------|--------|------|
| `/health` | GET | - | `{status, solver, initialized}` |
| `/init` | POST | `{obs, config}` | `{status, makespan, total_actions, total_transfers}` |
| `/plan` | POST | `{}` 或 `{obs, config}` (首次) | `{machine_actions, transfer_requests, time_stamp, remaining_*}` |
| `/reset` | POST | - | `{status}` |

### 4.4 obs JSON 格式

```json
{
  "jobs": [
    {
      "job_id": 0,
      "ops": [
        {"op_id": 0, "machine_options": [0, 2, 4], "proc_time": 3.0},
        {"op_id": 1, "machine_options": [1, 3],    "proc_time": 5.0}
      ],
      "release": 0.0,
      "due": null
    }
  ],
  "machines": [
    {"id": 0, "location": [0, 0]},
    {"id": 1, "location": [1, 0]}
  ]
}
```

### 4.5 端口分配

| 算法 | 默认端口 |
|------|---------|
| DE | 5001 |
| PSO | 5002 |
| DRL | 5003 |
| 新算法 | 5004+ |

### 4.6 参考实现

- DE: `FJSP-master/DE_solver/de_solver_server.py`
- PSO: `FJSP-master/PSO_solver/pso_solver_server.py`

---

## Step 5：Dockerfile + Compose

### 5.1 Dockerfile 模板

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy flask

# 只 COPY 该算法目录下的 3 个文件
COPY xxx_solver.py .
COPY online_xxx_solver.py .
COPY xxx_solver_server.py .

EXPOSE <端口>

CMD ["python", "xxx_solver_server.py", "--host", "0.0.0.0", "--port", "<端口>"]
```

### 5.2 docker-compose.yaml 中添加 service

```yaml
  xxx:
    build:
      context: FJSP-master/<算法名>_solver
      dockerfile: Dockerfile
    image: skyengine-fjsp-xxx:latest
    ports:
      - "<端口>:<端口>"
```

### 5.3 构建与运行

```bash
# 单独构建
cd FJSP-master/<算法名>_solver && docker build -t skyengine-fjsp-xxx .

# 通过 compose
docker compose build xxx

# 运行
docker compose up xxx
```

### 5.4 参考实现

- DE Dockerfile: `FJSP-master/DE_solver/Dockerfile`
- PSO Dockerfile: `FJSP-master/PSO_solver/Dockerfile`
- docker-compose: `docker-compose.yaml`

---

## 快速检查清单

新算法接入前，逐项确认：

- [ ] **判断类型**: 离线 or 在线？离线需要 Step 2，在线跳过
- [ ] **Solver 类**: `__init__` 接收 data 矩阵 + 规模参数，`solve()` 返回最优编码，`get_schedule()` 返回调度详情
- [ ] **OnlineSolver**: 首次 `plan()` 触发离线求解（或逐步推理），后续按 time_stamp 吐出
- [ ] **obs 反序列化**: Server 中的 `deserialize_obs()` 正确还原 Op/Job/Machine 对象
- [ ] **transfer_requests**: 同工件连续工序不同机器时生成搬运请求
- [ ] **RoutingTask 格式**: source/destination 为 tuple，序列化时转 list
- [ ] **端口不冲突**: 检查 docker-compose 中的端口分配
- [ ] **Dockerfile**: 只 COPY 必要的 3 个 .py 文件
- [ ] **测试**: 用 mock obs 完成完整生命周期测试 (init → plan×N → reset)

---

## 完整目录结构参考

```
FJSP/
├── skills.md                              # 本文档
├── readme.md                              # 接口规范
├── step.md                                # 详细实施步骤
├── docker-compose.yaml                    # 统一编排
├── dockerfiles/
│   ├── base.dockerfile                    # 基础镜像
│   ├── DE.dockerfile                      # ← 已弃用，改用各算法目录内 Dockerfile
│   └── PSO.dockerfile
│
├── FJSP-master/                           # DE + PSO 原始代码 + 重构
│   ├── data/
│   │   ├── data_first.txt                 # J10P5M6
│   │   ├── data_second.txt                # J20P10M10
│   │   └── data_third.txt                 # J20P20M15
│   ├── test_online_solvers.py             # 离线测试 (mock obs Python 对象)
│   ├── test_servers.py                    # 在线测试 (HTTP 请求)
│   │
│   ├── DE_solver/
│   │   ├── de_solver.py                   # Step 2: DESolver 类
│   │   ├── online_de_solver.py            # Step 3: OnlineDESolver
│   │   ├── de_solver_server.py            # Step 4: Flask HTTP Server
│   │   └── Dockerfile                     # Step 5: 容器化
│   │
│   └── PSO_solver/
│       ├── pso_solver.py                  # Step 2: PSOSolver 类
│       ├── online_pso_solver.py           # Step 3: OnlinePSOSolver
│       ├── pso_solver_server.py           # Step 4: Flask HTTP Server
│       └── Dockerfile                     # Step 5: 容器化
│
├── End-to-end-DRL-for-FJSP-main/          # DRL 原始代码
│   └── ...
│
└── data/
    └── fjsp-master/
        ├── convert.py                     # txt → JSON 转换
        ├── J10P5M6.json
        ├── J20P10M10.json
        └── J20P20M15.json
```
