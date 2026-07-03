# FJSP 柔性作业车间调度算法

复现不同的 FJSP（Flexible Job Shop Scheduling Problem）算法，统一 HTTP 接口，Docker 打包后作为黑盒服务融入 SkyEngine。

## 已实现的算法

| 算法 | 镜像 | 类型 | 需要 GPU | 说明 |
|------|------|------|----------|------|
| CP-SAT (OR-Tools) | `skyengine-fjsp-best` | 精确求解 | 否 | 最优解基准，速度慢 |
| DE | `skyengine-fjsp-de` | 元启发式 | 否 | 差分进化 |
| PSO | `skyengine-fjsp-pso` | 元启发式 | 否 | 粒子群优化 |
| DRL (Multi-PPO) | `skyengine-fjsp-drl` | 深度强化学习 | **是** | 两阶段 GNN+PPO |
| GraphGRPO | `skyengine-fjsp-graphgrpo` | 深度强化学习 | 否 | GNN + GRPO（无 critic），权重随仓库提供 |

## 项目结构

```
FJSP/
├── docker-compose.yaml          # 编排所有算法镜像
├── dockerfiles/                  # 各算法 Dockerfile
│   ├── fjsp.dockerfile          #   基础镜像 (numpy + flask)
│   ├── de.dockerfile            #   DE
│   ├── pso.dockerfile           #   PSO
│   ├── best.dockerfile          #   CP-SAT
│   ├── drl.dockerfile           #   DRL (PyTorch + CUDA)
│   └── test.dockerfile          #   黑盒测试
├── offline_test.py              # 黑盒测试脚本
├── data/                         # 测试数据集
│   ├── fjsp-master/             #   自定义实例 (J10P5M6 等)
│   └── fjsp-instances-main/    #   标准 benchmark
├── FJSP-master/                  # DE / PSO 源码
│   ├── DE_solver/
│   └── PSO_solver/
├── OR-solver/                    # CP-SAT 源码
├── End-to-end-DRL-for-FJSP-main/ # DRL 源码
├── GraphGRPO/                    # GraphGRPO 源码 + 预训练权重 (agent_model.pt)
└── readme.md
```

## HTTP API 接口

所有 FJSP 求解器遵循相同的 REST API，监听端口 **8002**。

### `GET /health` — 健康检查

**响应**：
```json
{
  "status": "ok",
  "solver": "DE",
  "initialized": false
}
```

### `POST /init` — 初始化并离线求解

接收问题实例，执行离线优化，生成完整调度方案。

**请求**：
```json
{
  "obs": {
    "jobs": [
      {
        "job_id": 0,
        "ops": [
          {
            "op_id": 0,
            "machine_options": [0, 1, 2],
            "proc_times": {"0": 3.0, "1": 6.0, "2": 2.0}
          }
        ]
      }
    ],
    "machines": [
      {"id": 0, "location": [0, 0]},
      {"id": 1, "location": [1, 0]}
    ]
  },
  "config": { ... }
}
```

**obs 字段说明**：
- `jobs`: 工单列表，每个 job 含有序工序列表 `ops`
- `ops[].machine_options`: 该工序可选的机器 ID 列表（体现柔性）
- `ops[].proc_times`: `{机器ID: 加工时间}` 的映射（key 为字符串，JSON 序列化后 int key 会变 string）
- `machines`: 机器列表，含 ID 和物理位置

**响应**：
```json
{
  "status": "initialized",
  "makespan": 16.0,
  "total_actions": 40,
  "total_transfers": 40
}
```

### `POST /plan` — 逐步获取调度结果

每步返回当前时刻应执行的机器动作和搬运请求。

**请求**（首次可自动触发 init）：
```json
{}
```

**响应**：
```json
{
  "machine_actions": [
    {
      "machine_id": 0,
      "job_id": 1,
      "op_id": 2,
      "start_time": 10.0,
      "expected_end": 18.0
    }
  ],
  "transfer_requests": [
    {
      "job_id": 0,
      "op_id": 1,
      "from_machine": -1,
      "to_machine": 3,
      "ready_time": 0.0
    }
  ],
  "time_stamp": 10.0,
  "remaining_actions": 35,
  "remaining_transfers": 38
}
```

**transfer_requests 字段说明**：
- `from_machine: -1` 表示从 depot 出发（首个工序）
- `from_machine >= 0` 表示从上一台机器搬运到当前机器

### `POST /reset` — 重置

清空求解器状态，准备接收新的问题实例。

**响应**：
```json
{"status": "reset"}
```

## 各算法配置参数

### DE（差分进化）

```json
{
  "config": {
    "init_strategy": "extreme",  // random | roulette | extreme
    "popsize": 30,
    "maxgen": 100,
    "F": 0.1,
    "Cr": 0.1,
    "seed": 42
  }
}
```

### PSO（粒子群优化）

```json
{
  "config": {
    "init_strategy": "extreme",  // random | roulette | extreme
    "popsize": 30,
    "maxgen": 100,
    "w": 0.9,
    "lr": [2, 2],
    "seed": 42
  }
}
```

### CP-SAT（OR-Tools 精确求解）

```json
{
  "config": {
    "time_limit": 60.0,
    "num_workers": 4,
    "seed": 42
  }
}
```

### DRL（深度强化学习 Multi-PPO）

```json
{
  "config": {
    "device": "cuda",
    "model_dir": null,
    "seed": 42
  }
}
```

自动匹配 `FJSP_MultiPPO/saved_network/` 下的预训练模型，按问题规模 (n_jobs, n_machines) 选择最接近的模型。

### GraphGRPO（GNN + GRPO）

```json
{
  "config": {
    "device": "cpu",
    "seed": 42,
    "n_agvs": null,
    "spread_machines": true
  }
}
```

加载 `GraphGRPO/models/agent_model.pt`（GNN 编码器 + 排序/路由/AGV 三个 Actor，纯 CPU 推理）。
内部以事件驱动仿真驱动 agent 逐步决策，产出完整调度后按时间步释放。

- `n_agvs`：仿真中的 AGV 数量，`null` 表示等于机器数（充足，运输不阻塞调度）
- `spread_machines`：当所有机器位置为 `[0,0]` 时自动网格化布置，避免 GNN 位置/运输特征退化

## Docker 使用

### 构建

```bash
cd FJSP

# 构建全部
docker compose build

# 构建单个
docker compose build de
docker compose build drl
docker compose build graphgrpo
```

### 单独测试某个算法

```bash
# 启动求解器
docker compose up -d de

# 跑黑盒测试（全部数据集）
docker compose run --rm test de

# 指定数据集
docker compose run --rm test de J10P5M6
```

### 测试 DRL（需要 GPU）

```bash
docker compose up -d drl
docker compose run --rm test drl J10P5M6
```

## 预训练权重（DRL）

DRL (Multi-PPO) 的预训练权重文件**已包含在仓库中**，位于：

```
End-to-end-DRL-for-FJSP-main/
├── FJSP_MultiPPO/saved_network/   # Multi-PPO 模型，~17MB
└── FJSP_RealWorld/saved_network/  # RealWorld 模型，~3MB
```

权重来源为原作者发布（参见 `End-to-end-DRL-for-FJSP-main/README.md`）：

> https://github.com/LeiK-unsw/End-to-end-DRL-for-FJSP

覆盖的问题规模：`J10M10`、`J10M15`、`J15M15`、`J15M30`、`J20M405`、`J50M205`、`J100M405` 等。系统会按 `(n_jobs, n_machines)` 自动匹配最接近的模型。

## 注意事项

- **proc_times key 类型**：JSON 序列化会将 int key 转为 string，求解器内部已做兼容处理
- **DRL 模型匹配**：当问题规模与预训练模型不一致时，会自动 pad 矩阵到模型尺寸，虚拟机器加工时间设为 0（不可用）
- **DRL 需要 GPU**：`drl.dockerfile` 安装 CUDA 版 PyTorch，docker-compose 中已配置 GPU 资源
