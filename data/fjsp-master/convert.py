"""
将 FJSP-master 的 txt 数据文件转换为标准 JSON 格式

txt 格式说明（来自 FJSP-master/README.md）：
  - 横向表示工序，纵向表示机器，每个数值表示机器加工工序的耗时
  - 以 data_first.txt（J10P5M6）为例：
    前5行 = 工件1的5道工序分别在6台机器上的加工时间
    第6-10行 = 工件2的5道工序分别在6台机器上的加工时间
    以此类推
  - "-" 表示该机器不可用于加工该工序

JSON 输出格式（参考 fjsp-instances-main/barnes/mt10c1.json）：
  {
    "machines": <机器数>,
    "jobs": [
      [                          # 工件1
        [                        # 工序1
          {"machine": 0, "processing": 29},   # 可选机器及其加工时间
          {"machine": 10, "processing": 29}
        ],
        [                        # 工序2
          {"machine": 1, "processing": 78}
        ],
        ...
      ],
      ...
    ]
  }
"""

import json
import os
import sys


# 三个数据文件的规模参数
INSTANCES = {
    "data_first.txt":  {"jobs": 10, "ops_per_job": 5,  "machines": 6,  "name": "J10P5M6"},
    "data_second.txt": {"jobs": 20, "ops_per_job": 10, "machines": 10, "name": "J20P10M10"},
    "data_third.txt":  {"jobs": 20, "ops_per_job": 20, "machines": 15, "name": "J20P20M15"},
}


def parse_txt(filepath: str, n_jobs: int, n_ops: int, n_machines: int) -> dict:
    """解析 txt 文件为标准 dict 结构

    Args:
        filepath: txt 文件路径
        n_jobs: 工件数
        n_ops: 每个工件的工序数
        n_machines: 机器数

    Returns:
        {"machines": int, "jobs": [[{"machine": int, "processing": int}, ...], ...]}
    """
    # 读取原始矩阵
    raw = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(line.split())

    # 校验行数
    expected_rows = n_jobs * n_ops
    if len(raw) != expected_rows:
        print(f"警告: {filepath} 期望 {expected_rows} 行，实际 {len(raw)} 行")

    # 逐工件、逐工序解析
    jobs = []
    for job_idx in range(n_jobs):
        operations = []
        for op_idx in range(n_ops):
            row_idx = job_idx * n_ops + op_idx
            row = raw[row_idx]

            # 校验列数
            if len(row) != n_machines:
                print(f"警告: {filepath} 第{row_idx+1}行期望 {n_machines} 列，实际 {len(row)} 列")

            # 收集该工序的可用机器（跳过 "-"）
            alternatives = []
            for mch_idx, val in enumerate(row):
                if val == "-":
                    continue
                alternatives.append({
                    "machine": mch_idx,
                    "processing": int(val),
                })

            if not alternatives:
                print(f"警告: 工件{job_idx+1} 工序{op_idx+1} (第{row_idx+1}行) 没有任何可用机器!")

            operations.append(alternatives)
        jobs.append(operations)

    return {"machines": n_machines, "jobs": jobs}


def validate(result: dict, n_jobs: int, n_ops: int, n_machines: int, name: str):
    """校验转换结果的基本合法性"""
    assert result["machines"] == n_machines, f"[{name}] machines 不匹配"
    assert len(result["jobs"]) == n_jobs, f"[{name}] jobs 数量不匹配"

    for j, job in enumerate(result["jobs"]):
        assert len(job) == n_ops, f"[{name}] 工件{j} 的工序数不匹配: 期望{n_ops}, 实际{len(job)}"
        for o, op in enumerate(job):
            assert len(op) > 0, f"[{name}] 工件{j} 工序{o} 没有可用机器"
            for alt in op:
                assert 0 <= alt["machine"] < n_machines, f"[{name}] 工件{j} 工序{o} machine={alt['machine']} 越界"
                assert alt["processing"] > 0, f"[{name}] 工件{j} 工序{o} processing={alt['processing']} 非正"

    print(f"[{name}] 校验通过: {n_jobs} 工件 × {n_ops} 工序 × {n_machines} 机器")


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = src_dir  # 输出到同一目录

    for filename, meta in INSTANCES.items():
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            print(f"跳过: {src_path} 不存在")
            continue

        # 解析
        result = parse_txt(src_path, meta["jobs"], meta["ops_per_job"], meta["machines"])

        # 校验
        validate(result, meta["jobs"], meta["ops_per_job"], meta["machines"], meta["name"])

        # 写入 JSON
        out_name = f"{meta['name']}.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 统计信息
        total_ops = meta["jobs"] * meta["ops_per_job"]
        avg_flex = sum(len(op) for job in result["jobs"] for op in job) / total_ops
        print(f"  → 已写入 {out_path} (平均柔性度: {avg_flex:.2f} 台机器/工序)")


if __name__ == "__main__":
    main()
