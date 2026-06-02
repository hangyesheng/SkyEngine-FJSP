"""
Benchmark Test Adapter for End-to-end DRL FJSP Solver

读取 fjsp-instances-main 中的 benchmark 实例，转换为 DRL 模型期望的格式，
运行预训练模型求解，并与已知 optimum 进行比较。

用法:
  # 在容器内运行（测试所有有 optimum 值的实例）
  python benchmark_test.py

  # 只测试特定规模的实例
  python benchmark_test.py --min_jobs 10 --max_jobs 15 --min_machines 5 --max_machines 10

  # 指定 benchmark 路径和模型路径
  python benchmark_test.py --benchmark_dir /path/to/fjsp-instances-main

  # 测试单个实例
  python benchmark_test.py --instance mk01
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import torch
from copy import deepcopy

# ── 数据格式转换 ───────────────────────────────────────────────────

def parse_fjs_file(filepath):
    """
    解析 .fjs / .txt 格式的 FJSP benchmark 文件。

    格式:
      第一行: n_jobs n_machines [avg_machines_per_op]
      后续每行一个 job:
        n_ops {n_alt_machines machine_id time ...} ...

    返回:
      dict: {
        'n_jobs': int,
        'n_machines': int,
        'max_ops': int,  # 所有 job 中最多的 operation 数
        'jobs': [
          [  # job 0
            {machine_id: processing_time, ...},  # operation 0
            {machine_id: processing_time, ...},  # operation 1
            ...
          ],
          ...
        ]
      }
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    first_line = lines[0].split()
    n_jobs = int(first_line[0])
    n_machines = int(first_line[1])

    jobs = []
    line_idx = 1

    for _ in range(n_jobs):
        if line_idx >= len(lines):
            break
        parts = list(map(int, lines[line_idx].split()))
        line_idx += 1

        n_ops = parts[0]
        operations = []

        idx = 1
        for _ in range(n_ops):
            if idx >= len(parts):
                break
            n_alt = parts[idx]
            idx += 1
            machine_time = {}
            for _ in range(n_alt):
                if idx + 1 >= len(parts):
                    break
                # benchmark 文件中 machine id 通常从 1 开始
                mid = parts[idx] - 1  # 转为 0-indexed
                ptime = parts[idx + 1]
                machine_time[mid] = float(ptime)
                idx += 2
            operations.append(machine_time)

        jobs.append(operations)

    max_ops = max(len(j) for j in jobs)

    return {
        'n_jobs': n_jobs,
        'n_machines': n_machines,
        'max_ops': max_ops,
        'jobs': jobs,
    }


def benchmark_to_env_matrix(parsed, n_j, n_m):
    """
    将解析后的 benchmark 数据转换为 DRL env 期望的矩阵格式。

    DRL env 期望: shape (n_j, n_m, n_m) 的矩阵
      matrix[job][op][machine] = processing_time, 0 表示不可在该机器上加工

    关键处理：
      - env 假设每个 job 有恰好 n_m 个 operations
      - 如果 benchmark 的某个 job ops 数 < n_m，则用 dummy operations 填充
      - dummy operations 的所有机器加工时间为 1（极小值），确保 env reset 中的
        durmch.min() / durmch.mean() 不会遇到空数组
      - 如果 benchmark 的 n_machines < n_m，扩展列并填 0

    参数:
      parsed: parse_fjs_file 的输出
      n_j: env 配置的 job 数
      n_m: env 配置的 machine 数

    返回:
      np.ndarray of shape (n_j, n_m, n_m)
    """
    assert parsed['n_jobs'] <= n_j, f"benchmark has {parsed['n_jobs']} jobs, but env expects {n_j}"
    assert parsed['n_machines'] <= n_m, f"benchmark has {parsed['n_machines']} machines, but env expects {n_m}"

    # 初始化：所有位置填 1（表示 dummy op 可在所有机器上以时间 1 完成）
    # 然后把真实数据覆盖进去，真实数据中 0 表示不可用
    matrix = np.ones((n_j, n_m, n_m), dtype=np.float32)

    # 先把所有位置清零，然后逐个填入
    matrix.fill(0)

    for job_idx, job_ops in enumerate(parsed['jobs']):
        for op_idx, machine_time in enumerate(job_ops):
            if op_idx >= n_m:
                break
            for mid, ptime in machine_time.items():
                if mid < n_m:
                    matrix[job_idx][op_idx][mid] = ptime

        # 对于 padding 的 dummy operations（op_idx >= len(job_ops)），
        # 标记为在所有机器上都可用，加工时间为 1
        for op_idx in range(len(job_ops), n_m):
            matrix[job_idx][op_idx][:parsed['n_machines']] = 1.0

    return matrix


def load_instances_json(benchmark_dir):
    """加载 instances.json 索引文件"""
    json_path = os.path.join(benchmark_dir, 'instances.json')
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        return json.load(f)


# ── 模型求解 ───────────────────────────────────────────────────

def find_matching_model(n_jobs, n_machines, saved_network_dir):
    """
    根据实例规模匹配最合适的预训练模型。
    模型目录命名格式: FJSP_J{jobs}M{machines}
    """
    # 精确匹配
    exact = f"FJSP_J{n_jobs}M{n_machines}"
    model_dir = os.path.join(saved_network_dir, exact)
    if os.path.isdir(model_dir):
        return model_dir, n_jobs, n_machines

    # 查找所有可用模型，按规模排序
    available = []
    for d in os.listdir(saved_network_dir):
        if d.startswith("FJSP_J") and os.path.isdir(os.path.join(saved_network_dir, d)):
            parts = d.replace("FJSP_J", "").split("M")
            if len(parts) == 2:
                j, m = int(parts[0]), int(parts[1])
                available.append((j, m, d))

    if not available:
        return None, None, None

    # 选择 jobs >= n_jobs 且 machines >= n_machines 的最小模型
    candidates = [(j, m, d) for j, m, d in available if j >= n_jobs and m >= n_machines]
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        j, m, d = candidates[0]
        return os.path.join(saved_network_dir, d), j, m

    # 没有足够大的模型，选最大的
    available.sort(key=lambda x: (x[0], x[1]), reverse=True)
    j, m, d = available[0]
    print(f"  [WARN] 没有足够大的模型匹配 {n_jobs}x{n_machines}，使用最大模型 {d}")
    return os.path.join(saved_network_dir, d), j, m


def find_best_model_file(model_dir):
    """在模型目录中找到 best_value 或最后的 checkpoint"""
    # 搜索所有子目录，找到含有 policy_job.pth 和 policy_mch.pth 的
    candidates = []

    # 先找 best_value 开头的目录
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
        # 优先选 best_value，否则选最后一个
        best = [c for c in candidates if c[0]]
        if best:
            # 多个 best_value 时选数字最大的
            best.sort(key=lambda x: int(x[1].replace('best_value', '') or '0'), reverse=True)
            return best[0][2], best[0][3]
        return candidates[-1][2], candidates[-1][3]

    # 直接在目录下找
    job_path = os.path.join(model_dir, 'policy_job.pth')
    mch_path = os.path.join(model_dir, 'policy_mch.pth')
    if os.path.exists(job_path) and os.path.exists(mch_path):
        return job_path, mch_path

    return None, None


def solve_instance(data_matrix, n_j, n_m, policy_job, policy_mch, device, configs):
    """
    用训练好的模型求解单个 FJSP 实例。

    参数:
      data_matrix: shape (n_j, n_m, n_m) 的 numpy 数组
      n_j, n_m: 环境参数
      policy_job, policy_mch: 训练好的策略网络
      device: torch device
      configs: 参数配置

    返回:
      float: makespan
    """
    from FJSP_Env import FJSP
    from mb_agg import aggr_obs, g_pool_cal

    # 添加 batch 维度: (1, n_j, n_m, n_m)
    batch_data = np.expand_dims(data_matrix, axis=0)

    env = FJSP(n_j, n_m)
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
        n_nodes=n_j * n_m,
        device=device,
    )

    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(batch_data)

    with torch.no_grad():
        pool = None
        while True:
            env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), n_j * n_m)
            env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
            env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
            env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
            env_mask = torch.from_numpy(np.copy(mask)).to(device)
            env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)

            action, _, _, action_node, _, mask_mch_action, hx = policy_job(
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

            pi_mch, pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time)
            _, mch_a = pi_mch.squeeze(-1).max(1)

            adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(
                action.cpu().numpy(), mch_a
            )

            if env.done():
                break

    makespan = env.mchsEndTimes.max(-1).max(-1)
    return makespan[0]


# ── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark test for DRL FJSP solver')
    parser.add_argument('--benchmark_dir', type=str,
                        default='/workspace/FJSP/data/fjsp-instances-main',
                        help='Path to fjsp-instances-main directory')
    parser.add_argument('--model_dir', type=str,
                        default='/workspace/FJSP/FJSP_MultiPPO/saved_network',
                        help='Path to saved models directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
    parser.add_argument('--min_jobs', type=int, default=0,
                        help='Filter: minimum number of jobs')
    parser.add_argument('--max_jobs', type=int, default=999,
                        help='Filter: maximum number of jobs')
    parser.add_argument('--min_machines', type=int, default=0,
                        help='Filter: minimum number of machines')
    parser.add_argument('--max_machines', type=int, default=999,
                        help='Filter: maximum number of machines')
    parser.add_argument('--instance', type=str, default=None,
                        help='Test a single instance by name (e.g. mk01)')
    parser.add_argument('--no_optimum_only', action='store_true',
                        help='Include instances without known optimum')
    args = parser.parse_args()

    # 加载 instances 索引
    instances = load_instances_json(args.benchmark_dir)
    if not instances:
        print(f"ERROR: 未找到 instances.json 在 {args.benchmark_dir}")
        sys.exit(1)

    # 筛选实例
    filtered = []
    for inst in instances:
        if args.instance and inst['name'] != args.instance:
            continue
        if not args.no_optimum_only and inst.get('optimum') is None:
            continue
        if inst['jobs'] < args.min_jobs or inst['jobs'] > args.max_jobs:
            continue
        if inst['machines'] < args.min_machines or inst['machines'] > args.max_machines:
            continue
        filtered.append(inst)

    if not filtered:
        print("没有匹配的实例。尝试加 --no_optimum_only 参数？")
        sys.exit(0)

    print(f"共筛选出 {len(filtered)} 个实例待测试")
    print(f"{'='*80}")

    # 导入模块 (需要在 FJSP_MultiPPO 工作目录下)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Params import configs as default_configs
    from PPOwithValue import PPO
    from mb_agg import g_pool_cal

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 按 (n_j, n_m) 分组，避免重复加载模型
    from collections import defaultdict
    grouped = defaultdict(list)
    for inst in filtered:
        grouped[(inst['jobs'], inst['machines'])].append(inst)

    all_results = []

    for (n_jobs_bench, n_machines_bench), inst_list in sorted(grouped.items()):
        print(f"\n{'─'*80}")
        print(f"  规模: {n_jobs_bench} jobs × {n_machines_bench} machines  ({len(inst_list)} 个实例)")
        print(f"{'─'*80}")

        # 找到匹配的模型
        model_path, model_n_j, model_n_m = find_matching_model(
            n_jobs_bench, n_machines_bench, args.model_dir
        )

        if model_path is None:
            print(f"  [SKIP] 没有可用的预训练模型")
            for inst in inst_list:
                all_results.append({
                    'name': inst['name'],
                    'jobs': inst['jobs'],
                    'machines': inst['machines'],
                    'optimum': inst.get('optimum'),
                    'drl_makespan': None,
                    'gap': None,
                    'time_sec': None,
                    'status': 'no_model',
                })
            continue

        job_pth, mch_pth = find_best_model_file(model_path)
        if job_pth is None:
            print(f"  [SKIP] 模型文件不完整: {model_path}")
            for inst in inst_list:
                all_results.append({
                    'name': inst['name'],
                    'jobs': inst['jobs'],
                    'machines': inst['machines'],
                    'optimum': inst.get('optimum'),
                    'drl_makespan': None,
                    'gap': None,
                    'time_sec': None,
                    'status': 'model_incomplete',
                })
            continue

        print(f"  模型: {os.path.basename(model_path)} (trained on {model_n_j}x{model_n_m})")
        print(f"  权重: {os.path.basename(job_pth)}")

        # 重要：PPO 内部使用 configs.n_j / configs.n_m 来创建 Actor 网络
        # 必须在创建 PPO 之前修改 configs 值以匹配模型的训练规模
        default_configs.n_j = model_n_j
        default_configs.n_m = model_n_m

        # 加载模型
        ppo = PPO(
            lr=default_configs.lr,
            gamma=default_configs.gamma,
            k_epochs=default_configs.k_epochs,
            eps_clip=default_configs.eps_clip,
            n_j=model_n_j,
            n_m=model_n_m,
            num_layers=default_configs.num_layers,
            neighbor_pooling_type=default_configs.neighbor_pooling_type,
            input_dim=default_configs.input_dim,
            hidden_dim=default_configs.hidden_dim,
            num_mlp_layers_feature_extract=default_configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=default_configs.num_mlp_layers_actor,
            hidden_dim_actor=default_configs.hidden_dim_actor,
            num_mlp_layers_critic=default_configs.num_mlp_layers_critic,
            hidden_dim_critic=default_configs.hidden_dim_critic,
        )

        ppo.policy_job.load_state_dict(torch.load(job_pth, map_location=device), strict=False)
        ppo.policy_mch.load_state_dict(torch.load(mch_pth, map_location=device), strict=False)
        ppo.policy_job.eval()
        ppo.policy_mch.eval()

        # 逐个实例测试
        for inst in inst_list:
            filepath = os.path.join(args.benchmark_dir, inst['path'])
            if not os.path.exists(filepath):
                print(f"  [SKIP] {inst['name']}: 文件不存在 {filepath}")
                all_results.append({
                    'name': inst['name'],
                    'jobs': inst['jobs'],
                    'machines': inst['machines'],
                    'optimum': inst.get('optimum'),
                    'drl_makespan': None,
                    'gap': None,
                    'time_sec': None,
                    'status': 'file_missing',
                })
                continue

            try:
                parsed = parse_fjs_file(filepath)
                data_matrix = benchmark_to_env_matrix(parsed, model_n_j, model_n_m)

                t_start = time.time()
                makespan = solve_instance(
                    data_matrix, model_n_j, model_n_m,
                    ppo.policy_job, ppo.policy_mch, device, default_configs
                )
                elapsed = time.time() - t_start

                makespan_val = makespan.item() if isinstance(makespan, torch.Tensor) else float(makespan)
                optimum = inst.get('optimum')
                gap = None
                if optimum is not None and optimum > 0:
                    gap = (makespan_val - optimum) / optimum * 100

                all_results.append({
                    'name': inst['name'],
                    'jobs': inst['jobs'],
                    'machines': inst['machines'],
                    'optimum': optimum,
                    'drl_makespan': round(makespan_val, 2),
                    'gap': round(gap, 2) if gap is not None else None,
                    'time_sec': round(elapsed, 3),
                    'status': 'ok',
                })

                gap_str = f"{gap:+.2f}%" if gap is not None else "N/A"
                opt_str = str(optimum) if optimum else "N/A"
                print(f"  {inst['name']:15s}  DRL={makespan_val:8.2f}  OPT={opt_str:>8s}  gap={gap_str:>8s}  time={elapsed:.3f}s")

            except Exception as e:
                print(f"  [ERROR] {inst['name']}: {e}")
                all_results.append({
                    'name': inst['name'],
                    'jobs': inst['jobs'],
                    'machines': inst['machines'],
                    'optimum': inst.get('optimum'),
                    'drl_makespan': None,
                    'gap': None,
                    'time_sec': None,
                    'status': f'error: {e}',
                })

    # ── 汇总报告 ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  汇总报告")
    print(f"{'='*80}")

    ok_results = [r for r in all_results if r['status'] == 'ok' and r['gap'] is not None]
    skip_results = [r for r in all_results if r['status'] != 'ok']

    if ok_results:
        gaps = [r['gap'] for r in ok_results]
        times = [r['time_sec'] for r in ok_results]
        print(f"\n  成功求解: {len(ok_results)}/{len(all_results)} 个实例")
        print(f"  GAP 统计:  min={min(gaps):+.2f}%  avg={sum(gaps)/len(gaps):+.2f}%  max={max(gaps):+.2f}%")
        print(f"  耗时统计:  min={min(times):.3f}s  avg={sum(times)/len(times):.3f}s  max={max(times):.3f}s")

        # 按 gap 排序输出
        print(f"\n  {'实例':15s} {'Jobs':>5s} {'Machines':>9s} {'DRL':>8s} {'OPT':>8s} {'GAP':>8s} {'Time':>8s}")
        print(f"  {'-'*15} {'-'*5} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in sorted(ok_results, key=lambda x: x['gap']):
            print(f"  {r['name']:15s} {r['jobs']:5d} {r['machines']:9d} "
                  f"{r['drl_makespan']:8.2f} {str(r['optimum']):>8s} "
                  f"{r['gap']:+7.2f}% {r['time_sec']:7.3f}s")

    if skip_results:
        print(f"\n  跳过/失败: {len(skip_results)} 个")
        for r in skip_results:
            print(f"    {r['name']:15s}  reason: {r['status']}")

    # 保存结果到 JSON
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
