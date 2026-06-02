"""
将 FJSP-master 的 txt 数据文件转为服务器需要的 obs JSON 格式，
保存到 test_data/ 目录供离线测试使用。

用法: python generate_test_data.py
"""
import json
import os

INSTANCES = {
    "data_first.txt":  {"jobs": 10, "ops_per_job": 5,  "machines": 6,  "name": "J10P5M6"},
    "data_second.txt": {"jobs": 20, "ops_per_job": 10, "machines": 10, "name": "J20P10M10"},
    "data_third.txt":  {"jobs": 20, "ops_per_job": 20, "machines": 15, "name": "J20P20M15"},
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fjsp-master")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def txt_to_obs_json(filepath, n_jobs, n_ops, n_machines):
    """txt → obs JSON（与 test_servers.py 中 load_obs_json 一致的格式）"""
    raw = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(line.split())

    jobs = []
    for j in range(n_jobs):
        ops = []
        for o in range(n_ops):
            row = raw[j * n_ops + o]
            mch_opts = []
            proc_times = {}
            for m in range(n_machines):
                if row[m] != "-":
                    mch_opts.append(m)
                    proc_times[m] = float(row[m])
            ops.append({
                "op_id": o,
                "machine_options": mch_opts,
                "proc_times": proc_times,
            })
        jobs.append({"job_id": j, "ops": ops})

    machines = [{"id": m, "location": [0, 0]} for m in range(n_machines)]
    return {"obs": {"jobs": jobs, "machines": machines}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for filename, meta in INSTANCES.items():
        src = os.path.join(DATA_DIR, filename)
        if not os.path.exists(src):
            print(f"跳过: {src} 不存在")
            continue

        obs = txt_to_obs_json(src, meta["jobs"], meta["ops_per_job"], meta["machines"])

        out_path = os.path.join(OUT_DIR, f"{meta['name']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obs, f, indent=2, ensure_ascii=False)

        n_jobs = meta["jobs"]
        n_ops  = meta["ops_per_job"]
        n_mch  = meta["machines"]
        total_ops = n_jobs * n_ops
        avg_flex = sum(
            len(op["machine_options"])
            for job in obs["obs"]["jobs"]
            for op in job["ops"]
        ) / total_ops

        print(f"[{meta['name']}] {n_jobs}×{n_ops}×{n_mch}  "
              f"平均柔性度={avg_flex:.2f}  → {out_path}")

    print("\n全部完成。测试数据已写入 test_data/")


if __name__ == "__main__":
    main()
