"""
测试 OnlineDE/PSO Solver — 用 mock obs 模拟 SkyEngine 调用流程
"""
import sys, os
import numpy as np

# Mock SkyEngine 数据结构
class Op:
    def __init__(self, op_id, machine_options, proc_time):
        self.op_id = op_id
        self.machine_options = machine_options  # 可用机器 id 列表
        self.proc_time = proc_time              # float
        self.assigned_machine = None
        self.status = "PENDING"

class Job:
    def __init__(self, job_id, ops):
        self.job_id = job_id
        self.ops = ops
        self.release = 0.0
        self.due = None
        self.completion_time = 0.0

class Machine:
    def __init__(self, mid, loc=(0,0)):
        self.id = mid
        self.location = loc
        self.current_op = None
        self.total_work_time = 0

def load_mock_obs(txt_path, n_jobs, n_ops, n_machines):
    """从 txt 数据文件构造 mock obs dict"""
    contents = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                contents.append(line.split())

    jobs = []
    for j in range(n_jobs):
        ops = []
        for o in range(n_ops):
            row = contents[j * n_ops + o]
            mch_opts = []
            pt = 1.0
            for m in range(n_machines):
                if row[m] != "-":
                    mch_opts.append(m)
                    pt = float(row[m])
            ops.append(Op(op_id=o, machine_options=mch_opts, proc_time=pt))
        jobs.append(Job(job_id=j, ops=ops))

    machines = [Machine(mid=m) for m in range(n_machines)]
    return {"jobs": jobs, "machines": machines}


def test_solver(solver_cls, solver_name, txt_path, n_jobs, n_ops, n_machines):
    print(f"\n{'='*55}")
    print(f"  Testing: {solver_name}")
    print(f"{'='*55}")

    obs = load_mock_obs(txt_path, n_jobs, n_ops, n_machines)
    solver = solver_cls(init_strategy="extreme", popsize=30, maxgen=100, seed=42)

    # 模拟 SkyEngine 逐时间步调用
    max_steps = 200
    total_actions = 0
    total_transfers = 0

    for step in range(1, max_steps + 1):
        result = solver.plan(obs)
        ma = result["machine_actions"]
        tr = result["transfer_requests"]

        if ma or tr:
            for a in ma:
                print(f"  t={step:3d} | M{a['machine_id']} 加工 J{a['job_id']}-O{a['op_id']} "
                      f"[{a['start_time']:.0f} ~ {a['expected_end']:.0f}]")
            for t in tr:
                print(f"  t={step:3d} | 搬运 J{t['job_id']}-O{t['op_id']} "
                      f"M{t['source'][0]} → M{t['destination'][0]} (ready={t['ready_time']:.0f})")

        total_actions += len(ma)
        total_transfers += len(tr)

        # 所有任务都输出了就结束
        if (not solver._machine_actions and not solver._transfer_requests
                and solver.initialized):
            print(f"  t={step}: 所有调度已输出完毕")
            break

    print(f"\n  统计: {total_actions} actions, {total_transfers} transfers, "
          f"{step} 时间步")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    txt = os.path.join(data_dir, "data_first.txt")

    sys.path.insert(0, os.path.join(base, "DE_solver"))
    sys.path.insert(0, os.path.join(base, "PSO_solver"))

    from online_de_solver import OnlineDESolver
    from online_pso_solver import OnlinePSOSolver

    # 用最小的 J10P5M6 做快速验证
    test_solver(OnlineDESolver, "OnlineDE (J10P5M6)", txt, 10, 5, 6)
    test_solver(OnlinePSOSolver, "OnlinePSO (J10P5M6)", txt, 10, 5, 6)
