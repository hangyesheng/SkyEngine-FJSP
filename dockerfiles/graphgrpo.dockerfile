# 打包 GraphGRPO (GNN + GRPO) 求解器
# 纯 CPU 镜像（模型很小，~1.7MB；resolve_device 在 CUDA 不可用时自动回退 CPU）
#
# 构建：docker build -t skyengine-fjsp-graphgrpo:latest -f dockerfiles/graphgrpo.dockerfile .
# 运行：docker run --rm -p 8002:8002 skyengine-fjsp-graphgrpo:latest

FROM python:3.11-slim-bookworm

# 构建工具（torch_geometric 的部分扩展可能需要编译；通常 PyG 2.3+ 的 SAGEConv
# 有纯 torch 实现无需 torch_scatter，这里保留 build-essential 以防万一）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 CPU 版 PyTorch，再装 PyG（让 PyG 匹配已安装的 torch ABI）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch_geometric numpy flask

# 复制 GraphGRPO solver 源码与权重（agent_model.pt 位于 models/）
COPY GraphGRPO/ ./

EXPOSE 8002

CMD ["python", "graphgrpo_solver_server.py", "--host", "0.0.0.0", "--port", "8002"]
