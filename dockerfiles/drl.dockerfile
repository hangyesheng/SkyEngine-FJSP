# 打包 End-to-end-DRL-for-FJSP (Multi-PPO) 深度强化学习求解器
# 需要 GPU 支持
#
# 构建：docker build -t skyengine-fjsp-drl:latest -f dockerfiles/drl.dockerfile .
# 运行：docker run --gpus all -p 8002:8002 skyengine-fjsp-drl:latest

FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch (CUDA 12.8)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Flask + 其余依赖
RUN pip install --no-cache-dir flask numpy matplotlib gym seaborn tqdm

# 复制 DRL solver 源码（server + online solver 在根目录，模型代码在子目录）
COPY End-to-end-DRL-for-FJSP-main/ ./

EXPOSE 8002

CMD ["python", "drl_solver_server.py", "--host", "0.0.0.0", "--port", "8002"]
