# docker build -t fjsp-drl -f fjsp.dockerfile . # 打包
#  docker run --gpus all -it fjsp-drl bash      # 启动
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/FJSP

# PyTorch (CUDA 12.8)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/FJSP

WORKDIR /workspace/FJSP/FJSP_MultiPPO
CMD ["python", "validation.py"]
