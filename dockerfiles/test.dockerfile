# 测试镜像 — 对 FJSP solver 服务做黑盒测试

FROM skyengine-fjsp-base:latest

RUN pip install --no-cache-dir requests

WORKDIR /app

COPY offline_test.py ./
COPY data/ ./data/

ENTRYPOINT ["python", "offline_test.py"]
