# 打包 OR-solver (CP-SAT 精确求解器)

FROM skyengine-fjsp-base:latest

RUN pip install --no-cache-dir ortools

WORKDIR /app

# 复制源码
COPY OR-solver/ ./

EXPOSE 8002

CMD ["python", "best_solver_server.py", "--host", "0.0.0.0", "--port", "8002"]
