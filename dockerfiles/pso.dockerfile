# 打包 PSO_solver 粒子群优化求解器

FROM skyengine-fjsp-base:latest

WORKDIR /app

# 复制源码
COPY FJSP-master/PSO_solver/ ./

EXPOSE 8002

CMD ["python", "pso_solver_server.py", "--host", "0.0.0.0", "--port", "8002"]
