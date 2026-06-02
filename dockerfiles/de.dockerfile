# 打包 DE_solver 差分进化求解器

FROM skyengine-fjsp-base:latest

WORKDIR /app

# 复制源码
COPY FJSP-master/DE_solver/ ./

EXPOSE 8002

CMD ["python", "de_solver_server.py", "--host", "0.0.0.0", "--port", "8002"]

