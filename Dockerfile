# CAST-PdM (React + FastAPI) 배포용 (Render 무료 웹 서비스, 2026-10-22)
#   Render: New → Web Service → 이 저장소·가지 → Language: Docker (이 파일을 자동으로 쓴다)
#   원자료(data/)·모델 파일(models/)·작업 기록(actions/)·비밀값(.streamlit/secrets.toml)은 넣지 않는다 (.dockerignore).
#   화면은 app_data/(미리 계산한 결과)만 읽는다. 비밀값은 Render의 Environment에 CAST_SECRETS(secrets.toml 내용 전체)로 넣는다

# 1) 화면 빌드 (React)
FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# 2) 서버 (FastAPI)
FROM python:3.14-slim
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app PATH=/home/app/.local/bin:$PATH PYTHONUNBUFFERED=1 CAST_HTTPS=1
WORKDIR /home/app/srv
COPY --chown=app deploy/requirements-deploy.txt ./
RUN pip install --no-cache-dir --user -r requirements-deploy.txt
COPY --chown=app api/ api/
COPY --chown=app src/ src/
COPY --chown=app app_data/ app_data/
COPY --chown=app ontology/ ontology/
COPY --chown=app --from=web /web/dist web/dist
# 로그인을 켜지 않았을 때 쓰는 로컬 저장소 (서버가 다시 켜지면 지워진다. 로그인하면 공용 DB를 쓴다)
RUN mkdir -p actions
# Render가 PORT를 정해 준다 (없으면 10000). --proxy-headers: 앞단(https) 주소로 구글 로그인 돌아올 주소를 만든다
CMD ["sh", "-c", "exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips '*'"]
