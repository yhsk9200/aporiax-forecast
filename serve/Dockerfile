# 텔레메트리 예측 서빙 이미지 (ADR-0010 D1).
# 빌드 context = repo 루트:  docker build -f serve/Dockerfile .
#
# 중요: 이 이미지는 **베이킹 워크플로우에서만** 빌드된다. 베이킹이 champion
# 모델을 ./model/ 로 내려받은 뒤 빌드하므로 아래 `COPY model/`이 성립한다.
# 일반 CI(docker-build-check)는 serve를 빌드하지 않는다 — 레포에 모델이 없어
# 실패하기 때문. "모델을 굽는 주체 = 베이킹 워크플로우"라는 소유권의 표현이다.
FROM python:3.12-slim

WORKDIR /app

COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 공유 특징·조회 로직 (학습과 동일 소스 → skew 없음)
COPY forecaster/ ./forecaster/
COPY serve/app.py .

# baked 모델 — 베이킹 워크플로우가 champion을 여기로 내려받은 상태에서 빌드.
# 런타임엔 MLflow/NAS를 보지 않는다(모델이 이미지 안에 있음).
COPY model/ ./model/

RUN useradd --create-home --uid 10001 svc
USER svc

EXPOSE 8000
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
