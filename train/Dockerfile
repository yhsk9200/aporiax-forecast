# 텔레메트리 예측 학습 이미지 (ADR-0010 C2).
# OCI 클러스터 CronJob이 실행. Prometheus·MLflow는 클러스터 내부(ClusterIP)로 접근.
# 멀티아치 빌드(amd64+arm64)는 .github/workflows/build-train-image.yaml에서 수행.
FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 (레이어 캐시). arm64/amd64 모두 PyPI manylinux 휠 사용 → 컴파일 없음.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_train.py .

# MLflow가 GitPython으로 소스 git SHA를 찾다 실패해 경고를 뱉는다. 이미지엔
# git이 없고 필요도 없으니 조용히 넘긴다.
ENV GIT_PYTHON_REFRESH=quiet

# 비루트 실행
RUN useradd --create-home --uid 10001 trainer
USER trainer

ENTRYPOINT ["python", "run_train.py"]
