"""텔레메트리 예측 서빙 (ADR-0010 D1).

모델은 이미지에 구워져 있다(/app/model). 시작 시 로드하고, 런타임엔
MLflow/NAS를 보지 않는다 — 그래서 아티팩트 저장소(NAS) 장애가 서비스 중
모델에 무영향이고, 롤백은 이전 이미지 태그로 git revert면 끝난다.

예측 입력(특징)은 현재 Prometheus에서 만든다. 특징 로직은 학습과 공유
(forecaster.features)하여 train/serve skew를 배제한다.

엔드포인트:
  GET /healthz  — probe (모델 로드 여부)
  GET /predict  — 지금 기준 30분 후 예측(JSON) + persistence 베이스라인
  GET /metrics  — 위 예측을 Prometheus 게이지로 노출(스크레이프 시점 계산)
"""
import json
import os

import joblib
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily

from forecaster.features import build_latest_features
from forecaster.prom import fetch_recent

PROM_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://platform-monitoring-promet-prometheus.platform-monitoring.svc.cluster.local:9090",
)
TARGET_QUERY = os.environ.get(
    "TARGET_QUERY", "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes"
)
STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "300"))
HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "30"))
# lag_60 + roll_60 창을 채우려면 최소 60분+; 여유 있게 120분.
LOOKBACK_MIN = int(os.environ.get("LOOKBACK_MIN", "120"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

# 시작 시 baked 모델 로드 (실패하면 컨테이너가 기동 못 함 = readiness 실패).
_model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
try:
    with open(os.path.join(MODEL_DIR, "meta.json")) as fh:
        _meta = json.load(fh)
except FileNotFoundError:
    _meta = {}

app = FastAPI(title="telemetry-forecaster serving")


def _current_forecast() -> dict:
    """현재 Prometheus로 특징을 만들어 30분 후 예측."""
    df = fetch_recent(PROM_URL, TARGET_QUERY, LOOKBACK_MIN, STEP_SECONDS)
    X = build_latest_features(df, STEP_SECONDS)
    predicted = float(_model.predict(X)[0])
    baseline = float(X["lag_0"].iloc[0])  # persistence = 현재값
    return {"predicted_mb": predicted, "baseline_mb": baseline, "as_of": df.index[-1].isoformat()}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_version": _meta.get("version")}


@app.get("/predict")
def predict() -> dict:
    f = _current_forecast()
    return {
        "horizon_min": HORIZON_MIN,
        "predicted_mb": f["predicted_mb"],
        "baseline_mb": f["baseline_mb"],
        "as_of": f["as_of"],
        "model_version": _meta.get("version"),
        "run_id": _meta.get("run_id"),
    }


class _ForecastCollector:
    """스크레이프 시점에 예측을 계산해 게이지로 노출."""

    def collect(self):
        info = GaugeMetricFamily(
            "forecast_model_info", "Baked model metadata", labels=["version", "run_id"]
        )
        info.add_metric([str(_meta.get("version", "")), str(_meta.get("run_id", ""))], 1)
        yield info

        err = GaugeMetricFamily(
            "forecast_scrape_error", "1 if the last scrape failed to produce a forecast"
        )
        try:
            f = _current_forecast()
        except Exception:  # noqa: BLE001 — 스크레이프는 앱을 죽이면 안 됨
            err.add_metric([], 1)
            yield err
            return

        g = GaugeMetricFamily(
            "forecast_node_memory_mb",
            "Predicted node memory (MB) at now + horizon",
            labels=["horizon"],
        )
        g.add_metric([f"{HORIZON_MIN}m"], f["predicted_mb"])
        yield g

        b = GaugeMetricFamily(
            "forecast_baseline_mb", "Persistence baseline = current value (MB)"
        )
        b.add_metric([], f["baseline_mb"])
        yield b

        err.add_metric([], 0)
        yield err


_registry = CollectorRegistry()
_registry.register(_ForecastCollector())


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(_registry), media_type=CONTENT_TYPE_LATEST)
