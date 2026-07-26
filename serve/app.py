"""텔레메트리 예측 서빙 (ADR-0010 D1).

모델은 이미지에 구워져 있다(/app/model). 시작 시 로드하고, 런타임엔
MLflow/NAS를 보지 않는다 — 그래서 아티팩트 저장소(NAS) 장애가 서비스 중
모델에 무영향이고, 롤백은 이전 이미지 태그로 git revert면 끝난다.

모델은 30분 후의 **변화량**을 예측한다. 절대 예측은 `현재값 + 변화량`으로
복원한다(설계 근거는 forecaster/features.py). 특징 로직은 학습과 공유하여
train/serve skew를 배제하고, 그 위에 스키마 검증을 하나 더 둔다 — baked 모델의
feature_set이 코드의 FEATURE_SET과 다르면 기동을 거부한다. 구 스키마 모델을
새 코드로 서빙하면 에러 없이 조용히 틀린 예측이 나오기 때문이다(fail closed).

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

from forecaster.features import FEATURE_SET, build_latest_features
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
# d_60 + roll_60 창을 채우려면 최소 60분+; 여유 있게 120분.
LOOKBACK_MIN = int(os.environ.get("LOOKBACK_MIN", "120"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

# 시작 시 baked 모델 로드 (실패하면 컨테이너가 기동 못 함 = readiness 실패).
_model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
try:
    with open(os.path.join(MODEL_DIR, "meta.json")) as fh:
        _meta = json.load(fh)
except FileNotFoundError:
    _meta = {}

# 코드-모델 짝 검증. 특징 스키마가 어긋난 조합은 조용히 틀린 답을 내므로 기동 거부.
_baked_fs = _meta.get("feature_set")
if _baked_fs != FEATURE_SET:
    raise RuntimeError(
        f"baked 모델의 feature_set={_baked_fs!r}이 서빙 코드의 {FEATURE_SET!r}와 다릅니다. "
        "해당 스키마로 학습된 모델을 베이킹해 승격하세요."
    )

app = FastAPI(title="telemetry-forecaster serving")


def _current_forecast() -> dict:
    """현재 Prometheus로 특징을 만들어 30분 후 예측(변화량 → 절대값 복원)."""
    df = fetch_recent(PROM_URL, TARGET_QUERY, LOOKBACK_MIN, STEP_SECONDS)
    X, current = build_latest_features(df, STEP_SECONDS)
    delta = float(_model.predict(X)[0])
    return {
        "predicted_mb": current + delta,
        "predicted_delta_mb": delta,
        "baseline_mb": current,          # persistence = 현재값
        "as_of": df.index[-1].isoformat(),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_version": _meta.get("version"),
        "feature_set": FEATURE_SET,
    }


@app.get("/predict")
def predict() -> dict:
    f = _current_forecast()
    return {
        "horizon_min": HORIZON_MIN,
        "predicted_mb": f["predicted_mb"],
        "predicted_delta_mb": f["predicted_delta_mb"],
        "baseline_mb": f["baseline_mb"],
        "as_of": f["as_of"],
        "model_version": _meta.get("version"),
        "model_name": _meta.get("model_name"),
        "feature_set": FEATURE_SET,
        "run_id": _meta.get("run_id"),
    }


class _ForecastCollector:
    """스크레이프 시점에 예측을 계산해 게이지로 노출."""

    def collect(self):
        info = GaugeMetricFamily(
            "forecast_model_info", "Baked model metadata",
            labels=["model_name", "version", "run_id", "feature_set"],
        )
        info.add_metric([
            str(_meta.get("model_name", "")), str(_meta.get("version", "")),
            str(_meta.get("run_id", "")), FEATURE_SET,
        ], 1)
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

        d = GaugeMetricFamily(
            "forecast_predicted_delta_mb",
            "Predicted change (MB) over the horizon",
            labels=["horizon"],
        )
        d.add_metric([f"{HORIZON_MIN}m"], f["predicted_delta_mb"])
        yield d

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
