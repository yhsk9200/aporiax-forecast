#!/usr/bin/env python3
"""텔레메트리 예측 학습 — 노드 메모리 사용량, 30분 지평.

Prometheus에서 노드 메모리 사용량 시계열을 당겨 lag/rolling/시각 특징을
만들고, gradient-boosting 회귀를 학습해 persistence 베이스라인과 비교한 뒤,
model version을 MLflow 레지스트리에 metric과 함께 등록한다.

접근은 전부 클러스터 내부(ClusterIP)다 — tailnet 불필요, S3 크레덴셜 불필요
(MLflow --serve-artifacts가 아티팩트 업로드를 프록시한다). 대상 series는
실측(acf 5분 0.58 / 일주기 0.35)으로 확정했다: lag 특징이 예측력을 갖고
일주기가 실재하는 유일한 후보 — CPU/load(노이즈)·pods(상수)·fs(persistence가
자명하게 이김)와 대비된다.
"""
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from prometheus_api_client import PrometheusConnect
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- 설정 (환경변수로 덮어쓰기 가능) ---
PROM_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://platform-monitoring-promet-prometheus.platform-monitoring.svc.cluster.local:9090",
)
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.platform-mlops.svc.cluster.local:5000",
)
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "telemetry-forecaster")
MODEL_NAME = os.environ.get("MODEL_NAME", "telemetry-forecaster")
# 대상: 노드 메모리 사용량(bytes). 학습 안에서 MB로 스케일해 metric 가독성 확보.
TARGET_QUERY = os.environ.get(
    "TARGET_QUERY", "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes"
)
LOOKBACK_DAYS = float(os.environ.get("LOOKBACK_DAYS", "6"))  # 보존 ~5.2일 → 6일 요청
STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "300"))    # 5분 격자
HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "30"))       # 30분 후 예측
LAG_MINUTES = [0, 5, 15, 30, 60]


def fetch_series() -> pd.DataFrame:
    """Prometheus query_range → 균일 격자 시계열(MB)."""
    prom = PrometheusConnect(url=PROM_URL, disable_ssl=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    data = prom.custom_query_range(
        query=TARGET_QUERY,
        start_time=start,
        end_time=end,
        step=str(STEP_SECONDS),
    )
    if not data:
        raise SystemExit(f"Prometheus가 빈 결과를 반환: query={TARGET_QUERY!r}")
    values = data[0]["values"]  # [[ts, "val"], ...]
    df = pd.DataFrame(values, columns=["ts", "y"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    df["y"] = df["y"].astype(float) / 1e6  # bytes → MB
    df = df[~df["ts"].duplicated(keep="last")].set_index("ts").sort_index()
    return df


def build_features(df: pd.DataFrame):
    """lag / rolling / 시각(sin·cos) 특징 + H분 후 라벨."""
    per_min = 60 / STEP_SECONDS
    lag = lambda m: int(round(m * per_min))  # noqa: E731
    h = lag(HORIZON_MIN)

    f = pd.DataFrame(index=df.index)
    for m in LAG_MINUTES:
        f[f"lag_{m}"] = df["y"].shift(lag(m))
    f["roll_mean_30"] = df["y"].rolling(lag(30)).mean()
    f["roll_std_30"] = df["y"].rolling(lag(30)).std()
    f["roll_mean_60"] = df["y"].rolling(lag(60)).mean()
    hod = df.index.hour + df.index.minute / 60.0
    f["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    f["target"] = df["y"].shift(-h)  # H분 후 값
    return f.dropna(), h


def main() -> None:
    df = fetch_series()
    feats, h = build_features(df)
    if len(feats) < 100:
        raise SystemExit(f"학습 샘플 부족: {len(feats)}행 (보존/스텝 확인)")

    feature_cols = [c for c in feats.columns if c != "target"]
    X, y = feats[feature_cols], feats["target"]

    # 시간순 분할 (마지막 20%가 테스트) — 시계열 누수 방지
    k = int(len(feats) * 0.8)
    Xtr, Xte, ytr, yte = X.iloc[:k], X.iloc[k:], y.iloc[:k], y.iloc[k:]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, random_state=0,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)

    # persistence 베이스라인: 예측 = 현재값(lag_0). 모델이 실제로 가치가 있는지 증명.
    base = Xte["lag_0"].to_numpy()
    base_mae = mean_absolute_error(yte, base)
    base_rmse = mean_squared_error(yte, base) ** 0.5
    improvement = 1 - mae / base_mae if base_mae else float("nan")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "target_query": TARGET_QUERY,
            "horizon_min": HORIZON_MIN,
            "step_seconds": STEP_SECONDS,
            "unit": "MB",
            "adr": "0010",
        })
        mlflow.log_params({
            "lookback_days": LOOKBACK_DAYS,
            "step_seconds": STEP_SECONDS,
            "horizon_min": HORIZON_MIN,
            "lag_minutes": LAG_MINUTES,
            "model": "GradientBoostingRegressor",
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
        })
        mlflow.log_metrics({
            "mae_mb": mae,
            "rmse_mb": rmse,
            "r2": r2,
            "baseline_mae_mb": base_mae,
            "baseline_rmse_mb": base_rmse,
            "mae_improvement_vs_baseline": improvement,
            "n_samples": float(len(feats)),
            "n_features": float(len(feature_cols)),
        })
        signature = infer_signature(Xtr, model.predict(Xtr))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=Xtr.iloc[:2],
        )
        run_id = run.info.run_id

    print(
        f"[ok] run={run_id} registered={MODEL_NAME} "
        f"MAE={mae:.4g}MB (baseline {base_mae:.4g}MB, +{improvement:.1%}) "
        f"RMSE={rmse:.4g}MB R2={r2:.3f} n={len(feats)}"
    )


if __name__ == "__main__":
    main()
