#!/usr/bin/env python3
"""텔레메트리 예측 학습 — 노드 메모리 사용량, 30분 지평 (ADR-0010 C2).

Prometheus에서 시계열을 당겨(forecaster.prom) 차분 특징을 만들고
(forecaster.features) Ridge 회귀로 30분 후 **변화량**을 학습한다. 절대 예측은
`현재값 + 변화량`으로 복원하므로 레벨 이동에 구조적으로 면역이다(설계 근거는
forecaster/features.py 참조). 특징 로직은 서빙과 공유한다(skew 방지).

검증은 **walk-forward**로 한다 — 시간순 단일 분할은 학습·테스트가 같은 레짐 안에
들어가 성적을 부풀린다(v1의 +27%가 그렇게 나왔고 레짐이 바뀌자 즉시 붕괴했다).
등록되는 metric은 여러 fold의 통계이며, persistence 베이스라인과 fold별로 비교한
승률까지 남긴다.

접근은 전부 클러스터 내부(ClusterIP) — tailnet·S3 크레덴셜 불필요
(MLflow --serve-artifacts가 아티팩트 업로드를 프록시).
"""
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from forecaster.features import FEATURE_COLS, FEATURE_SET, build_training_frame
from forecaster.prom import fetch_recent


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
TARGET_QUERY = os.environ.get(
    "TARGET_QUERY", "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes"
)
LOOKBACK_DAYS = float(os.environ.get("LOOKBACK_DAYS", "6"))  # 보존 5d → 6일 요청
STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "300"))    # 5분 격자
HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "30"))       # 30분 후 예측
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", "1.0"))
N_FOLDS = int(os.environ.get("N_FOLDS", "8"))
FOLD_HOURS = float(os.environ.get("FOLD_HOURS", "6"))        # fold별 테스트 구간
MIN_TRAIN_HOURS = float(os.environ.get("MIN_TRAIN_HOURS", "36"))


def _fit(X, y):
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X, y)
    return model


def walk_forward(feats):
    """확장창 walk-forward. fold별 (모델 MAE, 베이스라인 MAE)를 절대값 기준으로."""
    test_len = int(FOLD_HOURS * 3600 / STEP_SECONDS)
    min_train = int(MIN_TRAIN_HOURS * 3600 / STEP_SECONDS)
    n = len(feats)
    folds = max(1, min(N_FOLDS, (n - min_train) // test_len))
    rows = []
    for k in range(folds):
        te_end = n - (folds - 1 - k) * test_len
        te_start = te_end - test_len
        if te_start < min_train:
            continue
        tr, te = feats.iloc[:te_start], feats.iloc[te_start:te_end]
        model = _fit(tr[FEATURE_COLS], tr["target"])
        cur = te["current"].to_numpy()
        true_abs = cur + te["target"].to_numpy()
        pred_abs = cur + model.predict(te[FEATURE_COLS])
        rows.append((
            mean_absolute_error(true_abs, pred_abs),
            mean_absolute_error(true_abs, cur),       # persistence
            mean_squared_error(true_abs, pred_abs) ** 0.5,
            r2_score(true_abs, pred_abs),
        ))
    return np.array(rows)


def main() -> None:
    df = fetch_recent(PROM_URL, TARGET_QUERY, int(LOOKBACK_DAYS * 24 * 60), STEP_SECONDS)
    feats = build_training_frame(df, STEP_SECONDS, HORIZON_MIN)
    if len(feats) < 500:
        raise SystemExit(f"학습 샘플 부족: {len(feats)}행 (보존/스텝 확인)")

    cv = walk_forward(feats)
    if len(cv) == 0:
        raise SystemExit("walk-forward fold를 만들 수 없습니다 (데이터 길이 확인)")
    mae, base_mae, rmse, r2 = cv[:, 0], cv[:, 1], cv[:, 2], cv[:, 3]
    wins = int((mae <= base_mae).sum())
    improvement = 1 - mae.mean() / base_mae.mean() if base_mae.mean() else float("nan")

    # 등록 모델은 전체 데이터로 재학습한 것
    model = _fit(feats[FEATURE_COLS], feats["target"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "target_query": TARGET_QUERY,
            "horizon_min": HORIZON_MIN,
            "step_seconds": STEP_SECONDS,
            "unit": "MB",
            "adr": "0010",
            "feature_set": FEATURE_SET,   # 서빙이 코드-모델 짝을 검증하는 데 쓴다
            "model_family": "Ridge",
            "prediction_mode": "delta",   # 출력=변화량, 절대값은 현재값+변화량
            "validation": "walk-forward",
        })
        mlflow.log_params({
            "lookback_days": LOOKBACK_DAYS,
            "step_seconds": STEP_SECONDS,
            "horizon_min": HORIZON_MIN,
            "model": "Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "n_folds_requested": N_FOLDS,
            "fold_hours": FOLD_HOURS,
            "min_train_hours": MIN_TRAIN_HOURS,
            "feature_set": FEATURE_SET,
        })
        mlflow.log_metrics({
            # 헤드라인 metric은 walk-forward 평균 — 단일 분할보다 보수적이고 정직하다
            "mae_mb": float(mae.mean()),
            "mae_median_mb": float(np.median(mae)),
            "mae_worst_mb": float(mae.max()),
            "rmse_mb": float(rmse.mean()),
            "r2": float(r2.mean()),
            "baseline_mae_mb": float(base_mae.mean()),
            "baseline_mae_worst_mb": float(base_mae.max()),
            "mae_improvement_vs_baseline": float(improvement),
            "folds_won_vs_baseline": float(wins),
            "n_folds": float(len(cv)),
            "n_samples": float(len(feats)),
            "n_features": float(len(FEATURE_COLS)),
        })
        signature = infer_signature(feats[FEATURE_COLS], model.predict(feats[FEATURE_COLS]))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=feats[FEATURE_COLS].iloc[:2],
        )
        run_id = run.info.run_id

    print(
        f"[ok] run={run_id} registered={MODEL_NAME} features={FEATURE_SET} "
        f"MAE={mae.mean():.4g}MB (baseline {base_mae.mean():.4g}MB, {improvement:+.1%}) "
        f"worst={mae.max():.4g}MB folds_won={wins}/{len(cv)} n={len(feats)}"
    )


if __name__ == "__main__":
    main()
