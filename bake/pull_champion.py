#!/usr/bin/env python3
"""champion 모델을 MLflow에서 내려받아 서빙 빌드 컨텍스트(./model/)에 놓는다
(ADR-0010 D1 베이킹 1단계).

sklearn 추정기만 joblib로 저장하고(런타임에 mlflow 불필요 → 서빙 이미지 경량),
버전·run_id·metric을 meta.json에 남긴다. tailnet에 임시 편입된 러너에서
MLFLOW_TRACKING_URI(NodePort)로 접근한다.

env:
  MLFLOW_TRACKING_URI  (필수)
  MODEL_NAME           기본 telemetry-forecaster
  ALIAS                기본 champion   — alias로 승격 대상 지정
  MODEL_VERSION        (선택) 지정 시 alias 대신 이 버전을 직접 pull (테스트/핀)
"""
import json
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def main() -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name = os.environ.get("MODEL_NAME", "telemetry-forecaster")
    alias = os.environ.get("ALIAS", "champion")
    version_override = os.environ.get("MODEL_VERSION")

    client = MlflowClient()
    if version_override:
        mv = client.get_model_version(name, version_override)
        model_uri = f"models:/{name}/{version_override}"
        picked = f"version {version_override}"
    else:
        mv = client.get_model_version_by_alias(name, alias)
        model_uri = f"models:/{name}@{alias}"
        picked = f"alias {alias} → version {mv.version}"

    model = mlflow.sklearn.load_model(model_uri)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.joblib")

    run = client.get_run(mv.run_id)
    meta = {
        "model_name": name,
        "version": mv.version,
        "run_id": mv.run_id,
        "alias": None if version_override else alias,
        "metrics": run.data.metrics,
    }
    with open("model/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"version={mv.version}\n")
            fh.write(f"run_id={mv.run_id}\n")
            fh.write(f"mae_mb={run.data.metrics.get('mae_mb')}\n")

    print(f"pulled {name} ({picked}), run {mv.run_id}, mae_mb={run.data.metrics.get('mae_mb')}")


if __name__ == "__main__":
    main()
