# aporiax-forecast — 클러스터 텔레메트리 예측 MLOps 제품 테넌트

이 레포는 [gitops-infra](https://github.com/yhsk9200/gitops-infra) 플랫폼의 제품 테넌트입니다. 경계는 gitops-infra의 ADR-0007(첫 제품 테넌트 온보딩 패턴)과 ADR-0010(첫 MLOps 제품 테넌트 + 승격 메커니즘)을 따릅니다.

## 하는 일

클러스터가 자기 텔레메트리(Prometheus 메트릭)를 학습해 예측 모델을 만듭니다. 모델 배포는 이미지 베이킹 PR로 이루어져, 코드 배포와 동일한 감사·롤백 경로를 탑니다 — 모델을 컨테이너 이미지에 구워 넣고 GHCR에 올린 뒤 서빙 Deployment 태그를 범프하는 PR을 머지하는 방식입니다. 롤백은 `git revert` 한 줄로 끝납니다.

## 구조

```
deploy/manifests/   ArgoCD가 동기하는 배포 매니페스트
forecaster/         학습·서빙 공유 특징·조회 로직 (train/serve skew 방지)
train/              학습 코드 + 이미지
serve/              서빙 코드 + 이미지 (모델은 베이킹 시점에 주입)
bake/               champion 모델을 서빙 이미지에 굽는 스크립트
```

## 학습 (train/)

노드 메모리 사용량을 30분 후까지 예측하는 소형 회귀 모델을 학습합니다. Prometheus에서 시계열을 당겨(`query_range`, 5분 격자) lag·rolling·시각(sin/cos) 특징을 만들고, gradient-boosting 회귀를 학습한 뒤 persistence 베이스라인과 비교해 MLflow 레지스트리에 model version을 metric(MAE/RMSE/R² + baseline 대비 개선율)과 함께 등록합니다.

대상 series는 실측으로 정했습니다 — lag 특징이 예측력을 갖고(acf 5분 ≈ 0.58) 일주기가 실재하는(≈ 0.35) 유일한 후보입니다. CPU·load는 노이즈, pod 수는 상수, 파일시스템 사용률은 persistence가 자명하게 이겨 학습 가치가 없었습니다.

접근은 전부 클러스터 내부(ClusterIP)입니다 — Prometheus·MLflow에 tailnet 없이 닿고, MLflow `--serve-artifacts`가 아티팩트 업로드를 프록시해 S3 크레덴셜도 불필요합니다. `train/`의 설정은 모두 환경변수로 덮어쓸 수 있습니다(`TARGET_QUERY`·`HORIZON_MIN`·`LOOKBACK_DAYS` 등).

## 서빙 · 승격 (serve/, bake/)

서빙 앱(FastAPI)은 **모델을 이미지에 구워** 실행합니다 — 시작 시 `/app/model`을 로드하고 런타임엔 MLflow/NAS를 보지 않습니다. 그래서 아티팩트 저장소 장애가 서비스 중 모델에 무영향이고, 롤백은 이전 이미지 태그로 되돌리는 `git revert` 한 줄입니다. 예측 입력(특징)은 현재 Prometheus에서 만들며, 특징 로직은 학습과 `forecaster/`를 공유해 skew를 배제합니다.

- `GET /predict` — 지금 기준 30분 후 예측(JSON) + persistence 베이스라인 + 모델 버전
- `GET /metrics` — 같은 예측을 Prometheus 게이지(`forecast_node_memory_mb` 등)로 노출
- `GET /healthz` — probe

**승격 = 이미지 베이킹 PR** (`.github/workflows/bake.yaml`). 사람이 MLflow에서 `champion` alias를 지정한 뒤 워크플로우를 수동 실행하면: 러너를 tailnet에 임시 편입 → champion 모델 pull → 서빙 이미지에 굽기 → GHCR push → 서빙 태그를 범프하는 PR 자동 생성. 사람이 PR을 머지하면 ArgoCD가 롤아웃합니다. 즉 모델 배포가 코드 배포와 동일한 CI·리뷰·롤백 경로를 탑니다.

베이킹 실행에는 시크릿 2개가 필요합니다(수동 발급): `TS_AUTHKEY`(러너 tailnet 편입), `PROMOTE_PAT`(승격 PR이 CI를 트리거하도록 — GITHUB_TOKEN이 만든 PR은 워크플로우를 트리거하지 못함).

## 배포

gitops-infra의 `product-forecast-serving` Application이 이 레포의 `deploy/manifests`를 동기합니다. namespace는 `product-forecast`, AppProject는 `product-forecast`입니다.

## 참고

설계 근거와 단계별 계획은 gitops-infra의 [`docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md`](https://github.com/yhsk9200/gitops-infra/blob/main/docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md)를 참고하세요.
