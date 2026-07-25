# aporiax-forecast — 클러스터 텔레메트리 예측 MLOps 제품 테넌트

이 레포는 [gitops-infra](https://github.com/yhsk9200/gitops-infra) 플랫폼의 제품 테넌트입니다. 경계는 gitops-infra의 ADR-0007(첫 제품 테넌트 온보딩 패턴)과 ADR-0010(첫 MLOps 제품 테넌트 + 승격 메커니즘)을 따릅니다.

## 하는 일

클러스터가 자기 텔레메트리(Prometheus 메트릭)를 학습해 예측 모델을 만듭니다. 모델 배포는 이미지 베이킹 PR로 이루어져, 코드 배포와 동일한 감사·롤백 경로를 탑니다 — 모델을 컨테이너 이미지에 구워 넣고 GHCR에 올린 뒤 서빙 Deployment 태그를 범프하는 PR을 머지하는 방식입니다. 롤백은 `git revert` 한 줄로 끝납니다.

## 구조

```
deploy/manifests/   ArgoCD가 동기하는 배포 매니페스트
train/              학습 코드 + 이미지
serve/              서빙 코드 (예정)
```

## 학습 (train/)

노드 메모리 사용량을 30분 후까지 예측하는 소형 회귀 모델을 학습합니다. Prometheus에서 시계열을 당겨(`query_range`, 5분 격자) lag·rolling·시각(sin/cos) 특징을 만들고, gradient-boosting 회귀를 학습한 뒤 persistence 베이스라인과 비교해 MLflow 레지스트리에 model version을 metric(MAE/RMSE/R² + baseline 대비 개선율)과 함께 등록합니다.

대상 series는 실측으로 정했습니다 — lag 특징이 예측력을 갖고(acf 5분 ≈ 0.58) 일주기가 실재하는(≈ 0.35) 유일한 후보입니다. CPU·load는 노이즈, pod 수는 상수, 파일시스템 사용률은 persistence가 자명하게 이겨 학습 가치가 없었습니다.

접근은 전부 클러스터 내부(ClusterIP)입니다 — Prometheus·MLflow에 tailnet 없이 닿고, MLflow `--serve-artifacts`가 아티팩트 업로드를 프록시해 S3 크레덴셜도 불필요합니다. `train/`의 설정은 모두 환경변수로 덮어쓸 수 있습니다(`TARGET_QUERY`·`HORIZON_MIN`·`LOOKBACK_DAYS` 등).

## 배포

gitops-infra의 `product-forecast-serving` Application이 이 레포의 `deploy/manifests`를 동기합니다. namespace는 `product-forecast`, AppProject는 `product-forecast`입니다.

## 참고

설계 근거와 단계별 계획은 gitops-infra의 [`docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md`](https://github.com/yhsk9200/gitops-infra/blob/main/docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md)를 참고하세요.
