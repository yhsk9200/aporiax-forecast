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

노드 메모리 사용량을 30분 후까지 예측하는 소형 회귀 모델을 학습합니다. Prometheus에서 시계열을 당겨(`query_range`, 5분 격자) **차분(변화량) 특징**을 만들고, Ridge 회귀로 30분 후의 변화량을 학습합니다. 절대 예측은 `현재값 + 예측된 변화량`으로 복원합니다. 검증은 **walk-forward**(확장창, fold별 6시간 테스트)로 하고, persistence 베이스라인과 fold별 승률까지 metric으로 남깁니다.

특징에 절대 레벨을 넣지 않는 이유는 초기 설계가 프로덕션에서 두 번 고장난 경험에서 왔습니다. gradient-boosting을 절대 레벨에 학습시키면 학습창의 레벨 분포를 암기하고 그 바깥에선 평균 쪽으로 수축해, 레벨이 조금만 이동해도 예측이 천장에 갇힙니다. walk-forward로 네 설계를 비교한 결과(6일·8 fold):

| 설계 | MAE 평균 | 최악 fold | baseline 대비 | 승리 fold |
|---|---|---|---|---|
| persistence | 73.6 | 92.4 | — | — |
| abs-GBR (초기 설계) | 85.2 | 265.2 | −15.8% | 6/8 |
| abs-Ridge | 61.3 | 80.8 | +16.6% | 7/8 |
| diff-GBR | 62.8 | 81.0 | +14.6% | 7/8 |
| **diff-Ridge (채택)** | **59.5** | **74.2** | **+19.1%** | **8/8** |

예측 가능한 신호는 lag의 지속성이 아니라 **평균 회귀**입니다 — 차분 계열의 자기상관이 −0.47(음수)로, 변화가 다음 스텝에 되돌려집니다. `dev_mean_30/60`(이동평균으로부터의 이탈)이 그 신호를 담고, 그래서 선형 모델이 트리보다 잘 맞습니다.

타깃 선정 시 주의할 점도 실측으로 배웠습니다. 원계열의 자기상관(0.83)은 당시 추세(+49MB/일)가 만든 가짜 상관이었고, 추세가 사라지자 초기 모델이 붕괴했습니다. 비정상 시계열의 acf를 그대로 읽으면 예측 가능성을 과대평가하므로, 이후 판단은 **차분 계열의 acf**로 합니다. 같은 기준으로 재보면 CPU·load·pod 수·네트워크는 차분 acf가 음수이거나 변동계수가 과대해 신호가 없고, 파일시스템 사용률만 약한 양의 차분 acf(+0.12)와 실제 추세(+0.39%p/일)를 갖지만 선형 외삽으로 충분해 ML 가치가 낮습니다.

접근은 전부 클러스터 내부(ClusterIP)입니다 — Prometheus·MLflow에 tailnet 없이 닿고, MLflow `--serve-artifacts`가 아티팩트 업로드를 프록시해 S3 크레덴셜도 불필요합니다. `train/`의 설정은 모두 환경변수로 덮어쓸 수 있습니다(`TARGET_QUERY`·`HORIZON_MIN`·`LOOKBACK_DAYS` 등).

## 서빙 · 승격 (serve/, bake/)

서빙 앱(FastAPI)은 **모델을 이미지에 구워** 실행합니다 — 시작 시 `/app/model`을 로드하고 런타임엔 MLflow/NAS를 보지 않습니다. 그래서 아티팩트 저장소 장애가 서비스 중 모델에 무영향이고, 롤백은 이전 이미지 태그로 되돌리는 `git revert` 한 줄입니다. 예측 입력(특징)은 현재 Prometheus에서 만들며, 특징 로직은 학습과 `forecaster/`를 공유해 skew를 배제합니다.

그 위에 스키마 검증을 하나 더 둡니다. 학습은 특징 스키마 식별자(`feature_set`)를 MLflow 태그로 남기고, 베이킹이 그것을 `meta.json`에 옮기며, 서빙은 기동 시 자기 코드의 `FEATURE_SET`과 일치하는지 확인해 다르면 **기동을 거부합니다**(fail closed). 구 스키마 모델을 새 코드로 서빙하면 예외 없이 조용히 틀린 예측이 나오기 때문입니다.

- `GET /predict` — 지금 기준 30분 후 예측(JSON) + 예측 변화량 + persistence 베이스라인 + 모델 버전·스키마
- `GET /metrics` — 같은 예측을 Prometheus 게이지(`forecast_node_memory_mb` 등)로 노출
- `GET /healthz` — probe

**승격 = 이미지 베이킹 PR** (`.github/workflows/bake.yaml`). 사람이 MLflow에서 `champion` alias를 지정한 뒤 워크플로우를 수동 실행하면: 러너를 tailnet에 임시 편입 → champion 모델 pull → 서빙 이미지에 굽기 → GHCR push → 서빙 태그를 범프하는 PR 자동 생성. 사람이 PR을 머지하면 ArgoCD가 롤아웃합니다. 즉 모델 배포가 코드 배포와 동일한 CI·리뷰·롤백 경로를 탑니다.

베이킹 실행에는 시크릿 2개가 필요합니다(수동 발급): `TS_AUTHKEY`(러너 tailnet 편입), `PROMOTE_PAT`(승격 PR이 CI를 트리거하도록 — GITHUB_TOKEN이 만든 PR은 워크플로우를 트리거하지 못함).

## 배포

gitops-infra의 `product-forecast-serving` Application이 이 레포의 `deploy/manifests`를 동기합니다. namespace는 `product-forecast`, AppProject는 `product-forecast`입니다.

## 참고

설계 근거와 단계별 계획은 gitops-infra의 [`docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md`](https://github.com/yhsk9200/gitops-infra/blob/main/docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md)를 참고하세요.
