# aporiax-forecast — 클러스터 텔레메트리 예측 MLOps 제품 테넌트

이 레포는 [gitops-infra](https://github.com/yhsk9200/gitops-infra) 플랫폼의 제품 테넌트입니다. 경계는 gitops-infra의 ADR-0007(첫 제품 테넌트 온보딩 패턴)과 ADR-0010(첫 MLOps 제품 테넌트 + 승격 메커니즘)을 따릅니다.

## 하는 일

클러스터가 자기 텔레메트리(Prometheus 메트릭)를 학습해 예측 모델을 만듭니다. 모델 배포는 이미지 베이킹 PR로 이루어져, 코드 배포와 동일한 감사·롤백 경로를 탑니다 — 모델을 컨테이너 이미지에 구워 넣고 GHCR에 올린 뒤 서빙 Deployment 태그를 범프하는 PR을 머지하는 방식입니다. 롤백은 `git revert` 한 줄로 끝납니다.

## 구조

```
deploy/manifests/   ArgoCD가 동기하는 배포 매니페스트
train/              학습 코드 (예정)
serve/              서빙 코드 (예정)
```

## 배포

gitops-infra의 `product-forecast-serving` Application이 이 레포의 `deploy/manifests`를 동기합니다. namespace는 `product-forecast`, AppProject는 `product-forecast`입니다.

## 참고

설계 근거와 단계별 계획은 gitops-infra의 [`docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md`](https://github.com/yhsk9200/gitops-infra/blob/main/docs/adr/0010-telemetry-forecaster-tenant-and-promotion.md)를 참고하세요.
