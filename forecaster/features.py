"""특징 계산 — train·serve 공유 (ADR-0010).

모델 입력을 원본 시계열에서 만드는 유일한 소스. 학습과 서빙이 모두 이 함수를
쓰므로 특징 정의가 어긋날 수 없다(train/serve skew 방지).

## 설계 이력 — 왜 차분(differencing)인가

v1 설계는 **절대 레벨**을 입력·출력으로 썼다(lag_0..lag_60 → t+30m의 절대값).
프로덕션에서 두 번 같은 방식으로 고장났다: 트리 앙상블은 학습창의 레벨 분포를
암기하고 그 바깥에선 평균 쪽으로 수축하므로, 레벨이 조금만 이동해도 예측이
천장에 갇혀 체계적 편향이 생긴다(v1은 8시간, v4는 40분 만에 persistence보다 열위).

walk-forward 비교(6일, 8 fold, 6시간 테스트 블록)로 네 설계를 실측했다:

| 설계 | MAE 평균 | 최악 fold | baseline 대비 | 승리 fold |
|---|---|---|---|---|
| persistence | 73.6 | 92.4 | — | — |
| abs-GBR (v1 설계) | 85.2 | 265.2 | -15.8% | 6/8 |
| abs-Ridge | 61.3 | 80.8 | +16.6% | 7/8 |
| diff-GBR | 62.8 | 81.0 | +14.6% | 7/8 |
| diff-Ridge (채택) | 59.5 | 74.2 | +19.1% | 8/8 |

v1 설계는 평균적으로 베이스라인보다 **나쁘다** — 중앙값은 멀쩡한데(61.3)
레짐이 이동한 fold에서 265까지 튀어 평균을 망친다. 프로덕션 고장의 재현이다.

차분 설계는 레벨 정보를 입력에서 제거하고 **변화량**만 다룬다. 예측은
`현재값 + 예측된 변화량`으로 복원하므로 레벨을 항상 따라간다 — 레벨 이동에
구조적으로 면역이다.

## 신호의 정체 — 평균 회귀

차분 계열의 자기상관이 **-0.47**(음수)이다. 변화가 다음 스텝에 되돌려진다는
뜻이고, 이는 "느리게 움직이는 레벨 주위의 진동"이다. 그래서 예측 가능한 신호는
lag의 지속성이 아니라 **평균 회귀**다 — `dev_mean_30/60`(최근 평균으로부터의
이탈)이 그 신호를 담는 핵심 특징이고, 선형 모델(Ridge)이 트리보다 잘 맞는 이유다.

주의: 원계열 acf(0.83)는 추세(당시 +49MB/일)가 만든 가짜 상관이었다. 비정상
계열의 acf를 그대로 읽으면 예측 가능성을 과대평가한다 — v1 타깃 선정이 그 함정에
빠졌다. 이후 판단은 차분 계열의 acf로 한다.
"""
import numpy as np
import pandas as pd

# 특징 스키마 식별자. 모델과 코드가 짝이 맞는지 서빙이 검증하는 데 쓴다
# (구 스키마 모델을 새 코드로 서빙하면 에러 없이 조용히 틀린 예측이 나온다).
# 스키마를 바꾸면 이 값을 함께 올릴 것.
FEATURE_SET = "v2-diff"

# 열 순서는 학습·서빙에서 반드시 동일해야 한다(모델은 열 이름이 아니라 순서로
# 입력을 받는다). 이 리스트가 그 계약이다.
FEATURE_COLS = [
    "d_5", "d_15", "d_30", "d_60",      # 최근 m분간 변화량
    "roll_std_30",                       # 최근 변동성
    "dev_mean_30", "dev_mean_60",        # 이동평균으로부터의 이탈 = 평균 회귀 신호
    "hod_sin", "hod_cos",                # 시각(일주기)
]
DIFF_MINUTES = [5, 15, 30, 60]


def _steps(minutes: int, step_seconds: int) -> int:
    """분 → 격자 스텝 수."""
    return int(round(minutes * 60 / step_seconds))


def add_feature_columns(df: pd.DataFrame, step_seconds: int) -> pd.DataFrame:
    """df(index=UTC ts, 열 'y') → 특징 열(가장자리는 NaN).

    레벨(절대값)은 특징에 넣지 않는다 — 그게 v1의 고장 원인이었다.
    """
    y = df["y"]
    f = pd.DataFrame(index=df.index)
    for m in DIFF_MINUTES:
        f[f"d_{m}"] = y - y.shift(_steps(m, step_seconds))
    f["roll_std_30"] = y.rolling(_steps(30, step_seconds)).std()
    f["dev_mean_30"] = y - y.rolling(_steps(30, step_seconds)).mean()
    f["dev_mean_60"] = y - y.rolling(_steps(60, step_seconds)).mean()
    hod = df.index.hour + df.index.minute / 60.0
    f["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    return f


def build_training_frame(df: pd.DataFrame, step_seconds: int, horizon_min: int) -> pd.DataFrame:
    """학습용 프레임. 열 = FEATURE_COLS + 'target'(변화량) + 'current'(복원·베이스라인용).

    target은 절대값이 아니라 **horizon분 후의 변화량**이다. 절대 예측은
    current + target으로 복원한다.
    """
    f = add_feature_columns(df, step_seconds)
    y = df["y"]
    f["current"] = y
    f["target"] = y.shift(-_steps(horizon_min, step_seconds)) - y
    return f.dropna()[FEATURE_COLS + ["current", "target"]]


def build_latest_features(df: pd.DataFrame, step_seconds: int):
    """서빙용. (특징 1행, 현재값)을 반환 — 학습과 동일 로직.

    호출자는 `현재값 + model.predict(특징)`으로 절대 예측을 복원한다.
    """
    f = add_feature_columns(df, step_seconds)
    f["current"] = df["y"]
    f = f.dropna()
    if f.empty:
        raise ValueError("특징 생성에 데이터가 부족합니다 (최소 d_60+roll_60 창 필요)")
    row = f.iloc[[-1]]
    return row[FEATURE_COLS], float(row["current"].iloc[0])
