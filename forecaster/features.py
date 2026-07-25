"""특징 계산 — train·serve 공유 (ADR-0010).

모델 입력(10개 특징)을 원본 시계열에서 만드는 유일한 소스. 학습과 서빙이
모두 이 함수를 쓰므로 특징 정의가 어긋날 수 없다(train/serve skew 방지).
"""
import numpy as np
import pandas as pd

# 특징 순서는 학습·서빙에서 반드시 동일해야 한다(모델은 열 이름이 아니라
# 순서로 입력을 받는다). 이 리스트가 그 계약이다.
FEATURE_COLS = [
    "lag_0", "lag_5", "lag_15", "lag_30", "lag_60",
    "roll_mean_30", "roll_std_30", "roll_mean_60",
    "hod_sin", "hod_cos",
]
LAG_MINUTES = [0, 5, 15, 30, 60]


def _steps(minutes: int, step_seconds: int) -> int:
    """분 → 격자 스텝 수."""
    return int(round(minutes * 60 / step_seconds))


def add_feature_columns(df: pd.DataFrame, step_seconds: int) -> pd.DataFrame:
    """df(index=UTC ts, 열 'y') → 특징 열(가장자리는 NaN)."""
    f = pd.DataFrame(index=df.index)
    for m in LAG_MINUTES:
        f[f"lag_{m}"] = df["y"].shift(_steps(m, step_seconds))
    f["roll_mean_30"] = df["y"].rolling(_steps(30, step_seconds)).mean()
    f["roll_std_30"] = df["y"].rolling(_steps(30, step_seconds)).std()
    f["roll_mean_60"] = df["y"].rolling(_steps(60, step_seconds)).mean()
    hod = df.index.hour + df.index.minute / 60.0
    f["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    return f


def build_training_frame(df: pd.DataFrame, step_seconds: int, horizon_min: int) -> pd.DataFrame:
    """학습용: 특징 + horizon분 후 target, NaN 제거. 열 = FEATURE_COLS + 'target'."""
    f = add_feature_columns(df, step_seconds)
    f["target"] = df["y"].shift(-_steps(horizon_min, step_seconds))
    return f.dropna()[FEATURE_COLS + ["target"]]


def build_latest_features(df: pd.DataFrame, step_seconds: int) -> pd.DataFrame:
    """서빙용: 가장 최근의 완전한 특징 1행(target 없음). 학습과 동일 로직."""
    f = add_feature_columns(df, step_seconds).dropna()
    if f.empty:
        raise ValueError("특징 생성에 데이터가 부족합니다 (최소 lag_60+roll_60 창 필요)")
    return f.iloc[[-1]][FEATURE_COLS]
