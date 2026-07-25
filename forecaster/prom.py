"""Prometheus 조회 — train·serve 공유 (ADR-0010).

query_range 결과를 균일 격자 시계열(MB)로 만든다. 학습과 서빙이 같은 단위·
같은 방식으로 데이터를 읽도록 단일 소스로 둔다.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
from prometheus_api_client import PrometheusConnect


def fetch_series(prom_url: str, query: str, start, end, step_seconds: int) -> pd.DataFrame:
    """[start, end] 구간 query_range → DataFrame(index=UTC ts, y in MB)."""
    prom = PrometheusConnect(url=prom_url, disable_ssl=True)
    data = prom.custom_query_range(
        query=query, start_time=start, end_time=end, step=str(step_seconds)
    )
    if not data:
        raise RuntimeError(f"Prometheus 빈 결과: query={query!r}")
    values = data[0]["values"]  # [[ts, "val"], ...]
    df = pd.DataFrame(values, columns=["ts", "y"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    df["y"] = df["y"].astype(float) / 1e6  # bytes → MB (학습·서빙 동일 단위)
    return df[~df["ts"].duplicated(keep="last")].set_index("ts").sort_index()


def fetch_recent(prom_url: str, query: str, minutes: int, step_seconds: int) -> pd.DataFrame:
    """최근 N분 구간을 당긴다(서빙: 현재 특징 계산용)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return fetch_series(prom_url, query, start, end, step_seconds)
