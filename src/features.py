"""이상 탐지 입력을 만드는 함수 모음.

모든 함수는 로트 단위로 계산한다. 로트 사이는 측정 공백이라 이어 붙이지 않는다.
빈 시각은 채우지 않고 NaN으로 둔다.
"""

import numpy as np
import pandas as pd

# 가열·균열존 온도·제어출력 (EDA 범위)
CORE_SENSORS = [
    f'{prefix}-Z{zone}' for zone in range(3, 7) for prefix in ('TC', 'OP')
]

# 모델 입력: Z3는 큰 진폭이 점수를 지배해서 뺀다 (탐색 14절)
MODEL_FEATURES = [
    f'{prefix}-Z{zone}' for zone in range(4, 7) for prefix in ('TC', 'OP')
]

# 로트 시작부터 쌓아야 하는 최소 분 수 (탐색 17~19절의 기본값)
DEFAULT_WARMUP = 30

# MAD를 정규분포 표준편차 척도로 맞추는 상수
MAD_TO_STD = 1.4826


def candidate_rows(df_cal_labeled, drop_missing=CORE_SENSORS):
    """온도 유지 비교 후보 구간만 남긴다.

    drop_missing에 있는 센서 중 하나라도 결측인 행은 뺀다(탐색 12절).
    그런 행은 99행(0.03%)이고, 그중 75행이 240044의 로트 시작에 몰려 있다.
    """
    data = df_cal_labeled.loc[df_cal_labeled['COMPARE_CANDIDATE']]
    if drop_missing:
        data = data.dropna(subset=list(drop_missing))
    return data.copy()


def minute_means(rows, features=MODEL_FEATURES):
    """로트별 1분 평균. 인덱스는 (LOT_NO, MEAS_DT)다.

    1초 떨림은 제어기 반응이라(EDA 9절) 1분 평균으로 걸러 낸다.
    데이터가 없는 분은 NaN으로 남는다.
    """
    return (
        rows.set_index('MEAS_DT')
        .groupby('LOT_NO')[list(features)]
        .resample('1min')
        .mean()
    )


def lot_centered(rows, features=MODEL_FEATURES):
    """각 변수에서 로트 중앙값을 뺀다 (탐색 15절).

    로트 수준 차이(강종 baseline 이동 등)를 빼고 로트 안의 변화만 남긴다.
    로트 전체 중앙값은 미래 값도 쓰므로 사후 분석용이다.
    """
    return rows[list(features)].sub(
        rows.groupby('LOT_NO')[list(features)].transform('median')
    )


def lot_robust_z(minute):
    """로트 전체 중앙값·MAD로 만든 robust z (탐색 17-2절).

    사후 분석용 기준 점수의 재료다. 로트 전체를 다 본 뒤에야 계산할 수 있다.
    """
    center = minute.groupby(level='LOT_NO').transform('median')
    scale = (
        (minute - center).abs()
        .groupby(level='LOT_NO').transform('median') * MAD_TO_STD
    )
    return (minute - center) / scale


def nan_mad(values):
    """NaN을 무시한 MAD. 롤링 창 안에 shift로 생긴 NaN이 있어도 계산한다."""
    return np.nanmedian(np.abs(values - np.nanmedian(values)))


def past_center(minute, window='expanding', warmup=DEFAULT_WARMUP):
    """각 분의 기준값: 직전까지의 중앙값. 현재 1분은 넣지 않는다.

    window='expanding'이면 로트 시작부터, 정수면 직전 window분이다.
    값이 warmup개 이상 쌓여야 계산한다.
    """
    def per_lot(series):
        past = series.shift(1)
        if window == 'expanding':
            return past.expanding(min_periods=warmup).median()
        return past.rolling(window, min_periods=warmup).median()

    return minute.groupby(level='LOT_NO').transform(per_lot)


def past_scale(minute, window, warmup=DEFAULT_WARMUP):
    """직전 window분의 MAD × 1.4826 (이동 척도). 현재 1분은 넣지 않는다."""
    def per_lot(series):
        return (
            series.shift(1)
            .rolling(window, min_periods=warmup)
            .apply(nan_mad, raw=True)
        )

    return minute.groupby(level='LOT_NO').transform(per_lot) * MAD_TO_STD


def sum_of_squares(z):
    """변수별 z의 제곱합. 모든 변수의 z가 있는 분만 남긴다.

    반환값은 LOT_NO, MEAS_DT, SUMSQ 컬럼을 가진 표다.
    """
    z = z.dropna()
    return (z ** 2).sum(axis=1).rename('SUMSQ').reset_index()
