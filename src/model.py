"""이상 탐지 모델과 평가 함수 모음.

정비 기록은 데이터사전상 [가정]으로 붙인 기록이라 정답 라벨로 쓰지 않는다.
그래서 평가는 "사후 분석 점수의 사건을 실시간 점수가 얼마나 재현하나"로 한다.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features import (
    DEFAULT_WARMUP, MAD_TO_STD, MODEL_FEATURES,
    lot_centered, lot_robust_z, past_center, sum_of_squares,
)

# 이상 기록이 있어 학습에서 빼고 평가에만 쓰는 로트 (탐색 13절)
HOLDOUT_LOTS = [240044, 240069, 240117, 240133]

# 경보 기준: 학습 로트 점수의 상위 1%
ALARM_QUANTILE = 0.99

# 사건을 서로 비교할 때 앞뒤로 두는 여유
EVENT_TOLERANCE = pd.Timedelta('2min')


# ---------------------------------------------------------------------------
# 사건 묶기와 비교
# ---------------------------------------------------------------------------

def mark_set(scored, holdout_lots=HOLDOUT_LOTS):
    """학습(train)·평가(holdout) 구분 컬럼 SET을 붙인다."""
    scored = scored.copy()
    scored['SET'] = np.where(
        scored['LOT_NO'].isin(holdout_lots), 'holdout', 'train'
    )
    return scored


def train_quantile(scored, score_col, q=ALARM_QUANTILE):
    """학습 로트 점수의 분위수. 경보 기준으로 쓴다."""
    return scored.loc[scored['SET'].eq('train'), score_col].quantile(q)


def make_events(scored, score_col, limit, holdout_lots=HOLDOUT_LOTS):
    """1분 점수가 limit를 넘는 연속 구간을 사건 하나로 묶는다.

    1분 넘게 끊기면 다른 사건으로 본다. 반환 컬럼:
    LOT_NO, SET, START, END, MINUTES, PEAK
    """
    rows = []

    for lot_no, lot in scored.groupby('LOT_NO'):
        lot = lot.sort_values('MEAS_DT').reset_index(drop=True)
        over = lot[score_col] > limit

        # 앞 분이 기준을 넘지 않았거나 1분 넘게 떨어져 있으면 새 사건
        new_event = over & ~(
            over.shift(fill_value=False)
            & lot['MEAS_DT'].diff().eq(pd.Timedelta('1min'))
        )
        event_id = new_event.cumsum()

        for _, event in lot.loc[over].groupby(event_id.loc[over]):
            rows.append({
                'LOT_NO': lot_no,
                'SET': 'holdout' if lot_no in holdout_lots else 'train',
                'START': event['MEAS_DT'].min(),
                'END': event['MEAS_DT'].max(),
                'MINUTES': len(event),
                'PEAK': event[score_col].max(),
            })

    columns = ['LOT_NO', 'SET', 'START', 'END', 'MINUTES', 'PEAK']
    return pd.DataFrame(rows, columns=columns)


def overlaps(event, others, tolerance=EVENT_TOLERANCE):
    """같은 로트의 다른 사건 목록과 앞뒤 여유를 두고 겹치는지."""
    same_lot = others.loc[others['LOT_NO'].eq(event['LOT_NO'])]
    return bool((
        (same_lot['START'] <= event['END'] + tolerance)
        & (same_lot['END'] >= event['START'] - tolerance)
    ).any())


def event_recall(reference, events, first_scored):
    """기준 사건 중 events가 잡은 수와 평가 대상 수.

    first_scored(로트별 첫 점수 시각)보다 먼저 끝난 기준 사건은 뺀다.
    실시간 점수가 아직 없는 로트 시작 구간이라 잡을 수 없기 때문이다.
    """
    target = reference.loc[
        reference['END'] >= reference['LOT_NO'].map(first_scored)
    ]
    hit = target.apply(lambda e: overlaps(e, events), axis=1)
    return int(hit.sum()), len(target)


def rank_corr(scored, reference, col='SUMSQ', ref_col='SUMSQ'):
    """같은 (LOT_NO, MEAS_DT)끼리 맞춘 스피어만 순위 상관."""
    merged = scored[['LOT_NO', 'MEAS_DT', col]].merge(
        reference[['LOT_NO', 'MEAS_DT', ref_col]].rename(columns={ref_col: '_REF'}),
        on=['LOT_NO', 'MEAS_DT']
    )
    return merged[col].corr(merged['_REF'], method='spearman')


# ---------------------------------------------------------------------------
# 사후 분석용 점수 (비교 기준)
# ---------------------------------------------------------------------------

def lot_z_score(minute, holdout_lots=HOLDOUT_LOTS):
    """로트 기준 robust z 제곱합 (탐색 17-2절).

    로트 전체를 본 뒤 계산하는 사후 분석용 점수다. 15절 IsolationForest와
    순위 상관이 약 0.91이라 실시간 모델의 비교 기준으로 쓴다.
    """
    scored = mark_set(sum_of_squares(lot_robust_z(minute)), holdout_lots)
    limit = train_quantile(scored, 'SUMSQ')
    events = make_events(scored, 'SUMSQ', limit, holdout_lots)
    return scored, limit, events


def iforest_lot_centered(rows, features=MODEL_FEATURES,
                         holdout_lots=HOLDOUT_LOTS, random_state=42):
    """로트 중앙값을 뺀 입력의 IsolationForest (탐색 15절).

    반환값
    - minute: 1분 평균 이상 점수 (LOT_NO, MEAS_DT, ANOMALY_SCORE)
    - limit: 학습 로트 행 단위 점수의 상위 1%
    - events: 1분 평균이 limit를 넘는 사건
    - row_scores: 행 단위 이상 점수 (rows와 같은 순서의 배열)
    """
    centered = lot_centered(rows, features)
    is_train = ~rows['LOT_NO'].isin(holdout_lots)

    model = IsolationForest(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(centered.loc[is_train])

    # score_samples는 정상일수록 크므로 부호를 바꿔 클수록 이상
    scored = rows[['LOT_NO', 'MEAS_DT']].copy()
    scored['ANOMALY_SCORE'] = -model.score_samples(centered)
    limit = scored.loc[is_train.to_numpy(), 'ANOMALY_SCORE'].quantile(ALARM_QUANTILE)

    minute = (
        scored.set_index('MEAS_DT')
        .groupby('LOT_NO')['ANOMALY_SCORE']
        .resample('1min')
        .mean()
        .dropna()
        .reset_index()
    )
    events = make_events(minute, 'ANOMALY_SCORE', limit, holdout_lots)
    return minute, limit, events, scored['ANOMALY_SCORE'].to_numpy()


# ---------------------------------------------------------------------------
# 실시간 기준 모델
# ---------------------------------------------------------------------------

class RealtimeBaseline:
    """실시간 기준 모델: 누적·고정 척도 robust z 제곱합 (탐색 18~19절).

    각 분의 점수는 과거 값만으로 계산한다.
      기준값  로트 시작부터 직전 1분까지의 중앙값
      척도    학습 로트에서 변수별로 정한 고정값
      점수    변수별 z의 제곱합
      경보    학습 로트 점수의 상위 1% 초과

    fit_warmup: 척도와 경보 기준을 정할 때 쓰는 최소 기준 구간(분).
        기준값이 안정된 30분 이후의 학습 데이터로 정한다.
    score_warmup: 점수를 내기 시작하는 최소 기준 구간(분).
        10분으로 줄여도 오경보가 늘지 않았다(탐색 19절).
    """

    def __init__(self, fit_warmup=DEFAULT_WARMUP, score_warmup=10,
                 quantile=ALARM_QUANTILE):
        self.fit_warmup = fit_warmup
        self.score_warmup = score_warmup
        self.quantile = quantile
        self.scale_ = None
        self.limit_ = None

    def fit(self, minute, train_lots):
        """학습 로트로 변수별 고정 척도와 경보 기준을 정한다."""
        is_train = minute.index.get_level_values('LOT_NO').isin(train_lots)
        train = minute.loc[is_train]

        deviation = train - past_center(train, 'expanding', self.fit_warmup)
        self.scale_ = deviation.abs().median() * MAD_TO_STD

        z = deviation / self.scale_
        self.limit_ = sum_of_squares(z)['SUMSQ'].quantile(self.quantile)
        return self

    def score(self, minute, warmup=None):
        """1분 단위 점수. 반환 컬럼: LOT_NO, MEAS_DT, SUMSQ"""
        warmup = self.score_warmup if warmup is None else warmup
        center = past_center(minute, 'expanding', warmup)
        return sum_of_squares((minute - center) / self.scale_)

    def zscores(self, minute, warmup=None):
        """변수별 z. 경보가 났을 때 어느 변수가 벗어났는지 볼 때 쓴다."""
        warmup = self.score_warmup if warmup is None else warmup
        center = past_center(minute, 'expanding', warmup)
        return ((minute - center) / self.scale_).dropna()
