# [markdown]
# # 02. 이상 점수 방법 비교
#
# 실시간 기준 모델을 정하기 전에 확인한 내용을 다시 계산한다.
#
# 1. 원래 값으로 학습한 IsolationForest는 고장보다 로트·강종 차이를 먼저 잡는다.
# 2. 로트 중앙값을 빼면 로트 차이가 사라지고, 여러 구역이 함께 움직이는 짧은 사건이 남는다.
# 3. 그 IsolationForest 점수는 단순한 "로트 기준 robust z 제곱합"과 거의 같다(순위 상관 약 0.91).
#    직전 30분 기준 z는 다른 것을 본다(0.48).
#
# 정비 기록은 정답 라벨로 쓰지 않는다(데이터사전: 고장모드는 배정한 값).
# 실행: 프로젝트 루트에서 `python analysis/02_model_compare.py`

# 설정
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # 대화형 창에서 셀 단위로 실행할 때
    ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import IsolationForest

from src.preprocess import load_data, attach_grade, label_operating_state
from src.features import (
    CORE_SENSORS, MODEL_FEATURES, DEFAULT_WARMUP,
    candidate_rows, minute_means, past_center, past_scale, sum_of_squares,
)
from src.model import (
    HOLDOUT_LOTS, iforest_lot_centered, lot_z_score,
    make_events, mark_set, train_quantile, event_recall, rank_corr,
)
from src.plot_utils import set_korean_font, save_figure

set_korean_font()
pd.set_option('display.width', 200)

# 데이터 준비
df_cal, df_run_evt_cal, _ = load_data(ROOT / 'data')
df_cal_labeled = label_operating_state(df_cal, attach_grade(df_cal, df_run_evt_cal))

# 행 단위 입력: 비교 후보 구간에서 TC/OP-Z3~Z6 결측 행을 뺀 것
rows = candidate_rows(df_cal_labeled)
# 1분 평균 입력: 비교 후보 구간 전체 (빈 분은 NaN)
minute = minute_means(candidate_rows(df_cal_labeled, drop_missing=None))

is_train = ~rows['LOT_NO'].isin(HOLDOUT_LOTS)
print('행 단위 입력:', len(rows), '/ 학습:', int(is_train.sum()), '/ 평가:', int((~is_train).sum()))
print('평가 로트(이상 기록이 있어 학습에서 뺌):', HOLDOUT_LOTS)

# 1. IsolationForest 입력 세 가지
def iforest_raw(features):
    """원래 값으로 학습한 IsolationForest의 행 단위 점수 (클수록 이상)"""
    model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(rows.loc[is_train, features])
    return -model.score_samples(rows[features])


if_scores = pd.DataFrame({'LOT_NO': rows['LOT_NO'].to_numpy()})
if_scores['원래 값 (Z3~Z6)'] = iforest_raw(CORE_SENSORS)
if_scores['원래 값 (Z4~Z6)'] = iforest_raw(MODEL_FEATURES)

# 로트 중앙값을 뺀 입력 (탐색 15절). 사건 비교에도 이 모델을 쓴다
minute_c, limit_c, events_c, row_scores_c = iforest_lot_centered(rows)
if_scores['로트 중앙값 제거 (Z4~Z6)'] = row_scores_c

score_cols = ['원래 값 (Z3~Z6)', '원래 값 (Z4~Z6)', '로트 중앙값 제거 (Z4~Z6)']
over_pct = pd.DataFrame({
    col: if_scores.groupby('LOT_NO')[col].apply(
        lambda s, col=col: (s > if_scores.loc[is_train.to_numpy(), col].quantile(0.99)).mean() * 100
    )
    for col in score_cols
}).round(1)
over_pct.insert(0, 'GRADE', rows.groupby('LOT_NO')['GRADE'].first())
over_pct.insert(1, 'SET', np.where(over_pct.index.isin(HOLDOUT_LOTS), '평가', '학습'))

print('로트별 기준 초과 비율(%): 학습 로트 점수 상위 1% 기준')
print(over_pct.to_string())
print('\n원래 값 모델은 240038(SPCEN, OP-Z4 약 70)과 SPHC 로트를 이상으로 본다.')
print('로트 중앙값을 빼면 이 차이가 크게 줄어든다.')

# 그림 4: 로트별 점수 분포
lot_order = (
    rows.groupby('LOT_NO')
    .agg(GRADE=('GRADE', 'first'), START=('MEAS_DT', 'min'))
    .sort_values(['GRADE', 'START'])
)
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
for ax, col in zip(axes, score_cols):
    boxes = ax.boxplot(
        [if_scores.loc[if_scores['LOT_NO'] == lot_no, col] for lot_no in lot_order.index],
        patch_artist=True, showfliers=False
    )
    for box, lot_no in zip(boxes['boxes'], lot_order.index):
        box.set_facecolor('tomato' if lot_no in HOLDOUT_LOTS else 'lightsteelblue')
    ax.axhline(if_scores.loc[is_train.to_numpy(), col].quantile(0.99),
               color='gray', linestyle='--', label='학습 99% 기준')
    ax.set_title(f'IsolationForest 이상 점수: {col} (빨강: 평가 로트)')
    ax.set_ylabel('이상 점수')
    ax.grid(axis='y', alpha=0.3)
axes[0].legend()
axes[-1].set_xticks(range(1, len(lot_order) + 1))
axes[-1].set_xticklabels([f'{r.GRADE}\n{lot_no}' for lot_no, r in lot_order.iterrows()], fontsize=8)
plt.tight_layout()
save_figure(fig, '04_iforest_score_by_lot')

# 2. 로트 중앙값 제거 IsolationForest의 사건
print(f'사건: {len(events_c)}개 (학습 {int(events_c["SET"].eq("train").sum())}, '
      f'평가 {int(events_c["SET"].eq("holdout").sum())}), 길이 '
      f'{events_c["MINUTES"].min()}~{events_c["MINUTES"].max()}분')
print(events_c.sort_values('PEAK', ascending=False).head(10).round({'PEAK': 3}).to_string(index=False))

# 3. 단순한 z 점수와 비교
# (가) 로트 기준 z 제곱합: 로트 전체 중앙값·MAD (사후 분석용)
scored_lotz, limit_lotz, events_lotz = lot_z_score(minute)

# (나) 직전 30분 기준 z: 이동 중앙값·이동 MAD (실시간 가능)
center_30 = past_center(minute, window=DEFAULT_WARMUP, warmup=DEFAULT_WARMUP)
scale_30 = past_scale(minute, window=DEFAULT_WARMUP, warmup=DEFAULT_WARMUP)
is_train_minute = ~minute.index.get_level_values('LOT_NO').isin(HOLDOUT_LOTS)
# MAD가 거의 0인 구간에서 z가 폭주하지 않게 학습 로트 척도의 하위 10%를 하한으로 둔다
scale_floor = scale_30.loc[is_train_minute].quantile(0.10)
z_30 = ((minute - center_30) / scale_30.clip(lower=scale_floor, axis=1)).dropna()

scored_max = mark_set(z_30.abs().max(axis=1).rename('MAX_ABS_Z').reset_index())
scored_sumsq_30 = mark_set(sum_of_squares(z_30))

events_max = make_events(scored_max, 'MAX_ABS_Z', train_quantile(scored_max, 'MAX_ABS_Z'))
events_sumsq_30 = make_events(scored_sumsq_30, 'SUMSQ', train_quantile(scored_sumsq_30, 'SUMSQ'))

compare_rows = []
for name, scored, col, events in [
    ('직전 30분 기준, 최대 |z|', scored_max, 'MAX_ABS_Z', events_max),
    ('직전 30분 기준, z 제곱합', scored_sumsq_30, 'SUMSQ', events_sumsq_30),
    ('로트 기준, z 제곱합', scored_lotz, 'SUMSQ', events_lotz),
]:
    first_scored = scored.groupby('LOT_NO')['MEAS_DT'].min()
    hit, total = event_recall(events_c, events, first_scored)
    compare_rows.append({
        '방법': name,
        '사건 수': len(events),
        'IF 사건 재현': f'{hit}/{total}',
        'IF 점수와 순위 상관': round(rank_corr(scored, minute_c, col, 'ANOMALY_SCORE'), 3),
    })

print('로트 중앙값 제거 IsolationForest(사건 24개)와 비교')
print(pd.DataFrame(compare_rows).to_string(index=False))
print('\n차이는 점수를 합치는 방식이 아니라 기준 구간(로트 전체 vs 직전 30분)에서 나온다.')
print('IF는 대체로 "로트 평소 수준에서 여러 변수가 얼마나 벗어났나"를 보고,')
print('로트 기준 z 제곱합으로 상당 부분 같은 일을 할 수 있다.')

# 그림 5: 로트 기준 z 제곱합과 IF 점수
merged = scored_lotz.merge(minute_c, on=['LOT_NO', 'MEAS_DT'])
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(merged['SUMSQ'], merged['ANOMALY_SCORE'], s=4, alpha=0.3, color='gray')
ax.axvline(limit_lotz, color='black', linestyle='--', label='로트 기준 z 경보 기준')
ax.axhline(limit_c, color='steelblue', linestyle='--', label='IF 경보 기준')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
ax.set_xlabel('로트 기준 z 제곱합 (로그 축)')
ax.set_ylabel('IF 이상 점수 (1분 평균)')
ax.set_title(f"로트 기준 z 제곱합과 IF 점수 (순위 상관 {merged['SUMSQ'].corr(merged['ANOMALY_SCORE'], method='spearman'):.2f})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, '05_lot_z_vs_iforest')
