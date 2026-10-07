# [markdown]
# 03. 실시간 기준 모델 (메인)
#
# 데이터 불러오기부터 실시간 기준 모델의 학습·평가까지 처음부터 끝까지 실행한다.
#
# **실시간 기준 모델: 누적·고정 척도 robust z 제곱합**
# - 입력: `TC/OP-Z4~Z6` 1분 평균 (온도 유지 비교 후보 구간)
# - 기준값: 로트 시작부터 직전 1분까지의 중앙값 (과거 값만 씀), 10분부터 점수
# - 척도: 학습 로트에서 정한 변수별 고정값
# - 점수: 6개 변수 z의 제곱합 / 경보: 학습 로트 점수의 상위 1% 초과
#
# **평가 방법:** 정비 기록은 정답 라벨로 쓸 수 없다(데이터사전: 고장모드는 배정한 값).
# 그래서 로트 전체를 본 뒤 계산하는 사후 분석 점수(로트 기준 z 제곱합)의 사건을
# 실시간 점수가 얼마나 재현하는지 본다. 사후 분석 점수는 IsolationForest와
# 순위 상관이 약 0.91이다(02_model_compare.py).
#
# 실행: 프로젝트 루트에서 `python analysis/03_realtime_baseline.py`

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
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy import stats

from src.preprocess import load_data, attach_grade, label_operating_state
from src.features import (
    DEFAULT_WARMUP, MAD_TO_STD,
    candidate_rows, minute_means, past_center, past_scale, sum_of_squares,
    lot_robust_z, MODEL_FEATURES,
)
from src.model import (
    HOLDOUT_LOTS, RealtimeBaseline, iforest_lot_centered, lot_z_score,
    make_events, mark_set, train_quantile, event_recall, rank_corr, overlaps,
)
from src.plot_utils import set_korean_font, save_figure

set_korean_font()
pd.set_option('display.width', 200)

# 1. 데이터 준비
df_cal, df_run_evt_cal, df_m_cal = load_data(ROOT / 'data')
df_cal_labeled = label_operating_state(df_cal, attach_grade(df_cal, df_run_evt_cal))

rows = candidate_rows(df_cal_labeled)                               # 행 단위 (IF용)
minute = minute_means(candidate_rows(df_cal_labeled, drop_missing=None))  # 1분 평균

lots = minute.index.get_level_values('LOT_NO')
train_lots = sorted(set(lots) - set(HOLDOUT_LOTS))
is_train_minute = lots.isin(train_lots)

print('1분 평균:', len(minute), '분 /', lots.nunique(), '로트')
print('학습 로트:', train_lots)
print('평가 로트:', HOLDOUT_LOTS, '(이상 기록이 있어 학습에서 뺌)')

# 2. 비교 기준: 사후 분석 점수의 사건
scored_lotz, limit_lotz, events_lotz = lot_z_score(minute)
_, _, events_if, _ = iforest_lot_centered(rows)

print(f'로트 기준 z 제곱합 사건: {len(events_lotz)}개 (경보 기준 {limit_lotz:.1f})')
print(f'IsolationForest(로트 중앙값 제거) 사건: {len(events_if)}개')


def evaluate(scored, events, name):
    """사후 분석 사건 재현율, 순위 상관, 경보 빈도"""
    first_scored = scored.groupby('LOT_NO')['MEAS_DT'].min()
    lotz_hit, lotz_total = event_recall(events_lotz, events, first_scored)
    if_hit, if_total = event_recall(events_if, events, first_scored)
    return {
        '설정': name,
        '순위 상관(로트 기준)': round(rank_corr(scored, scored_lotz), 3),
        '로트 기준 사건 재현': f'{lotz_hit}/{lotz_total}',
        'IF 사건 재현': f'{if_hit}/{if_total}',
        '사건 수': len(events),
        '시간당 사건': round(len(events) / (len(scored) / 60), 2),
        '점수 있는 분': len(scored),
    }


# 3. 실시간 설정 비교: 기준 구간과 척도 (탐색 18절)
# 과거 값만 쓰는 6가지 설정. 모두 로트 시작 30분부터 점수를 낸다
settings = {
    '30분·이동 척도': (30, 'rolling'),
    '60분·이동 척도': (60, 'rolling'),
    '120분·이동 척도': (120, 'rolling'),
    '60분·고정 척도': (60, 'fixed'),
    '120분·고정 척도': (120, 'fixed'),
    '누적·고정 척도': ('expanding', 'fixed'),
}

# 이동 척도의 하한: 직전 30분 척도의 학습 로트 하위 10%
scale_floor = past_scale(minute, DEFAULT_WARMUP).loc[is_train_minute].quantile(0.10)

setting_rows = []
for name, (window, scale_mode) in settings.items():
    deviation = minute - past_center(minute, window, DEFAULT_WARMUP)

    if scale_mode == 'rolling':
        scale = past_scale(minute, window).clip(lower=scale_floor, axis=1)
    else:
        # 학습 로트에서 변수별 고정 척도 하나
        scale = deviation.loc[is_train_minute].abs().median() * MAD_TO_STD

    scored = mark_set(sum_of_squares(deviation / scale))
    events = make_events(scored, 'SUMSQ', train_quantile(scored, 'SUMSQ'))
    setting_rows.append(evaluate(scored, events, name))

setting_summary = pd.DataFrame(setting_rows)
print('실시간 설정 비교 (재현율 분모: 로트 시작 30분 안에서 끝난 사건 제외)')
print(setting_summary.to_string(index=False))
print('\n척도를 고정하면 로트 기준 사건의 80~88%를 재현한다.')
print('이동 척도는 흔들림이 큰 구간에서 척도도 커져 몇 분에 걸친 변화를 덜 잡는다.')

# 그림 6: 설정별 순위 상관과 재현율
recall = setting_summary['로트 기준 사건 재현'].map(
    lambda s: int(s.split('/')[0]) / int(s.split('/')[1])
)
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(setting_summary))
ax.bar(x - 0.2, setting_summary['순위 상관(로트 기준)'], width=0.4, label='순위 상관(로트 기준)')
ax.bar(x + 0.2, recall, width=0.4, label='로트 기준 사건 재현율')
ax.set_xticks(x)
ax.set_xticklabels(setting_summary['설정'])
ax.set_ylim(0, 1)
ax.set_title('실시간 설정 비교: 사후 분석 점수를 얼마나 따라가나')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_figure(fig, '06_realtime_settings')

# 4. 실시간 기준 모델 학습: 누적·고정 척도
# 척도와 경보 기준은 기준값이 안정된 로트 시작 30분 이후 학습 데이터로 정한다
model = RealtimeBaseline(fit_warmup=DEFAULT_WARMUP, score_warmup=10).fit(minute, train_lots)

print('변수별 고정 척도')
print(model.scale_.round(3).to_string())
print(f'경보 기준 (z 제곱합): {model.limit_:.1f}')

# 5. 로트 시작 공백: 점수를 내기 시작하는 최소 기준 구간 (탐색 19절)
warmup_rows = []
for warmup in [5, 10, 15, 20, 30]:
    scored = mark_set(model.score(minute, warmup=warmup))
    events = make_events(scored, 'SUMSQ', model.limit_)
    row = evaluate(scored, events, f'{warmup}분')

    # 30분 설정에는 없던 초반 구간의 경보 비율
    minute_no = scored.groupby('LOT_NO').cumcount() + warmup
    early = scored.loc[minute_no < DEFAULT_WARMUP, 'SUMSQ']
    row['초반 경보 비율(%)'] = round((early > model.limit_).mean() * 100, 1) if len(early) else None
    warmup_rows.append(row)

warmup_summary = pd.DataFrame(warmup_rows).rename(columns={'설정': '최소 기준 구간'})
print(warmup_summary.to_string(index=False))
print('\n10분으로 줄여도 초반 경보 비율이 평소(1%)와 비슷하다. 240133 15:19 사건을 추가로 잡는다.')
print('로트 시작 직후(기준값이 생기기 전)의 변화는 실시간으로 볼 수 없다.')

# 6. 최종 평가 (최소 기준 구간 10분)
scored = mark_set(model.score(minute))
events = make_events(scored, 'SUMSQ', model.limit_)
final = evaluate(scored, events, '실시간 기준 모델')

print('실시간 기준 모델 평가')
for key, value in final.items():
    print(f'  {key}: {value}')

alarm_rate = scored.assign(ALARM=scored['SUMSQ'] > model.limit_).groupby('SET')['ALARM'].mean() * 100
print('\n경보 분 비율(%):', alarm_rate.round(2).to_dict())

# 사건마다 가장 크게 벗어난 변수
abs_z = model.zscores(minute).abs().reset_index()
features = [c for c in abs_z.columns if c not in ('LOT_NO', 'MEAS_DT')]


def top_variable(event):
    window = abs_z.loc[
        abs_z['LOT_NO'].eq(event['LOT_NO'])
        & abs_z['MEAS_DT'].between(event['START'], event['END'])
    ]
    return window[features].max().idxmax()


events['TOP_VAR'] = events.apply(top_variable, axis=1)
# 사후 분석 사건과 겹치는지
events['IN_LOT_Z'] = events.apply(lambda e: overlaps(e, events_lotz), axis=1)
print('\n실시간 모델 사건 목록')
print(events.round({'PEAK': 1}).to_string(index=False))
print('\n가장 크게 벗어난 변수:', events['TOP_VAR'].value_counts().to_dict())

# 7. 평가 로트 시간 흐름
fig, axes = plt.subplots(len(HOLDOUT_LOTS), 1, figsize=(14, 3.3 * len(HOLDOUT_LOTS)))
maint = df_m_cal.copy()

for ax, lot_no in zip(axes, HOLDOUT_LOTS):
    rt = scored.loc[scored['LOT_NO'].eq(lot_no)].set_index('MEAS_DT')
    ref = scored_lotz.loc[scored_lotz['LOT_NO'].eq(lot_no)].set_index('MEAS_DT')

    ax.plot(ref.index, ref['SUMSQ'], color='lightgray', label='로트 기준 (사후)')
    ax.plot(rt.index, rt['SUMSQ'], color='black', label='실시간 기준 모델')
    ax.axhline(model.limit_, color='red', linestyle=':', label='경보 기준')

    for e in events.loc[events['LOT_NO'].eq(lot_no)].itertuples():
        ax.axvspan(e.START, e.END + pd.Timedelta('1min'), color='red', alpha=0.15)

    # 정비 발견 기록 (정답 라벨이 아니라 참고 표시)
    for row in maint.loc[maint['DETC_DT'].between(ref.index.min(), ref.index.max())].itertuples():
        ax.axvline(row.DETC_DT, color='purple', linestyle='--', label='정비 발견 기록')

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_title(f'로트 {lot_no} | 실시간 기준 모델 점수 (빨강 음영: 경보 사건)')
    ax.set_ylabel('z 제곱합')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=8)

plt.tight_layout()
save_figure(fig, '07_realtime_baseline_holdout')

# 8. 진단: 사건이 Z6에 쏠리는 이유 (탐색 20절)
# 사건 대부분에서 Z6(TC-Z6, OP-Z6)가 가장 크게 벗어난다.
# 척도가 작아서 생긴 쏠림인지, 데이터에서 Z6가 실제로 크게 움직이는지 확인한다.
abs_z_train = abs_z.loc[abs_z['LOT_NO'].isin(train_lots), features]

# 척도의 공정성: |z| 중앙값은 모든 변수에서 약 0.67로 같다. 차이는 꼬리에 있다
z_quantiles = abs_z_train.quantile([0.5, 0.99, 0.999]).round(2)
print('학습 로트의 변수별 |z| 분위수')
print(z_quantiles.to_string())

# 원래 단위(°C, %)의 편차로도 Z6의 큰 이탈이 크다
deviation = (minute - past_center(minute, 'expanding', model.score_warmup)).dropna()
deviation_train = deviation.loc[deviation.index.get_level_values('LOT_NO').isin(train_lots)]
print('\n학습 로트의 편차 원값 |값 - 기준값| 분위수 (온도 °C, 출력 %)')
print(deviation_train.abs().quantile([0.5, 0.99]).round(3).to_string())

# 점수(z 제곱합)에 대한 변수별 기여 비율: 전체 분과 사건 분
z_sq = model.zscores(minute) ** 2
lot_idx = z_sq.index.get_level_values('LOT_NO')
time_idx = z_sq.index.get_level_values('MEAS_DT')
in_event = np.zeros(len(z_sq), dtype=bool)
for e in events.itertuples():
    in_event |= (lot_idx == e.LOT_NO) & (time_idx >= e.START) & (time_idx <= e.END)

contribution = pd.DataFrame({
    '전체 분': z_sq.sum() / z_sq.sum().sum(),
    '사건 분': z_sq.loc[in_event].sum() / z_sq.loc[in_event].sum().sum(),
}).round(3)
print('\nz 제곱합에 대한 변수별 기여 비율')
print(contribution.to_string())

# 사후 분석 기준(로트 기준 z)의 사건도 Z6가 주도하는지
lot_abs_z = lot_robust_z(minute).dropna().abs().reset_index()
lot_top = [
    lot_abs_z.loc[
        lot_abs_z['LOT_NO'].eq(e.LOT_NO) & lot_abs_z['MEAS_DT'].between(e.START, e.END),
        features
    ].max().idxmax()
    for e in events_lotz.itertuples()
]
print('\n가장 크게 벗어난 변수: 실시간 모델', events['TOP_VAR'].value_counts().to_dict())
print('                     로트 기준(사후)', pd.Series(lot_top).value_counts().to_dict())
print('\nZ6는 평소 흔들림은 다른 구역과 비슷하지만 큰 이탈이 더 자주 나온다.')
print('척도 때문이 아니라 데이터의 성질이므로 척도는 그대로 두고, 경보마다 TOP_VAR를 함께 본다.')
print('(존 배정은 데이터사전상 [가정]이다. Z6는 이 데이터의 한 컬럼 쌍으로 읽는다.)')

# 그림 8: 변수별 |z| 꼬리와 기여 비율
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
quant = z_quantiles.T
quant.columns = ['50%', '99%', '99.9%']
quant.plot.bar(ax=axes[0], rot=0)
axes[0].set_title('학습 로트의 변수별 |z| 분위수 (고정 척도)')
axes[0].set_ylabel('|z|')
axes[0].grid(axis='y', alpha=0.3)

contribution.plot.bar(ax=axes[1], rot=0)
axes[1].set_title('z 제곱합에 대한 변수별 기여 비율')
axes[1].set_ylabel('비율')
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
save_figure(fig, '08_variable_contribution')

# 9. 강종이 바뀌어도 "정상"의 기준은 같은가 (탐색 22절)
# 정상의 수준(평소 값)과 변동 폭(평소 흔들림)을 강종별로 비교한다.
# 둘 다 고정 척도 단위로 맞춘다.
lot_grade_map = df_cal_labeled.groupby('LOT_NO')['GRADE'].first()

# 수준: 로트별 중앙값 - 14개 로트 중앙값 (척도 단위)
lot_level = minute.groupby(level='LOT_NO').median()
level_units = (lot_level - lot_level.median()) / model.scale_

# 변동 폭: 로트 기준값을 뺀 편차의 로트별 MAD / 고정 척도 (1이면 학습 로트 전체와 같음)
lot_dev = (minute - past_center(minute, 'expanding', model.score_warmup)).dropna()
spread_ratio = lot_dev.groupby(level='LOT_NO').apply(
    lambda d: (d - d.median()).abs().median() * MAD_TO_STD
) / model.scale_

print('로트별 수준 (14개 로트 중앙값 대비, 척도 단위)')
print(level_units.round(1).assign(GRADE=lot_grade_map).sort_values('GRADE').to_string())
print('\n로트별 변동 폭 비율 (로트 MAD / 고정 척도)')
print(spread_ratio.round(2).assign(GRADE=lot_grade_map).sort_values('GRADE').to_string())

# 강종이 설명하는 비율(η²): 로트가 2개 이상인 강종만 쓴다
# 4개 강종·11개 로트에서는 아무 관계가 없어도 평균 약 0.3이 나온다
multi_lots = lot_grade_map[lot_grade_map.map(lot_grade_map.value_counts()) > 1].index


def grade_share(table):
    shares = {}
    for col in table.columns:
        values = table.loc[multi_lots, col]
        groups = lot_grade_map.loc[multi_lots]
        total = ((values - values.mean()) ** 2).sum()
        between = sum(
            len(part) * (part.mean() - values.mean()) ** 2
            for _, part in values.groupby(groups)
        )
        shares[col] = between / total
    return pd.Series(shares)


share_table = pd.DataFrame({
    '수준': grade_share(level_units),
    '변동 폭': grade_share(spread_ratio),
}).round(2)
print(f'\n강종이 설명하는 비율 (η², {len(multi_lots)}개 로트, 관계가 없어도 약 0.3)')
print(share_table.to_string())

# 강종별 사건 빈도
lot_events = scored.groupby('LOT_NO').agg(MINUTES=('SUMSQ', 'size'))
lot_events['EVENTS'] = events.groupby('LOT_NO').size().reindex(lot_events.index, fill_value=0)
grade_events = lot_events.groupby(lot_grade_map).agg(
    LOTS=('MINUTES', 'size'), MINUTES=('MINUTES', 'sum'), EVENTS=('EVENTS', 'sum')
)
grade_events['EVENTS_PER_10H'] = grade_events['EVENTS'] / (grade_events['MINUTES'] / 600)
print('\n강종별 사건 빈도')
print(grade_events.round(2).to_string())

print('\n정상의 수준은 강종마다 다르다(온도는 강종이 거의 결정, 출력은 강종과 로트가 함께 결정).')
print('정상의 변동 폭은 강종과 거의 관계없다. 그래서 로트마다 기준값을 새로 잡고')
print('척도는 공통으로 쓰는 실시간 기준 모델 구조가 맞다.')
print('(로트가 1개뿐인 강종 3개는 비교에서 빠진다. 강종 baseline 이동은 데이터사전상 [가정]이다.)')

# 그림 9: 로트별 수준과 변동 폭 (같은 색 = 같은 강종)
order = lot_grade_map.sort_values().index
grade_color = dict(zip(lot_grade_map.loc[order].unique(), plt.get_cmap('tab10').colors))
colors = [grade_color[lot_grade_map[lot]] for lot in order]
labels = [f'{lot_grade_map[lot]}\n{lot}' for lot in order]

fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
for row, col in enumerate(['TC-Z5', 'OP-Z4']):
    axes[row, 0].bar(range(len(order)), level_units.loc[order, col], color=colors)
    axes[row, 0].axhline(0, color='gray', linewidth=0.8)
    axes[row, 0].set_title(f'{col} 로트 수준 (14개 로트 중앙값 대비, 척도 단위)')
    axes[row, 0].set_ylabel('척도 단위')

    axes[row, 1].bar(range(len(order)), spread_ratio.loc[order, col], color=colors)
    axes[row, 1].axhline(1, color='gray', linewidth=0.8)
    axes[row, 1].set_title(f'{col} 로트 변동 폭 (고정 척도 대비 비율)')
    axes[row, 1].set_ylabel('비율')

for ax in axes.flat:
    ax.grid(axis='y', alpha=0.3)
for ax in axes[1]:
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)

fig.suptitle('강종별 정상의 수준과 변동 폭 (같은 색: 같은 강종)')
plt.tight_layout()
save_figure(fig, '09_grade_level_vs_spread')

# 10. 공정 조건 변화와 설비 이상을 구분할 수 있는가 (탐색 23절)
# 운전 상태 정보 없이 전체 시간(승온·감온 포함)에 점수를 내고,
# 경보가 어떤 공정 조건 변화로 설명되는지 분류한다. 모델·척도·경보 기준은 위와 같다.
#   1. 운전 상태 전환 중: 승온·감온·감온 전 변동 구간 (정기수리 전후)
#   2. 전환 뒤 기준값 오염: 온도 유지 구간인데 기준값이 승온 구간 값에 끌려가 있음
#      (온도 변수 하나라도 기준값과 온도 유지 수준이 3 척도 단위 넘게 다름)
#   3. 강종변경 직후: 로트 시작(= 강종변경 시각) 후 30분 안
#   4. 설명 안 됨: 나머지
full_minute = (
    df_cal_labeled.set_index('MEAS_DT')
    .groupby('LOT_NO')[MODEL_FEATURES]
    .resample('1min').mean()
)
full_state = (
    df_cal_labeled.set_index('MEAS_DT')
    .groupby('LOT_NO')['OP_STATE']
    .resample('1min').first()
)

# 운전 상태 정보 없이: 기준값은 로트 시작부터의 중앙값
full_center = past_center(full_minute, 'expanding', model.score_warmup)
full_sumsq = (((full_minute - full_center) / model.scale_).dropna() ** 2).sum(axis=1)
full_alarm = full_sumsq > model.limit_

lot_idx = full_sumsq.index.get_level_values('LOT_NO')
time_idx = full_sumsq.index.get_level_values('MEAS_DT')
lot_first = df_cal_labeled.groupby('LOT_NO')['MEAS_DT'].min()
minutes_from_start = (time_idx - lot_idx.map(lot_first)).total_seconds() / 60

holding = full_state.reindex(full_minute.index).eq('온도 유지 중심·비교 후보').to_numpy()
holding_level = full_minute.loc[holding].groupby(level='LOT_NO').median()
temp_cols = ['TC-Z4', 'TC-Z5', 'TC-Z6']
baseline_gap = (
    full_center.reindex(full_sumsq.index)[temp_cols].to_numpy()
    - holding_level.reindex(lot_idx)[temp_cols].to_numpy()
) / model.scale_[temp_cols].to_numpy()

category = pd.Series('설명 안 됨', index=full_sumsq.index)
category[np.asarray(minutes_from_start) < 30] = '강종변경 직후'
category[(np.abs(baseline_gap) > 3).any(axis=1)] = '전환 뒤 기준값 오염'
category[full_state.reindex(full_sumsq.index).isin(
    ['승온 구간 추정', '감온 구간 추정', '감온 전 변동 구간']).to_numpy()] = '운전 상태 전환 중'

alarm_summary = pd.DataFrame({
    '분': category.value_counts(),
    '경보 분': category[full_alarm].value_counts(),
}).fillna(0).astype(int)
alarm_summary['경보 비율(%)'] = (alarm_summary['경보 분'] / alarm_summary['분'] * 100).round(1)
alarm_summary['경보 중 몫(%)'] = (alarm_summary['경보 분'] / full_alarm.sum() * 100).round(1)

print(f'운전 상태 정보 없이: 점수 있는 분 {len(full_sumsq)}, 경보 분 {int(full_alarm.sum())}')
print(alarm_summary.to_string())

# 운전 상태 정보를 쓸 때 = 위의 실시간 기준 모델 (온도 유지 구간만, 구간 시작부터 기준값)
held_alarm = int((scored['SUMSQ'] > model.limit_).sum())
after_grade_change = (events['START'] - events['LOT_NO'].map(lot_first)).dt.total_seconds() / 60 < 30
print(f'\n운전 상태 정보를 쓸 때: 점수 있는 분 {len(scored)}, 경보 분 {held_alarm} '
      f'({held_alarm / len(scored):.1%}), 사건 {len(events)} '
      f'(강종변경 직후 {int(after_grade_change.sum())}, 설명 안 됨 {int((~after_grade_change).sum())})')

print('\n운전 상태 정보 없이 경보를 내면 경보의 대부분이 정기수리 전후의 운전 상태 변화에서 나온다.')
print('강종변경은 로트 시작에서 기준값을 새로 잡으므로 경보를 거의 만들지 않는다.')
print('공정 조건 변화를 걸러 내고 남는 경보는 설계한 오경보 수준(학습 로트 상위 1%)과 비슷하다.')
print('정답 라벨이 없으므로 이 경보는 "설명되지 않은 이상 후보"까지만 말할 수 있다.')

# 그림 10: 승온 로트의 운전 상태 정보 없는 점수와 경보 분류
scored_full = full_sumsq.rename('SUMSQ').reset_index()
scored_full['CATEGORY'] = category.to_numpy()
category_color = {
    '운전 상태 전환 중': 'orange',
    '전환 뒤 기준값 오염': 'purple',
    '강종변경 직후': 'green',
    '설명 안 됨': 'red',
}

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
for ax, lot_no in zip(axes, [240061, 240117]):
    part = scored_full.loc[scored_full['LOT_NO'].eq(lot_no)]
    ax.plot(part['MEAS_DT'], part['SUMSQ'], color='black', linewidth=1)
    ax.axhline(model.limit_, color='gray', linestyle=':')
    for name, color in category_color.items():
        over = part.loc[part['CATEGORY'].eq(name) & (part['SUMSQ'] > model.limit_)]
        ax.scatter(over['MEAS_DT'], over['SUMSQ'], s=10, color=color, label=name)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_title(f'로트 {lot_no} | 운전 상태 정보 없이 낸 점수와 경보 분류')
    ax.set_ylabel('z 제곱합')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
save_figure(fig, '10_full_time_alarm_category')

# 11. 시계열 ①: 로트 안의 느린 추세(드리프트) (탐색 24절)
# 온도 유지 구간에서 몇 시간에 걸쳐 수준이 서서히 바뀌는지 본다.
# - 로트별 1분 평균에 Theil-Sen 기울기(중앙값 기반, 짧은 사건에 덜 흔들림)를 맞춘다.
# - 로트 전체 변화량 = 기울기 x 로트 길이 / 고정 척도 (1이면 평소 흔들림 하나만큼)
# - 1분 데이터는 앞뒤 값이 이어져 p값을 믿기 어려우므로 변화량의 크기로 판단한다.
# - 5시간 미만 로트는 로트 끝·시작의 사건에 끌려가므로 드리프트로 판단하지 않는다.
drift_rows = []
third_rows = []
for lot_no, lot in minute.groupby(level='LOT_NO'):
    lot = lot.droplevel('LOT_NO')
    hours = (lot.index - lot.index[0]).total_seconds() / 3600
    row = {'LOT_NO': lot_no, 'GRADE': lot_grade_map[lot_no], 'HOURS': round(hours.max(), 1)}
    for col in MODEL_FEATURES:
        valid = lot[col].notna().to_numpy()
        slope = stats.theilslopes(lot[col].to_numpy()[valid], hours[valid])[0]
        row[col] = slope * hours.max() / model.scale_[col]
    drift_rows.append(row)

    # 확인용: 마지막 1/3 중앙값 - 처음 1/3 중앙값
    third = len(lot) // 3
    third_rows.append(
        ((lot.iloc[-third:].median() - lot.iloc[:third].median()) / model.scale_).rename(lot_no)
    )

drift = pd.DataFrame(drift_rows).set_index('LOT_NO').sort_values('HOURS', ascending=False)
third_diff = pd.DataFrame(third_rows).loc[drift.index]
long_lots = drift.index[drift['HOURS'] >= 5]

print('로트 전체 기간 동안의 변화량 (척도 단위)')
print(drift.round(1).to_string())
print('\n확인용: 마지막 1/3 중앙값 - 처음 1/3 중앙값 (척도 단위)')
print(third_diff.round(1).to_string())
print(f'\n5시간 이상 로트({len(long_lots)}개)의 변화량 범위 (척도 단위)')
print(drift.loc[long_lots, MODEL_FEATURES].agg(['min', 'max']).round(1).to_string())

# 실시간 기준 모델이 드리프트를 잡을 수 있는가
# 기준값이 로트 시작부터의 중앙값이라 직선 드리프트를 절반쯤 따라간다.
# 로트 끝에서 z는 대략 (전체 변화량 / 2)이므로, 경보에는 변화량이 2 x sqrt(경보 기준) 이상 필요하다.
single_var_limit = np.sqrt(model.limit_)
print(f'\n드리프트만으로 경보가 나려면 로트 전체 변화량이 약 {2 * single_var_limit:.1f} 척도 단위 이상이어야 한다.')
print(f'5시간 이상 로트에서 관찰된 최대 변화량: '
      f'{drift.loc[long_lots, MODEL_FEATURES].abs().max().max():.1f} 척도 단위')

print('\n온도는 매우 평평하고, 출력만 조금 움직인다. 같은 온도에서 출력이 꾸준히 오른 로트는 240069 하나다.')
print('이 정도 드리프트는 실시간 기준 모델로 잡을 수 없으므로 별도 드리프트 지표가 필요하다.')

# 그림 11: 출력이 같은 온도에서 오른 로트(240069)와 같은 강종의 긴 로트(240091)
fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharey='row')
for col_idx, lot_no in enumerate([240069, 240091]):
    lot = minute.loc[lot_no].resample('10min').mean()
    hours = (lot.index - lot.index[0]).total_seconds() / 3600
    for row_idx, col in enumerate(['TC-Z4', 'OP-Z4']):
        ax = axes[row_idx, col_idx]
        ax.plot(lot.index, lot[col], color='black', linewidth=1)
        valid = lot[col].notna()
        slope, intercept = stats.theilslopes(lot[col][valid], hours[valid])[:2]
        ax.plot(lot.index, intercept + slope * hours, color='red', linestyle='--',
                label=f'Theil-Sen 추세: 전체 {drift.loc[lot_no, col]:+.1f} 척도 단위')
        ax.set_title(f'로트 {lot_no} ({lot_grade_map[lot_no]}) | {col} 10분 평균')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[0, 0].set_ylabel('온도 (°C)')
axes[1, 0].set_ylabel('제어출력')
plt.tight_layout()
save_figure(fig, '11_within_lot_drift')

# 12. 시계열 ②: 변동성의 시간 변화 (탐색 25절)
# 고정 척도는 평소 흔들림이 시간에 따라 같다고 가정한다. 이 가정이 맞는지 본다.
# - 1분 평균의 1차 차분(이번 분 - 직전 분)으로 느린 추세와 수준 차이를 뺀다.
# - 시간별 변동성 = 1시간 블록 안 차분의 MAD / 전체 로트의 같은 값 (1이면 평소)
# - 공통 변동성 = 6개 변수 시간별 변동성의 기하평균
# - 로트 안 추세와 지속성은 같은 로트의 시간 순서를 섞은 우연 범위와 비교한다.
def block_mad(values, min_count=20):
    values = values[~np.isnan(values)]
    if len(values) < min_count:  # 분이 20개 미만인 블록은 뺀다
        return np.nan
    return np.median(np.abs(values - np.median(values))) * MAD_TO_STD


minute_diff = minute.groupby(level='LOT_NO').diff()
diff_ref = minute_diff.apply(lambda s: block_mad(s.to_numpy()))

vol_blocks = []
for lot_no, lot in minute_diff.groupby(level='LOT_NO'):
    lot = lot.droplevel('LOT_NO')
    hourly = lot.resample('60min').agg(lambda s: block_mad(s.to_numpy())) / diff_ref
    hourly['MINUTES'] = lot.resample('60min').size()
    hourly['LOT_NO'] = lot_no
    vol_blocks.append(hourly.dropna(subset=MODEL_FEATURES, how='all'))
hourly_vol = pd.concat(vol_blocks).reset_index().set_index(['LOT_NO', 'MEAS_DT'])
hourly_vol['COMMON'] = np.exp(np.log(hourly_vol[MODEL_FEATURES]).mean(axis=1))

print('시간 블록', len(hourly_vol), '/ 공통 변동성 범위',
      hourly_vol['COMMON'].agg(['min', 'median', 'max']).round(2).to_dict())

# 변수끼리 같은 시간에 함께 흔들리는가 (로트 평균을 뺀 log 변동성의 순위 상관)
log_vol = np.log(hourly_vol[MODEL_FEATURES])
log_vol_c = log_vol - log_vol.groupby(level='LOT_NO').transform('mean')
print('\n변수 사이 시간별 변동성 상관 (로트 안)')
print(log_vol_c.corr(method='spearman').round(2).to_string())

# 로트 안 추세: 처음 대비 끝 배율 (5시간 이상 로트)
rng = np.random.default_rng(0)


def trend_ratio(log_values):
    n = len(log_values)
    return np.exp(stats.theilslopes(log_values, np.arange(n))[0] * (n - 1))


trend_rows = []
for lot_no, part in hourly_vol.groupby(level='LOT_NO'):
    values = np.log(part['COMMON'].to_numpy())
    if len(values) < 5:
        continue
    observed = trend_ratio(values)
    null = np.array([trend_ratio(rng.permutation(values)) for _ in range(2000)])
    trend_rows.append({
        'LOT_NO': lot_no, 'GRADE': lot_grade_map[lot_no], 'BLOCKS': len(values),
        'RATIO': observed,
        'CHANCE_LO': np.quantile(null, 0.025), 'CHANCE_HI': np.quantile(null, 0.975),
        'P': np.mean(np.abs(np.log(null)) >= abs(np.log(observed))),
    })
vol_trend = pd.DataFrame(trend_rows).set_index('LOT_NO')
print('\n로트 안 공통 변동성 추세 (끝/처음 배율, 우연 범위는 시간 순서를 섞은 95%)')
print(vol_trend.round(2).to_string())


# 지속성: 흔들리는 시간이 다음 시간으로 이어지는가 (1시간 시차 자기상관, 로트 안)
def lag1_corr(log_series):
    num = den = 0.0
    for _, part in log_series.groupby(level='LOT_NO'):
        values = part.to_numpy() - part.mean()
        if len(values) < 3:
            continue
        num += (values[1:] * values[:-1]).sum()
        den += (values ** 2).sum()
    return num / den


log_common = np.log(hourly_vol['COMMON'])
lag1_observed = lag1_corr(log_common)
lag1_null = np.array([
    lag1_corr(log_common.groupby(level='LOT_NO').transform(
        lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)))
    for _ in range(1000)
])
print(f'\n1시간 시차 자기상관 {lag1_observed:.2f} '
      f'(우연 범위 {np.quantile(lag1_null, 0.025):.2f} ~ {np.quantile(lag1_null, 0.975):.2f}, '
      f'p = {np.mean(lag1_null >= lag1_observed):.2f})')

# 경보와의 관계: 공통 변동성 4분위별 경보 분 비율
score_hour = scored.assign(
    MEAS_DT=scored['MEAS_DT'].dt.floor('60min'),
    ALARM=scored['SUMSQ'] > model.limit_,
).groupby(['LOT_NO', 'MEAS_DT'])['ALARM'].sum()
hourly_vol['ALARM_MIN'] = score_hour.reindex(hourly_vol.index).fillna(0).astype(int)
hourly_vol['VOL_BAND'] = pd.qcut(hourly_vol['COMMON'], 4, labels=['하위 25%', '25~50%', '50~75%', '상위 25%'])

band_alarm = hourly_vol.groupby('VOL_BAND', observed=True).agg(
    블록=('MINUTES', 'size'), 분=('MINUTES', 'sum'), 경보_분=('ALARM_MIN', 'sum'),
    변동성_하한=('COMMON', 'min'), 변동성_상한=('COMMON', 'max'),
)
band_alarm['경보_비율(%)'] = band_alarm['경보_분'] / band_alarm['분'] * 100
print('\n공통 변동성 4분위별 경보')
print(band_alarm.round(2).to_string())

print('\n변동성은 시간마다 0.6~2배로 오르내리고 구역이 함께 흔들린다. 로트 안에서 꾸준히 커지는 로트는 뚜렷하지 않다.')
print('경보는 흔들리는 시간에 몰린다(변동성과 경보는 같은 편차에서 계산하므로 일부는 당연히 겹친다).')
print('척도를 시간별로 바꾸면 흔들리는 시간 자체를 못 보므로 고정 척도를 유지하고,')
print('경보를 볼 때 최근 1시간 공통 변동성을 함께 보여 준다.')

# 그림 12: 5시간 이상 로트의 시간별 공통 변동성 (빨간 막대: 경보가 있던 시간)
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
for ax, lot_no in zip(axes.flat, vol_trend.index):
    part = hourly_vol.loc[lot_no]
    colors = np.where(part['ALARM_MIN'] > 0, 'tab:red', 'tab:gray')
    ax.bar(np.arange(len(part)), part['COMMON'], color=colors)
    ax.axhline(1, color='black', linewidth=0.8)
    ax.set_title(f'로트 {lot_no} ({lot_grade_map[lot_no]}) | 끝/처음 {vol_trend.loc[lot_no, "RATIO"]:.2f}배',
                 fontsize=10)
    ax.set_xlabel('로트 시작 후 시간 블록')
    ax.grid(axis='y', alpha=0.3)
for ax in axes[:, 0]:
    ax.set_ylabel('공통 변동성 (평소 = 1)')
fig.suptitle('시간별 공통 변동성 (빨간 막대: 실시간 기준 모델 경보가 있던 시간)')
plt.tight_layout()
save_figure(fig, '12_hourly_volatility')
