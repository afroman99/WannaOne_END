# [markdown]
# 01. 연속소둔로(CAL01) EDA
#
# EDA 결론의 근거를 다시 계산한다. 범위는 가열·균열존 `TC/OP-Z3~Z6`, 14개 로트다.
#
# 데이터 전제(데이터사전): KAMP 공개 데이터를 재구성한 교육용 데이터다.
# 존 배정과 파일 간 연결은 [가정]이다. 결과는 설비 특성이 아니라 데이터상 패턴으로 읽는다.
#
# 실행: 프로젝트 루트에서 `python analysis/01_eda.py`
# 그림은 figures/에 저장된다.

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

from src.preprocess import (
    load_data, attach_grade, label_operating_state,
    link_maintenance, find_missing_segments,
    TRANSITION_BOUNDARIES, PRE_COOLING_STARTS, SENSOR_PREFIXES,
)
from src.features import CORE_SENSORS, candidate_rows
from src.plot_utils import set_korean_font, save_figure

set_korean_font()
pd.set_option('display.width', 200)

# 1. 데이터 불러오기와 강종 연결
df_cal, df_run_evt_cal, df_m_cal = load_data(ROOT / 'data')
lot_grade = attach_grade(df_cal, df_run_evt_cal)

print('측정 행 수:', len(df_cal), '/ 로트 수:', df_cal['LOT_NO'].nunique())
print('\n로트별 강종 (측정 기간 안의 추가 강종변경 수 포함)')
print(lot_grade[
    ['LOT_NO', 'START_DT', 'END_DT', 'START_GRADE', 'CHANGE_COUNT_INSIDE']
].to_string(index=False))

# 2. 데이터 품질
# 시간·로트번호 결측과 중복
print('MEAS_DT 결측:', df_cal['MEAS_DT'].isna().sum(),
      '/ LOT_NO 결측:', df_cal['LOT_NO'].isna().sum())
print('같은 로트·시각 중복:', df_cal.duplicated(['LOT_NO', 'MEAS_DT']).sum())

# 로트 안의 측정 간격 (로트 사이는 측정 공백이라 보지 않음)
interval = (
    df_cal.sort_values(['LOT_NO', 'MEAS_DT'])
    .groupby('LOT_NO')['MEAS_DT'].diff().dt.total_seconds()
)
print('\n로트별 최대 측정 간격(초)')
print(interval.groupby(df_cal['LOT_NO']).max().to_string())

# 결측 구간: 모든 결측은 로트 시작에서 시작한다. OP=0은 결측이 아니다
missing_runs = find_missing_segments(df_cal)
sensor_cols = [c for c in df_cal.columns if c.startswith(SENSOR_PREFIXES)]

print('\n센서 결측 구간 수:', len(missing_runs))
print('센서 결측값 수:', int(df_cal[sensor_cols].isna().sum().sum()))
print('로트 중간에서 시작한 결측 구간:', int((~missing_runs['AT_LOT_START']).sum()))
print('\n가장 긴 결측 구간 5개')
print(missing_runs.sort_values('MISSING_ROWS', ascending=False).head(5).to_string(index=False))

# 3. 정비이력과 로트 연결
# 데이터사전상 고장모드는 배정한 값이다. 정비 기록은 센서 이상의 정답 라벨로 쓰지 않는다
lot_maint_summary = link_maintenance(lot_grade, df_m_cal)
print(lot_maint_summary[
    ['LOT_NO', 'GRADE', 'OVERLAP_COUNT', 'DETC_INSIDE_COUNT',
     'NEXT_MNT_TYPE', 'HOURS_TO_NEXT_DETC']
].to_string(index=False))

# 정비 기록 개요와 부품교체 연결
print('\nCAL 정비 작업:', len(df_m_cal), '건', df_m_cal['MNT_TYPE'].value_counts().to_dict())
part_change = pd.read_csv(ROOT / 'data' / 'G-02_부품교체이력.csv')
part_master = pd.read_csv(ROOT / 'data' / 'G-01_부품마스터.csv')
cal_parts = (
    df_m_cal[['WO_NO', 'MNT_TYPE', 'PART_CHG']]
    .merge(part_change, on='WO_NO')
    .merge(part_master[['PART_CD', 'PART_NM']], on='PART_CD', how='left')
)
flag_y = set(df_m_cal.loc[df_m_cal['PART_CHG'].eq('Y'), 'WO_NO'])
print('부품교체:', cal_parts['WO_NO'].nunique(), '건 /',
      cal_parts['PART_NM'].value_counts().to_dict())
print('PART_CHG=Y와 부품교체이력이 다른 작업:', len(flag_y ^ set(cal_parts['WO_NO'])))

# 발견 시각은 교대 시각(06~07, 14~15, 22~23시)에 몰려 있다
# 실제 발생 시점이 아니라 교대 때의 기록 시각으로 본다
shift_hours = [6, 7, 14, 15, 22, 23]
maint_all = pd.read_csv(ROOT / 'data' / 'G-02_정비이력.csv', parse_dates=['DETC_DT'])
print(f"\n발견 시각이 교대 시각대인 비율: CAL {df_m_cal['DETC_DT'].dt.hour.isin(shift_hours).mean():.0%}"
      f" / 전체 설비 {maint_all['DETC_DT'].dt.hour.isin(shift_hours).mean():.0%} (고르다면 25%)")

# 로트 측정 중의 정비 작업과 센서: 정지 시간이 기록돼 있는데 로는 운전 중이다
lot_span = df_cal.groupby('LOT_NO')['MEAS_DT'].agg(['min', 'max'])
lot_hours = (lot_span['max'] - lot_span['min']).dt.total_seconds().sum() / 3600
print(f'로트 측정 시간 합계: {lot_hours:.1f}시간 (2024년의 {lot_hours / 8784:.1%})')

overlap_rows = []
for work in df_m_cal.itertuples():
    for lot_no, span in lot_span.iterrows():
        start, end = max(work.STRT_DT, span['min']), min(work.END_DT, span['max'])
        if start >= end:
            continue
        sensor = df_cal.loc[
            df_cal['LOT_NO'].eq(lot_no) & df_cal['MEAS_DT'].between(start, end),
            ['TC-Z4', 'OP-Z4']
        ]
        overlap_rows.append({
            'WO_NO': work.WO_NO, 'MNT_TYPE': work.MNT_TYPE, 'LOT_NO': lot_no,
            '작업': f'{work.STRT_DT:%m-%d %H:%M}~{work.END_DT:%H:%M}',
            '정지(분)': work.DOWN_MIN,
            'TC-Z4 중앙값': round(sensor['TC-Z4'].median(), 1),
            'OP-Z4 중앙값': round(sensor['OP-Z4'].median(), 1),
        })

print('\n로트 측정 중에 기록된 정비 작업과 그 시간의 센서')
print(pd.DataFrame(overlap_rows).to_string(index=False))
print('정지 시간이 기록돼 있지만 로는 약 860°C를 유지하며 운전 중이다.')
print('정비 기록은 센서 데이터와 맞지 않으므로 설비 이상의 검증 수단으로 쓰지 않는다.')

# 4. 운전 상태: 승온·감온·감온 전 변동 구간
df_cal_labeled = label_operating_state(df_cal, lot_grade)

print('운전 상태별 행 수')
print(df_cal_labeled['OP_STATE'].value_counts().to_string())
n_candidate = int(df_cal_labeled['COMPARE_CANDIDATE'].sum())
print(f'\n온도 유지 비교 후보: {n_candidate:,}행 ({n_candidate / len(df_cal_labeled):.1%})')
print('비교 후보는 온도 유지 구간이라는 뜻이고, 설비가 정상이라는 판정이 아니다.')

# 그림 1: 승온·감온 로트의 1분 평균 온도·출력과 경계
transition_lots = list(TRANSITION_BOUNDARIES)
fig, axes = plt.subplots(2, len(transition_lots), figsize=(20, 7), sharex='col')

for col, lot_no in enumerate(transition_lots):
    kind, boundary = TRANSITION_BOUNDARIES[lot_no]
    lot = (
        df_cal.loc[df_cal['LOT_NO'] == lot_no]
        .set_index('MEAS_DT')[CORE_SENSORS]
        .resample('1min').mean()
    )

    for zone in range(4, 7):
        axes[0, col].plot(lot.index, lot[f'TC-Z{zone}'], label=f'TC-Z{zone}')
        axes[1, col].plot(lot.index, lot[f'OP-Z{zone}'], label=f'OP-Z{zone}')

    for ax in axes[:, col]:
        ax.axvline(boundary, color='black', linestyle='--', linewidth=1)
        if lot_no in PRE_COOLING_STARTS:
            ax.axvspan(PRE_COOLING_STARTS[lot_no], boundary, color='orange', alpha=0.2)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    axes[0, col].set_title(f'로트 {lot_no} | {kind} (점선: 경계, 주황: 감온 전 변동 구간)', fontsize=10)

axes[0, 0].set_ylabel('온도 (°C)')
axes[1, 0].set_ylabel('제어출력')
axes[0, 0].legend(fontsize=8)
axes[1, 0].legend(fontsize=8)
fig.suptitle('승온·감온 로트의 운전 상태 경계 (1분 평균)')
plt.tight_layout()
save_figure(fig, '01_operating_state_boundaries')

# 5. 로트·강종별 수준 차이
candidate = candidate_rows(df_cal_labeled, drop_missing=None)
model_cols = [f'{p}-Z{z}' for z in range(4, 7) for p in ('TC', 'OP')]

lot_level = candidate.groupby(['GRADE', 'LOT_NO'])[model_cols].median().round(2)
print('로트별 중앙값 (비교 후보 구간)')
print(lot_level.to_string())
print('\nSPHC는 TC-Z4~Z6이 약 859.7°C로 다른 강종(약 860.3~860.5°C)보다 낮다.')
print('데이터사전상 강종이 바뀌면 baseline이 이동한다(주로 제어출력, [가정]).')

# 그림 2: 로트별 OP-Z4 분포 (강종별 색)
lot_order = (
    candidate.groupby('LOT_NO')
    .agg(GRADE=('GRADE', 'first'), START=('MEAS_DT', 'min'))
    .sort_values(['GRADE', 'START'])
)
grades = lot_order['GRADE'].unique()
colors = dict(zip(grades, plt.get_cmap('tab10').colors))

fig, ax = plt.subplots(figsize=(14, 5))
boxes = ax.boxplot(
    [candidate.loc[candidate['LOT_NO'] == lot_no, 'OP-Z4'] for lot_no in lot_order.index],
    patch_artist=True, showfliers=False
)
for box, grade in zip(boxes['boxes'], lot_order['GRADE']):
    box.set_facecolor(colors[grade])
    box.set_alpha(0.7)

ax.set_xticks(range(1, len(lot_order) + 1))
ax.set_xticklabels([f'{r.GRADE}\n{lot_no}' for lot_no, r in lot_order.iterrows()], fontsize=8)
ax.set_ylabel('OP-Z4 제어출력')
ax.set_title('로트·강종별 OP-Z4 분포 (비교 후보 구간)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_figure(fig, '02_op_z4_by_lot')

# 6. 상관과 시차 상관
# 같은 구역의 온도와 출력은 모든 로트에서 음의 상관이다
pairs = [(f'TC-Z{z}', f'OP-Z{z}') for z in range(3, 7)] + [('TC-Z3', 'TC-Z4')]
pair_table = pd.DataFrame({
    f'{x} / {y}': candidate.groupby('LOT_NO').apply(lambda g, x=x, y=y: g[x].corr(g[y]))
    for x, y in pairs
}).round(2)
print('로트별 피어슨 상관 (1초 데이터, 서술용)')
print(pair_table.to_string())

# 시차 상관: corr(TC(t), OP(t + k)). k < 0이면 출력이 먼저 움직임
lags = sorted(set(range(-180, 181, 5)) | set(range(-10, 11)))
xcorr_rows = []

for lot_no, lot in candidate.groupby('LOT_NO'):
    # 1초 격자로 맞추되 빈 시각은 채우지 않음
    grid = lot.set_index('MEAS_DT')[CORE_SENSORS].resample('1s').mean()
    for zone in range(3, 7):
        tc, op = grid[f'TC-Z{zone}'], grid[f'OP-Z{zone}']
        for kind, x, y in [('값 자체', tc, op), ('1초 변화량', tc.diff(), op.diff())]:
            for k in lags:
                xcorr_rows.append({'LOT_NO': lot_no, 'ZONE': f'Z{zone}', 'KIND': kind,
                                   'LAG_S': k, 'CORR': x.corr(y.shift(-k))})

xcorr = pd.DataFrame(xcorr_rows)
median_xcorr = xcorr.groupby(['KIND', 'ZONE', 'LAG_S'])['CORR'].median()

print('\n1초 변화량: 지연 0초의 로트 중앙값 (같은 시각에 반대로 움직임 = 제어기 반응)')
print(median_xcorr.loc['1초 변화량'].xs(0, level='LAG_S').round(2).to_string())

peak_lag = (
    xcorr.loc[xcorr['KIND'].eq('값 자체')]
    .loc[lambda d: d.groupby(['ZONE', 'LOT_NO'])['CORR'].idxmin()]
    .groupby('ZONE')['LAG_S'].agg(lambda s: s.value_counts().idxmax())
)
print('\n값 자체: 상관이 가장 음수인 지연(초, 로트 최빈값). Z3만 출력이 약 40초 앞선다')
print(peak_lag.to_string())

# 그림 3: 시차 상관
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey='row')
for row, kind in enumerate(['값 자체', '1초 변화량']):
    for col, zone in enumerate(['Z3', 'Z4', 'Z5', 'Z6']):
        ax = axes[row, col]
        part = xcorr.loc[xcorr['KIND'].eq(kind) & xcorr['ZONE'].eq(zone)]
        for _, lot in part.groupby('LOT_NO'):
            ax.plot(lot['LAG_S'], lot['CORR'], color='lightgray', linewidth=0.8)
        med = median_xcorr.loc[(kind, zone)]
        ax.plot(med.index, med.values, color='black', linewidth=1.5, label='로트 중앙값')
        ax.axvline(0, color='red', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='gray', linewidth=0.5)
        if kind == '1초 변화량':
            ax.set_xlim(-10, 10)
        ax.set_title(f'{kind} | TC-{zone} vs OP-{zone}')
        ax.set_xlabel('지연 k (초): t+k 시각의 출력')
        ax.grid(alpha=0.3)
    axes[row, 0].set_ylabel('상관계수')
axes[0, 0].legend(loc='lower left')
fig.suptitle('온도와 제어출력의 시차 상관 (회색: 로트별)')
plt.tight_layout()
save_figure(fig, '03_lag_correlation')

# 7. CP-Z4M과 TC-Z5
# 데이터사전: CP-Z4M은 TC-Z5와 거의 같은 정보를 갖는 중복 컬럼이다
both = df_cal[['TC-Z5', 'CP-Z4M']].dropna()
print(f"상관계수: {both['TC-Z5'].corr(both['CP-Z4M']):.10f}")
print('값이 정확히 같은 행:', int((both['TC-Z5'] == both['CP-Z4M']).sum()))
print('같은 TC-Z5 값에 여러 CP-Z4M 값이 대응한 경우:',
      int((both.groupby('TC-Z5')['CP-Z4M'].nunique() > 1).sum()))
print('두 컬럼을 독립 신호로 보지 않고 TC-Z5를 쓴다.')
