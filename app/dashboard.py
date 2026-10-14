"""연주 설비 24시간 조기경보 대시보드 (시연용).

    python analysis/05_predict_test.py      # 처음 한 번: 모델 학습·저장 (models/risk24.joblib)
    streamlit run app/dashboard.py          # 브라우저에서 http://localhost:8501

- 저장된 모델로 예측만 한다(다시 학습하지 않는다).
- '지금 시각'을 옮기면 그 시각까지의 데이터만으로 본 상태를 보여 준다(재생 방식 시연).
  실제 서비스라면 이 자리에 새로 들어온 센서 데이터가 들어간다(src/serving.py 참고).
- 데이터 세트: 테스트(정답 없음, 실제 운영과 같은 상황) / 검증(정답 있음, 실제 고장과 비교 가능)

2단계 경보 (결과보고서 7-5절)
  주의  지금 열화 중일 확률 ≥ 주의 기준 → 장비 상태 확인, 설비가 돌면 다음 정비 때 함께 정비, 두 부품 재고 확인
  경보  24시간 안 고장 확률 ≥ 경보 기준 → 교체

화면
  1. 전체 현황    설비(시나리오)별 지금 단계(정상·주의·경보), 위험한 순서, 부품 소요 예측
  2. 설비 상세    위험 확률·열화 확률과 네 신호의 흐름, 정비 기록, 경보 이유, 조치 안내
  3. 모델 정보    경보·주의 모델과 기준, 검증 세트 성능
  4. 비용 계산기  원인 부품을 맞힐 확률에 따른 비용 (팀 결정 C)
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.posko_data import load_scenarios
from src.serving import cost_curve, explain, failure_hours, load_bundle, score_scenarios

st.set_page_config(page_title='연주 설비 조기경보', page_icon='⚠️', layout='wide')

SIGNAL_NAMES = {'Z_MLD-AMP': '몰드 레벨 진폭', 'Z_CONTROL_DELAY': '제어 지연',
                'Z_WDL': '인발 부하', 'Z_OSC-STK': '진동 스트로크'}
SPLITS = {'테스트 (정답 없음)': 'test', '검증 (정답 있음)': 'validation'}
ALARM_RED, CAUTION_YELLOW, OK_GRAY, MNT_GRAY = '#d64545', '#e0a526', '#9e9e9e', '#bdbdbd'
# 화면에 보여 줄 단계 이름: 정상(NORMAL) → 주의(CAUTION, 열화 감지) → 경보(WARNING, 24시간 고장 위험)
# 모델 출력(STAGE)은 '경보'/'주의'/'정상' 그대로 두고 화면에서만 이름을 바꾼다 (결과보고서 7-7절 ⑥)
LABEL = {'경보': '경보 (WARNING)', '주의': '주의 (CAUTION)', '정상': '정상 (NORMAL)'}
WARN, CAU, NORM = LABEL['경보'], LABEL['주의'], LABEL['정상']
STAGE_COLORS = alt.Scale(domain=[WARN, CAU, NORM], range=[ALARM_RED, CAUTION_YELLOW, OK_GRAY])


# 데이터 불러오기 (처음 한 번만 계산하고 기억해 둔다)

@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner='센서 데이터를 요약하고 예측하는 중...')
def get_data(split):
    bundle = get_bundle()
    ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
    pts, hr = score_scenarios(ids, bundle)
    contrib = explain(bundle, pts)
    hr = hr.assign(DONE_H=hr['HOUR'] + 1)          # 1시간 구간이 끝난 시각 = 그 값을 알 수 있게 되는 시각
    return pts, hr[['SCENARIO_ID', 'DONE_H', 'SCORE', *SIGNAL_NAMES]], contrib, failure_hours(ids)


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(f'저장된 모델이 없습니다. 터미널에서 `python analysis/05_predict_test.py`를 먼저 실행하세요.\n\n{e}')
    st.stop()
TH = bundle['threshold']
CAUTION = bundle.get('caution')          # 주의 모델이 없는 예전 모델 파일이면 경보 1단계만 보여 준다
CTH = CAUTION['threshold'] if CAUTION else None


# 사이드바: 화면, 데이터 세트, 지금 시각

st.sidebar.title('연주 설비 조기경보')
page = st.sidebar.radio('화면', ['전체 현황', '설비 상세', '모델 정보', '비용 계산기'])
split_label = st.sidebar.radio('데이터 세트', list(SPLITS))
split = SPLITS[split_label]
now = st.sidebar.slider('지금 시각 (시나리오 시작 후 경과 시간, h)', 0, 72, 72, step=1,
                        help='이 시각까지의 데이터만으로 본 상태를 보여 줍니다. 왼쪽으로 옮기면 과거로 돌아가 재생할 수 있습니다.')
show_truth = split == 'validation' and st.sidebar.checkbox(
    '실제 고장 시각 미리 보기 (정답)', help='검증 세트에만 있습니다. 모델은 이 정보를 보지 않습니다.')
st.sidebar.caption(f'경보 모델: {bundle["final"]["model"]} · {bundle["final"]["features"]}\n\n'
                   + (f'주의 기준: 지금 열화 중일 확률 ≥ {CTH:.2f}\n\n' if CAUTION else '')
                   + f'경보 기준: 24시간 안 고장 확률 ≥ {TH:.2f}\n\n모델 저장: {bundle["created_at"]}')

pts, hr, contrib, mnt = get_data(split)
pts_now = pts[pts['T_H'] <= now]
hr_now = hr[hr['DONE_H'] <= now]
mnt_now = mnt[mnt['START_H'] <= now].assign(
    END_SHOWN=lambda d: d['END_H'].clip(upper=now),
    시작=lambda d: d['START_H'].map(lambda x: f'{x:.1f}h'),
    종료=lambda d: d['END_H'].map(lambda x: f'{x:.1f}h' if x <= now else '진행 중 (종료 미확정)'))   # 미래의 종료 시각은 보여 주지 않는다
last_mnt_start = mnt_now.groupby('SCENARIO_ID')['START_H'].max()     # 지금까지 마지막으로 시작된 정비


def after_last_maintenance(df, time_col):
    """마지막 정비가 시작된 뒤의 행만 남긴다. 정비 전 값은 지금 상태가 아니기 때문이다."""
    start = df['SCENARIO_ID'].map(last_mnt_start).fillna(-np.inf)
    return df[df[time_col] >= start]


def status_table():
    """시나리오마다 지금 시각 기준 상태 한 줄"""
    last = pts_now.sort_values('T_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
    first_alarm = pts_now[pts_now['ALARM']].groupby('SCENARIO_ID')['T_H'].min()
    first_caution = pts_now[pts_now['STAGE'] != '정상'].groupby('SCENARIO_ID')['T_H'].min()
    cur = after_last_maintenance(hr_now.assign(HOUR_START=hr_now['DONE_H'] - 1), 'HOUR_START')
    last_hr = cur.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1)
    worst = last_hr.set_index('SCENARIO_ID')[list(SIGNAL_NAMES)].idxmax(axis=1).map(SIGNAL_NAMES)
    in_mnt = set(mnt[(mnt['START_H'] <= now) & (mnt['END_H'] > now)]['SCENARIO_ID'])
    # 마지막 예측 뒤에 정비가 시작됐으면 그 예측은 정비 전 상태라 쓰지 않는다
    stale = set(last_mnt_start[last_mnt_start > last['T_H'].reindex(last_mnt_start.index).fillna(-1)].index) - in_mnt
    t = pd.DataFrame({'설비 (시나리오)': sorted(pts['SCENARIO_ID'].unique())}).set_index('설비 (시나리오)', drop=False)
    t['위험 확률'] = last['PROB']
    if CAUTION:
        t['열화 확률'] = last['DEGRADED_PROB']
    t['마지막 예측 (h)'] = last['T_H']
    t['상태'] = np.select([t.index.isin(in_mnt), t.index.isin(stale)], ['정비 중', '정비 후 예측 대기'],
                         default=last['STAGE'].reindex(t.index).fillna('정상'))
    t.loc[t['상태'].isin(['정비 중', '정비 후 예측 대기']), ['위험 확률', '열화 확률'] if CAUTION else ['위험 확률']] = np.nan
    if CAUTION:
        t['첫 주의 (h)'] = first_caution
    t['첫 경보 (h)'] = first_alarm
    t['가장 많이 벗어난 신호'] = worst
    if show_truth:
        fails = mnt.dropna(subset=['FAIL_H'])
        nxt = fails[fails['FAIL_H'] > now].groupby('SCENARIO_ID')['FAIL_H'].min()
        t['다음 실제 고장까지 (h, 정답)'] = (nxt - now).round(1)
    order = t['상태'].map({'경보': 0, '주의': 1, '정상': 2}).fillna(3)
    t['상태'] = t['상태'].replace(LABEL)
    return t.assign(_o=order).sort_values(['_o', '위험 확률'], ascending=[True, False]).drop(columns='_o')


def mnt_layer(sid, y_field_max):
    """정비 기간(회색 띠)과 고장 시각(빨간 선)"""
    m = mnt_now[mnt_now['SCENARIO_ID'] == sid]
    band = alt.Chart(m).mark_rect(color=MNT_GRAY, opacity=0.35).encode(
        x='START_H:Q', x2='END_SHOWN:Q', tooltip=['종류', 'PART_CD', '시작', '종료'])
    f = m.dropna(subset=['FAIL_H'])
    fail = alt.Chart(f).mark_rule(color=ALARM_RED, strokeWidth=2).encode(x='FAIL_H:Q', tooltip=['종류', 'FAIL_H'])
    layers = [band, fail]
    if show_truth:
        fut = mnt[(mnt['SCENARIO_ID'] == sid) & (mnt['FAIL_H'] > now)].dropna(subset=['FAIL_H'])
        layers.append(alt.Chart(fut).mark_rule(color=ALARM_RED, strokeDash=[4, 4]).encode(
            x='FAIL_H:Q', tooltip=[alt.Tooltip('FAIL_H', title='실제 고장 (정답)')]))
    return layers


# 1. 전체 현황

if page == '전체 현황':
    st.header(f'전체 현황 · 지금 {now}시간 · {split_label}')
    t = status_table()
    n_alarm, n_caution = int((t['상태'] == WARN).sum()), int((t['상태'] == CAU).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('설비 수', len(t))
    c2.metric('지금 경보 (WARNING)', n_alarm)
    c3.metric('지금 주의 (CAUTION)', n_caution if CAUTION else '—', help=None if CAUTION else '주의 모델이 없는 모델 파일입니다')
    c4.metric('정비 중·예측 대기', int(t['상태'].isin(['정비 중', '정비 후 예측 대기']).sum()))
    st.caption((f'두 확률은 6시간마다 계산합니다. **주의 (CAUTION, 열화 감지)**: 지금 열화 중일 확률 ≥ {CTH:.2f} · ' if CAUTION else '위험 확률은 6시간마다 계산합니다. ')
               + f'**경보 (WARNING, 24시간 고장 위험)**: 24시간 안 고장 확률(위험 확률) ≥ {TH:.2f}. '
               '가장 많이 벗어난 신호는 마지막 1시간 기준이며, 표준화 점수 1이 정상 범위의 경계입니다.')

    if CAUTION:
        st.subheader('부품 소요 예측')
        st.caption('센서로는 원인 부품을 구분할 수 없어, 경보가 나면 두 부품을 모두 교체합니다(팀 결정 C). '
                   '그래서 설비 한 대에 밸브 1개·진동장치 1개가 필요하다고 셉니다.')
        parts = pd.DataFrame({'부품': ['밸브', '진동장치'], '지금 교체 (경보 설비)': [n_alarm] * 2,
                              '1~2일 안에 필요할 수 있음 (주의 설비)': [n_caution] * 2})
        parts['합계'] = parts.iloc[:, 1:].sum(axis=1)
        st.dataframe(parts, hide_index=True, width='content')
        cv = CAUTION['validation']
        st.caption(f'재고가 이보다 적으면 지금 발주를 시작하세요. 주의 중 일부(검증 세트 {cv["주의 묶음"]}번 중 '
                   f'{cv["열화 아닌 설비의 주의"]}번)는 열화가 아니었던 헛걸음입니다.')

    st.subheader('설비별 상태 (경보 WARNING → 주의 CAUTION → 정상 NORMAL 순)')
    st.dataframe(
        t, hide_index=True, height=520,
        column_config={
            '위험 확률': st.column_config.ProgressColumn('위험 확률 (24시간)', format='%.2f', min_value=0, max_value=1),
            '열화 확률': st.column_config.ProgressColumn('열화 확률 (지금)', format='%.2f', min_value=0, max_value=1),
            '마지막 예측 (h)': st.column_config.NumberColumn(format='%d'),
            '첫 주의 (h)': st.column_config.NumberColumn(format='%d', help='경보도 주의에 포함해 처음 켜진 시각'),
            '첫 경보 (h)': st.column_config.NumberColumn(format='%d'),
        })
    st.info('설비를 자세히 보려면 왼쪽에서 **설비 상세**를 고르세요.')


# 2. 설비 상세

elif page == '설비 상세':
    t = status_table()
    sid = st.selectbox('설비 (시나리오, 지금 위험한 순서)', t.index.tolist(),
                       format_func=lambda s: f'{s} · {t.loc[s, "상태"]} · 확률 {t.loc[s, "위험 확률"]:.2f}'
                       if pd.notna(t.loc[s, '위험 확률']) else s)
    row = t.loc[sid]
    st.header(f'{sid} · 지금 {now}시간')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('상태', row['상태'])
    c2.metric('24시간 안 고장 확률', '—' if pd.isna(row['위험 확률']) else f'{row["위험 확률"]:.2f}',
              help=f'마지막 예측: {row["마지막 예측 (h)"]:.0f}시간' if pd.notna(row['마지막 예측 (h)']) else None)
    if CAUTION:
        c3.metric('지금 열화 중일 확률', '—' if pd.isna(row['열화 확률']) else f'{row["열화 확률"]:.2f}')
        c4.metric('첫 주의 / 첫 경보', ' / '.join('—' if pd.isna(row[k]) else f'{row[k]:.0f}h' for k in ['첫 주의 (h)', '첫 경보 (h)']))
    else:
        c3.metric('첫 경보', '—' if pd.isna(row['첫 경보 (h)']) else f'{row["첫 경보 (h)"]:.0f}시간')

    if row['상태'] == WARN:
        st.error(
            '**조치 안내** · 24시간 안에 고장 날 위험이 높습니다.\n'
            '- 교체에 4시간이 걸리므로, 늦어도 고장 예상 4시간 전에는 교체를 시작해야 합니다.\n'
            '- 센서로는 원인 부품(진동장치·밸브)을 구분할 수 없습니다. 두 부품을 모두 점검하세요.\n'
            '- 고장 원인과 상관없이 같은 신호가 오르므로, 아래 신호 흐름으로 열화가 계속되는지 확인하세요.')
    elif row['상태'] == CAU:
        st.warning(
            '**조치 안내 · 주의** · 열화가 시작된 것으로 보입니다. 아직 24시간 안 고장 위험은 낮습니다.\n'
            '- 장비 상태를 확인하세요. 설비가 정상적으로 돌아가면 **다음 정비 때 이 부분까지 함께 정비**합니다.\n'
            '- 두 부품(진동장치·밸브)의 재고를 확인하고, 없으면 지금 발주를 시작하세요. 경보까지 보통 반나절 정도 남습니다.\n'
            f'- 주의 중 일부는 열화가 아닌 헛걸음입니다(검증 세트 {CAUTION["validation"]["주의 묶음"]}번 중 '
            f'{CAUTION["validation"]["열화 아닌 설비의 주의"]}번). 점검에서 이상이 없으면 그대로 운전합니다.')
    elif row['상태'] in ('정비 중', '정비 후 예측 대기'):
        st.warning('정비가 시작돼 정비 전 예측은 더 이상 쓰지 않습니다. 재가동 뒤 다음 예측 시점(6시간마다)에 다시 예측합니다.')
    else:
        st.success('지금은 열화 신호가 없고, 24시간 안 고장 위험이 낮습니다.' if CAUTION else '지금은 24시간 안 고장 위험이 낮습니다.')

    # 위험 확률 흐름
    p = pts_now[pts_now['SCENARIO_ID'] == sid].rename(columns={'STAGE': '단계'})
    p['단계'] = p['단계'].replace(LABEL)
    x = alt.X('T_H:Q', title='시나리오 시작 후 경과 시간 (h)', scale=alt.Scale(domain=[0, 72], nice=False))
    tip = [alt.Tooltip('T_H', title='시각(h)'), alt.Tooltip('PROB', title='24시간 안 고장 확률', format='.2f'), '단계']
    if CAUTION:
        tip.insert(2, alt.Tooltip('DEGRADED_PROB', title='열화 확률', format='.2f'))
    line = alt.Chart(p).mark_line(color='#2f5d8a').encode(x=x, y=alt.Y('PROB:Q', title='확률', scale=alt.Scale(domain=[0, 1])))
    dots = alt.Chart(p).mark_circle(size=90).encode(x=x, y='PROB:Q', color=alt.Color('단계:N', scale=STAGE_COLORS), tooltip=tip)
    layers = [*mnt_layer(sid, 1), line, dots,
              alt.Chart(pd.DataFrame({'y': [TH]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')]
    if CAUTION:
        layers += [alt.Chart(p).mark_line(color=CAUTION_YELLOW, strokeDash=[2, 2]).encode(x=x, y='DEGRADED_PROB:Q', tooltip=tip),
                   alt.Chart(pd.DataFrame({'y': [CTH]})).mark_rule(color=CAUTION_YELLOW, strokeDash=[6, 3]).encode(y='y:Q')]
    st.subheader('위험 확률과 열화 확률' if CAUTION else '위험 확률')
    st.altair_chart(alt.layer(*layers).properties(height=280), width='stretch')
    st.caption(('파란 선과 점: 24시간 안 고장 확률(점 색 = 단계) · 노란 점선: 지금 열화 중일 확률 · '
                f'빨간 가로 점선: 경보 기준 {TH:.2f} · 노란 가로 점선: 주의 기준 {CTH:.2f}' if CAUTION
                else f'점선: 경보 기준 {TH:.2f}')
               + ' · 회색 띠: 정비 · 빨간 실선: 고장'
               + (' · 빨간 점선: 앞으로 일어날 실제 고장 (정답)' if show_truth else ''))

    # 네 신호 흐름
    h = hr_now[hr_now['SCENARIO_ID'] == sid].rename(columns=SIGNAL_NAMES)
    long = h.melt(id_vars=['DONE_H'], value_vars=list(SIGNAL_NAMES.values()), var_name='신호', value_name='점수')
    sig = alt.Chart(long).mark_line(opacity=0.8).encode(
        x=alt.X('DONE_H:Q', title='시나리오 시작 후 경과 시간 (h)', scale=alt.Scale(domain=[0, 72], nice=False)),
        y=alt.Y('점수:Q', title='표준화 점수 (1 = 정상 경계)'), color='신호:N',
        tooltip=['신호', alt.Tooltip('DONE_H', title='시각(h)'), alt.Tooltip('점수', format='.2f')])
    avg = alt.Chart(h).mark_line(color='black', strokeWidth=2.5).encode(
        x='DONE_H:Q', y='SCORE:Q', tooltip=[alt.Tooltip('SCORE', title='네 신호 평균', format='.2f')])
    one = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='black', strokeDash=[2, 2]).encode(y='y:Q')
    st.subheader('네 신호의 흐름 (1시간마다)')
    st.altair_chart(alt.layer(*mnt_layer(sid, None), sig, avg, one).properties(height=300), width='stretch')
    st.caption('검은 굵은 선: 네 신호 평균 점수 · 점선: 정상 범위 경계(1). 정비 후 재가동하면 처음 2시간은 기준을 잡는 시간이라 비어 있습니다.')

    # 지금 신호 상태와 모델 기여도
    left, right = st.columns(2)
    with left:
        st.subheader('지금 신호 상태')
        cur = after_last_maintenance(h.assign(HOUR_START=h['DONE_H'] - 1), 'HOUR_START')
        last = cur.dropna(subset=list(SIGNAL_NAMES.values())).tail(1)
        if row['상태'] == '정비 중':
            st.write('정비 중이라 신호가 없습니다.')
        elif last.empty:
            st.write('재가동 뒤 아직 계산된 신호가 없습니다. 재가동 후 처음 2시간은 기준을 잡는 시간입니다. '
                     '(정비 전 신호는 지금 상태가 아니라서 보여 주지 않습니다.)')
        else:
            bar = last.melt(value_vars=list(SIGNAL_NAMES.values()), var_name='신호', value_name='점수')
            bar['정상 범위'] = np.where(bar['점수'] > 1, '벗어남', '안')
            st.altair_chart(
                alt.layer(
                    alt.Chart(bar).mark_bar().encode(
                        y=alt.Y('신호:N', title=None, sort='-x'), x=alt.X('점수:Q', title='표준화 점수'),
                        color=alt.Color('정상 범위:N', scale=alt.Scale(domain=['벗어남', '안'], range=[ALARM_RED, OK_GRAY]))),
                    alt.Chart(pd.DataFrame({'x': [1]})).mark_rule(strokeDash=[2, 2]).encode(x='x:Q'),
                ).properties(height=180), width='stretch')
            st.caption(f'{last["DONE_H"].iloc[0]:.0f}시간까지의 1시간 값. 1을 넘으면 정상 시나리오의 95%보다 많이 벗어난 것입니다.')
    with right:
        with st.expander('자세히: 모델이 본 이유 (기여도)'):
            cp = after_last_maintenance(pts_now[pts_now['SCENARIO_ID'] == sid], 'T_H')
            if contrib is None:
                st.write(f'최종 모델({bundle["final"]["model"]})은 나무 여러 개의 투표로 확률을 내서, '
                         '특징별 기여도를 바로 계산할 수 없습니다. 왼쪽의 신호 상태와 위의 신호 흐름으로 경보 이유를 확인하세요.')
            elif cp.empty:
                st.write('아직 예측이 없습니다. 정비가 시작되면 정비 전 예측은 쓰지 않고, 재가동 뒤 다음 예측 시점(6시간마다)을 기다립니다.')
            else:
                cr = contrib.loc[cp.index[-1]].rename('기여도').reset_index().rename(columns={'index': '묶음'})
                st.altair_chart(alt.Chart(cr).mark_bar().encode(
                    y=alt.Y('묶음:N', title=None, sort='-x'), x=alt.X('기여도:Q', title='위험을 올린 정도 (로그 오즈)'),
                    color=alt.condition('datum.기여도 > 0', alt.value(ALARM_RED), alt.value('#2f5d8a'))
                ).properties(height=220), width='stretch')
                st.caption('양수는 위험을 올린 쪽, 음수는 내린 쪽입니다(0 = 학습 세트 평균 시점). '
                           '네 신호와 결합 점수는 서로 비슷한 정보라 모델 안에서 서로 상쇄될 수 있습니다. '
                           '묶음 하나의 크기보다 왼쪽의 신호 상태를 먼저 보세요.')


# 3. 모델 정보

elif page == '모델 정보':
    v = bundle['validation']
    st.header('모델 정보와 검증 세트 성능')
    st.markdown(
        f'- **모델**: {bundle["final"]["model"]} · 특징 {len(bundle["features"])}개 ({bundle["final"]["features"]})\n'
        f'- **학습**: 학습 세트 216개 시나리오의 예측 시점 {bundle["n_train_points"]:,}개\n'
        f'- **경보 기준**: 24시간 안 고장 확률 ≥ {TH:.2f}. 학습 세트 교차검증에서 '
        '점진 고장을 모두 잡고 진짜 오경보가 0인 값 중 가장 낮은 값(오경보 없이 가장 일찍 경보)\n'
        '- **검증 세트**: 72개 시나리오, 학습·선택에 쓰지 않고 마지막에 한 번만 평가')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('AP (순위 능력)', f'{v["point"]["AP"]:.3f}', help='기준 예제 0.806')
    c2.metric('정밀도 / 재현율', f'{v["point"]["정밀도"]:.2f} / {v["point"]["재현율 (점진 고장만)"]:.2f}',
              help=f'재현율은 점진 고장 기준. 신호가 없는 돌발 고장을 포함하면 {v["point"]["재현율"]:.2f}')
    c3.metric('고장 감지', f'{v["event"]["감지"]}/{v["event"]["고장"]}', help='돌발 고장 2건은 신호가 없어 못 잡음')
    c4.metric('리드타임 중앙값', f'{v["event"]["리드타임 중앙값(h)"]:.0f}시간')
    st.markdown(
        f'- 오경보 {v["point"]["FP"]}개 중 {v["point"]["FP 중 이른 경보"]}개는 고장 24시간보다 일찍 울린 **이른 경보**이고, '
        f'고장 예정이 없는데 울린 진짜 오경보는 {v["point"]["FP 중 진짜 오경보"]}개입니다.\n'
        f'- 놓친 시점 {v["point"]["FN"]}개 중 {v["point"]["FN 중 돌발 고장"]}개는 신호 없이 갑자기 고장 난 돌발 고장입니다.\n'
        '- 위 숫자는 정답을 알 수 있는 시점만으로 셌습니다. 관측 끝부분의 미확정 경보는 비용 계산기 화면에 있습니다.')
    if CAUTION:
        cv = CAUTION['validation']
        st.subheader('2단계 경보: 주의 단계')
        st.markdown(
            f'- **주의 모델**: {CAUTION["final"]["model"]} · {CAUTION["final"]["features"]} 특징 {len(CAUTION["features"])}개. '
            '"지금 열화 중인가"를 맞힙니다(정답: 진단 정답의 열화 구간, 학습 때만 사용)\n'
            f'- **주의 기준**: 지금 열화 중일 확률 ≥ {CTH:.2f}. 학습 세트 교차검증에서 주의 중 {CAUTION["min_precision"]:.0%} 이상이 '
            '실제 열화인 값 중 가장 낮은 값\n'
            '- **헛걸음을 조금 허용하는 이유**: 현장에서는 주의를 받으면 장비 상태를 확인하고, 설비가 돌면 다음 정비 때 함께 정비합니다. '
            '설비를 세우거나 바로 발주하지 않으므로 헛걸음 비용은 점검 한 번입니다.')
        d1, d2, d3, d4 = st.columns(4)
        d1.metric('주의 정밀도', f'{cv["주의 정밀도"]:.2f}', help=f'주의 중 실제로 열화 중이던 비율. 목표 {CAUTION["min_precision"]}')
        d2.metric('주의가 켜진 점진 고장', f'{cv["주의로 감지"]}/{cv["점진 고장"]}')
        d3.metric('경보보다 먼저 (평균)', f'{cv["추가 준비 시간 평균(h)"]:.1f}시간')
        d4.metric('첫 주의 (중앙값)', f'고장 {cv["주의 리드타임 중앙값(h)"]:.0f}시간 전')
        st.caption(f'검증 세트 주의 {cv["주의 묶음"]}번 중 헛걸음 {cv["열화 아닌 설비의 주의"]}번. 대부분은 열화가 시작되기 몇 시간 전이었고, '
                   '일부는 끝까지 정상인 설비였습니다. 자세한 근거는 reports/연주_조기경보_결과보고서.md 7-5절')

    st.subheader('검증 세트 고장별 결과')
    ev = v['events'].rename(columns={'SCENARIO_ID': '시나리오', 'TYPE': '고장 종류'})
    st.dataframe(ev[['시나리오', '고장 종류', '감지', '리드타임(h)']].sort_values(['고장 종류', '리드타임(h)'], ascending=[True, False]),
                 hide_index=True, column_config={'리드타임(h)': st.column_config.NumberColumn(format='%.1f')})
    st.caption('리드타임: 고장 4시간 전까지 난 경보 중 가장 이른 경보부터 고장까지의 시간. '
               '자세한 근거는 reports/연주_조기경보_결과보고서.md 4절과 부록 B')


# 4. 비용 계산기

else:
    v = bundle['validation']['event']
    n = dict(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
             n_false=v['고장과 무관한 경보 묶음: 오경보'])
    st.header('비용 계산기 · 원인 부품을 얼마나 맞힐 수 있으면 이득인가')
    st.markdown(
        f'검증 세트 결과(고장 {v["고장"]}건 중 {n["n_det"]}건 감지, 고장과 무관한 교체 {n["n_needed"]}번, '
        f'진짜 오경보 {n["n_false"]}번)에 데이터셋 비용 규칙을 적용합니다.\n'
        '- 경보가 나면 바로 교체합니다. 한 부품만 바꾸는데 틀리면, 불필요 정비가 되고 고장도 못 막습니다.\n'
        '- 센서로는 원인 부품을 구분할 수 없어(EDA E9), 부품을 맞힐 확률은 현장 점검 능력에 달려 있습니다.')
    p = st.slider('원인 부품을 맞힐 확률', 0.5, 1.0, 0.5, step=0.01,
                  help='0.5 = 둘 중 하나를 찍음, 1.0 = 항상 맞힘')
    cc = cost_curve(**n)
    one = float(np.interp(p, cc['p'], cc['한 부품만 교체']))
    both, none = cc['두 부품 모두 교체'].iloc[0], cc['아무것도 안 함'].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric('한 부품만 교체', f'{one:.0f}점', f'{one - none:+.0f} (아무것도 안 함 대비)', delta_color='inverse')
    c2.metric('두 부품 모두 교체', f'{both:.0f}점', f'{both - none:+.0f}', delta_color='inverse')
    c3.metric('아무것도 안 함', f'{none:.0f}점')
    u = bundle['validation'].get('unlabeled')
    if u:
        lo, hi = u['비용 범위']['두 부품']
        st.warning(
            f'**위 비용은 정답을 알 수 있는 구간만 센 값입니다.** 관측이 끝나 결과를 알 수 없는 시점 {u["시점"]}개에서도 '
            f'경보가 {u["경보 시점"]}번(경보 묶음 {u["경보 묶음"]}개) 울렸고, 실제 운영이라면 교체가 일어납니다. '
            f'두 부품 교체 기준으로 {lo:.0f}~{hi:.0f}점이 더 들 수 있습니다. 이 경보가 관측 뒤의 고장을 막았는지, '
            '"아무것도 안 함"에서 관측 뒤에 고장이 났는지는 알 수 없어 운영 비용은 확정할 수 없습니다.')
    even = cc.loc[cc['한 부품만 교체'] <= both, 'p']
    if len(even):
        st.info(f'부품을 맞힐 확률이 **{even.iloc[0]:.2f} 이상**이면 한 부품만 바꾸는 쪽이 두 부품 모두 바꾸는 것보다 쌉니다.')
    long = cc.melt(id_vars='p', var_name='방식', value_name='비용')
    chart = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X('p:Q', title='원인 부품을 맞힐 확률'), y=alt.Y('비용:Q', title='비용 (점수, 낮을수록 좋음)'),
        color=alt.Color('방식:N', scale=alt.Scale(domain=['한 부품만 교체', '두 부품 모두 교체', '아무것도 안 함'],
                                                range=['#2f5d8a', '#4a9b5e', OK_GRAY])))
    now_rule = alt.Chart(pd.DataFrame({'p': [p]})).mark_rule(strokeDash=[4, 4]).encode(x='p:Q')
    st.altair_chart((chart + now_rule).properties(height=360), width='stretch')
