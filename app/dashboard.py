"""연주 설비 24시간 조기경보 대시보드 (시연용).

    python analysis/05_predict_test.py      # 처음 한 번: 모델 학습·저장 (models/risk24.joblib)
    streamlit run app/dashboard.py          # 브라우저에서 http://localhost:8501

- 저장된 모델로 예측만 한다(다시 학습하지 않는다).
- '지금 시각'을 옮기면 그 시각까지의 데이터만으로 본 상태를 보여 준다(재생 방식 시연).
  실제 서비스라면 이 자리에 새로 들어온 센서 데이터가 들어간다(src/serving.py 참고).
- 데이터 세트: 테스트(정답 없음, 실제 운영과 같은 상황) / 검증(정답 있음, 실제 고장과 비교 가능)

화면
  1. 전체 현황    설비(시나리오)별 지금 위험도, 위험한 순서
  2. 설비 상세    위험 확률과 네 신호의 흐름, 정비 기록, 경보 이유, 조치 안내
  3. 모델 정보    모델·임계값과 검증 세트 성능
  4. 비용 계산기  원인 부품을 맞힐 확률에 따른 비용 (팀 논의 항목 C)
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
ALARM_RED, OK_GRAY, MNT_GRAY = '#d64545', '#9e9e9e', '#bdbdbd'


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


# 사이드바: 화면, 데이터 세트, 지금 시각

st.sidebar.title('연주 설비 조기경보')
page = st.sidebar.radio('화면', ['전체 현황', '설비 상세', '모델 정보', '비용 계산기'])
split_label = st.sidebar.radio('데이터 세트', list(SPLITS))
split = SPLITS[split_label]
now = st.sidebar.slider('지금 시각 (시나리오 시작 후 경과 시간, h)', 0, 72, 72, step=1,
                        help='이 시각까지의 데이터만으로 본 상태를 보여 줍니다. 왼쪽으로 옮기면 과거로 돌아가 재생할 수 있습니다.')
show_truth = split == 'validation' and st.sidebar.checkbox(
    '실제 고장 시각 미리 보기 (정답)', help='검증 세트에만 있습니다. 모델은 이 정보를 보지 않습니다.')
st.sidebar.caption(f'모델: {bundle["final"]["model"]} · {bundle["final"]["features"]}\n\n'
                   f'경보 기준: 24시간 안 고장 확률 ≥ {TH:.2f}\n\n모델 저장: {bundle["created_at"]}')

pts, hr, contrib, mnt = get_data(split)
pts_now = pts[pts['T_H'] <= now]
hr_now = hr[hr['DONE_H'] <= now]
mnt_now = mnt[mnt['START_H'] <= now].assign(END_SHOWN=lambda d: d['END_H'].clip(upper=now))


def status_table():
    """시나리오마다 지금 시각 기준 상태 한 줄"""
    last = pts_now.sort_values('T_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
    first_alarm = pts_now[pts_now['ALARM']].groupby('SCENARIO_ID')['T_H'].min()
    last_hr = hr_now.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1)
    worst = last_hr.set_index('SCENARIO_ID')[list(SIGNAL_NAMES)].idxmax(axis=1).map(SIGNAL_NAMES)
    in_mnt = set(mnt[(mnt['START_H'] <= now) & (mnt['END_H'] > now)]['SCENARIO_ID'])
    # 마지막 예측 뒤에 정비가 시작됐으면 그 예측은 정비 전 상태라 쓰지 않는다
    last_mnt = mnt_now.groupby('SCENARIO_ID')['START_H'].max()
    stale = set(last_mnt[last_mnt > last['T_H'].reindex(last_mnt.index).fillna(-1)].index) - in_mnt
    t = pd.DataFrame({'설비 (시나리오)': sorted(pts['SCENARIO_ID'].unique())}).set_index('설비 (시나리오)', drop=False)
    t['위험 확률'] = last['PROB']
    t['마지막 예측 (h)'] = last['T_H']
    t['상태'] = np.select([t.index.isin(in_mnt), t.index.isin(stale),
                          last.reindex(t.index)['ALARM'].fillna(False).astype(bool)],
                         ['정비 중', '정비 후 예측 대기', '경보'], default='정상')
    t.loc[t['상태'].isin(['정비 중', '정비 후 예측 대기']), '위험 확률'] = np.nan
    t['첫 경보 (h)'] = first_alarm
    t['가장 많이 벗어난 신호'] = worst
    if show_truth:
        fails = mnt.dropna(subset=['FAIL_H'])
        nxt = fails[fails['FAIL_H'] > now].groupby('SCENARIO_ID')['FAIL_H'].min()
        t['다음 실제 고장까지 (h, 정답)'] = (nxt - now).round(1)
    return t.sort_values('위험 확률', ascending=False)


def mnt_layer(sid, y_field_max):
    """정비 기간(회색 띠)과 고장 시각(빨간 선)"""
    m = mnt_now[mnt_now['SCENARIO_ID'] == sid]
    band = alt.Chart(m).mark_rect(color=MNT_GRAY, opacity=0.35).encode(
        x='START_H:Q', x2='END_SHOWN:Q', tooltip=['종류', 'PART_CD', 'START_H', 'END_H'])
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('설비 수', len(t))
    c2.metric('지금 경보', int((t['상태'] == '경보').sum()))
    c3.metric('정비 중·예측 대기', int(t['상태'].isin(['정비 중', '정비 후 예측 대기']).sum()))
    c4.metric('지금까지 경보가 난 설비', int(t['첫 경보 (h)'].notna().sum()))
    st.caption(f'위험 확률은 6시간마다 계산합니다. 확률이 {TH:.2f} 이상이면 경보입니다. '
               '가장 많이 벗어난 신호는 마지막 1시간 기준이며, 표준화 점수 1이 정상 범위의 경계입니다.')
    st.dataframe(
        t, hide_index=True, height=560,
        column_config={
            '위험 확률': st.column_config.ProgressColumn('위험 확률', format='%.2f', min_value=0, max_value=1),
            '마지막 예측 (h)': st.column_config.NumberColumn(format='%d'),
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

    c1, c2, c3 = st.columns(3)
    c1.metric('24시간 안 고장 확률', '—' if pd.isna(row['위험 확률']) else f'{row["위험 확률"]:.2f}',
              help=f'마지막 예측: {row["마지막 예측 (h)"]:.0f}시간' if pd.notna(row['마지막 예측 (h)']) else None)
    c2.metric('상태', row['상태'])
    c3.metric('첫 경보', '—' if pd.isna(row['첫 경보 (h)']) else f'{row["첫 경보 (h)"]:.0f}시간')

    if row['상태'] == '경보':
        st.error(
            '**조치 안내** · 24시간 안에 고장 날 위험이 높습니다.\n'
            '- 교체에 4시간이 걸리므로, 늦어도 고장 예상 4시간 전에는 교체를 시작해야 합니다.\n'
            '- 센서로는 원인 부품(진동장치·밸브)을 구분할 수 없습니다. 두 부품을 모두 점검하세요.\n'
            '- 고장 원인과 상관없이 같은 신호가 오르므로, 아래 신호 흐름으로 열화가 계속되는지 확인하세요.')
    elif row['상태'] in ('정비 중', '정비 후 예측 대기'):
        st.warning('정비가 시작돼 정비 전 예측은 더 이상 쓰지 않습니다. 재가동 뒤 다음 예측 시점(6시간마다)에 다시 예측합니다.')
    else:
        st.success('지금은 24시간 안 고장 위험이 낮습니다.')

    # 위험 확률 흐름
    p = pts_now[pts_now['SCENARIO_ID'] == sid].assign(경보=lambda d: np.where(d['ALARM'], '경보', '정상'))
    x = alt.X('T_H:Q', title='시나리오 시작 후 경과 시간 (h)', scale=alt.Scale(domain=[0, 72], nice=False))
    line = alt.Chart(p).mark_line(color='#2f5d8a').encode(x=x, y=alt.Y('PROB:Q', title='24시간 안 고장 확률',
                                                                          scale=alt.Scale(domain=[0, 1])))
    dots = alt.Chart(p).mark_circle(size=90).encode(
        x=x, y='PROB:Q', color=alt.Color('경보:N', scale=alt.Scale(domain=['경보', '정상'], range=[ALARM_RED, OK_GRAY])),
        tooltip=[alt.Tooltip('T_H', title='시각(h)'), alt.Tooltip('PROB', title='확률', format='.2f'), '경보'])
    rule = alt.Chart(pd.DataFrame({'y': [TH]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')
    st.subheader('위험 확률')
    st.altair_chart(alt.layer(*mnt_layer(sid, 1), line, dots, rule).properties(height=260), width='stretch')
    st.caption(f'점선: 경보 기준 {TH:.2f} · 회색 띠: 정비 · 빨간 실선: 고장'
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
        last = h.dropna(subset=list(SIGNAL_NAMES.values())).tail(1)
        if last.empty:
            st.write('아직 계산된 신호가 없습니다 (시작 직후 또는 정비 직후).')
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
            cp = pts_now[pts_now['SCENARIO_ID'] == sid]
            if cp.empty:
                st.write('아직 예측이 없습니다.')
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
    c2.metric('정밀도 / 재현율', f'{v["point"]["정밀도"]:.2f} / {v["point"]["재현율"]:.2f}')
    c3.metric('고장 감지', f'{v["event"]["감지"]}/{v["event"]["고장"]}', help='돌발 고장 2건은 신호가 없어 못 잡음')
    c4.metric('리드타임 중앙값', f'{v["event"]["리드타임 중앙값(h)"]:.0f}시간')
    st.markdown(
        f'- 오경보 {v["point"]["FP"]}개는 모두 고장 24시간보다 이른 **이른 경보**이고, 고장 예정이 없는데 울린 진짜 오경보는 '
        f'{v["point"]["FP 중 진짜 오경보"]}개입니다.\n'
        f'- 놓친 시점 {v["point"]["FN"]}개 중 {v["point"]["FN 중 돌발 고장"]}개는 신호 없이 갑자기 고장 난 돌발 고장입니다.')
    st.subheader('검증 세트 고장별 결과')
    ev = v['events'].rename(columns={'SCENARIO_ID': '시나리오', 'TYPE': '고장 종류'})
    st.dataframe(ev[['시나리오', '고장 종류', '감지', '리드타임(h)']].sort_values(['고장 종류', '리드타임(h)'], ascending=[True, False]),
                 hide_index=True, column_config={'리드타임(h)': st.column_config.NumberColumn(format='%.1f')})
    st.caption('리드타임: 고장 4시간 전까지 난 경보 중 가장 이른 경보부터 고장까지의 시간. '
               '자세한 근거는 reports/연주_조기경보_모델_결과.md')


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
