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
  3. 의사결정     경보 설비마다 지금 교체 / 정기 교체 때 함께의 기대 비용 비교와 추천 (src/dss.py), 가정값 조정
  4. 모델 정보    경보·주의 모델과 기준, 검증 세트 성능
  5. 비용 계산기  원인 부품을 맞힐 확률에 따른 비용 (팀 결정 C)
"""

import sys
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.posko_data import load_scenarios
from src.dss import fail_prob, field_costs, options, plans, recommend
from src import app_data
from src.auth import require_login
from src.serving import cost_curve, explain, failure_hours, load_bundle, score_scenarios

st.set_page_config(page_title='연주 설비 조기경보', page_icon=':material/precision_manufacturing:', layout='wide',
                   initial_sidebar_state='collapsed')

# 실행 방식 (reports/보안_점검.md)
#   로컬: data/posko/ 원자료와 models/risk24.joblib로 직접 계산한다 (내 컴퓨터 안에서만 열림, 비밀번호 없음)
#   클라우드: 원자료가 없고 app_data/만 있다. 미리 계산한 결과만 읽고(모델 파일을 불러오지 않음), 비밀번호를 요구한다
CLOUD = not (ROOT / 'data' / 'posko').exists() and app_data.available()
require_login(st, CLOUD)

SIGNAL_NAMES = {'Z_MLD-AMP': '몰드 레벨 진폭', 'Z_CONTROL_DELAY': '제어 지연',
                'Z_WDL': '인발 부하', 'Z_OSC-STK': '진동 스트로크'}
SPLITS = {'테스트 (정답 없음)': 'test', '검증 (정답 있음)': 'validation'}   # 화면 위 선택지
ALARM_RED, CAUTION_YELLOW, OK_GRAY, MNT_GRAY = '#d64545', '#e0a526', '#9e9e9e', '#bdbdbd'
# 화면에 보여 줄 단계 이름: 정상(NORMAL) → 주의(CAUTION, 열화 감지) → 경보(WARNING, 24시간 고장 위험)
# 모델 출력(STAGE)은 '경보'/'주의'/'정상' 그대로 두고 화면에서만 이름을 바꾼다 (결과보고서 7-7절 ⑥)
LABEL = {'경보': '경보 (WARNING)', '주의': '주의 (CAUTION)', '정상': '정상 (NORMAL)'}
WARN, CAU, NORM = LABEL['경보'], LABEL['주의'], LABEL['정상']
STAGE_COLORS = alt.Scale(domain=[WARN, CAU, NORM], range=[ALARM_RED, CAUTION_YELLOW, OK_GRAY])


# 데이터 불러오기 (처음 한 번만 계산하고 기억해 둔다)

@st.cache_resource
def get_app_data():
    return app_data.load()


@st.cache_resource
def get_bundle():
    return get_app_data()[0] if CLOUD else load_bundle()


@st.cache_data(show_spinner='센서 데이터를 요약하고 예측하는 중...')
def get_data(split):
    if CLOUD:
        pts, hr, contrib, unit, mnt, _ = get_app_data()[1][split]
        return pts, hr, contrib, unit, mnt
    bundle = get_bundle()
    ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
    pts, hr = score_scenarios(ids, bundle)
    contrib, unit = explain(bundle, pts)
    hr = hr.assign(DONE_H=hr['HOUR'] + 1)          # 1시간 구간이 끝난 시각 = 그 값을 알 수 있게 되는 시각
    return pts, hr[['SCENARIO_ID', 'DONE_H', 'SCORE', *SIGNAL_NAMES]], contrib, unit, failure_hours(ids)


try:
    bundle = get_bundle()
except (FileNotFoundError, ValueError) as e:   # 없는 파일, 저장 뒤 바뀐 파일, 빠진 값
    print(f'[dashboard] 모델·데이터를 불러오지 못함: {e!r}', file=sys.stderr)   # 자세한 내용(파일 경로 등)은 서버 로그에만
    st.error('모델·데이터 파일을 쓸 수 없습니다. 관리자에게 문의하세요.' if CLOUD else
             '모델 파일을 쓸 수 없습니다. 터미널에서 `python analysis/05_predict_test.py`를 실행해 다시 만드세요. '
             '(자세한 내용은 터미널 로그)')
    st.stop()
TH = bundle['threshold']
STEP = bundle.get('step_h', 6)           # 예측 간격(시간). 예전 모델 파일은 6시간
CAUTION = bundle.get('caution')          # 주의 모델이 없는 예전 모델 파일이면 경보 1단계만 보여 준다
CTH = CAUTION['threshold'] if CAUTION else None


# 화면 위: 보는 사람 → 화면 → 데이터 세트·지금 시각 (모든 화면에 같은 자리)
#   현장 작업    현장 실무자: 할 일·순서·이유(시간과 위험)·안전 경고. 비용과 가정값은 보여 주지 않는다
#   정비 책임자  의사결정 주체: 비용·가정값·정책 시뮬레이션까지
ROLE_PAGES = {'현장 작업': ['오늘 할 일', '설비 상세'],
              '정비 책임자': ['전체 현황', '의사결정', '설비 상세', '비용 계산기', '모델 정보']}

ICONS = {'현장 작업': ':material/engineering:', '정비 책임자': ':material/manage_accounts:',
         '오늘 할 일': ':material/checklist:', '설비 상세': ':material/monitoring:', '전체 현황': ':material/dashboard:',
         '의사결정': ':material/balance:', '비용 계산기': ':material/calculate:', '모델 정보': ':material/model_training:'}
with_icon = lambda x: f'{ICONS.get(x, "")} {x}'

st.markdown('### :material/precision_manufacturing: 연주 설비 조기경보')
with st.container(horizontal=True, vertical_alignment='bottom', gap='medium'):
    st.session_state.setdefault('role', '현장 작업')     # 기본값은 세션 상태로만 정한다 (표에서 설비를 누르면 화면을 바꾸기 때문)
    role = st.segmented_control('보는 사람', list(ROLE_PAGES), key='role', required=True, format_func=with_icon)
    st.session_state.setdefault(f'page_{role}', ROLE_PAGES[role][0])
    page = st.segmented_control('화면', ROLE_PAGES[role], key=f'page_{role}', required=True, format_func=with_icon)
c1, c2, c3 = st.columns([1.6, 2.6, 1.2], vertical_alignment='center')
split_label = c1.selectbox('데이터 세트', list(SPLITS), key='split_label',
                       help='테스트: 정답이 없는 실제 운영과 같은 상황 · 검증: 정답이 있어 실제 고장과 비교할 수 있음')
split = SPLITS[split_label]
now = c2.slider('지금 시각 (시나리오 시작 후 h)', 0, 72, 72, step=1, key='now',
                help='이 시각까지의 데이터만으로 본 상태를 보여 줍니다. 왼쪽으로 옮기면 과거로 돌아가 재생할 수 있습니다.')
show_truth = c3.toggle('정답 보기', key='truth', disabled=split != 'validation',
                       help='검증 세트에서만 켤 수 있습니다. 앞으로 일어날 실제 고장 시각(정답)을 겹쳐 봅니다. 모델은 이 정보를 보지 않습니다.') and split == 'validation'

pts, hr, contrib, contrib_unit, mnt = get_data(split)


@st.cache_data
def get_plans(split):
    """정기 교체 계획 (calendar 표, 미리 안다고 가정): 시나리오 시작 후 시각(h)과 계획된 부품"""
    if CLOUD:
        return get_app_data()[1][split][5]
    ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
    start = load_scenarios().set_index('SCENARIO_ID')['START_AT']
    pl = plans(ids)
    pl['PLAN_H'] = (pl['PLAN_START'] - pl['SCENARIO_ID'].map(start)).dt.total_seconds() / 3600
    return pl


PART_NAMES = {'valve': '밸브', 'oscillator': '진동장치'}
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
    # 경보 유지: 마지막 정비 시작 이후 한 번이라도 경보였으면, 지금 확률이 내려가도 정비 전까지 경보로 둔다
    since = after_last_maintenance(pts_now, 'T_H')
    latched = set(since.loc[since['ALARM'], 'SCENARIO_ID']) - set(last.index[last['ALARM']])
    stage = last['STAGE'].reindex(t.index).fillna('정상').where(~t.index.isin(latched), '경보')
    t['상태'] = np.select([t.index.isin(in_mnt), t.index.isin(stale)], ['정비 중', '정비 후 예측 대기'], default=stage)
    t['경보 유지'] = np.where(t.index.isin(latched) & (t['상태'] == '경보'), '유지 (확률 내려감)', '')
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



# 의사결정 계산 (정비 책임자 '의사결정'과 현장 '오늘 할 일'이 같이 쓴다)

def assumptions():
    """정비 책임자가 정한 가정값 (없으면 기본값 = 모델 파일의 DSS 가정값 + 팀 결정: 정비팀 2팀, 팀당 하루 8공수)"""
    D = bundle.get('dss')
    base, bc = D['config'], D['config'].costs
    default = {'safety': float(base.safety_max), 'prep': int(base.prep_h), 'crew_wait': int(base.crew_wait_h),
               'n_crews': 2, 'crew_day_h': 8, 'down_h': float(bc.downtime_h), 'penalty': float(bc.failure_penalty),
               'p_other': float(base.p_other_needed), 'max_wait': int(base.max_wait_h), 'delay_h': 0,
               'delay_rate': float(bc.downtime_h),
               # 비용 기준: 데이터셋 점수 / 현장 가정(만 원, 정건우 현장 경험 2026-10-15)
               'mode': COST_MODES[0], 'f_cons': 100.0, 'f_part': 200.0, 'f_labor_part': 20.0, 'f_labor_ph': 3.0,
               'f_crew': 4, 'f_down': 500.0, 'f_fail': 300.0, 'f_delay_rate': 500.0}
    return {**default, **st.session_state.get('assume', {})}


COST_MODES = ['데이터셋 점수', '현장 가정 (만 원)']


def make_cfg(A):
    base = bundle['dss']['config']
    if A['mode'] == COST_MODES[1]:
        costs = field_costs(consumables=A['f_cons'], major_part=A['f_part'], labor_per_part=A['f_labor_part'],
                            labor_per_person_h=A['f_labor_ph'], crew_size=A['f_crew'], downtime_h=A['f_down'],
                            failure_extra=A['f_fail'])
    else:
        costs = replace(base.costs, downtime_h=A['down_h'], failure_penalty=A['penalty'])
    cfg = replace(base, safety_max=A['safety'], prep_h=A['prep'], crew_wait_h=A['crew_wait'], p_other_needed=A['p_other'],
                  max_wait_h=A['max_wait'], costs=costs)
    return costs, cfg


def work_plan(A, t):
    """경보 설비마다 추천 선택지와 정비팀 일정 (우선순위, 하루 작업 한도 포함)"""
    costs, cfg = make_cfg(A)
    n_crews, crew_day_h = A['n_crews'], A['crew_day_h']
    delay_h = A['delay_h']
    delay_rate = A['f_delay_rate'] if A['mode'] == COST_MODES[1] else A['delay_rate']
    curve = bundle['dss']['curve']
    pl = get_plans(split)
    since = after_last_maintenance(pts_now, 'T_H')
    first_alarm_now = since[since['ALARM']].groupby('SCENARIO_ID')['T_H'].min()
    dly = delay_h if delay_h > 0 else None

    def evaluate_options(sid, crew_extra=0.0, p_override=None):
        t_a = first_alarm_now.get(sid, np.nan)
        nxt = pl[(pl['SCENARIO_ID'] == sid) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').head(1)
        wait_h = float(nxt['PLAN_H'].iloc[0] - now) if len(nxt) else np.nan
        part = PART_NAMES.get(nxt['PLAN_PART'].iloc[0], nxt['PLAN_PART'].iloc[0]) if len(nxt) else None
        elapsed = max(0.0, now - t_a) if pd.notna(t_a) else 0.0
        c = replace(cfg, crew_wait_h=cfg.crew_wait_h + crew_extra)
        opts = options(wait_h, curve, c, elapsed, delay_h=dly, delay_downtime_h=delay_rate, p_override=p_override)
        best, urgent = recommend(opts)
        return {'t_a': t_a, 'wait_h': wait_h, 'part': part, 'elapsed': elapsed, 'opts': opts, 'best': best, 'urgent': urgent}

    alarm_ids = t.index[t['상태'] == WARN].tolist()
    first = {sid: evaluate_options(sid) for sid in alarm_ids}
    # 우선순위: 지금 정비가 필요한 설비를 위험한 순서로(한 차례 미루면 고장 확률이 큰 순서) 정비팀에 배정한다
    need_crew = sorted([s for s in alarm_ids if not first[s]['best']['선택지'].startswith('다음 정기')],
                       key=lambda s: -fail_prob(curve, first[s]['elapsed'], first[s]['opts'][0]['정비 완료까지(h)'] + costs.replace_h))
    per_day = max(1, int(crew_day_h // costs.replace_h))     # 팀 하나가 하루에 할 수 있는 교체 수 (8공수 ÷ 4시간 = 2)

    def crew_delay(k):
        """k번째(0부터) 정비 작업이 시작될 때까지 기다리는 시간: 팀 수만큼 동시에, 팀당 하루 per_day번, 넘으면 다음 날"""
        rnd = k // n_crews                 # 몇 번째 차례인가 (모든 팀이 동시에 한 대씩)
        day, within = divmod(rnd, per_day)
        return day * 24.0 + within * costs.replace_h

    rows = []
    for k, sid in enumerate(need_crew + [s for s in alarm_ids if s not in need_crew]):
        wait_crew = crew_delay(k) if sid in need_crew else 0.0
        slot = wait_crew
        e = evaluate_options(sid, crew_extra=wait_crew) if wait_crew else first[sid]
        b = e['best']
        rows.append({'순위': k + 1 if sid in need_crew else None, '설비 (시나리오)': sid, '추천': b['선택지'],
                     '정비 완료까지 (h)': b['정비 완료까지(h)'], '기대 총비용': b['기대 총비용'], '완료 전 고장 확률': b['고장 확률'],
                     '제약': ('즉시 조치: 지금 해도 완료 전 고장 확률이 안전 기준 초과' if first[sid]['urgent'] else
                            ('안전 기준 초과: 정비팀 하루 작업 한도 초과 → 다음 날 (팀·공수 추가 필요)' if wait_crew >= 24 else
                             '안전 기준 초과: 정비팀 부족 (팀 추가 필요)') if e['urgent'] else
                            (f'충족 (정비팀 차례 대기 {wait_crew:.0f}시간' + (', 다음 날 작업' if wait_crew >= 24 else '') + ')'
                             if wait_crew else '충족')),
                     '첫 경보 (h)': e['t_a'],
                     '다음 정기 교체': f'{e["wait_h"]:.0f}시간 뒤 ({e["part"]})' if pd.notna(e['wait_h']) else '계획 없음',
                     '_e': e, '_slot': slot,
                     '_start': (e['wait_h'] if b['선택지'].startswith('다음 정기') else b['정비 완료까지(h)'] - costs.replace_h)})
    dec = pd.DataFrame(rows)
    return {'dec': dec, 'need_crew': need_crew, 'per_day': per_day, 'evaluate_options': evaluate_options,
            'cfg': cfg, 'costs': costs, 'curve': curve, 'pl': pl}


def risk_word(p, safety):
    """고장 확률을 현장 말로 (비용 없이 위험만)"""
    if pd.isna(p):
        return '—'
    return '높음' if p > safety else ('보통' if p >= 0.05 else '낮음')


def open_detail(table_key, ids):
    """표에서 고른 설비로 '설비 상세' 화면을 연다 (st.dataframe 선택 콜백)"""
    rows = st.session_state[table_key].selection.rows
    if rows:
        st.session_state['sid'] = ids[rows[0]]
        st.session_state[f'page_{st.session_state["role"]}'] = '설비 상세'


STAGE_STYLE = {WARN: 'background-color: rgba(214,69,69,.18)', CAU: 'background-color: rgba(224,165,38,.22)',
               '정비 중': 'background-color: rgba(158,158,158,.18)', '정비 후 예측 대기': 'background-color: rgba(158,158,158,.18)'}


def style_stage(df, col):
    return df.style.map(lambda v: STAGE_STYLE.get(v, ''), subset=[col])


# 0. 현장 작업: 오늘 할 일

if page == '오늘 할 일':
    st.header('오늘 할 일', icon=ICONS['오늘 할 일'])
    st.caption(f'{split_label} · 지금 {now}시간 기준')
    if not bundle.get('dss'):
        st.warning('이 모델 파일에는 정비 계획 정보가 없습니다. python analysis/05_predict_test.py 로 다시 만드세요.')
        st.stop()
    A = assumptions()
    t = status_table()
    W = work_plan(A, t)
    dec, cfg, costs = W['dec'], W['cfg'], W['costs']

    # 안전 경고가 맨 위
    urgent = dec[dec['제약'].str.startswith('즉시')]['설비 (시나리오)'].tolist() if len(dec) else []
    short = dec[dec['제약'].str.startswith('안전 기준 초과')]['설비 (시나리오)'].tolist() if len(dec) else []
    last_p = after_last_maintenance(pts_now, 'T_H').sort_values('T_H').groupby('SCENARIO_ID').tail(1)
    sensor = last_p.loc[last_p.get('OOR_N', pd.Series(0, index=last_p.index)) > 0, 'SCENARIO_ID'].tolist() if 'OOR_N' in last_p else []
    if urgent:
        st.error(', '.join(urgent) + ' — 경보가 켜진 지 오래됐습니다. 가장 먼저 교체하세요.', title='즉시 조치', icon=':material/emergency:')
    if short:
        st.warning(', '.join(short) + ' — 차례를 기다리면 고장 위험이 높아집니다. 정비 책임자에게 팀·공수 추가를 요청하세요.',
                   title='일손 부족', icon=':material/group_add:')
    if sensor:
        st.warning(', '.join(sensor) + ' — 입력 값이 평소 범위를 벗어났습니다. 센서와 설비를 직접 확인하세요.',
                   title='센서 확인', icon=':material/sensors:')

    now_jobs = dec[~dec['추천'].str.startswith('다음 정기')] if len(dec) else dec
    plan_jobs = dec[dec['추천'].str.startswith('다음 정기')] if len(dec) else dec
    cau = t.index[t['상태'] == CAU].tolist()
    k1, k2, k3, k4 = st.columns(4, border=True)
    k1.metric('교체할 설비', len(now_jobs))
    k2.metric('정기 교체 때 함께', len(plan_jobs))
    k3.metric('점검할 설비 (주의)', len(cau))
    k4.metric('정비 중', int((t['상태'] == '정비 중').sum()))

    st.subheader('작업 순서', icon=':material/format_list_numbered:')
    if dec.empty:
        st.success('지금 교체할 설비가 없습니다.', icon=':material/check_circle:')
    else:
        jobs = []
        for _, r in dec.sort_values(['순위'], na_position='last').iterrows():
            e = r['_e']
            plan = r['추천'].startswith('다음 정기')
            p_fail = r['완료 전 고장 확률']
            if plan:
                why = f'{e["wait_h"]:.0f}시간 뒤 정기 교체({e["part"]})가 있고, 그때까지 고장 위험 {risk_word(p_fail, cfg.safety_max)}'
            elif r['제약'].startswith('즉시'):
                why = '경보가 켜진 지 오래됨 · 지금 바로'
            else:
                why = (f'24시간 안 고장 위험 · 다음 정기 교체 {"없음" if pd.isna(e["wait_h"]) else f"{e["wait_h"]:.0f}시간 뒤라 기다릴 수 없음"}')
            jobs.append({'순서': '정기 교체 때' if plan else f'{int(r["순위"])}',
                         '설비': r['설비 (시나리오)'],
                         '할 일': '정기 교체 때 두 부품 함께 교체' if plan else '설비 정지 · 두 부품(밸브·진동장치) 교체',
                         '시작 (h)': now + r['_start'], '끝 (h)': now + r['_start'] + costs.replace_h,
                         '끝나기 전 고장 위험': risk_word(p_fail, cfg.safety_max), '이유': why})
        jt = pd.DataFrame(jobs)
        st.dataframe(jt.style.map(lambda v: {'높음': 'color: #d64545; font-weight: 600'}.get(v, ''), subset=['끝나기 전 고장 위험']),
                     hide_index=True, key='tbl_jobs', on_select=lambda: open_detail('tbl_jobs', jt['설비'].tolist()),
                     selection_mode='single-row',
                     column_config={'시작 (h)': st.column_config.NumberColumn(format='%d시간', help='시나리오 시작 후 시각'),
                                    '끝 (h)': st.column_config.NumberColumn(format='%d시간'),
                                    '이유': st.column_config.TextColumn(width='large')})
        st.caption(f'정비팀 {A["n_crews"]}팀 · 팀당 하루 {A["crew_day_h"]}공수(교체 {W["per_day"]}번)로 순서를 정했습니다. '
                   '줄을 누르면 그 설비의 신호 흐름을 볼 수 있습니다. 순서와 기준은 정비 책임자가 정합니다.')

    if cau:
        st.subheader('점검할 설비', icon=':material/search_check:')
        st.caption('주의: 열화가 시작됐지만 아직 24시간 안 고장 위험은 낮습니다')
        pl = W['pl']
        nxt = pl[pl['SCENARIO_ID'].isin(cau) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').groupby('SCENARIO_ID').head(1).set_index('SCENARIO_ID')
        ct = pd.DataFrame({'설비': cau})
        ct['다음 정기 교체'] = ct['설비'].map(lambda x: f'{nxt.loc[x, "PLAN_H"] - now:.0f}시간 뒤 '
                                        f'({PART_NAMES.get(nxt.loc[x, "PLAN_PART"], nxt.loc[x, "PLAN_PART"])})' if x in nxt.index else '없음')
        ct['할 일'] = np.where(ct['다음 정기 교체'] != '없음', '장비 점검 · 정기 교체 때 두 부품 함께 교체 준비',
                              '장비 점검 · 두 부품 재고 확인 (경보가 켜지면 교체)')
        st.dataframe(ct, hide_index=True, key='tbl_check', on_select=lambda: open_detail('tbl_check', ct['설비'].tolist()),
                     selection_mode='single-row')
    with st.expander('이 화면 보는 법', icon=':material/help:'):
        st.markdown('- **작업 순서**: 고장 위험이 큰 설비부터 정비팀에 배정했습니다. 정기 교체가 곧 있는 설비는 그때 함께 교체합니다.\n'
                    '- **끝나기 전 고장 위험**: 교체가 끝나기 전에 고장 날 가능성입니다(낮음 · 보통 · 높음).\n'
                    '- **점검할 설비**: 열화가 시작됐지만 아직 24시간 안 고장 위험은 낮습니다. 상태를 확인하고 부품을 준비합니다.\n'
                    '- 이 화면은 지나간 데이터를 재생합니다. 실제로는 조치하지 않은 경보가 쌓여 보일 수 있습니다.')

# 1. 전체 현황

elif page == '전체 현황':
    st.header('전체 현황', icon=ICONS['전체 현황'])
    st.caption(f'{split_label} · 지금 {now}시간 기준')
    t = status_table()
    n_alarm, n_caution = int((t['상태'] == WARN).sum()), int((t['상태'] == CAU).sum())
    c1, c2, c3, c4 = st.columns(4, border=True)
    c1.metric('설비 수', len(t))
    c2.metric('지금 경보', n_alarm, help='WARNING: 24시간 안 고장 위험')
    c3.metric('지금 주의', n_caution if CAUTION else '—', help='CAUTION: 열화 감지' if CAUTION else '주의 모델이 없는 모델 파일입니다')
    c4.metric('정비 중', int(t['상태'].isin(['정비 중', '정비 후 예측 대기']).sum()), help='정비 중이거나 정비 뒤 다음 예측을 기다리는 설비')
    with st.expander('이 화면 보는 법', icon=':material/help:'):
        st.markdown((f'- 두 확률은 {STEP}시간마다 계산합니다.\n- **주의 (CAUTION, 열화 감지)**: 지금 열화 중일 확률 ≥ {CTH:.2f}\n' if CAUTION else '')
                    + f'- **경보 (WARNING, 24시간 고장 위험)**: 24시간 안 고장 확률(위험 확률) ≥ {TH:.2f}\n'
                    '- **경보 유지**: 경보가 한 번 켜지면 확률이 내려가도 정비가 시작될 때까지 경보로 둡니다(켜졌다 꺼졌다 하지 않게).\n'
                    '- **가장 많이 벗어난 신호**: 마지막 1시간 기준. 표준화 점수 1이 정상 범위의 경계입니다.\n'
                    '- 표의 줄을 누르면 그 설비의 상세 화면이 열립니다.')

    if CAUTION:
        st.subheader('부품 소요 예측', icon=':material/inventory_2:')
        st.caption('센서로는 원인 부품을 구분할 수 없어, 경보가 나면 두 부품을 모두 교체합니다(팀 결정 C). '
                   '그래서 설비 한 대에 밸브 1개·진동장치 1개가 필요하다고 셉니다.')
        parts = pd.DataFrame({'부품': ['밸브', '진동장치'], '지금 교체 (경보 설비)': [n_alarm] * 2,
                              '1~2일 안에 필요할 수 있음 (주의 설비)': [n_caution] * 2})
        parts['합계'] = parts.iloc[:, 1:].sum(axis=1)
        st.dataframe(parts, hide_index=True, width='content')
        cv = CAUTION['validation']
        st.caption(f'재고가 이보다 적으면 지금 발주를 시작하세요. 주의 중 일부(검증 세트 {cv["주의 묶음"]}번 중 '
                   f'{cv["열화 아닌 설비의 주의"]}번)는 열화가 아니었던 헛걸음입니다.')

    st.subheader('설비별 상태', icon=':material/list_alt:')
    st.caption('위험한 순서 · 줄을 누르면 상세 화면')
    show = [c for c in ['설비 (시나리오)', '상태', '위험 확률', '열화 확률', '첫 주의 (h)', '첫 경보 (h)', '가장 많이 벗어난 신호',
                        '경보 유지', '다음 실제 고장까지 (h, 정답)'] if c in t.columns]
    tv = t[show].reset_index(drop=True)
    st.dataframe(
        style_stage(tv, '상태'), hide_index=True, height=520, key='tbl_status', selection_mode='single-row',
        on_select=lambda: open_detail('tbl_status', tv['설비 (시나리오)'].tolist()),
        column_config={
            '위험 확률': st.column_config.ProgressColumn('위험 확률 (24시간)', format='%.2f', min_value=0, max_value=1),
            '열화 확률': st.column_config.ProgressColumn('열화 확률 (지금)', format='%.2f', min_value=0, max_value=1),
            '마지막 예측 (h)': st.column_config.NumberColumn(format='%d'),
            '첫 주의 (h)': st.column_config.NumberColumn(format='%d', help='경보도 주의에 포함해 처음 켜진 시각'),
            '첫 경보 (h)': st.column_config.NumberColumn(format='%d'),
        })
    st.caption('줄을 누르면 그 설비의 상세 화면이 열립니다.')


# 2. 설비 상세

elif page == '설비 상세':
    t = status_table()
    if st.session_state.get('sid') not in t.index:
        st.session_state.pop('sid', None)
    sid = st.selectbox('설비 (지금 위험한 순서)', t.index.tolist(), key='sid',
                       format_func=lambda s: f'{s} · {t.loc[s, "상태"]} · 확률 {t.loc[s, "위험 확률"]:.2f}'
                       if pd.notna(t.loc[s, '위험 확률']) else f'{s} · {t.loc[s, "상태"]}')
    row = t.loc[sid]
    st.header(f'{sid}', icon=ICONS['설비 상세'])
    st.caption(f'{split_label} · 지금 {now}시간 기준')

    c1, c2, c3, c4 = st.columns(4, border=True)
    c1.metric('상태', row['상태'].split(' (')[0], help=row['상태'])   # 칸이 좁아 영문 이름은 도움말로
    c2.metric('24시간 안 고장 확률', '—' if pd.isna(row['위험 확률']) else f'{row["위험 확률"]:.2f}',
              help=f'마지막 예측: {row["마지막 예측 (h)"]:.0f}시간' if pd.notna(row['마지막 예측 (h)']) else None)
    if CAUTION:
        c3.metric('지금 열화 중일 확률', '—' if pd.isna(row['열화 확률']) else f'{row["열화 확률"]:.2f}')
        c4.metric('첫 주의 / 첫 경보', ' / '.join('—' if pd.isna(row[k]) else f'{row[k]:.0f}h' for k in ['첫 주의 (h)', '첫 경보 (h)']))
    else:
        c3.metric('첫 경보', '—' if pd.isna(row['첫 경보 (h)']) else f'{row["첫 경보 (h)"]:.0f}시간')

    if row['상태'] == WARN:
        st.error(
            '24시간 안에 고장 날 위험이 높습니다.\n'
            '- 교체에 4시간이 걸리므로, 늦어도 고장 예상 4시간 전에는 교체를 시작해야 합니다.\n'
            '- 센서로는 원인 부품(진동장치·밸브)을 구분할 수 없습니다. 두 부품을 모두 점검하세요.\n'
            '- 고장 원인과 상관없이 같은 신호가 오르므로, 아래 신호 흐름으로 열화가 계속되는지 확인하세요.',
            title='조치 안내: 교체', icon=':material/build:')
        if row['경보 유지']:
            st.caption('지금 확률은 경보 기준보다 낮지만, 이번 운전 구간에서 이미 경보가 켜져 정비 전까지 경보로 유지합니다.')
    elif row['상태'] == CAU:
        st.warning(
            '열화가 시작된 것으로 보입니다. 아직 24시간 안 고장 위험은 낮습니다.\n'
            '- 장비 상태를 확인하세요. 설비가 정상적으로 돌아가면 **다음 정비 때 이 부분까지 함께 정비**합니다.\n'
            '- 두 부품(진동장치·밸브)의 재고를 확인하고, 없으면 지금 발주를 시작하세요. 경보까지 보통 반나절 정도 남습니다.\n'
            f'- 주의 중 일부는 열화가 아닌 헛걸음입니다(검증 세트 {CAUTION["validation"]["주의 묶음"]}번 중 '
            f'{CAUTION["validation"]["열화 아닌 설비의 주의"]}번). 점검에서 이상이 없으면 그대로 운전합니다.',
            title='조치 안내: 점검', icon=':material/search_check:')
    elif row['상태'] in ('정비 중', '정비 후 예측 대기'):
        st.warning(f'정비가 시작돼 정비 전 예측은 더 이상 쓰지 않습니다. 재가동 뒤 다음 예측 시점({STEP}시간마다)에 다시 예측합니다.')
    else:
        st.success('지금은 열화 신호가 없고, 24시간 안 고장 위험이 낮습니다.' if CAUTION else '지금은 24시간 안 고장 위험이 낮습니다.')
    last_p = after_last_maintenance(pts_now[pts_now['SCENARIO_ID'] == sid], 'T_H').tail(1)
    if 'OOR_N' in last_p and len(last_p) and last_p['OOR_N'].iloc[0] > 0:
        st.warning(f'마지막 예측({last_p["T_H"].iloc[0]:.0f}시간)의 특징 {int(last_p["OOR_N"].iloc[0])}개가 '
                   f'학습 때 본 범위(0.1~99.9%) 밖입니다: {last_p["OOR_FEATS"].iloc[0]}. '
                   '아주 심한 열화이거나 센서 이상일 수 있어, 모델 확률만 믿지 말고 현장에서 센서와 설비를 확인하세요.',
                   title='입력 감시: 센서 확인', icon=':material/sensors:')

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
    st.subheader('위험 확률과 열화 확률' if CAUTION else '위험 확률', icon=':material/trending_up:')
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
    st.subheader('네 신호의 흐름', icon=':material/ssid_chart:')
    st.altair_chart(alt.layer(*mnt_layer(sid, None), sig, avg, one).properties(height=300), width='stretch')
    st.caption('검은 굵은 선: 네 신호 평균 점수 · 점선: 정상 범위 경계(1). 정비 후 재가동하면 처음 2시간은 기준을 잡는 시간이라 비어 있습니다.')

    # 지금 신호 상태와 모델 기여도
    left, right = st.columns(2)
    with left:
        st.subheader('지금 신호 상태', icon=':material/sensors:')
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
                st.write(f'최종 모델({bundle["final"]["model"]})은 특징별 기여도를 계산할 수 없는 종류입니다. '
                         '왼쪽의 신호 상태와 위의 신호 흐름으로 경보 이유를 확인하세요.')
            elif cp.empty:
                st.write(f'아직 예측이 없습니다. 정비가 시작되면 정비 전 예측은 쓰지 않고, 재가동 뒤 다음 예측 시점({STEP}시간마다)을 기다립니다.')
            else:
                cr = contrib.loc[cp.index[-1]].rename('기여도').reset_index().rename(columns={'index': '묶음'})
                st.altair_chart(alt.Chart(cr).mark_bar().encode(
                    y=alt.Y('묶음:N', title=None, sort='-x'), x=alt.X('기여도:Q', title=f'위험을 올린 정도 ({contrib_unit})'),
                    color=alt.condition('datum.기여도 > 0', alt.value(ALARM_RED), alt.value('#2f5d8a'))
                ).properties(height=220), width='stretch')
                st.caption(f'{cp["T_H"].iloc[-1]:.0f}시간 예측 기준. 양수는 위험을 올린 쪽, 음수는 내린 쪽입니다. '
                           + ('나무마다 결정 경로를 따라 확률이 바뀐 양을 그 분기에 쓰인 특징 묶음에 나눠 준 값으로, '
                              '기준값에 모두 더하면 위의 위험 확률이 됩니다. ' if contrib_unit.startswith('확률')
                              else '0은 학습 세트 평균 시점입니다. ')
                           + '네 신호와 결합 점수는 서로 비슷한 정보라 모델 안에서 서로 상쇄될 수 있습니다. '
                           '묶음 하나의 크기보다 왼쪽의 신호 상태를 먼저 보세요.')


# 3. 의사결정 (DSS)

elif page == '의사결정':
    D = bundle.get('dss')
    st.header('의사결정', icon=ICONS['의사결정'])
    st.caption(f'{split_label} · 지금 {now}시간 기준')
    if not D:
        st.warning('이 모델 파일에는 의사결정 정보가 없습니다. 터미널에서 `python analysis/05_predict_test.py`로 다시 만드세요.')
        st.stop()
    with st.expander('이 화면 보는 법', icon=':material/help:'):
        st.markdown('- 경보 설비마다 정비 선택지(지금 / N시간 뒤 / 다음 정기 교체 때)의 **기대 총비용 = 정비 비용 + 생산 차질 비용 + '
                    '고장 확률 × 고장 시 손실**을 비교해 추천합니다.\n'
                    '- 안전 기준과 실행 가능성(정비팀·부품 준비)은 비용보다 먼저 지킵니다.\n'
                    '- 판단은 검증 가능한 예측값과 계산·규칙으로만 하며, 최종 결정은 정비 책임자가 합니다.')

    A = assumptions()
    A['mode'] = st.segmented_control('비용 기준', COST_MODES, default=A['mode'], required=True,
                         help='데이터셋 점수: 과제의 비용 규칙(모든 팀 공통). 현장 가정: 소모품·중요 부품·공임·인력·생산 차질·고장 손실을 만 원으로 넣은 값(정건우 현장 경험)')
    FIELD = A['mode'] == COST_MODES[1]
    U = '만 원' if FIELD else '점'

    def won(x):
        return f'{x:,.0f}{U}' if not FIELD or abs(x) < 10000 else f'{x / 10000:,.2f}억 원'

    # ① 이 시스템이 없었다면 → 있으면 (검증 세트 시뮬레이션)
    sim = D.get('validation_field') if FIELD else D.get('validation')
    if FIELD and not sim:
        st.info('이 모델 파일에는 현장 가정 시뮬레이션이 없습니다. python analysis/05_predict_test.py 로 다시 만드세요.')
    if sim:
        st.subheader('이 시스템이 없었다면 → 있으면', icon=':material/compare_arrows:')
        st.caption('검증 세트 72대 시뮬레이션')
        r1, r2, r3 = sim['regular'], sim['now'], sim['dss']
        k1, k2, k3 = st.columns(3, border=True)
        k1.metric('① 시스템 없음: 정기 교체만', won(r1['비용']), f'고장 {r1["고장"]}건', delta_color='off')
        k2.metric('② 경보만: 켜지면 즉시 교체', won(r2['비용']), f'{won(r2["비용"] - r1["비용"])} · 고장 {r2["고장"]}건',
                  delta_color='inverse')
        k3.metric('③ 경보 + 의사결정 지원', won(r3['비용']), f'{won(r3["비용"] - r1["비용"])} · 고장 {r3["고장"]}건',
                  delta_color='inverse')
        st.caption(f'정비 관리자가 예측 없이 정기 교체만 하면 고장 {r1["고장"]}건이 그대로 일어납니다. 경보를 따라 바로 교체하면 고장이 '
                   f'{r2["고장"]}건으로 줄지만, 정기 교체 직전의 경보까지 따로 교체해 정비가 {r2["정비 횟수"]}번 듭니다. '
                   f'의사결정 지원은 그중 {r3["정기 교체 때"]}건을 정기 교체 때 함께 처리해 계획 밖 정지를 '
                   f'{r2["계획 밖 정지(h)"]:.0f} → {r3["계획 밖 정지(h)"]:.0f}시간으로 줄입니다. '
                   f'**시뮬레이션에 따른 예상 절감액: {won(r1["비용"] - r3["비용"])} ({(r1["비용"] - r3["비용"]) / r1["비용"]:.0%}, ① 대비)**. '
                   + ('현장 가정(만 원, 기본 가정값)으로 계산한 값으로 실제 운영 결과가 아닙니다. ' if FIELD else
                      '점수는 데이터셋 비용 규칙이며 실제 돈이 아닙니다. ')
                   + '정답을 알 수 없는 관측 끝부분의 경보는 빠져 있습니다(analysis/10_dss_policy.py).')

    # ② 가정값 (여기서 바꾸면 현장 작업 화면의 할 일도 같은 계획으로 바뀐다)
    with st.expander('가정값 (바꾸면 추천과 근거, 현장 할 일 목록이 다시 계산됩니다)', expanded=False, icon=':material/tune:'):
        st.markdown('**안전·실행 가능성 (비용보다 먼저 지키는 제약)**')
        a1, a2, a3, a4 = st.columns(4)
        A['safety'] = a1.slider('안전 기준: 정비 완료 전 고장 확률 상한', 0.0, 0.5, A['safety'], 0.01)
        A['prep'] = a2.slider('부품 준비 시간 (조달 + 교체, h)', 4, 36, A['prep'], 1,
                              help='데이터셋 비용 규칙은 교체 4시간. 부품을 조달해야 하면 더 길어집니다.')
        A['crew_wait'] = a3.slider('정비팀이 시작할 수 있을 때까지 (h)', 0, 24, A['crew_wait'], 1)
        A['n_crews'] = a4.number_input('동시에 교체할 수 있는 정비팀 수', 1, 10, A['n_crews'], 1,
                                       help='경보 설비가 이보다 많으면 위험한 순서대로 차례를 기다립니다.')
        A['crew_day_h'] = a4.number_input('정비팀 1팀의 하루 작업 한도 (공수, 시간)', 4, 24, A['crew_day_h'], 1,
                                          help='팀 결정(2026-10-15): 1팀당 하루 최대 8공수. 교체 1번이 4시간이라 팀당 하루 2대까지이고, 넘으면 다음 날로 넘어갑니다.')
        if FIELD:
            st.markdown('**비용 (만 원, 현장 가정)**')
            f1, f2, f3, f4 = st.columns(4)
            A['f_cons'] = f1.number_input('소모품 (교체 1번: 볼트·너트·O링·윤활유)', 0.0, 10000.0, A['f_cons'], 10.0)
            A['f_part'] = f2.number_input('중요 부품 (개당: 밸브·진동장치)', 0.0, 10000.0, A['f_part'], 10.0)
            A['f_labor_part'] = f3.number_input('공임 (파트당)', 0.0, 1000.0, A['f_labor_part'], 1.0)
            A['f_labor_ph'] = f4.number_input('정비 인력 (1명·1시간)', 0.0, 100.0, A['f_labor_ph'], 0.5)
            g1, g2, g3, g4 = st.columns(4)
            A['f_crew'] = g1.number_input('정비팀 인원 (명)', 1, 10, A['f_crew'], 1, help='현장: 보통 3~4인 1팀')
            A['f_down'] = g2.number_input('생산 차질 (정지 1시간)', 0.0, 100000.0, A['f_down'], 50.0)
            A['f_fail'] = g3.number_input('고장 1건 추가 손실 (불량·품질·안전 대응)', 0.0, 100000.0, A['f_fail'], 50.0)
        st.markdown('**비용 (점, 기본값 = 데이터셋 비용 규칙)**' if not FIELD else '**기타**')
        b1, b2, b3, b4 = st.columns(4)
        if not FIELD:
            A['down_h'] = b1.number_input('생산 차질 비용 (정지 1시간)', 0.0, 50.0, A['down_h'], 0.5)
            A['penalty'] = b2.number_input('고장 벌점 (안전·품질 영향)', 0.0, 200.0, A['penalty'], 5.0)
        A['p_other'] = b3.slider('함께 바꾸는 부품이 실제 원인일 확률', 0.0, 1.0, A['p_other'], 0.05,
                                 help='센서로 원인 부품을 구분할 수 없어 기본값은 0.5입니다.')
        A['max_wait'] = b4.slider('기다릴 수 있는 최대 시간 (h)', 0, 72, A['max_wait'], 1)
        st.markdown('**N시간 뒤 정비 선택지 (예: 생산 영향이 적은 시간대)**')
        d1, d2 = st.columns(2)
        A['delay_h'] = d1.slider('N시간 뒤 (0 = 쓰지 않음)', 0, 48, A['delay_h'], 1)
        if FIELD:
            A['f_delay_rate'] = d2.number_input('그 시간대 생산 차질 (정지 1시간, 만 원)', 0.0, 100000.0, A['f_delay_rate'], 50.0)
        else:
            A['delay_rate'] = d2.number_input('그 시간대 생산 차질 비용 (정지 1시간)', 0.0, 50.0, A['delay_rate'], 0.5)
        if st.button('기본값으로 되돌리기'):
            st.session_state.pop('assume', None)
            st.rerun()
    st.session_state['assume'] = dict(A)
    costs, cfg = make_cfg(A)
    now_cost = 2 * costs.part + costs.unnecessary + costs.replace_fixed + costs.replace_h * costs.downtime_h
    st.caption(f'지금 가정: 지금 두 부품 교체 **{won(now_cost)}** · 정기 교체 때 다른 부품 추가 **{won(costs.part)}** · '
               f'고장 시 손실 = 복구 {costs.repair:,.0f} + 복구 {costs.repair_h:.0f}시간 × (생산 차질 {costs.downtime_h:,.1f} + 인력 {costs.labor_h:,.1f}) '
               f'+ 고장 추가 손실 {costs.failure_penalty:,.0f} = **{won(costs.failure_loss)}** · 안전 기준 {cfg.safety_max:.2f} · '
               f'부품 준비 {cfg.prep_h:.0f}시간 · 정비팀 {A["n_crews"]}팀(팀당 하루 {A["crew_day_h"]}공수). 고장 확률은 학습 세트에서 경보 뒤 고장까지 걸린 시간 분포로 계산합니다.')

    t = status_table()
    W = work_plan(A, t)
    dec, need_crew, per_day, evaluate_options, cfg, costs, curve, pl = (W[k] for k in ['dec', 'need_crew', 'per_day', 'evaluate_options', 'cfg', 'costs', 'curve', 'pl'])
    n_crews, crew_day_h = A['n_crews'], A['crew_day_h']

    c1, c2, c3, c4 = st.columns(4, border=True)
    c1.metric('경보 설비', len(dec))
    c2.metric('지금 정비 (정비팀 배정)', len(need_crew))
    c3.metric('정기 교체 때 함께', int(dec['추천'].str.startswith('다음 정기').sum()) if len(dec) else 0)
    c4.metric('일정 제약 충족', f'{int(dec["제약"].str.startswith("충족").sum())}/{len(dec)}' if len(dec) else '—',
              help='정비팀 차례를 기다려도 정비가 끝나기 전 고장 확률이 안전 기준 이하인 설비 수')
    if len(dec) and dec['제약'].str.startswith('즉시').any():
        st.error(', '.join(dec.loc[dec['제약'].str.startswith('즉시'), '설비 (시나리오)'])
                 + ' — 지금 교체해도 완료 전 고장 확률이 안전 기준을 넘습니다. 가장 먼저 정비하세요.',
                 title='즉시 조치', icon=':material/emergency:')
        st.caption('지나간 데이터를 재생하는 화면이라, 실제로는 조치하지 않은 경보가 쌓여 보일 수 있습니다.')

    st.subheader('정비 결정 목록', icon=':material/event_note:')
    st.caption('우선순위와 정비팀 일정')
    if dec.empty:
        st.success('지금 경보가 켜진 설비가 없습니다.', icon=':material/check_circle:')
    else:
        st.dataframe(dec.drop(columns=['_e', '_slot', '_start']), hide_index=True, column_config={
            '순위': st.column_config.NumberColumn(format='%d', help='지금 정비할 설비의 정비팀 배정 순서 (위험한 순서)'),
            '추천': st.column_config.TextColumn(width='medium'),
            '정비 완료까지 (h)': st.column_config.NumberColumn(format='%.0f'),
            '기대 총비용': st.column_config.NumberColumn(f'기대 총비용 ({U})', format='localized'),
            '완료 전 고장 확률': st.column_config.NumberColumn(format='%.2f'),
            '첫 경보 (h)': st.column_config.NumberColumn(format='%d')})
        st.caption(f'정비팀 {n_crews}팀이 한 번에 {n_crews}대씩 교체({costs.replace_h:.0f}시간)하고, 팀당 하루 {crew_day_h:.0f}공수(교체 {per_day}번)까지 합니다. '
                   f'하루에 {n_crews * per_day}대를 넘는 설비는 다음 날로 넘어갑니다. 차례를 기다리는 설비는 그만큼 늦게 끝나는 것으로 다시 계산합니다.')

        # ④ 설비별 근거: 선택지 비교와 산정 근거
        sid = st.selectbox('설비를 골라 근거 보기', dec['설비 (시나리오)'].tolist())
        e = dec.set_index('설비 (시나리오)').loc[sid, '_e']
        slot = dec.set_index('설비 (시나리오)').loc[sid, '_slot']
        override = st.checkbox('고장 확률을 직접 입력 (현장 판단으로 덮어쓰기)', help='점검 결과 등 현장 판단이 있을 때 모델 확률 대신 씁니다.')
        if override:
            cols_o = st.columns(len(e['opts']))
            keys = {'지금 두 부품 교체': '지금 두 부품 교체', '다음 정기 교체 때 두 부품 함께': '정기 교체 때'}
            po = {}
            for col, o in zip(cols_o, e['opts']):
                key = keys.get(o['선택지'], 'N시간 뒤 교체')
                po[key] = col.number_input(f'{o["선택지"]}: 완료 전 고장 확률', 0.0, 1.0, float(round(o['고장 확률'], 2)), 0.01, key=f'po_{sid}_{key}')
            e = evaluate_options(sid, crew_extra=slot, p_override=po)
        b = e['best']
        msg = f'**{sid} · 추천: {b["선택지"]}**' + (f' ({e["urgent"]})' if e['urgent'] else '')
        (st.info if b['선택지'].startswith('다음 정기') else st.warning)(
            msg + f'\n\n기대 총비용 {won(b["기대 총비용"])} = (1 − 고장 확률 {b["고장 확률"]:.2f}) × (정비 {b["정비 비용"]:,.0f} + 생산 차질 {b["생산 차질 비용"]:,.0f})'
            f' + {b["고장 확률"]:.2f} × 고장 시 손실 {b["고장 시 손실"]:,.0f} ({U})')
        left, right = st.columns([1.3, 1])
        with left:
            ot = pd.DataFrame(e['opts'])
            ot['추천'] = np.where(ot['선택지'] == b['선택지'], '✔', '')
            st.dataframe(ot[['추천', '선택지', '정비 완료까지(h)', '정비 비용', '생산 차질 비용', '고장 확률', '고장 시 손실', '기대 총비용', '안전 기준 충족']],
                         hide_index=True, column_config={
                             '정비 완료까지(h)': st.column_config.NumberColumn(format='%.0f'),
                             '정비 비용': st.column_config.NumberColumn(format='localized'),
                             '생산 차질 비용': st.column_config.NumberColumn(format='localized'),
                             '고장 확률': st.column_config.NumberColumn('완료 전 고장 확률', format='%.2f'),
                             '고장 시 손실': st.column_config.NumberColumn(format='localized'),
                             '기대 총비용': st.column_config.NumberColumn(format='localized')})
            st.markdown('**산정 근거**\n' + '\n'.join(f'- {o["선택지"]}: {o["근거"]}' for o in e['opts'])
                        + f'\n- 고장 확률: 학습 세트에서 경보가 처음 켜진 뒤 실제 고장까지 걸린 시간 분포(카플란-마이어). '
                          f'이 설비는 경보 뒤 {e["elapsed"]:.0f}시간 동안 고장이 없었다는 조건으로 계산'
                        + ('\n- 고장 확률은 현장 판단으로 덮어쓴 값' if override else ''))
            wait = next((o for o in e['opts'] if o['선택지'].startswith('다음 정기')), None)
            if wait is not None:
                c_wait = wait['정비 비용'] + wait['생산 차질 비용']
                even = (e['opts'][0]['기대 총비용'] - c_wait) / (costs.failure_loss - c_wait)
                st.caption((f'추천이 바뀌는 경계: 안전 기준을 {wait["고장 확률"]:.2f}보다 낮추면 "지금 교체". ' if wait['고장 확률'] >= 0.005 else
                            '정기 교체 전까지 고장 확률이 거의 0이라 안전 기준과 상관없이 기다릴 수 있습니다. ')
                           + f'비용만 보면 정기 교체 전 고장 확률이 {even:.2f}를 넘을 때 지금 교체가 더 쌉니다. '
                           '부품 준비 시간이 정기 교체보다 길어지면 "지금 교체"로 바뀝니다.')
            elif pd.isna(e['wait_h']):
                st.caption('다음 정기 교체 계획이 없어 지금(또는 N시간 뒤) 교체만 가능합니다.')
        with right:
            cv = pd.DataFrame({'h': np.arange(0, 49)})
            cv['p'] = [fail_prob(curve, e['elapsed'], h) for h in cv['h']]
            marks = [alt.Chart(cv).mark_line(color='#2f5d8a').encode(
                x=alt.X('h:Q', title='지금부터 시간 (h)'), y=alt.Y('p:Q', title='그 안에 고장 날 확률', scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip('h', title='시간(h)'), alt.Tooltip('p', title='고장 확률', format='.2f')]),
                alt.Chart(pd.DataFrame({'y': [cfg.safety_max]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')]
            pts_o = pd.DataFrame([{'h': o['정비 완료까지(h)'], 'p': fail_prob(curve, e['elapsed'], o['정비 완료까지(h)']), '선택지': o['선택지']}
                                  for o in e['opts']])
            marks.append(alt.Chart(pts_o).mark_point(size=90, filled=True, color='#4a9b5e').encode(
                x='h:Q', y='p:Q', tooltip=['선택지', alt.Tooltip('h', title='정비 완료(h)'), alt.Tooltip('p', title='고장 확률', format='.2f')]))
            st.altair_chart(alt.layer(*marks).properties(height=240), width='stretch')
            st.caption('파란 선: 지금부터 h시간 안에 고장 날 확률 · 초록 점: 선택지별 정비 완료 시점 · 빨간 점선: 안전 기준')

    # ⑤ 주의 설비
    cau = t.index[t['상태'] == CAU].tolist()
    if cau:
        st.subheader('주의 설비 준비', icon=':material/search_check:')
        nxt = pl[pl['SCENARIO_ID'].isin(cau) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').groupby('SCENARIO_ID').head(1)
        cd = pd.DataFrame({'설비 (시나리오)': cau})
        cd['다음 정기 교체'] = cd['설비 (시나리오)'].map(
            nxt.set_index('SCENARIO_ID').apply(lambda x: f'{x["PLAN_H"] - now:.0f}시간 뒤 ({PART_NAMES.get(x["PLAN_PART"], x["PLAN_PART"])})', axis=1)
            if len(nxt) else {}).fillna('계획 없음')
        cd['할 일'] = np.where(cd['다음 정기 교체'] != '계획 없음', '장비 점검, 정기 교체 때 두 부품 함께 교체 준비',
                              f'장비 점검, 두 부품 재고 확인 (조달 {cfg.prep_h:.0f}시간 가정. 경보가 켜지면 지금 교체 대상)')
        st.dataframe(cd, hide_index=True)

    # ⑥ 정책 시뮬레이션 상세
    if sim:
        with st.expander('정책 시뮬레이션 상세 (검증 세트, 기본 가정값)'):
            sv = pd.DataFrame(list(sim.values())).set_index('정책')
            sv['예상 절감액 (① 대비)'] = sim['regular']['비용'] - sv['비용']
            st.dataframe(sv[['비용', '예상 절감액 (① 대비)', '고장', '고장 손실', '정비 횟수', '지금 교체', '정기 교체 때',
                             '기다리다 고장', '고장 전 조치 완료율', '계획 밖 정지(h)', '미확정 경보']],
                         column_config={'비용': st.column_config.NumberColumn(f'기대 총비용 ({U})', format='localized'),
                                        '예상 절감액 (① 대비)': st.column_config.NumberColumn(format='localized'),
                                        '고장 손실': st.column_config.NumberColumn(format='localized'),
                                        '고장 전 조치 완료율': st.column_config.NumberColumn(format='%.2f')})
            st.caption('같은 데이터·같은 모델 예측으로 세 정책을 돌려 실제 고장 시각으로 채점했습니다. '
                       '준비 시간·안전 기준을 바꾼 민감도는 analysis/10_dss_policy.py에 있습니다.')


# 4. 모델 정보

elif page == '모델 정보':
    v = bundle['validation']
    st.header('모델 정보와 검증 세트 성능', icon=ICONS['모델 정보'])
    st.caption(f'모델 저장: {bundle["created_at"]}' + (' · 클라우드: 미리 계산한 결과만 사용 (원자료·모델 파일 없음)' if CLOUD else ''))
    st.markdown(
        f'- **모델**: {bundle["final"]["model"]} · 특징 {len(bundle["features"])}개 ({bundle["final"]["features"]}) · {STEP}시간마다 예측\n'
        f'- **학습**: 학습 세트 216개 시나리오의 예측 시점 {bundle["n_train_points"]:,}개\n'
        f'- **경보 기준**: 24시간 안 고장 확률 ≥ {TH:.2f}. 학습 세트 교차검증에서 '
        '점진 고장을 모두 잡고 진짜 오경보가 0인 값 중 가장 낮은 값(오경보 없이 가장 일찍 경보)\n'
        '- **검증 세트**: 72개 시나리오, 학습에 쓰지 않고 평가에만 사용(모델·기준 선택은 학습 세트 교차검증. 수정 이력은 결과보고서 2절)')
    c1, c2, c3, c4 = st.columns(4, border=True)
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
        st.subheader('2단계 경보: 주의 단계', icon=':material/notifications:')
        st.markdown(
            f'- **주의 모델**: {CAUTION["final"]["model"]} · {CAUTION["final"]["features"]} 특징 {len(CAUTION["features"])}개. '
            '"지금 열화 중인가"를 맞힙니다(정답: 진단 정답의 열화 구간, 학습 때만 사용)\n'
            f'- **주의 기준**: 지금 열화 중일 확률 ≥ {CTH:.2f}. 학습 세트 교차검증에서 주의 중 {CAUTION["min_precision"]:.0%} 이상이 '
            '실제 열화인 값 중 가장 낮은 값\n'
            '- **헛걸음을 조금 허용하는 이유**: 현장에서는 주의를 받으면 장비 상태를 확인하고, 설비가 돌면 다음 정비 때 함께 정비합니다. '
            '설비를 세우거나 바로 발주하지 않으므로 헛걸음 비용은 점검 한 번입니다.')
        d1, d2, d3, d4 = st.columns(4, border=True)
        d1.metric('주의 정밀도', f'{cv["주의 정밀도"]:.2f}', help=f'주의 중 실제로 열화 중이던 비율. 목표 {CAUTION["min_precision"]}')
        d2.metric('주의가 켜진 점진 고장', f'{cv["주의로 감지"]}/{cv["점진 고장"]}')
        d3.metric('경보보다 먼저 (평균)', f'{cv["추가 준비 시간 평균(h)"]:.1f}시간')
        d4.metric('첫 주의 (중앙값)', f'고장 {cv["주의 리드타임 중앙값(h)"]:.0f}시간 전')
        st.caption(f'검증 세트 주의 {cv["주의 묶음"]}번 중 헛걸음 {cv["열화 아닌 설비의 주의"]}번. 대부분은 열화가 시작되기 몇 시간 전이었고, '
                   '일부는 끝까지 정상인 설비였습니다. 자세한 근거는 reports/연주_조기경보_결과보고서.md 7-5절')

    st.subheader('검증 세트 고장별 결과', icon=':material/fact_check:')
    ev = v['events'].rename(columns={'SCENARIO_ID': '시나리오', 'TYPE': '고장 종류'})
    st.dataframe(ev[['시나리오', '고장 종류', '감지', '리드타임(h)']].sort_values(['고장 종류', '리드타임(h)'], ascending=[True, False]),
                 hide_index=True, column_config={'리드타임(h)': st.column_config.NumberColumn(format='%.1f')})
    st.caption('리드타임: 고장 4시간 전까지 난 경보 중 가장 이른 경보부터 고장까지의 시간. '
               '자세한 근거는 reports/연주_조기경보_결과보고서.md 4절과 부록 B')


# 5. 비용 계산기

else:
    v = bundle['validation']['event']
    n = dict(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
             n_false=v['고장과 무관한 경보 묶음: 오경보'])
    st.header('비용 계산기', icon=ICONS['비용 계산기'])
    st.caption('원인 부품을 얼마나 맞힐 수 있으면 이득인가')
    st.markdown(
        f'검증 세트 결과(고장 {v["고장"]}건 중 {n["n_det"]}건 감지, 고장과 무관한 교체 {n["n_needed"]}번, '
        f'진짜 오경보 {n["n_false"]}번)에 데이터셋 비용 규칙을 적용합니다.\n'
        '- 경보가 나면 바로 교체합니다(정책: 경보 즉시 교체). 한 부품만 바꾸는데 틀리면, 불필요 정비가 되고 고장도 못 막습니다.\n'
        '- 정기 교체 직전에 울린 맞는 경보도 교체 한 번으로 셉니다. 예정 교체와 묶어 처리하면 이 비용은 줄어듭니다(의사결정 지원, 다음 단계).\n'
        '- 센서로는 원인 부품을 구분할 수 없어(EDA E9), 부품을 맞힐 확률은 현장 점검 능력에 달려 있습니다.')
    p = st.slider('원인 부품을 맞힐 확률', 0.5, 1.0, 0.5, step=0.01,
                  help='0.5 = 둘 중 하나를 찍음, 1.0 = 항상 맞힘')
    cc = cost_curve(**n)
    one = float(np.interp(p, cc['p'], cc['한 부품만 교체']))
    both, none = cc['두 부품 모두 교체'].iloc[0], cc['아무것도 안 함'].iloc[0]
    c1, c2, c3 = st.columns(3, border=True)
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
