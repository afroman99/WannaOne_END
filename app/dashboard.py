"""CAST-PdM: 정비 의사결정 지원 솔루션 (시연용). 팀원 설계 화면(원래 이름 CAST GUARD AI)과 합침 2026-10-15, 이름 변경 2026-10-16.

    python analysis/05_predict_test.py      # 처음 한 번: 모델 학습·저장 (models/risk24.joblib)
    streamlit run app/dashboard.py          # 브라우저에서 http://localhost:8501

- 저장된 모델로 예측만 한다(다시 학습하지 않는다). 원자료가 없으면 app_data/의 미리 계산한 결과만 읽는다(클라우드).
- 오른쪽 위 시연 설정에서 데이터 세트와 '지금 시각'을 고른다. 그 시각까지의 데이터만으로 본 상태를 보여 준다(재생 방식).
- 2단계 경보: 주의 = 열화 감지 → 점검·부품 준비, 경보 = 24시간 안 고장 위험 → 교체

화면 (화면마다 맨 위에 '지금 할 일' 하나와 주 버튼 하나: 콜투액션)
  작업자      내 작업    내 팀의 지금 할 일 → 작업 시작 → 완료 보고(결과·수행 내용·어느 부품이었나). 즉시 조치 설비는 배정 전에도 시작
                         아래 탭: 지금 · 지연 · 예정 · 완료·이력 (재생 시각 기준). 비용은 보여 주지 않는다
              설비 상세  위험 확률·열화 확률과 네 신호의 흐름, 정비 기록, 경보 이유, 근거 따라가기(온톨로지)
  설비관리자  홈 두 범주 (2026-10-21): 설비 상태(경보·주의·정상 칸, 검색) / 작업 지시 현황(배정할 경보 칸·카드, 작업 목록, 작업 수정하기)
              설비 상태 → (경보 칸) → 경보 설비 목록(작업 지시 상태) → 설비 화면
              작업 지시 현황 → (배정할 경보 칸) → 배정할 시나리오 목록 → 고르면 그 아래 근거(온톨로지)·비용(만 원) → 배정·보류
              작업 수정·배정 취소는 '작업 수정하기'를 눌러야 열리고, 사유를 적어야 저장된다
              하위 페이지(노션처럼 눌러 들어감): 설비 상세 · 정책·비용 · 부품 손익 · 모델 정보·한계

기록: 작업 지시는 src/tasks.py, 판단·진행 기록(승인·보류·발주·시작·완료·점검 결과)은 src/actions.py.
  로컬은 actions/ 아래 파일(커밋하지 않음), 공개 클라우드는 보는 사람의 세션에만 남긴다.
  구글 로그인(src/auth.py)과 Secrets [db]가 있으면 공용 DB(Supabase, src/db.py)에 남겨 노트북·폰이 같은 작업을 본다 (2차 2026-10-16)
  여러 사람이 같이 보는 저장소(공용 DB·로컬 SQLite)에서는 재생 시각도 함께 쓰고, 10초마다 바뀐 것이 있으면 화면을 다시 그린다.
온톨로지 (src/ontology.py, OWL/RDF): 지금 시각의 설비·예측·신호·경보·정비 결정·조치를 한 그래프로 잇는다.
"""

import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.posko_data import load_scenarios
from src.dss import PARTS, PartsPolicy, fail_prob, field_costs, missing_parts, options, plans, recommend
from src import actions, app_data, ontology, tasks
from src.limitations import LIMITS
from src.auth import require_login
from src.serving import cost_curve, explain, failure_hours, load_bundle, score_scenarios

st.set_page_config(page_title='CAST-PdM', page_icon=':material/shield:', layout='wide',
                   initial_sidebar_state='collapsed')
# 폰 홈 화면에 앱처럼 설치 (PWA식, 새 프론트엔드 없이): manifest·아이콘을 문서 머리에 한 번 붙인다
#   파일은 app/static/ (config.toml enableStaticServing). Streamlit Cloud는 앱을 한 겹 감싼 페이지로 보여 주므로 열 수 있으면 바깥 문서에도 붙인다
#   폰에서: iPhone Safari 공유 → 홈 화면에 추가 / Android Chrome ⋮ → 홈 화면에 추가
st.html("""<script>
(() => {
  const add = (doc) => {
    if (!doc || doc.querySelector('link[rel="manifest"]')) return;
    const base = location.origin + '/app/static/';
    const tags = [['link', {rel: 'manifest', href: base + 'manifest.json'}],
                  ['link', {rel: 'apple-touch-icon', href: base + 'apple-touch-icon.png'}],
                  ['meta', {name: 'apple-mobile-web-app-capable', content: 'yes'}],
                  ['meta', {name: 'mobile-web-app-capable', content: 'yes'}],
                  ['meta', {name: 'apple-mobile-web-app-title', content: 'CAST-PdM'}],
                  ['meta', {name: 'theme-color', content: '#080F1C'}]];
    for (const [tag, attrs] of tags) {
      const el = doc.createElement(tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      doc.head.appendChild(el);
    }
  };
  add(document);
  try { if (window.top !== window) add(window.top.document); } catch (e) {}   // 다른 출처면 건너뛴다
})();
</script>""", unsafe_allow_javascript=True, width='content')

# 실행 방식
#   로컬: data/posko/ 원자료와 models/risk24.joblib로 직접 계산한다 (내 컴퓨터 안에서만 열림, 비밀번호 없음)
#   클라우드: 원자료가 없고 app_data/만 있다. 미리 계산한 결과만 읽고(모델 파일을 불러오지 않음), 비밀번호를 요구한다
CLOUD = not (ROOT / 'data' / 'posko').exists() and app_data.available()
USER = require_login(st, CLOUD)     # 구글 로그인한 사람 {'email', 'name', 'role', 'team'} (로그인을 설정하지 않았으면 None)

SIGNAL_NAMES = {'Z_MLD-AMP': '몰드 레벨 진폭', 'Z_CONTROL_DELAY': '제어 지연',
                'Z_WDL': '인발 부하', 'Z_OSC-STK': '진동 스트로크'}
SPLITS = {'테스트 (정답 없음)': 'test', '검증 (정답 있음)': 'validation'}   # 화면 위 선택지
# 라이트·다크 테마 모두에서 구분되는 색: 경보 빨강 · 주의 노랑 · 정상 회색 · 정비 띠 · 선(파랑·초록) · 기준선(밝은 글자색)
ALARM_RED, CAUTION_YELLOW, OK_GRAY, MNT_GRAY = '#FF667A', '#F7B955', '#7D8FA9', '#3A4D66'
LINE_BLUE, LINE_GREEN, INK = '#60A5FA', '#2DD4A0', '#8392AB'   # INK: 라이트·다크 모두에서 보이는 회청색 (평균선·기준선)
# 화면에 보여 줄 단계 이름: 정상(NORMAL) → 주의(CAUTION, 열화 감지) → 경보(WARNING, 24시간 고장 위험)
# 모델 출력(STAGE)은 '경보'/'주의'/'정상' 그대로 두고 화면에서만 이름을 바꾼다
LABEL = {'경보': '경보 (WARNING)', '주의': '주의 (CAUTION)', '정상': '정상 (NORMAL)'}
WARN, CAU, NORM = LABEL['경보'], LABEL['주의'], LABEL['정상']
STAGE_COLORS = alt.Scale(domain=[WARN, CAU, NORM], range=[ALARM_RED, CAUTION_YELLOW, OK_GRAY])


def show_chart(chart, **kw):
    """차트 글자도 화면 글자처럼 크게 (축·범례·툴팁, 2026-10-21 고참 관리자도 읽기 쉽게)"""
    st.altair_chart(chart.configure_axis(labelFontSize=15, titleFontSize=15)
                    .configure_legend(labelFontSize=15, titleFontSize=15), **kw)


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


# 저장소: 작업 지시(src/tasks.py)와 판단·진행 기록(src/actions.py)
#   공용 DB    구글 로그인 + Secrets [db] url (Supabase Postgres, src/db.py). 노트북·폰이 같은 작업을 본다. 그릴 때마다 새로 읽는다
#   로컬       actions/ 아래 SQLite·CSV (커밋하지 않음). 발표장에서 노트북을 서버로 쓸 때도 이 경로 (같은 Wi-Fi·핫스팟의 폰이 접속)
#   공개 화면  로그인 없는 클라우드는 서버 파일·DB에 쓰지 않고 보는 사람의 세션에만 (누구나 쓰고 남의 기록을 보는 것을 막는다)
def db_url():
    try:
        return st.secrets['db']['url'] if 'db' in st.secrets else None
    except Exception:
        return None


@st.cache_resource(show_spinner='공용 DB에 연결하는 중...')
def get_db(url):
    from src import db
    return db.Postgres(url)


@st.cache_resource
def get_local_tasks():
    return tasks.TaskStore(ROOT / 'actions' / 'tasks.sqlite3')


@st.cache_resource
def get_local_log():
    return actions.ActionLog(ROOT / 'actions' / 'action_log.csv')


PG = None
if USER and db_url():               # 공용 DB는 로그인한 사람만 쓴다
    try:
        PG = get_db(db_url())
    except Exception as e:          # 접속 주소(비밀번호 포함)가 화면·로그에 나오지 않게 종류만 남긴다
        print(f'[dashboard] 공용 DB 연결 실패: {type(e).__name__}', file=sys.stderr)
        st.error('공용 DB에 연결하지 못했습니다. 잠시 뒤 새로 고침하세요. 계속되면 관리자에게 문의하세요.')
        st.stop()
if PG:
    TASKS, LOG = tasks.TaskStore(db=PG), actions.ActionLog(db=PG)
elif CLOUD:
    if 'task_store' not in st.session_state:
        st.session_state['task_store'] = tasks.TaskStore(None)
        st.session_state['action_log'] = actions.ActionLog(None)
    TASKS, LOG = st.session_state['task_store'], st.session_state['action_log']
else:
    TASKS, LOG = get_local_tasks(), get_local_log()
STORE_NOTE = ('기록은 공용 DB에 남아 설비관리자·작업자가 같이 봅니다.' if PG else
              '기록은 이 브라우저 세션에만 남습니다(공개 화면).' if CLOUD else '기록은 이 컴퓨터의 actions/tasks.sqlite3에 남습니다.')
st.session_state['_fresh'] = True   # 자동 갱신(맨 아래 watch_changes)에 '방금 전체를 다시 그렸다'고 알린다


def clock(h):
    """재생 시각(시나리오 시작 후 h)을 사람 말로: 25 → '2일차 01:00' (2026-10-19). 합성 데이터라 실제 날짜 대신 일차로 쓴다"""
    if h is None or pd.isna(h):
        return '—'
    m = int(round(float(h) * 60))
    d, m = divmod(m, 24 * 60)
    return f'{d + 1}일차 {m // 60:02d}:{m % 60:02d}'


def hour_pick(label, lo, hi, default, key):
    """보고 시각 고르기: [일차] + [시] 두 칸 (폼 안에서도 쓰도록 범위 검사는 pick_ok로 따로). 반환: 고른 h (정수)"""
    c1, c2 = st.columns(2)
    d = c1.selectbox(f'{label} · 일차', list(range(int(lo) // 24 + 1, int(hi) // 24 + 2)), index=int(default) // 24 - int(lo) // 24,
                     format_func=lambda x: f'{x}일차', key=f'{key}_{int(hi)}_d')   # 지금 시각이 바뀌면 기본값도 새로
    hh = c2.selectbox('시', list(range(24)), index=int(default) % 24, format_func=lambda x: f'{x:02d}:00', key=f'{key}_{int(hi)}_h')
    return (d - 1) * 24 + hh


def pick_ok(h, lo, hi):
    """고른 보고 시각이 범위(작업 시작 ~ 지금) 안인가. 아니면 화면에 알리고 False"""
    if lo <= h <= hi:
        return True
    st.error(f'보고 시각은 {clock(lo)} ~ {clock(hi)} 사이로 고르세요. (고른 시각 {clock(h)})', icon=':material/schedule:')
    return False


# 화면 위: CAST-PdM 머리말 · 시연 설정 · 보는 사람 (팀원 설계 화면과 합침 2026-10-15)
#   작업자      현장: 내 팀의 지금 할 일 하나(시작 → 완료 보고), 날짜 대신 재생 시각 기준 지금·지연·예정·완료. 비용 없음
#   설비관리자  의사결정: 홈(설비 상태 / 작업 지시 현황) → 목록 → 설비의 근거·비용·배정 → 하위 페이지
WORKER, MANAGER = '현장 작업', '정비 책임자'              # 기록(src/actions.py)에 쓰는 역할 이름
ROLE_LABEL = {WORKER: '작업자', MANAGER: '설비관리자'}
MANAGER_TABS = {'설비 상세': '설비 상세', '정책·비용': '의사결정', '부품 손익': '비용 계산기', '모델 정보·한계': '모델 정보'}
# 설비관리자 홈 범주와 화면 단계 (2026-10-21): 설비 상태 → (경보 칸) → 경보 설비 목록 → 설비 화면
#   / 작업 지시 현황 → (배정할 경보 칸) → 배정할 시나리오 목록 → 고르면 그 아래 근거·비용 → 배정·보류
EQUIP, ORDERS, STAFF = '설비 상태', '작업 지시 현황', '인원 현황'
MANAGER_VIEWS = ('경보', '주의', '정상', 'assign')
ICONS = {WORKER: ':material/engineering:', MANAGER: ':material/manage_accounts:',
         '내 작업': ':material/checklist:', '설비 상세': ':material/monitoring:', '정책·비용': ':material/balance:',
         '부품 손익': ':material/calculate:', '모델 정보·한계': ':material/model_training:'}

st.session_state.setdefault('role', WORKER)       # 기본값은 세션 상태로만 정한다 (버튼으로 화면을 바꾸기 때문)
st.session_state.setdefault('split_label', list(SPLITS)[0])
st.session_state.setdefault('now', 72)
st.session_state.setdefault('clock_at', None)
# 재생 시각 함께 쓰기: 같이 보는 저장소면 설비관리자가 정한 데이터 세트·시각을 모든 화면이 따른다
#   (따로 두면 관리자가 25시간에 배정한 작업을 작업자가 72시간에 시작해 관리자 화면에서는 '대기'로 남는다)
CAN_SET_CLOCK = USER is None or USER['role'] == 'manager'
if TASKS.shared:
    _split, _now = TASKS.setting('split'), TASKS.setting('now')
    _label = {v: k for k, v in SPLITS.items()}.get(_split)
    if _label:
        st.session_state['split_label'] = _label
    if _now and _now.isdigit() and 0 <= int(_now) <= 72:
        st.session_state['now'] = int(_now)
    _at = TASKS.setting('clock_at')
    try:
        st.session_state['clock_at'] = float(_at) if _at else None
    except ValueError:
        st.session_state['clock_at'] = None
# 시각이 흐른다 (2026-10-21 8차): '적용'을 누른 순간부터 실제 시간과 같은 속도로. 시연 끝(4일차 00:00 = 72h)에서 멈춘다.
#   now     = 판단 시각(정각): 예측 · 경보 · 비용은 1시간 요약본이라 정각마다 바뀐다
#   now_f   = 흐르는 시각(소수 h): 작업 · 진행 기록을 남기고 읽는 시각, 화면 위 시계와 고장 타이머
#   적용한 시연 시각(now)과 적용한 실제 시각(clock_at, 초)만 저장하고 흐르는 시각은 매번 계산한다 → 노트북 · 폰이 같은 값을 본다
#   clock_at이 없으면(목업 스크립트로 맞춘 직후 등) 흐르지 않고 그 시각에 멈춰 있다
END_H = 72


def flow_hour():
    at = st.session_state.get('clock_at')
    h0 = float(st.session_state['now'])
    return h0 if not at else min(float(END_H), h0 + max(0.0, time.time() - at) / 3600)


# 적용한 값(split_label·now)과 고르는 칸(split_pick·now_pick)을 나눈다 (2026-10-21 버그 수정)
#   전에는 슬라이더가 적용한 값을 바로 들고 있어서, 어느 화면의 슬라이더가 처음 값(1일차 00:00)을 보고하면
#   그 값이 공용 저장소에 저장돼 모든 화면이 처음으로 돌아갔다. 이제는 '적용' 버튼을 눌러야만 저장한다.
#   고르는 칸은 적용한 값이 바뀌었을 때만 따라간다 (적용 전에 옮기던 값은 그대로 둔다)
_applied = (st.session_state['split_label'], int(flow_hour()))     # 고르는 칸은 지금 흐르는 시각의 정각을 따라간다
if st.session_state.get('_clock_seen') != _applied:
    st.session_state['_clock_seen'] = _applied
    st.session_state['split_pick'], st.session_state['now_pick'] = _applied


def apply_clock():
    """시연 설정 '적용' (폼 제출 콜백): 고른 값을 적용하고, 같이 보는 저장소면 모두에게 저장한다"""
    if not CAN_SET_CLOCK:
        return
    st.session_state['split_label'] = st.session_state['split_pick']
    st.session_state['now'] = int(st.session_state['now_pick'])
    st.session_state['clock_at'] = time.time()      # 이 순간부터 흐른다
    if TASKS.shared:
        TASKS.set_setting('split', SPLITS[st.session_state['split_label']])
        TASKS.set_setting('clock_at', repr(st.session_state['clock_at']))
        TASKS.set_setting('now', str(st.session_state['now']))


split = SPLITS[st.session_state['split_label']]
now_f = flow_hour()
now = int(now_f)


def clock_s(h):
    """흐르는 시각을 초까지: 26.2 → '2일차 02:12:00'"""
    sec = int(float(h) * 3600 + 1e-6)
    d, sec = divmod(sec, 86400)
    return f'{d + 1}일차 {sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


# 화면에서 1초씩 움직이는 시계 · 고장 타이머 (브라우저가 같은 계산을 한다: 적용한 시연 시각 + 적용 뒤 흐른 실제 시간).
#   서버는 처음 값만 그리고, 다시 그리기는 정각이 바뀔 때만 한다. 숫자는 data- 속성으로만 넘긴다(사용자 글 없음).
#   st.html은 이 자리에서 스크립트를 빼므로, 보이지 않는 작은 칸(st.iframe, 같은 출처)에서 1초마다 화면의 칸을 고친다
TICK_FRAME = ('<script>var D=window.parent.document,p=function(n){n=Math.floor(n);return(n<10?"0":"")+n};'
              'setInterval(function(){var t=Date.now()/1000;'
              'D.querySelectorAll("[data-cast-h0]").forEach(function(e){'
              'var at=+e.dataset.castAt,h=+e.dataset.castH0;if(at>0)h=Math.min(+e.dataset.castEnd,h+Math.max(0,t-at)/3600);'
              'if(e.dataset.castDl!==undefined){var s=Math.max(0,Math.floor((+e.dataset.castDl-h)*3600));'
              'e.textContent=p(s/3600)+":"+p(s%3600/60)+":"+p(s%60);}'
              'else{var s=Math.floor(h*3600+1e-6),d=Math.floor(s/86400);s-=d*86400;'
              'e.textContent=(d+1)+"일차 "+p(s/3600)+":"+p(s%3600/60)+":"+p(s%60);}});},1000);</script>')


def tick_attrs(deadline=None):
    at = st.session_state.get('clock_at') or 0
    return (f'data-cast-h0="{float(st.session_state["now"])!r}" data-cast-at="{float(at)!r}" data-cast-end="{END_H}"'
            + ('' if deadline is None else f' data-cast-dl="{float(deadline)!r}"'))
with st.container(horizontal=True, vertical_alignment='center', gap='medium'):
    st.markdown('### :material/shield: CAST-:blue[PdM]')
    st.caption('정비 의사결정 지원 솔루션 · 시연')
    st.space('stretch')
    # 시연 설정: 실제 운영에는 없는 재생용 값이라 한 곳에 접어 둔다 (지금 값은 버튼 이름에 보인다)
    #   key를 줘서 이름(지금 시각)이 바뀌어도 같은 칸으로 남게 한다
    st.html(f'<span style="font-variant-numeric: tabular-nums; font-weight: 600;"><span {tick_attrs()}>{clock_s(now_f)}</span></span>'
            + ('' if st.session_state.get('clock_at') else ' <span style="opacity:.6">(멈춤 · 적용하면 흐름)</span>')
            , width='content')
    with st.popover(f'{st.session_state["split_label"].split(" ")[0]} · 시연 설정', key='clock_pop',
                    icon=':material/tune:', help='시연 설정: 데이터 세트와 지금 시각'):
        with st.form('clock_form', border=False):
            st.selectbox('데이터 세트', list(SPLITS), key='split_pick', disabled=not CAN_SET_CLOCK,
                         help='테스트: 정답이 없는 실제 운영과 같은 상황 · 검증: 정답이 있어 실제 고장과 비교할 수 있음')
            st.select_slider('지금 시각', list(range(73)), key='now_pick', disabled=not CAN_SET_CLOCK, format_func=clock,
                             help='이 시각까지의 데이터만으로 본 상태를 보여 줍니다. 왼쪽으로 옮기면 과거로 돌아가 재생할 수 있습니다. '
                                  '1일차 00:00 = 시나리오 시작 (합성 데이터라 실제 날짜 대신 일차로 표시)')
            st.form_submit_button('적용', icon=':material/check:', type='primary', on_click=apply_clock, disabled=not CAN_SET_CLOCK,
                                  width='stretch')
        show_truth = st.toggle('정답 보기 (검증 세트)', key='truth', disabled=split != 'validation',
                               help='앞으로 일어날 실제 고장 시각(정답)을 겹쳐 봅니다. 모델은 이 정보를 보지 않습니다.') and split == 'validation'
        st.caption(f'지금: **{st.session_state["split_label"]} · {clock_s(now_f)}** · 옮긴 뒤 "적용"을 눌러야 바뀝니다. '
                   '적용한 순간부터 실제 시간과 같은 속도로 흐르고(4일차 00:00에서 멈춤), 예측 · 경보 · 비용은 1시간 요약본이라 정각마다 바뀝니다  \n'
                   '독립된 합성 시나리오를 재생합니다 · 실시간 설비 연결 아님'
                   + (' · 데이터 세트와 시각은 모든 화면이 함께 씁니다' + ('' if CAN_SET_CLOCK else ' (설비관리자가 바꿈)')
                      if TASKS.shared else ''))
    if USER:
        with st.popover(USER['name'].split('@')[0], icon=':material/account_circle:'):
            st.caption(USER['email'])
            st.markdown('설비관리자' if USER['role'] == 'manager' else f'작업자 · {USER["team"]}')
            st.button('로그아웃', icon=':material/logout:', on_click=st.logout)
if st.session_state.get('clock_at') and now < END_H:
    st.iframe(TICK_FRAME, height=1)     # 시계가 흐를 때만 (멈춰 있으면 고칠 것이 없다)
# 주소(URL)와 화면 위치를 맞춘다 → 브라우저·마우스의 뒤로 가기/앞으로 가기가 된다
#   ?view=m (설비관리자) &page=설비 상세|정책·비용|... (관리자 하위 페이지) &sid=S0294. 작업자는 내 작업 하나라 주소가 비어 있다
#   Streamlit은 st.query_params를 바꾸면 기록을 하나 쌓고(pushState), 뒤로 가기로 주소가 바뀌면 그 주소로 다시 실행한다.
#   그래서 주소가 '우리가 마지막으로 쓴 주소'와 다를 때만(뒤로·앞으로 가기, 주소 직접 입력) 주소 → 화면으로 옮기고,
#   나머지 경우는 맨 아래 sync_url()이 화면 → 주소로 쓴다. 기본 화면(작업자·내 작업)은 주소에 아무것도 넣지 않는다.
def url_state():
    return {k: v for k, v in st.query_params.to_dict().items() if k in ('view', 'page', 'mv', 'sid')}


def apply_url():
    q = url_state()
    if q == st.session_state.get('_nav_last', {}):
        return
    st.session_state['_nav_last'] = q
    role_ = MANAGER if q.get('view') == 'm' else WORKER
    page_ = q.get('page')
    sid_ = q.get('sid') if re.fullmatch(r'S\d{4}', q.get('sid', '')) else None
    before = (st.session_state.get('role'), st.session_state.get('msub'), st.session_state.get('mv'))
    if role_ == MANAGER and page_ not in MANAGER_TABS and q.get('mv') in MANAGER_VIEWS:
        st.session_state['mv'] = q['mv']
    else:
        st.session_state.pop('mv', None)
    st.session_state['role'] = role_
    if role_ == MANAGER:
        if page_ in MANAGER_TABS:
            st.session_state['msub'] = page_
        else:
            st.session_state.pop('msub', None)
    else:
        st.session_state.pop('msub', None)
    if sid_:
        st.session_state['sid'] = sid_
        if role_ == MANAGER and page_ not in MANAGER_TABS:
            st.session_state['decision_sid'] = sid_
            st.session_state['nav_pick'] = True
    elif role_ == MANAGER and page_ not in MANAGER_TABS:
        st.session_state.pop('nav_pick', None)
        st.session_state.pop('decision_sid', None)
    if before != (st.session_state.get('role'), st.session_state.get('msub'), st.session_state.get('mv')):
        st.session_state['scroll_top'] = True


def sync_url():
    """지금 화면 위치를 주소에 쓴다 (바뀌었을 때만 → 브라우저 기록 한 칸)"""
    q = {}
    if st.session_state.get('role') == MANAGER:
        q['view'] = 'm'
        if st.session_state.get('msub') in MANAGER_TABS:
            q['page'] = st.session_state['msub']
            if st.session_state['msub'] == '설비 상세' and st.session_state.get('sid'):
                q['sid'] = st.session_state['sid']
        else:
            if st.session_state.get('mv') in MANAGER_VIEWS:
                q['mv'] = st.session_state['mv']
            if st.session_state.get('nav_pick') and st.session_state.get('decision_sid'):
                q['sid'] = st.session_state['decision_sid']
    if q != url_state():
        st.query_params.from_dict(q)
    st.session_state['_nav_last'] = q


def pick_equipment(sid):
    """설비를 고른다 (설비관리자 목록·맨 위 카드): 비용 비교·배정·설비 페이지가 이 설비로, 주소에도 남긴다"""
    st.session_state.update({'decision_sid': sid, 'sid': sid, 'nav_pick': True, 'scroll_top': True})


def leave_equipment():
    """설비 화면에서 한 단계 위로 (상태별 설비 목록에서 왔으면 그 목록, 아니면 설비관리자 홈)"""
    st.session_state.pop('nav_pick', None)
    st.session_state['scroll_top'] = True


def open_view(mv):
    """설비관리자 화면 단계로 (경보·주의·정상 = 상태별 설비 목록, assign = 배정할 시나리오 목록)"""
    st.session_state['mv'] = mv
    st.session_state.pop('nav_pick', None)
    st.session_state['scroll_top'] = True


def go_home(tab):
    """설비관리자 홈의 한 범주로 (설비 상태 / 작업 지시 현황)"""
    st.session_state.pop('mv', None)
    st.session_state.pop('nav_pick', None)
    st.session_state['mhome'] = tab
    st.session_state['scroll_top'] = True


def open_assign(sid):
    """배정할 시나리오 목록에서 설비를 골라 그 아래에 근거·비용을 연다"""
    st.session_state['mv'] = 'assign'
    pick_equipment(sid)


def open_task_edit(tid=None):
    """'작업 수정하기': 작업 지시 현황으로 가서 작업 수정 칸을 연다 (tid가 있으면 그 작업을 골라 둔다)"""
    go_home(ORDERS)
    st.session_state['task_edit'] = True
    if tid is not None:
        st.session_state['edit_tid'] = tid


apply_url()


def close_sub():
    """설비관리자 하위 페이지에서 나오기 (역할을 바꿀 때도)"""
    if st.session_state.pop('msub', None):
        st.session_state['scroll_top'] = True


# 구글 로그인한 작업자는 자기 팀의 내 작업만 (주소에 ?view=m을 넣어도 설비관리자 화면으로 가지 않는다)
LOCKED_WORKER = USER is not None and USER['role'] == 'worker'
if LOCKED_WORKER:
    if USER['team'] not in tasks.TEAMS:
        st.error('등록된 팀 이름이 맞지 않습니다. 설비관리자에게 문의하세요.')
        st.stop()
    st.session_state['role'] = WORKER
    st.session_state.pop('msub', None)
    st.session_state['my_team'] = USER['team']
    role = WORKER
elif USER is not None and USER['role'] == 'manager':
    # 구글 로그인한 설비관리자는 설비관리자 화면만 (2026-10-21 8차): 작업자 화면의 체크·시작·완료는 실제 작업자만 하게.
    #   기록의 '현장 작업'이 실제 작업자가 한 일만 남는다. 로그인 없는 화면(로컬)은 두 역할을 그대로 고를 수 있다
    st.session_state['role'] = MANAGER
    role = MANAGER
else:
    role = st.segmented_control('보는 사람', [MANAGER, WORKER], key='role', required=True, label_visibility='collapsed',
                                on_change=close_sub, format_func=lambda r: f'{ICONS[r]} {ROLE_LABEL[r]}')

pts, hr, contrib, contrib_unit, mnt = get_data(split)


# 설비 이름 (2026-10-21): 'S0294' 같은 시나리오 번호는 현장 사람에게 뜻이 없어 가상 설비 이름을 크게, 번호는 작게 보여 준다
#   원자료의 설비 ID는 모든 시나리오가 같은 연주기 하나(P1-CC1-INS01)라, 시나리오 하나를 연주기 한 대로 가정해 이름을 붙인다.
#   테스트는 1연주, 검증은 2연주로 나눠 두 세트의 이름이 겹치지 않게 한다.
#   실제 설비 이름(예: 회사 설비 마스터 코드)을 쓰려면 app/equipment_names.csv (열: SCENARIO_ID, NAME)를 두면 그 이름을 쓴다
LINE_NO = {'test': 1, 'validation': 2}


@st.cache_data
def split_ids(split_):
    """데이터 세트의 시나리오 번호 (가상 설비 이름을 붙이는 순서)"""
    if CLOUD:
        return tuple(sorted(get_app_data()[1][split_][0]['SCENARIO_ID'].unique()))
    return tuple(sorted(load_scenarios().query('SPLIT == @split_')['SCENARIO_ID']))


@st.cache_data
def equipment_names(split_):
    names = {s: f'{LINE_NO[split_]}연주 #{i:02d}' for i, s in enumerate(split_ids(split_), 1)}
    f = ROOT / 'app' / 'equipment_names.csv'
    if f.exists():
        m = pd.read_csv(f, dtype=str).dropna()
        names.update({s: n.strip()[:30] for s, n in zip(m['SCENARIO_ID'], m['NAME']) if s in names and n.strip()})
    return names


NAMES = equipment_names(split)


def nm(sid):
    """화면에 크게 보일 설비 이름"""
    return NAMES.get(sid, sid)


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
    종료=lambda d: d['END_H'].map(lambda x: clock(x) if x <= now else '진행 중 (종료 미확정)'))   # 미래의 종료 시각은 보여 주지 않는다
last_mnt_start = mnt_now.groupby('SCENARIO_ID')['START_H'].max()     # 지금까지 마지막으로 시작된 정비


MNT_START = last_mnt_start.to_dict()


def part_breakeven():
    """원인 부품을 맞힐 확률이 이 값 이상이면 한 부품만 교체하는 쪽이 싸다 (비용 계산기와 같은 계산, 같은 만 원 가정값)"""
    v = bundle['validation']['event']
    cc = cost_curve(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
                    n_false=v['고장과 무관한 경보 묶음: 오경보'], costs=make_cfg(assumptions())[0])
    even = cc.loc[cc['한 부품만 교체'] <= cc['두 부품 모두 교체'].iloc[0], 'p']
    return float(even.iloc[0]) if len(even) else None


def feedback():
    """점검 되먹임: 점검으로 원인 부품을 맞힌 비율이 손익분기를 넘으면 '점검 후 한 부품'"""
    n, acc = LOG.inspection_accuracy()
    be = part_breakeven()
    return {'n': n, 'accuracy': acc, 'breakeven': be, 'ready': be is not None and actions.one_part_ready(n, acc, be)}


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



# 의사결정 계산 (설비관리자 화면과 작업자 '내 작업'이 같이 쓴다)

def assumptions():
    """설비관리자가 정한 가정값 (없으면 기본값 = 모델 파일의 DSS 가정값 + 팀 결정: 정비팀 2팀, 팀당 하루 8공수)"""
    D = bundle.get('dss')
    base, bc = D['config'], D['config'].costs
    bp = getattr(base, 'parts', PartsPolicy())
    default = {'safety': float(base.safety_max), 'prep': int(base.prep_h), 'crew_wait': int(base.crew_wait_h),
               'n_crews': 2, 'crew_day_h': 8,   # 정비는 상주 근무(주 5일, 하루 8시간)라 팀당 하루 8공수 (팀 결정 2026-10-15, 2026-10-16 확인)
               'down_h': float(bc.downtime_h), 'penalty': float(bc.failure_penalty),
               'p_other': float(base.p_other_needed), 'max_wait': int(base.max_wait_h), 'delay_h': 0,
               'delay_rate': float(bc.downtime_h),
               # 비용 기준: 서비스 화면은 현장 가정(만 원, 정건우 현장 경험 2026-10-15) 하나로 통일 (2026-10-15 사용자 요청).
               #   데이터셋 점수(과제 공통 비용 규칙)는 모델 평가용이라 결과보고서·발표에만 쓴다
               'mode': COST_MODES[1], 'f_cons': 100.0, 'f_part': 200.0, 'f_labor_part': 20.0, 'f_labor_ph': 3.0,
               'f_crew': 4, 'f_down': 500.0, 'f_fail': 300.0, 'f_delay_rate': 500.0,
               # 부품 조달 (데이터에 없어 미리 정한 가정값, 정건우 현장 경험 2026-10-15)
               'stock_v': int(bp.stock_valve), 'stock_o': int(bp.stock_oscillator), 'lead': float(bp.lead_h),
               'rush_h': float(bp.rush_h), 'rush_mk': float(bp.rush_markup)}
    return {**default, **st.session_state.get('assume', {}), 'mode': COST_MODES[1]}


COST_MODES = ['데이터셋 점수', '현장 가정 (만 원)']


def make_cfg(A):
    base = bundle['dss']['config']
    if A['mode'] == COST_MODES[1]:
        costs = field_costs(consumables=A['f_cons'], major_part=A['f_part'], labor_per_part=A['f_labor_part'],
                            labor_per_person_h=A['f_labor_ph'], crew_size=A['f_crew'], downtime_h=A['f_down'],
                            failure_extra=A['f_fail'])
    else:
        costs = replace(base.costs, downtime_h=A['down_h'], failure_penalty=A['penalty'])
    parts = PartsPolicy(stock_valve=A['stock_v'], stock_oscillator=A['stock_o'], lead_h=A['lead'], rush_h=A['rush_h'],
                        rush_markup=A['rush_mk'])
    cfg = replace(base, safety_max=A['safety'], prep_h=A['prep'], crew_wait_h=A['crew_wait'], p_other_needed=A['p_other'],
                  max_wait_h=A['max_wait'], costs=costs, parts=parts)
    return costs, cfg


# 인·시 작업량 (2026-10-21): 지금 비용 가정(4명이 교체 4시간, 고장 복구 8시간)을 작업량으로 본다.
#   n명이 맡으면 걸리는 시간 = 작업량 ÷ n. 인건비 합계는 인원과 상관없이 같고, 정지 시간·작업 중 고장 확률이 바뀐다
WORK_REPLACE, WORK_REPAIR = 16.0, 32.0     # 교체 · 고장 복구 (인·시)
SHIFT = (9, 17)                            # 정규 근무 9시~17시 (하루 8시간, 상주 근무)
OVERTIME = 1.5                             # 정규 근무 밖(야간·휴일 특근) 인건비 배율 (팀 결정 2026-10-21)


def overtime_share(start_h, dur):
    """start_h부터 dur시간 동안 정규 근무 밖인 비율 (재생 시각의 시 = 하루 안 시각)"""
    if dur <= 0:
        return 0.0
    hours = np.arange(start_h, start_h + dur, 0.25) % 24
    return float(np.mean((hours < SHIFT[0]) | (hours >= SHIFT[1])))


def crew_costs(base, n, start_h, labor_ph):
    """n명이 맡을 때의 비용 가정: 교체 16인·시 ÷ n시간, 고장 복구 32인·시 ÷ n시간, 정규 근무 밖은 인건비 1.5배"""
    n = max(1, int(n))
    rep_h, fix_h = WORK_REPLACE / n, WORK_REPAIR / n
    mult = 1 + (OVERTIME - 1) * overtime_share(start_h, rep_h)
    return replace(base, replace_h=rep_h, repair_h=fix_h, labor_h=labor_ph * n * mult)


def stock_now(cfg):
    """설비마다 지금 남은 예비 재고 {sid: {'valve': n, 'oscillator': n}}.
    처음 재고 − 고장 복구에 쓴 부품 − 작업 완료(두 부품 교체)에 쓴 부품 + 도착한 발주. 정기 교체 부품은 계획과 함께 들어온다고 본다"""
    out = {sid: cfg.parts.stock() for sid in pts['SCENARIO_ID'].unique()}
    for r in mnt_now[mnt_now['종류'] == '고장 후 복구'].itertuples():
        if r.PART_CD in PARTS:
            out[r.SCENARIO_ID][r.PART_CD] = max(0, out[r.SCENARIO_ID][r.PART_CD] - 1)
    lg = LOG.frame(split, now_f)
    # 작업 완료마다 실제 조치(기록의 '선택지')대로 부품을 쓴다: 두 부품 2개 · 한 부품 1개 · 교체 안 함 0개 (2026-10-21).
    #   예전 기록·설비관리자가 바꾼 완료는 선택지가 비어 있어 두 부품으로 본다.
    #   설비관리자가 완료를 되돌린 기록('… 완료 → …')은 두 부품만큼 돌려놓는다
    used = {}
    for _, r in lg[lg['종류'] == '작업 완료'].iterrows():
        for q in tasks.ACTION_PARTS.get(str(r['선택지']), PARTS):
            used[(r['설비'], q)] = used.get((r['설비'], q), 0) + 1
    for sid_, n_ in lg[lg['메모'].astype(str).str.contains('완료 → ', regex=False)].groupby('설비').size().items():
        for q in PARTS:
            used[(sid_, q)] = used.get((sid_, q), 0) - int(n_)
    for (sid_, q), n_ in used.items():
        if sid_ in out:
            out[sid_][q] = max(0, out[sid_][q] - max(0, n_))
    for _, r in lg[lg['종류'] == '부품 발주'].iterrows():
        arrive = r['시각(h)'] + (cfg.parts.rush_h if r['선택지'] == '긴급' else cfg.parts.lead_h)
        if arrive <= now and r['부품'] in PARTS:
            out[r['설비']][r['부품']] += 1
    return out


def in_transit(cfg):
    """아직 도착하지 않은 발주 {sid: [(부품, 도착 시각, 긴급 여부)]}"""
    lg = LOG.frame(split, now_f)
    out = {}
    for _, r in lg[lg['종류'] == '부품 발주'].iterrows():
        arrive = r['시각(h)'] + (cfg.parts.rush_h if r['선택지'] == '긴급' else cfg.parts.lead_h)
        if arrive > now:
            out.setdefault(r['설비'], []).append((r['부품'], arrive, r['선택지'] == '긴급'))
    return out


def procurement(cfg, stock, ids):
    """주의 설비의 부품 준비: 두 부품 교체에 모자라는 부품과 발주 권고 (발주 시점 = 주의 단계에서 재고 부족)"""
    trans = in_transit(cfg)
    rows = []
    for sid in ids:
        st_ = stock[sid]
        coming = {q for q, _, _ in trans.get(sid, [])}
        short = [q for q in missing_parts(st_, PARTS) if q not in coming]
        rows.append({'설비': sid, '재고': stock_text(st_),
                     '발주': ('발주 필요: ' + ', '.join(PART_NAMES[q] for q in short) + f' (정상 {cfg.parts.lead_h:.0f}시간)') if short else
                            ('도착 예정: ' + ', '.join(f'{PART_NAMES[q]} {clock(a)}' for q, a, _ in trans.get(sid, []))) if coming else '준비됨',
                     '_short': short})
    return pd.DataFrame(rows, columns=['설비', '재고', '발주', '_short'])


def stock_text(st_):
    return f'밸브 {st_["valve"]} · 진동장치 {st_["oscillator"]}'


def work_plan(A, t):
    """경보 설비마다 추천 선택지와 정비팀 일정 (우선순위, 하루 작업 한도 포함)"""
    costs, cfg = make_cfg(A)
    STOCK = stock_now(cfg)
    crew_day_h = A['crew_day_h']           # 한 사람의 하루 근무 시간 (정비팀 수는 명단(tasks.TEAMS)에서)
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
        plan_part = nxt['PLAN_PART'].iloc[0] if len(nxt) else None
        elapsed = max(0.0, now - t_a) if pd.notna(t_a) else 0.0
        c = replace(cfg, crew_wait_h=cfg.crew_wait_h + crew_extra)
        opts = options(wait_h, curve, c, elapsed, delay_h=dly, delay_downtime_h=delay_rate, p_override=p_override,
                       stock=STOCK.get(sid), plan_part=plan_part)
        best, urgent = recommend(opts)
        return {'t_a': t_a, 'wait_h': wait_h, 'part': part, 'plan_part': plan_part, 'elapsed': elapsed, 'opts': opts, 'best': best,
                'urgent': urgent}

    alarm_ids = t.index[t['상태'] == WARN].tolist()
    first = {sid: evaluate_options(sid) for sid in alarm_ids}
    # 우선순위: 지금 정비가 필요한 설비를 위험한 순서로(한 차례 미루면 고장 확률이 큰 순서) 정비팀에 배정한다
    need_crew = sorted([s for s in alarm_ids if not first[s]['best']['선택지'].startswith('다음 정기')],
                       key=lambda s: -fail_prob(curve, first[s]['elapsed'], first[s]['opts'][0]['정비 완료까지(h)'] + costs.replace_h))
    # 팀 하루 작업 한도 (2026-10-21, 인·시): 출근한 팀원 × 하루 8시간 ÷ 교체 1번 16인·시 = 팀당 하루 교체 수.
    #   전원 출근이면 5명 × 8 = 40인·시 → 2번 (예전 '팀당 하루 2대'와 같다). 부재가 있으면 줄어든다.
    #   순서·대기 계산은 작업 하나를 4명 · 4시간으로 본다 (팀 결정 A2: 고른 인원은 그 작업의 비용·시간에만)
    absent_now = TASKS.absent_at(now_f)
    team_jobs = {k: int(sum(m not in absent_now for m in v) * crew_day_h // WORK_REPLACE) for k, v in tasks.TEAMS.items()}
    per_day = max(1, max(team_jobs.values()))
    slots = sorted(d * 24.0 + j * costs.replace_h for d in range(30) for n_ in team_jobs.values() for j in range(n_))

    def crew_delay(k):
        """k번째(0부터) 정비 작업이 시작될 때까지 기다리는 시간: 팀마다 하루 한도만큼, 넘으면 다음 날 (모든 팀이 비면 아주 뒤)"""
        return slots[k] if k < len(slots) else 24.0 * 30

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
    if len(dec):
        state = LOG.job_state(split, now_f, MNT_START)
        dec.insert(3, '방법', '점검 후 한 부품' if feedback()['ready'] else '두 부품')
        dec.insert(1, '조치', dec['설비 (시나리오)'].map(lambda x: state.get(x, '대기')))
    if len(dec):
        dec.insert(4, '재고', dec['설비 (시나리오)'].map(lambda x: stock_text(STOCK[x])))
    return {'stock': STOCK, 'dec': dec, 'need_crew': need_crew, 'per_day': per_day, 'evaluate_options': evaluate_options,
            'cfg': cfg, 'costs': costs, 'curve': curve, 'pl': pl}


def risk_word(p, safety):
    """고장 확률을 현장 말로 (비용 없이 위험만)"""
    if pd.isna(p):
        return '—'
    return '높음' if p > safety else ('보통' if p >= 0.05 else '낮음')


def ontology_graph(A, t, with_decisions=True):
    """지금 시각의 상태를 온톨로지 그래프로 (src/ontology.py). 화면이 이미 계산한 값만 쓴다"""
    status = pd.DataFrame({
        'STAGE': t['상태'].str.split(' (', regex=False).str[0], 'PROB': t['위험 확률'],
        'DEGRADED_PROB': t['열화 확률'] if CAUTION else np.nan, 'PRED_H': t['마지막 예측 (h)'],
        'FIRST_ALARM_H': t['첫 경보 (h)'], 'FIRST_CAUTION_H': t['첫 주의 (h)'] if CAUTION else np.nan}, index=t.index)
    cur = after_last_maintenance(hr_now.assign(HOUR_START=hr_now['DONE_H'] - 1), 'HOUR_START')
    signals = cur.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
    decisions, unit, safety = [], None, None
    if with_decisions and bundle.get('dss'):
        W = work_plan(A, t)
        unit, safety = W['costs'].unit, W['cfg'].safety_max
        decisions = [{'sid': r['설비 (시나리오)'], 'rank': r['순위'], 'constraint': r['제약'], 'best': r['추천'],
                      'options': r['_e']['opts']} for _, r in W['dec'].iterrows()]
    return ontology.build(now, status, signals, mnt_now, get_plans(split), decisions, cost_unit=unit, safety_max=safety,
                          th=TH, cth=CTH, actions=LOG.frame(split, now_f), feedback=feedback())


def guide_content(row):
    """상태별 조치 안내: (색, 제목, 아이콘, 본문). 근거·비용 화면의 조치 안내 칸이 쓴다 (2026-10-21)"""
    if row['상태'] == WARN:
        return ('red', '조치 안내: 교체', ':material/build:',
                '24시간 안에 고장 날 위험이 높습니다.\n'
                '- 교체에 4시간이 걸리므로, 늦어도 고장 예상 4시간 전에는 교체를 시작해야 합니다.\n'
                '- 센서로는 원인 부품(진동장치·밸브)을 구분할 수 없습니다. 두 부품을 모두 점검하세요.\n'
                '- 고장 원인과 상관없이 같은 신호가 오르므로, 아래 분석 근거의 신호 흐름으로 열화가 계속되는지 확인하세요.')
    if row['상태'] == CAU:
        return ('orange', '조치 안내: 점검', ':material/search_check:',
                '열화가 시작된 것으로 보입니다. 아직 24시간 안 고장 위험은 낮습니다.\n'
                '- 장비 상태를 확인하세요. 설비가 정상적으로 돌아가면 **다음 정비 때 이 부분까지 함께 정비**합니다.\n'
                '- 두 부품(진동장치·밸브)의 재고를 확인하고, 없으면 지금 발주를 시작하세요. 경보까지 보통 반나절 정도 남습니다.\n'
                f'- 주의 중 일부는 열화가 아닌 헛걸음입니다(검증 세트 {CAUTION["validation"]["주의 묶음"]}번 중 '
                f'{CAUTION["validation"]["열화 아닌 설비의 주의"]}번). 점검에서 이상이 없으면 그대로 운전합니다.')
    if row['상태'] in ('정비 중', '정비 후 예측 대기'):
        return ('orange', '조치 안내: 정비 중', ':material/build:',
                f'정비가 시작돼 정비 전 예측은 더 이상 쓰지 않습니다. 재가동 뒤 다음 예측 시점({STEP}시간마다)에 다시 예측합니다.')
    return ('green', '조치 안내: 운전 계속', ':material/check_circle:',
            '지금은 열화 신호가 없고, 24시간 안 고장 위험이 낮습니다.' if CAUTION else '지금은 24시간 안 고장 위험이 낮습니다.')


def remaining_median(curve, elapsed, cap=240.0):
    """경보 뒤 elapsed시간 동안 고장이 없었을 때 고장 확률이 절반이 되는 남은 시간(h) = 고장 예상까지(중앙값).
    학습 세트의 '경보 뒤 고장까지 걸린 시간' 분포(비용 계산의 P와 같은 곡선)로 계산한다. cap 안에서 절반이 안 되면 None"""
    for h in np.arange(0.25, cap + 0.25, 0.25):
        if fail_prob(curve, elapsed, h) >= 0.5:
            return float(h)
    return None


def hms(h):
    """시간(h) → hh:mm:ss"""
    sec = int(round(h * 3600))
    return f'{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def ontology_dot(sid, srow, e=None, costs=None):
    """온톨로지에 넣은 연결을 그림으로 (설비 하나): 설비 → 단계 → 예측 → 신호, 단계 → 정비 결정 → 선택지 → 부품 (graphviz DOT)"""
    def q(x):
        return '"' + str(x).replace('"', "'") + '"'
    stage = srow['상태'].split(' (')[0]
    fill = {'경보': '#FFD6DC', '주의': '#FCE9C6', '정상': '#DDF3EA'}.get(stage, '#E6EAF0')
    out = ['digraph G {', 'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;',
           'node [shape=box, style="rounded,filled", fillcolor="#EEF2F8", color="#9AAFCB", fontname="Noto Sans KR", fontsize=11];',
           'edge [fontname="Noto Sans KR", fontsize=10, color="#8392AB", fontcolor="#8392AB"];',
           f'E [label={q(nm(sid) + chr(10) + sid)}, fillcolor="#D9E6FB"];',
           f'S [label={q("단계" + chr(10) + stage)}, fillcolor="{fill}"];', 'E -> S [label="상태"];']
    if pd.notna(srow['위험 확률']):
        pred_h, prob = srow['마지막 예측 (h)'], srow['위험 확률']
        out += [f'P [label={q(f"{pred_h:.0f}h 예측" + chr(10) + f"24시간 내 고장 위험 {prob:.0%}")}];',
                'E -> P [label="관측"]; S -> P [label="근거"];']
        cur = after_last_maintenance(hr_now[hr_now['SCENARIO_ID'] == sid].assign(HOUR_START=lambda d: d['DONE_H'] - 1), 'HOUR_START')
        last = cur.dropna(subset=list(SIGNAL_NAMES)).tail(1)
        for k, (col, name) in enumerate(SIGNAL_NAMES.items()):
            if len(last):
                v = float(last[col].iloc[0])
                out += [f'G{k} [label={q(f"{name}" + chr(10) + f"{v:.2f}" + (" (벗어남)" if v > 1 else ""))}, '
                        f'fillcolor="{"#FFD6DC" if v > 1 else "#EEF2F8"}"];', f'P -> G{k} [label="근거"];']
    if e is not None:
        out += ['D [label="정비 결정", fillcolor="#E6E0FA"];', 'S -> D [label="요청"];']
        for k, o in enumerate(e['opts']):
            rec = o is e['best']
            cost = f"{chr(10)}{o['기대 총비용']:,.0f}{costs.unit if costs is not None else ''}"
            out += [f'O{k} [label={q(o["선택지"] + cost)}, fillcolor="{"#CFE0FF" if rec else "#EEF2F8"}", penwidth={2 if rec else 1}];',
                    f'D -> O{k} [label="{"추천" if rec else "비교"}"];']
            if rec:
                out += ['V [label="밸브"]; X [label="진동장치"];', f'O{k} -> V [label="교체"]; O{k} -> X [label="교체"];']
    return '\n'.join(out + ['}'])
def equipment_detail(sid, t, nested=False):
    """설비 하나의 상태 자세히: 지표 네 칸 · 입력 감시 · 위험 확률 흐름 · 네 신호 흐름 · 지금 신호 상태 · 모델이 본 이유.
    설비 페이지(하위 페이지)와 근거·비용 화면의 '설비 페이지' 접힌 칸(2026-10-21, nested=True)이 같이 쓴다"""
    row = t.loc[sid]
    c1, c2, c3, c4 = st.columns(4, border=True)
    c1.metric('상태', row['상태'].split(' (')[0], help=row['상태'])   # 칸이 좁아 영문 이름은 도움말로
    c2.metric('24시간 내 고장 위험', '—' if pd.isna(row['위험 확률']) else f'{row["위험 확률"]:.0%}',
              help=f'마지막 예측: {clock(row["마지막 예측 (h)"])}' if pd.notna(row['마지막 예측 (h)']) else None)
    if CAUTION:
        c3.metric('지금 열화 중일 확률', '—' if pd.isna(row['열화 확률']) else f'{row["열화 확률"]:.2f}')
        c4.metric('첫 주의 / 첫 경보', ' / '.join(clock(row[k]) for k in ['첫 주의 (h)', '첫 경보 (h)']))
    else:
        c3.metric('첫 경보', clock(row['첫 경보 (h)']))

    # 조치 안내는 설비관리자 근거·비용 화면으로 옮겼다 (2026-10-21, action_guide)
    if row['상태'] == WARN and row['경보 유지']:
        st.caption('지금 확률은 경보 기준보다 낮지만, 이번 운전 구간에서 이미 경보가 켜져 정비 전까지 경보로 유지합니다.')
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
    line = alt.Chart(p).mark_line(color=LINE_BLUE).encode(x=x, y=alt.Y('PROB:Q', title='확률', scale=alt.Scale(domain=[0, 1])))
    dots = alt.Chart(p).mark_circle(size=90).encode(x=x, y='PROB:Q', color=alt.Color('단계:N', scale=STAGE_COLORS), tooltip=tip)
    layers = [*mnt_layer(sid, 1), line, dots,
              alt.Chart(pd.DataFrame({'y': [TH]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')]
    if CAUTION:
        layers += [alt.Chart(p).mark_line(color=CAUTION_YELLOW, strokeDash=[2, 2]).encode(x=x, y='DEGRADED_PROB:Q', tooltip=tip),
                   alt.Chart(pd.DataFrame({'y': [CTH]})).mark_rule(color=CAUTION_YELLOW, strokeDash=[6, 3]).encode(y='y:Q')]
    st.subheader('위험 확률과 열화 확률' if CAUTION else '위험 확률', icon=':material/trending_up:')
    show_chart(alt.layer(*layers).properties(height=280), width='stretch')
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
    avg = alt.Chart(h).mark_line(color=INK, strokeWidth=2.5).encode(
        x='DONE_H:Q', y='SCORE:Q', tooltip=[alt.Tooltip('SCORE', title='네 신호 평균', format='.2f')])
    one = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color=INK, strokeDash=[2, 2]).encode(y='y:Q')
    st.subheader('네 신호의 흐름', icon=':material/ssid_chart:')
    show_chart(alt.layer(*mnt_layer(sid, None), sig, avg, one).properties(height=300), width='stretch')
    st.caption('회색 굵은 선: 네 신호 평균 점수 · 점선: 정상 범위 경계(1). 정비 후 재가동하면 처음 2시간은 기준을 잡는 시간이라 비어 있습니다.')

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
            show_chart(
                alt.layer(
                    alt.Chart(bar).mark_bar().encode(
                        y=alt.Y('신호:N', title=None, sort='-x'), x=alt.X('점수:Q', title='표준화 점수'),
                        color=alt.Color('정상 범위:N', scale=alt.Scale(domain=['벗어남', '안'], range=[ALARM_RED, OK_GRAY]))),
                    alt.Chart(pd.DataFrame({'x': [1]})).mark_rule(strokeDash=[2, 2]).encode(x='x:Q'),
                ).properties(height=180), width='stretch')
            st.caption(f'{last["DONE_H"].iloc[0]:.0f}시간까지의 1시간 값. 1을 넘으면 정상 시나리오의 95%보다 많이 벗어난 것입니다.')
    with right:
        box = st.container() if nested else st.expander('자세히: 모델이 본 이유 (기여도)')   # 접힌 칸 안에 접힌 칸을 두지 않는다
        with box:
            if nested:
                st.markdown('**모델이 본 이유 (기여도)**')
            cp = after_last_maintenance(pts_now[pts_now['SCENARIO_ID'] == sid], 'T_H')
            if contrib is None:
                st.write(f'최종 모델({bundle["final"]["model"]})은 특징별 기여도를 계산할 수 없는 종류입니다. '
                         '왼쪽의 신호 상태와 위의 신호 흐름으로 경보 이유를 확인하세요.')
            elif cp.empty:
                st.write(f'아직 예측이 없습니다. 정비가 시작되면 정비 전 예측은 쓰지 않고, 재가동 뒤 다음 예측 시점({STEP}시간마다)을 기다립니다.')
            else:
                cr = contrib.loc[cp.index[-1]].rename('기여도').reset_index().rename(columns={'index': '묶음'})
                show_chart(alt.Chart(cr).mark_bar().encode(
                    y=alt.Y('묶음:N', title=None, sort='-x'), x=alt.X('기여도:Q', title=f'위험을 올린 정도 ({contrib_unit})'),
                    color=alt.condition('datum.기여도 > 0', alt.value(ALARM_RED), alt.value(LINE_BLUE))
                ).properties(height=220), width='stretch')
                st.caption(f'{cp["T_H"].iloc[-1]:.0f}시간 예측 기준. 양수는 위험을 올린 쪽, 음수는 내린 쪽입니다. '
                           + ('나무마다 결정 경로를 따라 확률이 바뀐 양을 그 분기에 쓰인 특징 묶음에 나눠 준 값으로, '
                              '기준값에 모두 더하면 위의 위험 확률이 됩니다. ' if contrib_unit.startswith('확률')
                              else '0은 학습 세트 평균 시점입니다. ')
                           + '네 신호와 결합 점수는 서로 비슷한 정보라 모델 안에서 서로 상쇄될 수 있습니다. '
                           '묶음 하나의 크기보다 왼쪽의 신호 상태를 먼저 보세요.')


def open_detail(table_key, ids):
    """표에서 고른 설비로 '설비 상세' 화면을 연다 (st.dataframe 선택 콜백)"""
    rows = st.session_state[table_key].selection.rows
    if rows:
        go_detail(ids[rows[0]])


def open_sub(key, sid=None):
    """설비관리자 하위 페이지 열기 (노션의 하위 페이지처럼)"""
    st.session_state['msub'] = key
    st.session_state['scroll_top'] = True
    if sid:
        st.session_state['sid'] = sid


def go_detail(sid):
    """버튼에서 설비관리자의 '설비 상세' 하위 페이지로 (콜백)"""
    st.session_state['sid'] = sid
    st.session_state['msub'] = '설비 상세'


SHOW_SUBPAGE_LIST = False   # 설비관리자 ③ 하위 페이지 목록 (2026-10-15 사용자 요청으로 숨김. 설비 페이지는 비용 비교 제목 옆 버튼으로 들어감)
PLAN_GUESS_H = 12   # 정기 교체 계획이 없을 때 비용 비교에만 쓰는 '계획 정비' 시점 (가정값, 추천에는 안 씀)
STAGE_DOT = {'경보': 'red', '주의': 'orange', '정상': 'green', '정비 중': 'gray', '정비 후 예측 대기': 'gray'}   # 설비 목록 상태 점 색


def act(kind, sid, role_, note_key=None, **kw):
    """조치 기록 버튼 콜백: 기록하고 알림 (note_key: 메모 입력칸의 key)"""
    LOG.record(kind, split, sid, flow_hour(), role_, note=st.session_state.get(note_key, '') if note_key else '', **kw)
    st.toast(f'{sid}: {kind}', icon=':material/check_circle:')


RISK_BADGE = {'높음': ':red-badge[:material/warning: 끝나기 전 고장 위험 높음]', '보통': ':orange-badge[끝나기 전 고장 위험 보통]',
              '낮음': ':green-badge[끝나기 전 고장 위험 낮음]', '—': ''}
PRIORITY_BADGE = {'긴급': ':red-badge[긴급]', '높음': ':orange-badge[높음]', '보통': ':gray-badge[보통]'}
TASK_BADGE = {'대기': ':gray-badge[대기]', '진행 중': ':blue-badge[:material/build: 진행 중]', '완료': ':green-badge[:material/task_alt: 완료]',
              '중단': ':orange-badge[:material/stop_circle: 중단]'}


def fmt_cost(x, costs):
    """비용 표시: 만 원 (2026-10-21 사용자 요청: 억 원으로 바꾸지 않고 만 원 단위로)"""
    return f'{x:,.0f}{"만 원" if costs.unit == "만 원" else "점"}'


def start_task(tid, sid):
    if TASKS.start(tid, flow_hour()):
        LOG.record('작업 시작', split, sid, flow_hour(), WORKER)
        st.toast(f'{sid}: 작업 시작', icon=':material/play_arrow:')


def urgent_create(sid, team, option):
    """즉시 조치 설비 (2026-10-21 7차): 배정 전이라도 현장이 작업을 만들고 그 작업 상세로 간다 (안전이 먼저).
    바로 시작하지 않는다: 관리자 구두 승인 · 설비 정지 · TBM 등 사전 절차를 체크해야 작업 시작이 켜진다"""
    absent = TASKS.absent_at(now_f)
    mem = [m for m in tasks.TEAMS[team] if m not in absent][:4]
    try:
        ok = TASKS.add(split, sid, f'{sid} 즉시 조치: 두 부품 교체', team, '긴급', flow_hour(), flow_hour(), tasks.DEFAULT_GUIDE,
                       '지금 교체해도 완료 전 고장 확률이 안전 기준을 넘음 → 배정 전 현장이 긴급 작업을 만듦', option, mem)
    except ValueError as e:
        st.toast(str(e), icon=':material/error:')
        return
    if not ok:
        st.toast(f'{nm(sid)}: 이미 열린 작업이 있습니다', icon=':material/info:')
        return
    w_open_task(int(TASKS.open_task(split, sid, now_f)['id']))
    st.toast(f'{nm(sid)}: 즉시 조치 작업을 만들었습니다 · 사전 절차를 체크하세요', icon=':material/emergency:')


def assign_task(sid, title, team, priority, due_h, guide, reason, option, members=(), leader=''):
    """작업 배정 = 정비 결정 승인 + 작업 지시 (같은 설비에 열린 작업이 있으면 배정하지 않는다)"""
    try:
        ok = TASKS.add(split, sid, title, team, priority, due_h, flow_hour(), guide, reason, option, members, leader)
    except ValueError as e:
        st.toast(str(e), icon=':material/error:')
        return
    if ok:
        LOG.record('승인', split, sid, flow_hour(), MANAGER, option=option)
        st.toast(f'{sid}: {team}에 배정', icon=':material/assignment_ind:')
    else:
        st.toast(f'{sid}: 이미 열린 작업이 있어 배정하지 않았습니다', icon=':material/block:')


def assign_box(sid, srow, sstage, job_, dec_i, W, cfg, load, border=True):
    """비용 비교 바로 아래의 실행 칸: 팀 고르고 배정(주 버튼 하나) · 보류 · 부품 발주. 세부 내용은 접어 둔다"""
    existing = TASKS.blocking(split, sid, now_f)
    if existing:
        when = '' if existing['created_h'] <= now_f else f' ({clock(existing["created_h"])}에 만든 작업)'
        st.warning(f'이미 작업 #{int(existing["id"])} · {existing["title"]} ({existing["team"]}){when}이 있어 다시 배정하지 않습니다. '
                   '아래 "작업 수정하기"에서 고치거나 배정을 취소할 수 있습니다.', icon=':material/block:')
        st.button('작업 수정하기', icon=':material/edit:', key='edit_from_eq', on_click=open_task_edit, args=(int(existing['id']),))
        return
    if job_ == '작업 완료':
        return
    alarm = sid in dec_i.index
    urgent = alarm and dec_i.loc[sid, '제약'].startswith('즉시')
    rec = dec_i.loc[sid, '_e']['best']['선택지'] if alarm else ('점검·부품 준비' if sstage == '주의' else '상태 점검')
    with st.form(f'assign_{sid}', border=border):
        st.markdown(f'**배정: {nm(sid)} {rec}**')
        c1, c2 = st.columns([1.3, 1], vertical_alignment='bottom')
        team_f = c1.selectbox('담당 팀', list(tasks.TEAMS), index=list(tasks.TEAMS).index(min(load, key=load.get)),
                              format_func=lambda k: f'{k} · 열린 작업 {load[k]}건')
        go = c2.form_submit_button('작업자 화면으로 배정', icon=':material/send:', type='primary', width='stretch')
        with st.expander('작업 내용 바꾸기 (작업명 · 우선순위 · 시작 시각 · 점검 가이드)'):
            title_f = st.text_input('작업명', value=f'{sid} {rec}', max_chars=tasks.LIMITS['title'])
            d1, d2 = st.columns(2)
            prio_f = d1.selectbox('우선순위', tasks.PRIORITIES, index=0 if urgent else ({'경보': 1, '주의': 1}.get(sstage, 2)))
            # 즉시 조치는 정비팀 차례를 기다리지 않고 바로 시작한다 (작업자 화면의 '바로 시작'과 같은 규칙, 2026-10-19)
            due0 = min(240, int(round(now + max(0.0, dec_i.loc[sid, '_start'])))) if alarm and not urgent else int(now)
            due_f = d2.selectbox('시작 예정', list(range(241)), index=due0, format_func=clock)
            guide_f = st.text_area('점검 가이드', value=tasks.DEFAULT_GUIDE, height=140, max_chars=tasks.LIMITS['guide'])
            reason_f = st.text_area('추천 이유', max_chars=tasks.LIMITS['reason'],
                                    value=f'{sid} · {srow["상태"]} · 추천 {rec}' + (f' · 24시간 안 고장 확률 {srow["위험 확률"]:.2f}'
                                                                                 if pd.notna(srow['위험 확률']) else ''))
        st.caption('배정하면 그 팀 작업자 화면의 "지금 할 일"에 바로 나타납니다.')
        if go:
            assign_task(sid, title_f, team_f, prio_f, due_f, guide_f, reason_f, rec)
            st.rerun()
    order_buttons(sid, sstage, alarm, rec, cfg, W)


def order_rush(sid, parts):
    """두 부품 긴급 조달 (콜백): 조달 중이 아닌 부품을 긴급으로 발주 기록"""
    for q in parts:
        LOG.record('부품 발주', split, sid, flow_hour(), MANAGER, part=q, option='긴급')
    st.toast(f'{nm(sid)}: ' + ', '.join(PART_NAMES[q] for q in parts) + ' 긴급 조달', icon=':material/bolt:')


def order_buttons(sid, sstage, alarm, rec, cfg, W):
    """보류 · 부품 보충 발주 · 두 부품 긴급 조달 버튼 줄 (정비 판단 상자 맨 아래)"""
    if sstage in ('경보', '주의'):
        coming_ = {x for x, _, _ in in_transit(cfg).get(sid, [])}
        with st.container(horizontal=True):
            if alarm:
                st.button('보류', icon=':material/pause_circle:', key='hold_btn', on_click=act,
                          args=('보류', sid, MANAGER), kwargs={'option': rec}, help='지금은 배정하지 않고 지켜봅니다. 배정할 경보 수에서 빠집니다.')
            for part in PARTS:
                st.button(f'{PART_NAMES[part]} ' + ('조달 중' if part in coming_ else f'보충 발주 ({cfg.parts.lead_h:.0f}시간)'),
                          icon=':material/local_shipping:', key=f'order_{part}', disabled=part in coming_, on_click=act,
                          args=('부품 발주', sid, MANAGER), kwargs={'part': part, 'option': '정상'})
            # 두 부품 긴급 조달 (2026-10-21): 센서로 원인 부품을 가릴 수 없어 교체는 늘 두 부품이라 한 번에. 조달 중인 부품은 건너뛴다
            rush_parts = [q for q in PARTS if q not in coming_]
            st.button('두 부품 긴급 조달' + (f' ({cfg.parts.rush_h:.0f}시간)' if rush_parts else ' · 조달 중'), icon=':material/bolt:',
                      key='order_rush', disabled=not rush_parts, on_click=order_rush, args=(sid, rush_parts),
                      help=f'{cfg.parts.rush_h:.0f}시간 안에 도착, 웃돈 부품값의 {cfg.parts.rush_markup:.0%}')


def assign_from_card(sid, team_key, members_key, leader_key, title, priority, due_h, reason, option):
    """비용 칸의 '작업자 화면으로 배정' (콜백, 2026-10-21): 그 칸의 선택지·작업자·작업책임자로 배정. 점검 가이드·덧붙일 말은 '작업 내용 바꾸기'에서"""
    team = st.session_state.get(team_key, list(tasks.TEAMS)[0])
    members = list(st.session_state.get(members_key, []))
    guide = st.session_state.get(f'guide_txt_{sid}') or tasks.DEFAULT_GUIDE
    note = str(st.session_state.get(f'reason_txt_{sid}', '')).strip()
    assign_task(sid, title, team, priority, min(240.0, max(0.0, float(due_h))), guide, reason + (f' · {note}' if note else ''), option,
                members, st.session_state.get(leader_key, ''))


def cancel_task(tid, sid, title, why_key):
    """배정 취소 (콜백): 완료 전 작업을 지우고 진행 기록에 '배정 취소'와 사유를 남긴다. 사유가 없으면 취소하지 않는다"""
    why = st.session_state.get(why_key, '').strip()
    if not why:
        st.toast('배정 취소 사유를 입력하세요', icon=':material/error:')
        return
    if TASKS.cancel(tid):
        LOG.record('배정 취소', split, sid, flow_hour(), MANAGER, note=f'#{tid} {title} · 사유: {why}')
        st.session_state.pop(why_key, None)
        st.session_state.pop('edit_tid', None)
        st.toast(f'{sid}: 배정을 취소했습니다. 다시 배정할 수 있습니다', icon=':material/undo:')
    else:
        st.toast('완료된 작업이거나 이미 없는 작업입니다', icon=':material/error:')


def team_load(tf):
    """팀마다 열린 작업 수 (배정 기본값: 가장 한가한 팀)"""
    o = tf[tf['상태'] != '완료']
    return {k: int((o['team'] == k).sum()) for k in tasks.TEAMS}


# 작업자 화면 단계 (2026-10-21 7차): 홈(지금·지연·예정·완료·중단 칸) → 구분별 목록 → 작업 상세(체크리스트) → 완료 보고 / 작업 중단
W_LISTS = {'지금': (':material/play_circle:', '예정 시각이 됐거나 이미 시작한 작업'),
           '지연': (':material/schedule:', '예정 시각이 지났는데 아직 시작하지 않은 작업'),
           '예정': (':material/event:', '아직 예정 시각 전인 작업 (보기만)'),
           '완료': (':material/task_alt:', '완료 보고한 작업'),
           '중단': (':material/stop_circle:', '현장에서 중단한 작업')}
W_SCREENS = ('detail', 'report', 'stop')

# 작업 상세 체크리스트 (2026-10-21 7차, 사용자 현장 설명대로)
#   사전 절차: ① 작업 허가서(전날 정비 범위 미팅에서 전산 입력, 당일 출력) → ② 설비 정지 → ③ TBM(하위 6개 순서 자유, 다 체크해야 'TBM 완료')
#             ④ 부품·공구(순서 자유). 전부 체크해야 [작업 시작]. 즉시 조치는 ①을 관리자 구두 승인으로 (허가서는 사후 처리)
#   정비 중: 작업책임자가 10대 안전철칙마다 준수 / 해당 없음을 고른다. 작업 시작 전에는 잠금
#   정비 후: ①~④ 순서대로, ⑤⑥은 자유. 정비 중 10개를 다 골라야 열린다. 전부 끝나야 [완료 보고 작성]
#   체크 상태는 작업 표 checks 칸(JSON)에 바로 저장한다 → 기기를 바꿔도 남고 설비관리자가 진행률을 본다
TBM_ITEMS = [('tbm_share', '작업 내용 · 위험 요인 · 역할 분담 공유'), ('tbm_ils', 'ILS 절차 준수 여부 확인'), ('tbm_press', '잔압 제거'),
             ('tbm_power', '전원 차단'), ('tbm_valve', '밸브 차단'), ('tbm_ppe', '보호구 착용 확인')]
PRE_KEYS = ['permit', 'halt', *[k for k, _ in TBM_ITEMS], 'tbm_done', 'parts']
SAFETY_RULES = [('절대 금지', '가동설비 접근 및 점검·수리 금지'), ('절대 금지', '추락위험개소 안전대 미체결 시 진입 금지'),
                ('절대 금지', '폭발·인화성 물질 주변 화기 취급 금지'), ('절대 금지', '중량물·중장비 작업반경 내 접근 금지'),
                ('절대 금지', '산소·유해가스 측정 전 밀폐공간 진입 금지'), ('절대 금지', '전원 차단 및 검전 전 전기 작업 금지'),
                ('절대 실행', '작업 중 지적확인 (위험 시 작업중지권 사용)'), ('절대 실행', '수리 작업 시 ILS 절차 준수'),
                ('절대 실행', '모든 작업 시 안전 CCTV 촬영'), ('절대 실행', '규정에 맞는 안전보호구 착용')]
SAFETY_KEYS = [f's{i}' for i in range(1, len(SAFETY_RULES) + 1)]
SAFETY_OPTS = ['준수', '해당 없음']
POST_ITEMS = [('clean', '① 공구·이물질 회수, 작업 구역 정리'), ('evac', '② 작업자 전원 철수 확인'), ('ils_off', '③ ILS 해제와 관계자 통보'),
              ('trial', '④ 시운전'), ('signal', '⑤ 몰드 레벨 · 제어 지연 · 인발 부하 · 진동 스트로크 신호 확인'),
              ('cause', '⑥ 원인 부품 확인 (완료 보고에 기록)')]
POST_KEYS = [k for k, _ in POST_ITEMS]
POST_ORDER = POST_KEYS[:4]
CHECK_TOTAL = len(PRE_KEYS) + len(SAFETY_KEYS) + len(POST_KEYS)


def load_checks(raw):
    """작업 표 checks 칸(JSON) → dict. 비었거나 깨졌으면 빈 dict"""
    try:
        d = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        d = {}
    return d if isinstance(d, dict) else {}


def check_count(d):
    """체크한 항목 수 (사전 · 정비 중 · 정비 후)"""
    return (sum(bool(d.get(k)) for k in PRE_KEYS) + sum(d.get(k) in SAFETY_OPTS for k in SAFETY_KEYS)
            + sum(bool(d.get(k)) for k in POST_KEYS))


def fix_checks(d):
    """순서 규칙: 앞 항목을 풀면 뒤 항목도 푼다 (사전 ①→②→③, TBM 하위 → TBM 완료, 정비 후 ①→④)"""
    if not d.get('permit'):
        d['halt'] = False
    if not d.get('halt'):
        d.update({k: False for k, _ in TBM_ITEMS})
    if not all(d.get(k) for k, _ in TBM_ITEMS):
        d['tbm_done'] = False
    for a, b in zip(POST_ORDER, POST_ORDER[1:]):
        if not d.get(a):
            d[b] = False
    return d


def save_checks(tid, extra):
    """체크 하나를 바꿀 때마다 (콜백): 순서 규칙을 맞추고 작업 표에 바로 저장한다. extra: 위젯이 아닌 저장값(허가서 사후 처리 등)"""
    pre = f'ck{tid}_'
    d = {k: bool(st.session_state.get(pre + k)) for k in PRE_KEYS + POST_KEYS}
    d.update({k: st.session_state.get(pre + k) for k in SAFETY_KEYS})
    d = fix_checks(d)
    for k in PRE_KEYS + POST_KEYS:
        st.session_state[pre + k] = d[k]
    TASKS.set_checks(tid, json.dumps({**extra, **{k: v for k, v in d.items() if v}}, ensure_ascii=False))


def w_go(view=None):
    """작업자 화면 단계로 (None = 홈, 지금·지연·예정·완료·중단 = 목록, detail·report·stop = 작업 하나)"""
    st.session_state['wv'] = view
    st.session_state['w_scroll'] = True


def w_open_task(tid):
    """'배정된 작업 보기': 작업 상세로 (돌아올 목록을 기억한다)"""
    if st.session_state.get('wv') in W_LISTS:
        st.session_state['w_back'] = st.session_state['wv']
    elif st.session_state.get('wv') is None:
        st.session_state['w_back'] = None
    st.session_state['w_tid'] = tid
    w_go('detail')


def is_urgent_job(r):
    return r['priority'] == '긴급' or '즉시 조치' in str(r['title'])


def people(r):
    """작업자 이름 (작업책임자 표시)"""
    lead = r['leader'] or tasks.pick_leader(r['members'], '')
    return ' · '.join(m + (' (작업책임자)' if m == lead else '') for m in str(r['members']).split(',') if m)


def task_line(r):
    """작업 한 줄 (작업자 목록 카드): 요약 + '배정된 작업 보기'"""
    with st.container(border=True):
        st.markdown(f'**#{int(r["id"])} {r["title"] if str(r["title"]).startswith(r["sid"]) else r["sid"] + " — " + r["title"]}**  \n'
                    + ' '.join([PRIORITY_BADGE.get(r['priority'], ''), TASK_BADGE.get(r['상태'], ''),
                                ':red-badge[지연]' if r['구분'] == '지연' else '', f':gray-badge[{r["team"]}]',
                                f':gray-badge[:material/group: {people(r)}]' if r['members'] else '',
                                f':blue-badge[{nm(r["sid"])}]',
                                f':violet-badge[:material/checklist: 체크 {check_count(load_checks(r["checks"]))}/{CHECK_TOTAL}]'
                                if r['상태'] in ('대기', '진행 중') and r['checks'] else '']))
        st.caption(f'예정 {clock(r["due_h"])} · 배정 {clock(r["created_h"])}'
                   + (f' · 완료 {clock(r["completed_h"])}' if pd.notna(r['completed_h']) and r['상태'] == '완료' else '')
                   + (f' · {r["reason"]}' if r['reason'] else ''))
        if r['상태'] == '완료':
            st.markdown(f'완료 보고: **{r["outcome"]}** · 실제 조치 {r["action"] or tasks.ACTIONS[0]} · '
                        f'점검 판단 {PART_NAMES.get(r["judged"], r["judged"])} · '
                        f'확인된 원인 {PART_NAMES.get(r["confirmed"], r["confirmed"])}  \n{r["details"]}'
                        + (f'  \n:orange[지시와 다름 · {r["action_why"]}]' if r['action_why'] else ''))
        elif r['상태'] == '중단':
            st.markdown(f':orange[작업 중단 {clock(r["stopped_h"])} · 사유: {r["stop_why"]}]')
        st.button('배정된 작업 보기', key=f'wopen_{int(r["id"])}', icon=':material/arrow_forward:',
                  type='primary' if r['구분'] in ('지금', '지연') else 'secondary', on_click=w_open_task, args=(int(r['id']),))


# 알림: 다른 사람이 한 일 (설비관리자 ← 완료 보고, 작업자 ← 새 배정). 재생 시각을 옮겨서 바뀐 것은 알리지 않는다
#   실제로 일어난 일(완료 보고 기록 시각·새 작업 번호)을 이 세션이 이미 본 것과 비교한다. 데이터 세트를 처음 볼 때는 기록만 한다

def notify():
    if not TASKS.shared:
        return
    allt = TASKS.all(split)
    seen = st.session_state.setdefault('_seen', {})
    if role == MANAGER:
        cur = {int(r.id): r for r in allt[allt['completed_at'].fillna('') != ''].itertuples()}
        key = ('done', split)
        for i in sorted(set(cur) - seen.get(key, set(cur))):
            r = cur[i]
            st.toast(f'완료 보고 · {nm(r.sid)} {r.outcome} ({r.team}, 작업 #{i})' + (' · 지시와 다름' if r.action_why else ''),
                     icon=':material/task_alt:', duration='long')
        seen[key] = set(cur)
        cur = {int(r.id): r for r in allt[allt['stopped_h'].notna()].itertuples()}
        key = ('stop', split)
        for i in sorted(set(cur) - seen.get(key, set(cur))):
            r = cur[i]
            st.toast(f'작업 중단 · {nm(r.sid)} ({r.team}, 작업 #{i}) · 사유: {r.stop_why}', icon=':material/stop_circle:', duration='long')
    else:
        team_ = st.session_state.get('my_team', list(tasks.TEAMS)[0])
        cur = {int(r.id): r for r in allt[allt['team'] == team_].itertuples()}
        key = ('new', split, team_)
        for i in sorted(set(cur) - seen.get(key, set(cur))):
            r = cur[i]
            st.toast(f'새 작업 · {r.title} ({r.priority})', icon=':material/assignment_ind:', duration='long')
    seen[key] = set(cur)


notify()


def close_task_edit():
    st.session_state['task_edit'] = False


def task_editor(tf):
    """작업 수정하기 (2026-10-21): 지시된 작업을 골라 수정하거나 배정을 취소한다. 수정·취소 모두 사유를 적어야 저장된다"""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment='center'):
            st.markdown('##### :material/edit: 작업 수정')
            st.space('stretch')
            st.button('닫기', icon=':material/close:', type='tertiary', key='task_edit_close', on_click=close_task_edit)
        tfi = tf.set_index('id')
        if st.session_state.get('edit_tid') not in tfi.index:
            st.session_state.pop('edit_tid', None)
        tid = st.selectbox('수정할 작업', tf['id'].tolist(), key='edit_tid',
                           format_func=lambda i: f'#{i} · {nm(tfi.loc[i, "sid"])} · {tfi.loc[i, "title"]} · {tfi.loc[i, "상태"]}')
        cur = tfi.loc[tid]
        done_ = cur['상태'] == '완료'
        if cur['상태'] == '중단':
            st.warning(f'작업자가 {clock(cur["stopped_h"])}에 중단한 작업입니다 · 사유: {cur["stop_why"]}  \n'
                       '다시 맡기려면 상태를 대기·진행 중으로 바꿔 저장하고, 맡기지 않으려면 아래에서 배정을 취소하세요.',
                       icon=':material/stop_circle:')
        elif cur['action_why']:
            st.info(f'실제 조치: {cur["action"]} · 지시와 다름: {cur["action_why"]}', icon=':material/info:')
        # 상태는 폼 밖에서 고른다: '완료'를 고르면 아래에 완료 보고 칸이 바로 나타나게. 중단한 작업은 다시 맡길 상태를 고른다
        e_state = st.segmented_control('작업 상태', tasks.STATUSES, default=cur['상태'] if cur['상태'] in tasks.STATUSES else '대기',
                                       required=True, key=f'edit_state_{tid}',
                                       help='대기·진행 중으로 되돌리면 완료 보고·중단이 지워집니다. 완료로 바꾸려면 완료 보고가 필요합니다.')
        with st.form(f'edit_task_{tid}', border=False):
            c1, c2, c3 = st.columns(3)
            e_team = c1.selectbox('담당 팀', list(tasks.TEAMS), index=list(tasks.TEAMS).index(cur['team']))
            cur_mem = [m for m in str(cur['members']).split(',') if m in tasks.TEAMS[cur['team']]]
            e_mem = st.multiselect('작업자 (팀을 바꾸면 저장할 때 새 팀원만 남깁니다)', [m for v in tasks.TEAMS.values() for m in v], default=cur_mem,
                                   format_func=lambda m: f'{m} · {tasks.TEAM_OF[m]}')
            all_people = [m for v in tasks.TEAMS.values() for m in v]
            cur_lead = cur['leader'] or tasks.pick_leader(cur['members'], '')
            e_lead = st.selectbox('작업책임자 (작업자 중 한 명, 아니면 저장할 때 반장 · 첫 사람으로 정해집니다)', all_people,
                                  index=all_people.index(cur_lead) if cur_lead in all_people else 0,
                                  format_func=lambda m: f'{m} · {tasks.TEAM_OF[m]}')
            e_prio = c2.selectbox('우선순위', tasks.PRIORITIES, index=tasks.PRIORITIES.index(cur['priority']))
            e_due = c3.selectbox('시작 예정', list(range(241)), index=min(240, int(round(cur['due_h']))), format_func=clock)
            e_guide = st.text_area('점검 가이드', value=cur['guide'], max_chars=tasks.LIMITS['guide'])
            e_reason = st.text_area('추천 이유', value=cur['reason'], max_chars=tasks.LIMITS['reason'])
            if e_state == '완료':
                st.markdown('**완료 보고**' + (f' (완료 {clock(cur["completed_h"])})' if done_ else ''))
                if not done_:   # 새로 완료로 바꿀 때만 완료 시각을 고른다 (이미 완료된 작업은 시각을 그대로 둔다)
                    lo_e = min(int(np.ceil(cur['started_h'] if pd.notna(cur['started_h']) else cur['created_h'])), int(now))
                    e_at = hour_pick('완료 시각', lo_e, now, now, f'edit_at_{tid}')
                    if e_at == now:     # 지금 정각을 고르면 흐르는 지금 시각으로 (시작보다 앞서지 않게)
                        e_at = now_f
                pick = lambda opts, v, d: opts.index(v) if v in opts else opts.index(d)
                e_out = st.selectbox('작업 결과', tasks.OUTCOMES, index=pick(tasks.OUTCOMES, cur['outcome'], '이상 발견'))
                e_jud = st.selectbox('점검으로 판단한 부품', tasks.JUDGED, index=pick(tasks.JUDGED, cur['judged'], '구분 못함'),
                                     format_func=lambda x: PART_NAMES.get(x, x))
                e_con = st.selectbox('교체 뒤 확인된 원인 부품', tasks.CONFIRMED,
                                     index=pick(tasks.CONFIRMED, cur['confirmed'], '아직 모름'), format_func=lambda x: PART_NAMES.get(x, x))
                e_det = st.text_area('수행 내용', value=cur['details'] if done_ else '', max_chars=tasks.LIMITS['details'])
            e_why = st.text_input('수정 사유 (필수)', max_chars=150, placeholder='예: 정비 1팀 인원이 부족해 2팀으로 넘김')
            if st.form_submit_button('수정 저장', icon=':material/save:', type='primary'):
                if not e_why.strip():
                    st.error('수정 사유를 입력해야 저장됩니다.', icon=':material/error:')
                elif e_state != '완료' or done_ or pick_ok(e_at, lo_e, now_f):
                    try:
                        TASKS.edit(tid, e_team, e_prio, e_due, e_guide, e_reason, [m for m in e_mem if tasks.TEAM_OF[m] == e_team], e_lead)
                        if e_state != cur['상태']:
                            # 상태 변경: 작업 기록을 바꾸고, 진행 단계 기록(src/actions.py)에도 남겨 화면 전체가 같은 상태를 본다
                            if e_state == '완료':
                                TASKS.set_status(tid, '완료', e_at, e_out, e_det, e_jud, e_con)
                                LOG.record('작업 완료', split, cur['sid'], e_at, MANAGER, note=f'상태 변경 #{tid}: {e_det}')
                                LOG.record('점검 결과', split, cur['sid'], e_at, MANAGER, judged=e_jud, confirmed=e_con, note=e_out)
                            else:
                                TASKS.set_status(tid, e_state, flow_hour())
                                if done_:   # 되돌린 완료의 점검 결과는 같은 시각에 '아직 모름'으로 덮는다 (되먹임은 마지막 기록만 센다)
                                    LOG.record('점검 결과', split, cur['sid'], float(cur['completed_h']), MANAGER, judged='구분 못함',
                                               confirmed='아직 모름', note=f'완료 되돌림 #{tid}')
                                LOG.record('작업 시작' if e_state == '진행 중' else '승인', split, cur['sid'], flow_hour(), MANAGER,
                                           note=f'상태 변경 #{tid}: {cur["상태"]} → {e_state}')
                        elif done_:
                            TASKS.edit_report(tid, e_out, e_det, e_jud, e_con)
                            if (e_out, e_jud, e_con) != (cur['outcome'], cur['judged'], cur['confirmed']):
                                LOG.record('점검 결과', split, cur['sid'], float(cur['completed_h']), MANAGER, judged=e_jud,
                                           confirmed=e_con, note=f'완료 보고 수정 #{tid}: {e_out}')
                        LOG.record('작업 수정', split, cur['sid'], flow_hour(), MANAGER, note=f'#{tid} 사유: {e_why.strip()}')
                        st.toast(f'#{tid} 수정했습니다' + (f' ({cur["상태"]} → {e_state})' if e_state != cur['상태'] else ''),
                                 icon=':material/check_circle:')
                        st.rerun()
                    except ValueError as err:
                        st.error(str(err))
        # 배정 취소: 실수로 배정했거나 계획이 바뀐 경우. 그 설비는 다시 '배정할 경보'로 돌아간다
        st.divider()
        if done_:
            st.caption('완료된 작업은 배정을 취소할 수 없습니다. 위에서 상태를 대기·진행 중으로 되돌린 뒤 취소하세요.')
        else:
            with st.form(f'cancel_form_{tid}', border=False):
                st.text_input(f'#{tid} 배정 취소 사유 (필수)', key=f'cancel_why_{tid}', max_chars=150,
                              placeholder='예: 정기 교체 일정과 겹쳐 그때 함께 하기로 함')
                st.form_submit_button('배정 취소', icon=':material/undo:', on_click=cancel_task,
                                      args=(tid, cur['sid'], cur['title'], f'cancel_why_{tid}'),
                                      help='작업 목록에서 빠지고, 설비는 다시 배정할 경보로 돌아갑니다')


# 누르는 지표 칸 모양 (st.metric 칸과 비슷하게: 왼쪽 위 이름, 아래 큰 숫자)과 조치 안내 칸 바탕색. 작업자·설비관리자가 같이 쓴다. 고정된 글만 넣는다
CARD_CSS = ('<style>'
            '[class*="st-key-card_"] button { min-height: 132px; justify-content: flex-start; align-items: flex-start; padding: 16px 20px; }'
            '[class*="st-key-card_"] button > div, [class*="st-key-card_"] button [data-testid="stMarkdownContainer"] '
            '{ width: 100%; justify-content: flex-start; text-align: left; }'
            '[class*="st-key-card_"] button p { line-height: 1.25; }'
            '[class*="st-key-card_"] button strong { font-size: 2.25rem; font-weight: 500; display: block; margin-top: 10px; }'
            # 조치 안내 칸: 알림 상자처럼 연한 바탕 (라이트·다크 모두 보이게 반투명)
            '.st-key-guide_red { background: rgba(255,102,122,0.10); border-color: rgba(255,102,122,0.45) !important; }'
            '.st-key-guide_orange { background: rgba(247,185,85,0.12); border-color: rgba(247,185,85,0.5) !important; }'
            '.st-key-guide_green { background: rgba(45,212,160,0.10); border-color: rgba(45,212,160,0.45) !important; }'
            '[class*="st-key-guide_"] [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }'
            '</style>')


def scroll_to_top():
    """화면 단계를 바꾸면 맨 위(경로)부터 보이게. 고정된 글(사용자 입력 없음)만 넣는다"""
    st.iframe('<script>const d = window.parent.document;'
              '["[data-testid=stMain]", "[data-testid=stAppViewContainer]", "section.main"]'
              '.forEach(q => { const e = d.querySelector(q); if (e) e.scrollTo(0, 0); });'
              'window.parent.scrollTo(0, 0);</script>', height=1)


# 0. 작업자: 내 작업 (2026-10-21 7차) 홈 칸 5개 → 구분별 목록 → 작업 상세(체크리스트 → 작업 시작) → 완료 보고 / 작업 중단

page = None
if role == WORKER:
    # 작업자는 내 작업만 본다 (2026-10-16: 설비 상세는 설비관리자의 하위 페이지로만 연다)
    if not bundle.get('dss'):
        st.warning('이 모델 파일에는 정비 계획 정보가 없습니다. python analysis/05_predict_test.py 로 다시 만드세요.')
        st.stop()
    A = assumptions()
    t = status_table()
    W = work_plan(A, t)
    dec, cfg, costs = W['dec'], W['cfg'], W['costs']
    tf = TASKS.frame(split, now_f)
    dec_i = dec.set_index('설비 (시나리오)') if len(dec) else pd.DataFrame()
    st.session_state.setdefault('my_team', list(tasks.TEAMS)[0])
    if st.session_state.pop('w_scroll', False):
        scroll_to_top()
    st.html(CARD_CSS)
    wv = st.session_state.get('wv')
    if wv not in (None, *W_LISTS, *W_SCREENS):
        wv = None
    if wv in W_SCREENS:     # 작업이 없어졌거나(배정 취소) 다른 팀 작업이면 홈으로
        mine_ids = set(tf.loc[tf['team'] == st.session_state['my_team'], 'id'])
        if st.session_state.get('w_tid') not in mine_ids:
            wv = st.session_state['wv'] = None

    # 머리말: 홈이면 '내 작업' + 팀, 아래 단계면 돌아가는 경로 + 팀 이름
    with st.container(horizontal=True, vertical_alignment='center', gap='small'):
        if wv is None:
            st.subheader('내 작업', icon=':material/checklist:')
        elif wv in W_LISTS:
            st.button('내 작업', icon=':material/arrow_back:', type='tertiary', key='w_back_home', on_click=w_go)
            st.markdown(f'/ &nbsp;{W_LISTS[wv][0]} **{wv}**')
        else:
            back = st.session_state.get('w_back')
            if wv == 'detail':
                st.button(back or '내 작업', icon=':material/arrow_back:', type='tertiary', key='w_back_list', on_click=w_go, args=(back,))
                st.markdown(f'/ &nbsp;**작업 #{st.session_state["w_tid"]}**')
            else:
                st.button(f'작업 #{st.session_state["w_tid"]}', icon=':material/arrow_back:', type='tertiary', key='w_back_detail',
                          on_click=w_go, args=('detail',))
                st.markdown('/ &nbsp;**' + ('완료 보고' if wv == 'report' else '작업 중단') + '**')
        st.space('stretch')
        # 44: 구글 로그인 작업자는 자기 팀 이름만 (버튼 없음). 로그인 없는 화면 · 설비관리자가 보는 작업자 화면은 두 팀을 고를 수 있다
        if LOCKED_WORKER or wv is not None:
            st.markdown(f':blue-badge[:material/groups: {st.session_state["my_team"]}]')
        else:
            # 고르는 칸(team_pick)과 고른 팀(my_team)을 나눈다: 아래 단계에서는 이 칸을 그리지 않아 칸 값이 지워지기 때문
            st.session_state['team_pick'] = st.session_state['my_team']
            st.segmented_control('내 팀', list(tasks.TEAMS), key='team_pick', required=True, label_visibility='collapsed',
                                 on_change=lambda: st.session_state.update(my_team=st.session_state['team_pick']),
                                 help='팀원: ' + ' / '.join(f'{k} ' + '·'.join(v) for k, v in tasks.TEAMS.items()))
    team = st.session_state['my_team']
    mine = tf[tf['team'] == team]

    if wv is None:
        # 배정 전 즉시 조치 설비: 홈 칸 위에 빨간 알림. 누르면 작업을 만들고 그 작업 상세로 (시작은 사전 절차 뒤)
        busy = TASKS.blocked_sids(split, now_f)
        js = LOG.job_state(split, now_f, MNT_START)
        done_ = {x for x, v in js.items() if v == '작업 완료'}     # 이번 운전 구간에서 이미 교체를 마친 설비
        urgent_un = [x for x in dec_i.index if dec_i.loc[x, '제약'].startswith('즉시') and x not in busy | done_] if len(dec_i) else []
        if urgent_un:
            with st.container(border=True, key='guide_red'):
                st.markdown(f'#### :red[:material/emergency: 배정 전 즉시 조치 {len(urgent_un)}대]')
                st.markdown('지금 교체해도 완료 전 고장 확률이 안전 기준을 넘습니다. 안전이 먼저라 설비관리자 배정을 기다리지 않고 '
                            '작업을 만들 수 있습니다. 관리자 구두 승인을 받은 뒤 사전 절차를 체크하세요 (작업 허가서는 사후 처리).')
                with st.container(horizontal=True):
                    for sid in urgent_un:
                        st.button(f'{nm(sid)} 즉시 조치 작업 만들기', key=f'urg_{sid}', icon=':material/emergency:', type='primary',
                                  on_click=urgent_create, args=(sid, team, dec_i.loc[sid, '추천']))
        cnt = mine['구분'].value_counts()
        for col, (k, (ic, help_)) in zip(st.columns(len(W_LISTS)), W_LISTS.items()):
            with col:
                st.button(f'{ic} {k} :material/chevron_right:\n\n**{cnt.get(k, 0)}**', key=f'card_w_{k}', width='stretch',
                          on_click=w_go, args=(k,), help=help_)
        st.caption('재생 시각 기준: 지금 = 예정 시각이 됐거나 이미 시작함 · 지연 = 예정 시각이 지났는데 시작 전 · 예정 = 아직 · '
                   + STORE_NOTE)

    elif wv in W_LISTS:
        rows = mine[mine['구분'] == wv]
        if wv == '지금':    # 진행 중인 작업 먼저
            rows = rows.assign(_r=(rows['상태'] != '진행 중').astype(int)).sort_values('_r', kind='stable')
        st.caption(W_LISTS[wv][1] + f' · {len(rows)}건')
        if rows.empty:
            st.markdown('##### :material/check_circle: 없습니다')
            st.caption('설비관리자가 배정하면 지금·예정에 나타납니다.' if wv in ('지금', '예정') else '')
        for _, r in rows.iterrows():
            task_line(r)

    else:
        r = mine.set_index('id').loc[st.session_state['w_tid']]
        tid, sid = int(st.session_state['w_tid']), r['sid']
        urgent_job = is_urgent_job(r)
        saved = load_checks(r['checks'])
        extra = {k: v for k, v in saved.items() if k not in PRE_KEYS + SAFETY_KEYS + POST_KEYS}

        if wv == 'detail':
            info = dec_i.loc[sid] if sid in dec_i.index else None
            plan_part = info['_e'].get('plan_part') if info is not None else None
            need = PARTS if not str(r['option']).startswith('다음 정기') else [q for q in PARTS if q != plan_part]
            miss_ = missing_parts(W['stock'][sid], need)
            with st.container(border=True):
                st.markdown(f'### {r["title"] if str(r["title"]).startswith(sid) else sid + " — " + r["title"]}')
                st.markdown(f':blue-badge[:material/precision_manufacturing: {nm(sid)}]'
                            + (f' :gray-badge[:material/group: {people(r)}]' if r['members'] else ''))
                st.markdown(' '.join(x for x in [
                    PRIORITY_BADGE.get(r['priority'], ''), TASK_BADGE.get(r['상태'], ''),
                    ':red-badge[:material/schedule: 예정 시각 지남]' if r['구분'] == '지연' else '',
                    ':red-badge[:material/emergency: 즉시 조치]' if urgent_job else '',
                    (RISK_BADGE.get(risk_word(info['완료 전 고장 확률'], cfg.safety_max), '') if info is not None else ''),
                    (f':orange-badge[:material/inventory_2: {", ".join(PART_NAMES[q] for q in miss_)} 재고 없음 · 긴급 조달 '
                     f'{cfg.parts.rush_h:.0f}시간]' if miss_ else ':green-badge[:material/inventory_2: 부품 준비됨]')] if x))
                st.markdown(f'**예정 {clock(r["due_h"])} → {clock(r["due_h"] + costs.replace_h)}** · {r["reason"]}')
                st.caption(f'시나리오 {sid} · 배정 {clock(r["created_h"])}'
                           + (f' · 시작 {clock(r["started_h"])}' if pd.notna(r['started_h']) else ''))
                with st.expander('점검 가이드 (설비관리자 메모)', icon=':material/fact_check:'):
                    st.markdown(r['guide'].replace('\n', '  \n'))
            if r['상태'] == '완료':
                st.success(f'완료 보고 {clock(r["completed_h"])}: **{r["outcome"]}** · 실제 조치 {r["action"] or tasks.ACTIONS[0]} · '
                           f'확인된 원인 {PART_NAMES.get(r["confirmed"], r["confirmed"])}  \n{r["details"]}'
                           + (f'  \n지시와 다름 · {r["action_why"]}' if r['action_why'] else ''), icon=':material/task_alt:')
            elif r['상태'] == '중단':
                st.warning(f'작업 중단 {clock(r["stopped_h"])} · 사유: {r["stop_why"]}', icon=':material/stop_circle:')
            elif r['구분'] == '예정':
                st.info(f'예정 시각({clock(r["due_h"])}) 전이라 보기만 합니다. 그 시각이 되면 "지금"에서 사전 절차를 체크할 수 있습니다.',
                        icon=':material/event:')

            # 체크리스트: 저장된 값으로 칸을 채운다 (처음 열 때만. 그 뒤로는 체크할 때마다 저장)
            pre = f'ck{tid}_'
            for k in PRE_KEYS + POST_KEYS:
                st.session_state.setdefault(pre + k, bool(saved.get(k)))
            for k in SAFETY_KEYS:
                st.session_state.setdefault(pre + k, saved.get(k) if saved.get(k) in SAFETY_OPTS else None)
            v = lambda k: st.session_state[pre + k]
            can_pre = r['상태'] == '대기' and r['구분'] in ('지금', '지연')
            running = r['상태'] == '진행 중'
            cb = dict(on_change=save_checks, args=(tid, extra))

            pre_left = sum(not v(k) for k in PRE_KEYS)
            with st.container(border=True):
                st.markdown('##### :material/assignment_turned_in: 사전 절차' + (' · 완료' if not pre_left else f' · {pre_left}개 남음'))
                st.checkbox('① 관리자 구두 승인 확인 (작업 허가서는 사후 처리)' if urgent_job else
                            '① 작업 허가서 출력·확인 (전날 정비 범위 미팅에서 전산 입력한 것)', key=pre + 'permit', disabled=not can_pre, **cb)
                st.checkbox('② 설비 정지 확인', key=pre + 'halt', disabled=not can_pre or not v('permit'), **cb)
                with st.container(border=True):
                    st.markdown('③ **TBM** (Tool Box Meeting) · 작업자 전원, 아래 6개는 순서 상관없음')
                    for k, label in TBM_ITEMS:
                        st.checkbox(label, key=pre + k, disabled=not can_pre or not v('halt'), **cb)
                    st.checkbox('**TBM 완료**', key=pre + 'tbm_done', disabled=not can_pre or not all(v(k) for k, _ in TBM_ITEMS), **cb)
                st.checkbox('④ 교체 부품·공구 준비', key=pre + 'parts', disabled=not can_pre, **cb)
                if r['상태'] == '대기':
                    st.button('작업 시작', icon=':material/play_arrow:', type='primary', key=f'wstart_{tid}', disabled=not can_pre or pre_left > 0,
                              on_click=start_task, args=(tid, sid),
                              help='사전 절차를 모두 체크해야 켜집니다' if pre_left else None)
                    if can_pre and pre_left:
                        st.caption(f'사전 절차 {pre_left}개 남음 · ①→②→③ 순서대로 켜집니다')

            safety_left = sum(v(k) not in SAFETY_OPTS for k in SAFETY_KEYS)
            with st.container(border=True):
                st.markdown('##### :material/health_and_safety: 정비 중 · 10대 안전철칙'
                            + (' · 완료' if not safety_left else f' · {safety_left}개 남음'))
                st.caption(f'작업책임자({r["leader"] or tasks.pick_leader(r["members"], "") or "미지정"})가 항목마다 준수 / 해당 없음을 고릅니다.'
                           + ('' if running or r['상태'] in ('완료', '중단') else ' 작업 시작 뒤에 열립니다.'))
                for i, (k, (kind, text)) in enumerate(zip(SAFETY_KEYS, SAFETY_RULES), 1):
                    if i == 1 or kind != SAFETY_RULES[i - 2][0]:
                        st.markdown(f'**{kind} 수칙**')
                    st.segmented_control(f'{i}. {text}', SAFETY_OPTS, key=pre + k, disabled=not running, **cb)

            post_ok = running and not safety_left
            with st.container(border=True):
                post_left = sum(not v(k) for k in POST_KEYS)
                st.markdown('##### :material/cleaning_services: 정비 후 절차' + (' · 완료' if not post_left else f' · {post_left}개 남음'))
                if running and safety_left:
                    st.caption('정비 중 10대 안전철칙을 모두 고르면 열립니다. ①~④는 순서대로 켜집니다.')
                for j, (k, label) in enumerate(POST_ITEMS):
                    first = j == 0 or j >= len(POST_ORDER)
                    st.checkbox(label, key=pre + k, disabled=not post_ok or not (first or v(POST_ORDER[j - 1])), **cb)

            if r['상태'] in ('대기', '진행 중'):
                with st.container(horizontal=True):
                    if running:
                        ready = not safety_left and not post_left
                        st.button('완료 보고 작성', icon=':material/edit_note:', type='primary', key=f'wreport_{tid}', disabled=not ready,
                                  on_click=w_go, args=('report',), help=None if ready else '정비 중 · 정비 후 항목을 모두 체크해야 켜집니다')
                    st.button('작업 중단', icon=':material/stop_circle:', key=f'wstop_{tid}', on_click=w_go, args=('stop',),
                              help='현장 상황이 지시와 달라 끝낼 수 없을 때 (언제든)')
            st.caption('현장 안전 규정과 작업 허가가 우선입니다. 체크는 바로 저장되어 설비관리자가 진행률을 봅니다. ' + STORE_NOTE)

        elif wv == 'report':
            if r['상태'] != '진행 중':
                st.info('진행 중인 작업만 완료 보고를 쓸 수 있습니다.', icon=':material/info:')
            else:
                st.markdown(f'#### {nm(sid)} · 완료 보고')
                st.caption(f'작업 #{tid} · {r["title"]} · 결과와 수행 내용을 남겨야 완료됩니다.')
                with st.form(f'complete_{tid}'):
                    c1, c2, c3 = st.columns([1.2, 1, 1])
                    outcome = c1.segmented_control('작업 결과', tasks.OUTCOMES, default='이상 발견', required=True)
                    judged = c2.segmented_control('점검으로 판단한 부품', tasks.JUDGED, default='구분 못함', required=True,
                                                  format_func=lambda x: PART_NAMES.get(x, x))
                    confirmed = c3.segmented_control('교체 뒤 확인된 원인 부품', tasks.CONFIRMED, default='아직 모름', required=True,
                                                     format_func=lambda x: PART_NAMES.get(x, x))
                    # 실제 조치 (2026-10-21): 현장에서는 지시와 다르게 할 수 있다. 재고는 실제 조치대로 뺀다
                    action = st.segmented_control('실제 조치 (지시: 두 부품 교체)', tasks.ACTIONS, default=tasks.ACTIONS[0], required=True)
                    action_why = st.text_input('지시와 다르게 한 이유 (실제 조치가 두 부품 교체가 아니면 필수)', max_chars=tasks.LIMITS['why'],
                                               placeholder='예: 열어 보니 밸브만 마모, 진동장치는 정상')
                    details = st.text_area('수행 내용 (필수)', max_chars=tasks.LIMITS['details'],
                                           placeholder='예: 진동장치 베어링 마모 확인, 두 부품 교체 후 신호 정상')
                    # 완료 시각 = 제출하는 지금 (흐르는 시각, 2026-10-21 8차). 재고·점검 집계가 이 시각을 쓴다
                    st.caption(f'완료 시각은 제출하는 순간의 시각으로 남습니다 (지금 {clock_s(now_f)})')
                    permit_after = st.checkbox('작업 허가서 사후 처리 완료 (즉시 조치는 필수)') if urgent_job else True
                    go_done = st.form_submit_button('완료 보고 제출', icon=':material/task_alt:', type='primary')
                if go_done:
                    if not permit_after:
                        st.error('즉시 조치 작업은 작업 허가서를 사후 처리한 뒤 완료 보고를 제출하세요.', icon=':material/error:')
                    else:
                        done_h = flow_hour()
                        try:
                            done = TASKS.complete(tid, done_h, outcome, details, judged, confirmed, action, action_why)
                        except ValueError as e:
                            st.error(str(e))
                        else:
                            if done:
                                if urgent_job:
                                    TASKS.set_checks(tid, json.dumps({**saved, 'permit_after': True}, ensure_ascii=False))
                                LOG.record('작업 완료', split, sid, done_h, WORKER, option=action,
                                           note=details + (f' · 지시와 다름: {action_why}' if action != tasks.ACTIONS[0] else '')
                                           + (' · 작업 허가서 사후 처리 완료' if urgent_job else ''))
                                LOG.record('점검 결과', split, sid, done_h, WORKER, judged=judged, confirmed=confirmed, note=outcome)
                                w_go()
                                st.toast(f'{nm(sid)}: 완료 보고', icon=':material/task_alt:')
                                st.rerun()
                            else:
                                st.warning('이미 완료된 작업입니다.')

        else:   # 작업 중단 화면
            if r['상태'] not in ('대기', '진행 중'):
                st.info('이미 완료됐거나 중단한 작업입니다.', icon=':material/info:')
            else:
                st.markdown(f'#### {nm(sid)} · 작업 중단')
                st.caption(f'작업 #{tid} · {r["title"]} · 중단하면 설비관리자에게 알리고, 그 설비는 다시 배정할 수 있게 됩니다.')
                with st.form(f'stop_{tid}'):
                    stop_why = st.text_area('중단 사유 (필수)', max_chars=tasks.LIMITS['why'],
                                            placeholder='예: 부품이 맞지 않아 교체 못 함, 안전 문제로 작업 허가 보류')
                    go_stop = st.form_submit_button('작업 중단 제출', icon=':material/stop_circle:', type='primary')
                if go_stop:
                    try:
                        stopped = TASKS.stop(tid, flow_hour(), stop_why)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        if stopped:
                            LOG.record('작업 중단', split, sid, flow_hour(), WORKER, note=f'#{tid} 사유: {stop_why.strip()}')
                            w_go()
                            st.toast(f'{nm(sid)}: 작업 중단 · 설비관리자에게 알렸습니다', icon=':material/stop_circle:')
                            st.rerun()
                        else:
                            st.warning('이미 완료됐거나 중단한 작업입니다.')


# 1. 설비관리자: 홈(설비 상태 / 작업 지시 현황) → 목록 → 설비의 근거·비용·배정 → 하위 페이지

else:
    if not bundle.get('dss'):
        st.warning('이 모델 파일에는 의사결정 정보가 없습니다. 터미널에서 `python analysis/05_predict_test.py`로 다시 만드세요.')
        st.stop()
    # 하위 페이지 (노션처럼): 설비 상세·정책·비용·부품 손익·모델 정보는 목록에서 눌러야 들어가고, 위의 경로로 돌아온다
    sub = st.session_state.get('msub')
    if st.session_state.pop('scroll_top', False):    # 페이지를 바꾸면 맨 위(경로)부터 보이게
        scroll_to_top()
    if sub in MANAGER_TABS:
        with st.container(horizontal=True, vertical_alignment='center', gap='small'):
            st.button('정비 의사결정', icon=':material/arrow_back:', type='tertiary', key='sub_back', on_click=close_sub)
            st.markdown(f'/ &nbsp;{ICONS[sub]} **{nm(st.session_state.get("sid", "")) if sub == "설비 상세" else sub}**')
        page = MANAGER_TABS[sub]
    else:
        A = assumptions()
        t = status_table()
        W = work_plan(A, t)
        dec, cfg, costs = W['dec'], W['cfg'], W['costs']
        dec_i = dec.set_index('설비 (시나리오)') if len(dec) else pd.DataFrame()
        tf = TASKS.frame(split, now_f)
        busy = TASKS.blocked_sids(split, now_f)      # 배정을 막는 작업이 있는 설비 (재생에서 미래에 만든 작업 포함, 배정 규칙과 같음)
        js = LOG.job_state(split, now_f, MNT_START)
        load = team_load(tf)
        # 배정할 경보: 열린 작업이 없고, 보류하지 않았고, 이번 운전 구간에서 작업을 마치지 않은 경보 설비
        #   (재생 데이터에는 우리가 한 교체가 반영되지 않아 경보가 남으므로, 완료한 설비는 정비가 시작될 때까지 다시 올리지 않는다)
        queue = dec[~dec['설비 (시나리오)'].isin(busy) & ~dec['설비 (시나리오)'].map(lambda x: js.get(x)).isin(['보류', '작업 완료'])] if len(dec) else dec
        if len(queue):    # 즉시 조치 먼저, 그다음 정비팀 배정 순서
            queue = queue.assign(_u=(~queue['제약'].str.startswith('즉시')).astype(int)).sort_values(['_u', '순위'], na_position='last')
        short = t['상태'].str.split(' (', regex=False).str[0]
        tf_last = tf.sort_values('id').groupby('sid').tail(1).set_index('sid') if len(tf) else pd.DataFrame()

        def order_state(x):
            """설비 하나의 작업 지시 상태: 배정 전 · 대기 · 진행 중 · 완료 · 보류 (경보가 아니면 작업 없음)"""
            if len(tf_last) and x in tf_last.index:
                return tf_last.loc[x, '상태']
            if js.get(x) == '보류':
                return '보류'
            return '배정 전' if short[x] == '경보' else '작업 없음'

        ORDER_BADGE = {'배정 전': 'red', '대기': 'blue', '진행 중': 'blue', '완료': 'green', '보류': 'orange', '중단': 'orange', '작업 없음': 'gray'}

        def manwon(x):
            return f'{x:,.0f}만 원' if costs.unit == '만 원' else f'{x:,.0f}점'

        def card(label, value, key, on_click, args=(), help_=None):
            """누르면 다음 화면으로 넘어가는 지표 칸 (칸 어디를 눌러도 된다)"""
            st.button(f'{label} :material/chevron_right:\n\n**{value}**', key=f'card_{key}', width='stretch',
                      on_click=on_click, args=args, help=help_)

        st.html(CARD_CSS)

        def decision_panel(sid):
            """설비 하나의 정비 판단 (2026-10-21 6차 피드백)
            조치 안내 칸(안내 · 분석 근거 · 근거(온톨로지 그림) | 고장 예상까지)
            → 정비 판단 상자: 즉시 정비 · 다음 정기 교체 · 사후정비 칸마다 비용과 담당 팀 · 배정 → 작업 내용 바꾸기 · 보류 · 부품 발주
            → 설명 → 접힌 칸(비용 가정 · 비용 계산 근거)"""
            srow = t.loc[sid]
            sstage = srow['상태'].split(' (')[0]
            st.subheader(nm(sid), icon=':material/precision_manufacturing:')
            st.markdown(' '.join(x for x in [
                f':{STAGE_DOT.get(sstage, "gray")}-badge[{sstage}]',
                f':{ORDER_BADGE.get(order_state(sid), "gray")}-badge[작업 지시: {order_state(sid)}]',
                f'24시간 내 고장 위험 **{srow["위험 확률"]:.0%}**' if pd.notna(srow['위험 확률']) else '',
                f':gray[시나리오 {sid} · {clock(now)} 기준]'] if x))
            # 이미 손댄 설비는 먼저 그 사실을 보여 준다 (재생 데이터에는 우리 교체가 반영되지 않아 경보가 그대로 남기 때문)
            job_ = '작업 있음' if sid in busy else js.get(sid)
            JOB_NOTE = {'작업 완료': '작업 완료 · 재가동 뒤 새 예측이 나올 때까지 경보 표시가 남습니다',
                        '작업 있음': '작업이 이미 배정돼 있습니다 · 작업 지시 현황에서 확인',
                        '작업 시작': '작업 진행 중', '보류': '보류한 설비 · 다시 배정하려면 아래에서 배정', '승인': '승인됨 · 작업 배정 전',
                        '작업 중단': '작업자가 중단한 설비 · 사유는 작업 지시 현황에서 확인, 다시 배정할 수 있습니다'}
            if job_ in JOB_NOTE:
                st.markdown(f':green-badge[:material/task_alt: {JOB_NOTE[job_]}]' if job_ == '작업 완료'
                            else f':orange-badge[:material/stop_circle: {JOB_NOTE[job_]}]' if job_ == '작업 중단'
                            else f':blue-badge[:material/info: {JOB_NOTE[job_]}]')
            alarm = sid in dec_i.index
            e = b = r_ = None
            if alarm:
                r_ = dec_i.loc[sid]
                e = r_['_e']
                b = e['best']
                p24 = fail_prob(W['curve'], e['elapsed'], 24)
                now_opt = next(o for o in e['opts'] if o['선택지'].startswith('지금'))
            # 조치 안내 칸 (알림 상자처럼 꾸민 테두리 칸: 알림 상자 st.error에는 토글을 넣을 수 없어서, 2026-10-21)
            tone, g_title, g_icon, g_body = guide_content(srow)
            remain = remaining_median(W['curve'], e['elapsed']) if alarm else None
            with st.container(border=True, key=f'guide_{tone}'):
                g_left, g_right = st.columns([3.2, 1], vertical_alignment='center') if alarm else (st.container(), None)
                with g_left:
                    st.markdown(f'**:{tone}[{g_icon} {g_title}]**')
                    st.markdown(g_body)
                    if sstage == '주의':
                        pr = procurement(cfg, W['stock'], [sid]).iloc[0]
                        st.caption(f'부품: {pr["발주"]} (재고 {pr["재고"]})')
                    # 분석 근거 (설비 페이지 내용): 그래프가 많아 펼칠 때만 그린다 (on_change='rerun' → 접혀 있으면 계산하지 않음)
                    eq_box = st.expander('분석 근거', icon=':material/monitoring:', key=f'eq_detail_{sid}', on_change='rerun')
                    if eq_box.open:
                        with eq_box:
                            equipment_detail(sid, t, nested=True)
                    # 근거 사슬 (온톨로지): 그림 + 번호 목록. 만들다 실패해도 이 칸만 알린다 (2026-10-21 배포 오류 재발 방지)
                    on_box = st.expander('근거: 경보 → 예측 → 신호 → 정비 결정 (온톨로지)', icon=':material/account_tree:',
                                         key=f'onto_{sid}', on_change='rerun')
                    if on_box.open:
                        with on_box:
                            try:
                                st.graphviz_chart(ontology_dot(sid, srow, e, costs), width='stretch',
                                                  alt=f'{nm(sid)}의 온톨로지 연결: 설비, 단계, 예측, 신호, 정비 결정, 선택지')
                                tr = ontology.trace(ontology_graph(A, t), sid, with_cost=alarm)
                            except Exception as err:
                                print(f'[dashboard] 근거 사슬 실패: {type(err).__name__}: {err}', file=sys.stderr)
                                tr = None
                            if tr is None:
                                st.caption('근거를 불러오지 못했습니다. 앱을 다시 시작(Reboot)하면 대부분 해결됩니다. 비용과 배정은 그대로 쓸 수 있습니다.')
                            else:
                                unit_ = '만 원' if costs.unit == '만 원' else '점'   # 비용 단위는 화면에서 붙인다 (모듈 버전과 상관없게)
                                tr = [(k, re.sub(r'(기대 총비용 [\d,]+)(?![\d,]|만 원|점)', rf'\1{unit_}', v)) for k, v in tr]
                                st.markdown('\n'.join(f'{i}. **{k}** · {v}' for i, (k, v) in enumerate(tr, 1)))
                                st.caption('설비·예측·신호·경보·정비 결정·정기 교체 계획을 하나의 그래프(온톨로지, OWL/RDF)로 잇고, 그 연결을 그림과 '
                                           '목록으로 보여 줍니다. 판단은 경보 기준과 의사결정 규칙이 하고, 그래프는 그 근거를 이어 보여 줍니다.')
                if g_right is not None:
                    with g_right:
                        # 1초씩 줄어드는 타이머 (2026-10-21 8차): 정각의 추정값(1시간 요약본)에서 흐르는 시각만큼 뺀다. 정각마다 새 추정으로 바뀐다
                        if remain is None:
                            st.metric('고장 예상까지', '—')
                        else:
                            st.html('<div style="font-size: 0.875rem; margin-bottom: 4px;">고장 예상까지</div>'
                                    f'<div style="font-size: 2.25rem; line-height: 1.2; font-variant-numeric: tabular-nums;">'
                                    f'<span {tick_attrs(now + remain)}>{hms(max(0.0, now + remain - now_f))}</span></div>'
                                    '<div style="font-size: 0.8rem; opacity: .65; margin-top: 4px;">추정 · 정각마다 다시 계산</div>')
            # 정비 판단 상자 (2026-10-21): 즉시 정비 · 다음 정기 교체 · 사후정비를 같은 기준으로 비교하고 그 칸에서 배정
            #   같은 기준 = 같은 비교 기간(지금 ~ 다음 정기 정지, 계획이 없으면 시연 끝 72h) 안의 기대 비용,
            #   같은 비용 항목(부품 · 인건비(인·시, 정규 근무 밖 1.5배) · 정지 생산 차질 · 고장 손실). 칸마다 고른 작업자 수로 다시 계산
            rec_kind, vals = None, {}
            with st.container(border=True):
                st.markdown(f'##### {nm(sid)} · 정비 판단')
                if alarm:
                    existing = TASKS.blocking(split, sid, now_f)
                    can_assign = existing is None and job_ != '작업 완료'
                    urgent = r_['제약'].startswith('즉시')
                    base_wait = cfg.crew_wait_h + r_['_slot']
                    now_start = 0.0 if urgent else max(0.0, now_opt['정비 완료까지(h)'] - costs.replace_h)
                    horizon = float(e['wait_h']) if pd.notna(e['wait_h']) else max(1.0, 72.0 - now)
                    ph = A['f_labor_ph']

                    def card_values(kind, n):
                        """칸 하나의 기대 비용 (n명이 맡을 때). 반환: cost, p(기간 안 또는 끝나기 전 고장 확률), sub, start"""
                        if kind == 'run':
                            c_n = crew_costs(costs, n, now, ph)
                            p_h = fail_prob(W['curve'], e['elapsed'], horizon)
                            return {'cost': p_h * c_n.failure_loss, 'p': p_h, 'start': 0.0,
                                    'sub': f'{horizon:.0f}시간 안 고장 확률 {p_h:.0%} · 그 뒤에도 고장 위험 남음 ({1 - p_h:.0%})',
                                    'fix_h': c_n.repair_h}
                        if kind == 'now':
                            c_n = crew_costs(costs, n, now + now_start, ph)
                            o = next(o for o in options(e['wait_h'], W['curve'], replace(cfg, crew_wait_h=base_wait, costs=c_n), e['elapsed'],
                                                        stock=W['stock'][sid], plan_part=e['plan_part']) if o['선택지'].startswith('지금'))
                            return {'cost': o['기대 총비용'], 'p': o['고장 확률'], 'start': now_start,
                                    'sub': f'{o["선택지"]} · 교체 {c_n.replace_h:.1f}시간'}
                        # 다음 정기 교체: 엔진 조건(최대 대기·부품·정비팀) 밖이거나 계획이 없으면 비교용으로만 계산한다
                        has_plan = pd.notna(e['wait_h'])
                        c_n = crew_costs(costs, n, now + (e['wait_h'] if has_plan else PLAN_GUESS_H), ph)
                        cfg_n = replace(cfg, crew_wait_h=base_wait, costs=c_n)
                        o = next((o for o in options(e['wait_h'], W['curve'], cfg_n, e['elapsed'], stock=W['stock'][sid],
                                                     plan_part=e['plan_part']) if not o['선택지'].startswith('지금')), None)
                        if o is not None:
                            return {'cost': o['기대 총비용'], 'p': o['고장 확률'], 'start': float(e['wait_h']), 'sub': o['선택지'], 'note': ''}
                        loose = replace(cfg_n, max_wait_h=1e9, prep_h=0.0)
                        guesses = tuple(d for d in (PLAN_GUESS_H, 2 * PLAN_GUESS_H) if d > base_wait) or (int(np.ceil(base_wait)) + PLAN_GUESS_H,)
                        for d_ in ((None,) if has_plan else ()) + guesses:
                            po_ = [o for o in options(e['wait_h'] if d_ is None else np.nan, W['curve'], loose, e['elapsed'], delay_h=d_,
                                                      stock=W['stock'][sid], plan_part=e['plan_part']) if not o['선택지'].startswith('지금')]
                            if po_:
                                note = (f'다음 정기 교체({e["wait_h"]:.0f}시간 뒤)는 최대 대기·부품·정비팀 조건 밖이라 비교만 합니다. ' if d_ is None else
                                        (f'다음 정기 교체({e["wait_h"]:.0f}시간 뒤)에는 부품을 맞출 수 없어' if has_plan else '정기 교체 계획이 없어')
                                        + f' 다음 정기 교체 칸은 {d_}시간 뒤 정지를 가정해 비교만 합니다. ')
                                return {'cost': po_[0]['기대 총비용'], 'p': po_[0]['고장 확률'], 'note': note,
                                        'start': float(e['wait_h']) if d_ is None else float(d_),
                                        'sub': (f'{e["wait_h"]:.0f}시간 뒤 · 조건 밖이라 비교만' if d_ is None else
                                                ('정기 교체 때 부품을 맞출 수 없음' if has_plan else '정기 교체 계획 없음') + f' · {d_}시간 뒤 정지 가정')}
                        return {'cost': None, 'p': np.nan, 'start': None, 'sub': '계산할 수 없음', 'note': ''}

                    absent_now = TASKS.absent_at(now_f)
                    open_ = tf[~tf['상태'].isin(['완료', '중단'])]
                    busy_people = {m for v in open_['members'] for m in str(v).split(',') if m}
                    team0 = list(tasks.TEAMS)[list(tasks.TEAMS).index(min(load, key=load.get))]
                    KIND = [('now', '즉시 정비', '경보를 보고 지금 두 부품을 교체'),
                            ('plan', '다음 정기 교체', '정해 둔 정기 교체(계획 정지) 때 두 부품을 함께 교체'),
                            ('run', '사후정비', '지금은 운전을 계속하고, 고장 나면 복구 (비교 기간 안 고장 확률 × 고장 손실)')]
                    picks = {}
                    for kind, _, _ in KIND:     # 칸마다 고른 팀·작업자 (위젯보다 먼저 읽어서 금액을 바로 맞춘다)
                        tk = f'team_{kind}_{sid}'
                        team_ = st.session_state.get(tk, team0)
                        avail = [m for m in tasks.TEAMS[team_] if m not in absent_now]
                        mk = f'mem_{kind}_{sid}_{team_}'
                        if mk not in st.session_state:
                            st.session_state[mk] = avail[:4]          # 기본: 반장부터 출근한 사람 4명
                        st.session_state[mk] = [m for m in st.session_state[mk] if m in avail]
                        picks[kind] = (tk, team_, avail, mk, st.session_state[mk])
                        vals[kind] = card_values(kind, max(1, len(st.session_state[mk])))
                    priced = {k: v['cost'] for k, v in vals.items() if v['cost'] is not None}
                    rec_kind = min(priced, key=priced.get) if priced else None
                    for col, (kind, label, help_) in zip(st.columns(3, border=True), KIND):
                        v = vals[kind]
                        tk, team_, avail, mk, sel = picks[kind]
                        with col:
                            st.metric(label + (' · 추천' if kind == rec_kind else ''), manwon(v['cost']) if v['cost'] is not None else '—', help=help_)
                            st.caption(v['sub'])
                            if pd.notna(v['p']) and v['p'] > cfg.safety_max:
                                st.markdown(f':orange-badge[:material/warning: 안전 기준 초과: 고장 확률 {v["p"]:.0%} > {cfg.safety_max:.0%}]')
                            if can_assign and v['cost'] is not None:
                                st.selectbox('담당 팀', list(tasks.TEAMS), key=tk, index=list(tasks.TEAMS).index(team_),
                                             format_func=lambda k: f'{k} · 열린 작업 {load[k]}건')
                                st.multiselect('작업자', avail, key=mk, placeholder='작업자를 고르세요',
                                               format_func=lambda m: m + (' (반장)' if m in tasks.LEADERS.values() else '')
                                               + (' (작업 중)' if m in busy_people else ''))
                                # 작업책임자 (2026-10-21 7차): 고른 작업자 중 한 명, 정비 중 10대 안전철칙을 확인한다. 기본 = 고른 사람 중 반장, 없으면 첫 사람
                                lk = f'lead_{kind}_{sid}_{team_}'
                                if sel:
                                    if st.session_state.get(lk) not in sel:
                                        st.session_state[lk] = tasks.pick_leader(','.join(sel), '')
                                    st.selectbox('작업책임자', sel, key=lk, help='정비 중 10대 안전철칙을 확인·체크하는 사람 (고른 작업자 중)')
                                off = [f'{m}({absent_now[m]})' for m in tasks.TEAMS[team_] if m in absent_now]
                                if off:
                                    st.caption('부재: ' + ', '.join(off))
                                if 0 < len(sel) < 4:
                                    st.markdown(f':orange-badge[:material/groups: 인원 부족 (4명 기준) · '
                                                f'{"고장 복구" if kind == "run" else "교체"} '
                                                f'{(WORK_REPAIR if kind == "run" else WORK_REPLACE) / len(sel):.1f}시간]')
                                title_ = {'now': f'{sid} 지금 두 부품 교체', 'plan': f'{sid} 다음 정기 교체 때 두 부품 교체',
                                          'run': f'{sid} 사후정비: 운전 계속 · 고장 나면 복구'}[kind]
                                prio_ = {'now': '긴급' if urgent else '높음', 'plan': '높음', 'run': '보통'}[kind]
                                st.button('작업자 화면으로 배정', icon=':material/send:', key=f'assign_{kind}_{sid}', width='stretch',
                                          type='primary' if kind == rec_kind else 'secondary', disabled=not sel, on_click=assign_from_card,
                                          args=(sid, tk, mk, lk, title_, prio_, now + (v['start'] or 0.0),
                                                f'{sid} · {srow["상태"]} · {label} 선택 · 기대 비용 {manwon(v["cost"])}', label))
                    if existing:
                        when = '' if existing['created_h'] <= now_f else f' ({clock(existing["created_h"])}에 만든 작업)'
                        st.warning(f'이미 작업 #{int(existing["id"])} · {existing["title"]} ({existing["team"]}){when}이 있어 다시 배정하지 않습니다. '
                                   '아래 "작업 수정하기"에서 고치거나 배정을 취소할 수 있습니다.', icon=':material/block:')
                        st.button('작업 수정하기', icon=':material/edit:', key='edit_from_eq', on_click=open_task_edit,
                                  args=(int(existing['id']),))
                    elif can_assign:
                        with st.expander('작업 내용 바꾸기 (점검 가이드 · 추천 이유 덧붙이기)'):
                            st.text_area('점검 가이드', value=tasks.DEFAULT_GUIDE, height=140, max_chars=tasks.LIMITS['guide'],
                                         key=f'guide_txt_{sid}')
                            st.text_input('추천 이유에 덧붙일 말', max_chars=200, key=f'reason_txt_{sid}')
                        st.caption('칸에서 배정하면 그 선택지와 작업자로 작업이 만들어지고, 그 팀 작업자 화면의 "지금 할 일"에 바로 나타납니다. '
                                   f'비교 기간: 지금 ~ {clock(now + horizon)} ({"다음 정기 정지" if pd.notna(e["wait_h"]) else "정기 정지 계획이 없어 시연 끝"}까지)')
                    order_buttons(sid, sstage, True, b['선택지'], cfg, W)
                else:
                    assign_box(sid, srow, sstage, job_, dec_i, W, cfg, load, border=False)
            # 상자 아래: 설명 → 접힌 칸
            if alarm:
                lab = {'now': '즉시 정비', 'plan': '다음 정기 교체', 'run': '사후정비'}
                if rec_kind is not None:
                    rv, run_v = vals[rec_kind], vals['run']
                    lines_ = [f'- 고장 나면 드는 돈(정지·생산 차질·복구, 사후정비 칸 작업자 기준): **{manwon(run_v["cost"] / run_v["p"]) if run_v["p"] > 0 else "—"}**',
                              f'- 추천({lab[rec_kind]})의 기대 비용: **{manwon(rv["cost"])}** · '
                              + (f'비교 기간 안 고장 확률 **{rv["p"]:.0%}**' if rec_kind == 'run' else f'끝나기 전에 고장 날 확률 **{rv["p"]:.0%}**')
                              + f' (안전 기준 {cfg.safety_max:.0%})']
                    if rec_kind != 'run' and run_v['cost'] > rv['cost']:
                        lines_.append(f'- 사후정비(기대 비용)보다 **{manwon(run_v["cost"] - rv["cost"])}** 적게 듭니다')
                    if rec_kind == 'run':
                        lines_.append('- 이 설비는 지금 세우는 것보다 **운전을 계속하고 고장 나면 복구하는 쪽**의 기대 비용이 낮습니다. '
                                      '다만 비교 기간 뒤에도 고장 위험이 남고, 안전 기준을 넘으면 현장 판단이 먼저입니다.')
                    st.markdown('\n'.join(lines_))
                with st.expander('이 설비 비용 가정', icon=':material/tune:'):
                    st.markdown(f'- 지금 두 부품 교체 비용: **{manwon(now_opt["정비 비용"])}** (교체 {costs.replace_h:.0f}시간)\n'
                                f'- 계획 밖 정지 1시간 생산 차질: **{manwon(costs.downtime_h)}**\n'
                                f'- 고장 시 손실: **{manwon(costs.failure_loss)}**\n'
                                f'- 안전 기준(완료 전 고장 확률): **{cfg.safety_max:.2f}** · 다음 정기 교체: **{r_["다음 정기 교체"]}**\n'
                                f'- 예비 재고: **{stock_text(W["stock"][sid])}** · 첫 경보 뒤 **{e["elapsed"]:.0f}시간** 지남')
                    st.caption('지금 가정값(기본값)으로 계산했습니다.')
                with st.expander('비용 계산 근거', icon=':material/calculate:'):
                    st.markdown('  \n'.join(f'- **{o["선택지"]}**: 고장 확률 P {o["고장 확률"]:.2f} · {o["근거"]}' for o in e['opts'])
                                + f'  \n- **24시간 그냥 두면**: 앞으로 24시간 안 고장 확률 {p24:.2f} × 고장 시 손실 = {manwon(p24 * costs.failure_loss)}'
                                + f'  \n- 경보가 켜진 뒤 실제 고장까지 걸린 시간(학습 세트)으로 보면, 이 설비가 앞으로 24시간 안에 고장 날 확률은 '
                                  f'{p24:.0%}입니다. 위의 "24시간 내 고장 위험"(모델 예측)과는 계산 방법이 다릅니다.'
                                + (f'  \n- **고장 예상까지 {hms(remain)}**: 같은 분포에서 경보 뒤 {e["elapsed"]:.0f}시간 동안 고장이 없었다는 조건으로 '
                                   '고장 확률이 절반이 되는 시간(중앙값)입니다. 확정된 고장 시각이 아니라 추정입니다.' if remain is not None else '')
                                + f'  \n- P는 학습 세트에서 경보 뒤 실제 고장까지 걸린 시간 분포(경보 뒤 {e["elapsed"]:.0f}시간 동안 고장 없음 조건).')
                    st.caption(f'기대 총비용 = (1 − P) × (정비 비용 + 생산 차질) + P × 고장 시 손실 · P = 정비를 마치기 전에 고장 날 확률. '
                               f'추천 = 완료 전 고장 확률이 안전 기준({cfg.safety_max:.2f}) 이하인 선택지 중 기대 총비용이 가장 낮은 것. '
                               + ('어떤 선택지도 안전 기준을 못 지켜 가장 빨리 할 수 있는 조치를 권합니다. ' if e['urgent'] else '')
                               + vals.get('plan', {}).get('note', '') + '최소 비용과 실제 권장 조치는 다를 수 있습니다. 안전·현장 절차 우선.')

        def equipment_list(ids, key_prefix, on_pick=pick_equipment, selected=None, compact=False):
            """설비 목록: 버튼 한 줄 = 설비 하나 (상태 색 점 · 이름 · 상태 · 24시간 내 고장 위험 · 작업 지시 상태 · 시나리오 번호).
            compact: 설비 상태 칸 안 목록 (첫 줄 이름, 둘째 줄부터 나머지. 칸 폭에 맞춰 줄바꿈하고 정보는 줄이지 않는다)"""
            if not ids:
                st.caption('해당하는 설비가 없습니다.')
            for x in ids:
                stg = short[x]
                p_ = t.loc[x, '위험 확률']
                o_ = order_state(x)
                risk_ = f'24시간 내 고장 위험 {p_:.0%}' if pd.notna(p_) else ''
                badge_ = f':{ORDER_BADGE.get(o_, "gray")}-badge[작업 지시: {o_}]'
                if compact:
                    label = f'**:{STAGE_DOT.get(stg, "gray")}[●] {nm(x)}** ' + ' · '.join(v for v in [f':gray[{x}]', stg, risk_, badge_] if v)
                else:
                    label = ' · '.join(v for v in [f':{STAGE_DOT.get(stg, "gray")}[●] **{nm(x)}**', stg, risk_, badge_, f':gray[{x}]'] if v)
                st.button(label, key=f'{key_prefix}_{x}', width='stretch', on_click=on_pick, args=(x,),
                          type='primary' if x == selected else 'secondary')

        mv = st.session_state.get('mv')
        sid_ = st.session_state.get('decision_sid')
        picked = bool(st.session_state.get('nav_pick')) and sid_ in t.index
        if mv == 'assign':
            # 배정할 시나리오 목록 → 고르면 바로 아래에 근거·비용 → 배정·보류
            st.button(ORDERS, icon=':material/arrow_back:', type='tertiary', key='back_assign', on_click=go_home, args=(ORDERS,))
            st.subheader(f'배정할 경보 {len(queue)}건 · 배정할 시나리오 고르기', icon=':material/assignment_late:')
            q_ids = queue['설비 (시나리오)'].tolist() if len(queue) else []
            if not q_ids:
                st.success('지금 배정할 경보가 없습니다.', icon=':material/check_circle:')
            else:
                st.caption('즉시 조치 → 정비팀 배정 순서 · 누르면 바로 아래에 그 설비의 근거와 비용이 열립니다')
                qi = queue.set_index('설비 (시나리오)')
                for x in q_ids:
                    rq = qi.loc[x]
                    label = ' · '.join(v for v in [
                        f':red[●] **{nm(x)}**', ':red-badge[즉시 조치]' if rq['제약'].startswith('즉시') else '',
                        f'추천: {rq["추천"]}', f'정비팀 순서 {int(rq["순위"])}번' if pd.notna(rq['순위']) else '정기 교체 때',
                        f':gray[{x}]'] if v)
                    st.button(label, key=f'q_{x}', width='stretch', on_click=open_assign, args=(x,),
                              type='primary' if picked and x == sid_ else 'secondary')
            if picked:
                st.divider()
                decision_panel(sid_)
            sid = sid_ if picked else (q_ids[0] if q_ids else t.index[0])
        elif picked:
            # 설비 화면 (상태별 설비 목록·검색에서 고른 설비)
            sid = sid_
            st.button(f'{mv} 설비 목록' if mv in ('경보', '주의', '정상') else '정비 의사결정', icon=':material/arrow_back:', type='tertiary',
                      key='eq_back', on_click=leave_equipment)
            decision_panel(sid)
        elif mv in ('경보', '주의', '정상'):
            # 상태별 설비 목록 (설비 상태의 지표 칸에서 들어옴)
            ids_ = t.index[short == mv].tolist()
            sid = ids_[0] if ids_ else t.index[0]
            st.button(EQUIP, icon=':material/arrow_back:', type='tertiary', key='back_stage', on_click=go_home, args=(EQUIP,))
            with st.container(horizontal=True, vertical_alignment='center'):
                st.subheader(f'{mv} 설비 {len(ids_)}대', icon=':material/format_list_bulleted:')
                st.space('stretch')
                st.button(f'{ORDERS} 보기', icon=':material/assignment:', key='to_orders', on_click=go_home, args=(ORDERS,))
            if mv == '경보':
                n_new = sum(order_state(x) == '배정 전' for x in ids_)
                st.caption(f'위험한 순서 · 작업 지시가 아직 없는 설비 {n_new}대 · 누르면 그 설비의 근거·비용과 배정이 열립니다')
            else:
                st.caption('위험한 순서 · 누르면 그 설비 화면이 열립니다')
            equipment_list(ids_, 'st')
        else:
            # 설비관리자 홈: 설비 상태 / 작업 지시 현황
            st.session_state.pop('nav_pick', None)
            sid = sid_ if sid_ in t.index else t.index[0]
            st.subheader('정비 의사결정', icon=':material/balance:')
            st.session_state.setdefault('mhome', EQUIP)    # 기본값은 세션에서만 (go_home이 이 값을 바꾼다)
            home = st.segmented_control('홈 범주', [EQUIP, ORDERS, STAFF], key='mhome', required=True, label_visibility='collapsed',
                                        format_func=lambda x: {EQUIP: f':material/sensors: {EQUIP}', ORDERS: f':material/assignment: {ORDERS}',
                                                               STAFF: f':material/badge: {STAFF}'}[x])
            if home == EQUIP:
                # 칸마다 그 상태의 설비 전부를 칸 안 스크롤 목록으로 (2026-10-21 강사 피드백). 칸 제목을 누르면 넓은 목록 화면
                st.html('<style>'
                        '[class*="st-key-stagebox_"] [class*="st-key-card_"] button { border: none; background: transparent; min-height: auto; '
                        'padding: 2px 4px 6px; }'
                        '[class*="st-key-hl"] button { justify-content: flex-start; padding: 8px 12px; }'
                        '[class*="st-key-hl"] button > div, [class*="st-key-hl"] button [data-testid="stMarkdownContainer"] '
                        '{ width: 100%; justify-content: flex-start; text-align: left; }'
                        '[class*="st-key-hl"] button strong { display: block; margin-bottom: 2px; }'
                        '</style>')
                HELP = {'경보': '24시간 안 고장 위험. 제목을 누르면 넓은 목록 화면', '주의': '열화 감지: 점검·부품 준비. 제목을 누르면 넓은 목록 화면',
                        '정상': '제목을 누르면 넓은 목록 화면'}
                for col, (stage_, k_) in zip(st.columns(3), [('경보', 'a'), ('주의', 'c'), ('정상', 'n')]):
                    ids_s = t.index[short == stage_].tolist()
                    with col.container(border=True, key=f'stagebox_{k_}'):
                        card(stage_, len(ids_s) if (stage_ != '주의' or CAUTION) else '—', stage_, open_view, (stage_,), HELP[stage_])
                        with st.container(height=440, border=False):
                            equipment_list(ids_s, f'hl{k_}', compact=True)
            elif home == ORDERS:
                k1, k2, k3, k4, k6, k5 = st.columns(6)
                with k6.container(border=True):
                    st.metric('중단', int((tf['상태'] == '중단').sum()) if len(tf) else 0, help='작업자가 현장 상황 때문에 끝내지 못하고 중단한 작업')
                with k1:
                    card('배정할 경보', len(queue), 'assign', open_view, ('assign',), '경보인데 아직 작업이 없는 설비 (보류 제외). 누르면 배정할 시나리오 목록')
                with k2.container(border=True):
                    st.metric('진행 중', int((tf['상태'] == '진행 중').sum()))
                with k3.container(border=True):
                    st.metric('지연', int((tf['구분'] == '지연').sum()), help='예정 시각이 지났는데 시작 전인 작업')
                with k4.container(border=True):
                    st.metric('완료', int((tf['상태'] == '완료').sum()))
                with k5.container(border=True):
                    st.metric('정비 중', int(short.isin(['정비 중', '정비 후 예측 대기']).sum()), help='정비 중이거나 재가동 뒤 예측을 기다리는(대기) 설비')
                # 배정할 경보 카드: 즉시 조치 → 정비팀 배정 순서의 첫 설비
                with st.container(border=True):
                    if len(queue):
                        r = queue.iloc[0]
                        fsid, fb_ = r['설비 (시나리오)'], r['_e']['best']
                        held = sum(v == '보류' for v in js.values())
                        st.caption(f'지금 먼저 볼 설비 · 배정할 경보 {len(queue)}건' + (f' · 보류 {held}건' if held else ''))
                        with st.container(horizontal=True, vertical_alignment='center'):
                            st.markdown(f'### {nm(fsid)} — {fb_["선택지"]}')
                            st.space('stretch')
                            st.button('이 설비 보기', icon=':material/arrow_forward:', type='primary', key='cta_open',
                                      on_click=open_assign, args=(fsid,))
                        st.markdown(' '.join(x for x in [
                            ':red-badge[:material/emergency: 즉시 조치]' if r['제약'].startswith('즉시') else '',
                            f':gray-badge[정비팀 순서 {int(r["순위"])}번]' if pd.notna(r['순위']) else ':gray-badge[정기 교체 때]',
                            f':gray[시나리오 {fsid}]'] if x))
                    else:
                        st.markdown('##### :material/check_circle: 지금 배정할 경보가 없습니다')
                    st.caption(f'열린 작업 {len(busy)}건 · 팀별 ' + ', '.join(f'{k} {v}건' for k, v in load.items()))
                # 지시된 작업 목록
                if tf.empty:
                    st.caption('아직 지시된 작업이 없습니다. "배정할 경보"를 눌러 배정하세요.')
                else:
                    f3 = st.segmented_control('작업 구분', ['전체', '중단', '지금', '지연', '예정', '완료'], default='전체', key='task_filter',
                                              label_visibility='collapsed')
                    shown = tf if f3 in (None, '전체') else tf[tf['구분'] == f3]
                    # 실제 조치가 지시(두 부품 교체)와 다르거나 중단한 작업은 사유를 같이 보여 준다
                    act_ = np.where(shown['상태'] == '완료', shown['action'].replace('', tasks.ACTIONS[0]), '')   # 예전 완료 기록은 두 부품
                    why_ = np.where(shown['상태'] == '중단', '중단: ' + shown['stop_why'],
                                    np.where(shown['action_why'] != '', '지시와 다름: ' + shown['action_why'], ''))
                    st.dataframe(shown.assign(confirmed=shown['confirmed'].map(lambda x: PART_NAMES.get(x, x)), due_h=shown['due_h'].map(clock),
                                              name=shown['sid'].map(nm), act=act_, why=why_,
                                              # 작업자 체크리스트 진행률 (2026-10-21 7차): 사전 · 정비 중(10대 안전철칙) · 정비 후
                                              chk=shown['checks'].map(lambda x: f'{check_count(load_checks(x))}/{CHECK_TOTAL}'),
                                              lead=[x or tasks.pick_leader(m, '') for x, m in zip(shown['leader'], shown['members'])])[
                        ['id', 'name', 'title', 'team', 'members', 'lead', 'priority', '상태', '구분', 'chk', 'due_h', 'act', 'outcome', 'confirmed', 'why']].rename(columns={
                        'id': '번호', 'name': '설비', 'title': '작업', 'team': '팀', 'members': '작업자', 'lead': '작업책임자', 'chk': '체크',
                        'priority': '우선순위', 'due_h': '예정', 'act': '실제 조치',
                        'outcome': '완료 결과', 'confirmed': '확인된 원인', 'why': '사유'}), hide_index=True,
                        column_config={'사유': st.column_config.TextColumn(width='large')})
                    with st.container(horizontal=True):
                        if not st.session_state.get('task_edit'):
                            st.button('작업 수정하기', icon=':material/edit:', key='task_edit_open', on_click=open_task_edit)
                        st.download_button('작업 이력 CSV', tasks.safe_csv(tf.drop(columns=['split'])), file_name=f'cast_pdm_tasks_{split}.csv',
                                           mime='text/csv', icon=':material/download:')
                    if st.session_state.get('task_edit'):
                        task_editor(tf)
            else:
                # 인원 현황 (2026-10-21): 사람마다 부재 종류·기간을 고른다. 저장하면 공용 저장소에 남아 모든 화면에 반영
                #   (작업자 고르기 · 팀 하루 작업 한도 · 칸별 비용). 처음은 전원 출근, 라이브 데모 때 시나리오에 맞춰 바꾼다
                absent_now = TASKS.absent_at(now_f)
                for col, (team_, mem) in zip(st.columns(len(tasks.TEAMS), border=True), tasks.TEAMS.items()):
                    on = [m for m in mem if m not in absent_now]
                    cap = len(on) * A['crew_day_h']
                    with col:
                        st.metric(f'{team_} 지금 출근', f'{len(on)} / {len(mem)}명')
                        st.caption(f'하루 작업 한도 {cap:.0f}인·시 (교체 {int(cap // WORK_REPLACE)}번) · '
                                   + (('부재: ' + ', '.join(f'{m}({absent_now[m]})' for m in mem if m in absent_now)) if len(on) < len(mem)
                                      else '전원 출근'))
                ab = TASKS.absences()
                day_h = lambda h: (int(h // 24) + 1, int(h % 24))
                rows = []
                for team_, mem in tasks.TEAMS.items():
                    for m in mem:
                        mine = ab[ab['name'] == m]
                        if mine.empty:
                            rows.append({'이름': m, '팀': team_, '부재 종류': '없음', '시작 일차': 1, '시작 시': 0, '끝 일차': 1, '끝 시': 0, '메모': ''})
                        for r in mine.itertuples():
                            (d1, h1), (d2, h2) = day_h(r.start_h), day_h(r.end_h)
                            rows.append({'이름': m, '팀': team_, '부재 종류': r.kind, '시작 일차': d1, '시작 시': h1,
                                         '끝 일차': d2, '끝 시': h2, '메모': r.note})
                st.markdown('##### :material/badge: 인원별 부재')
                st.caption(f'부재 종류를 고르고 기간(시연 시각 기준 일차 · 시)을 정한 뒤 저장하세요. 끝 시각은 그 시각부터 출근입니다. '
                           f'한 사람이 두 번 이상 빠지면 아래 + 로 줄을 더합니다. 지금 {clock(now)} 기준')
                edited = st.data_editor(
                    pd.DataFrame(rows), key=f'absence_editor_{st.session_state.get("absence_ver", 0)}', num_rows='dynamic', hide_index=True,
                    width='stretch',
                    column_config={
                        '이름': st.column_config.SelectboxColumn('이름', options=list(tasks.TEAM_OF), required=True),
                        '팀': st.column_config.TextColumn('팀', disabled=True, help='저장할 때 이름으로 정해집니다'),
                        '부재 종류': st.column_config.SelectboxColumn('부재 종류', options=['없음', *tasks.ABSENCE_KINDS], required=True),
                        '시작 일차': st.column_config.SelectboxColumn('시작 일차', options=[1, 2, 3, 4], required=True),
                        '시작 시': st.column_config.SelectboxColumn('시작 시', options=list(range(24)), required=True),
                        '끝 일차': st.column_config.SelectboxColumn('끝 일차', options=[1, 2, 3, 4, 5], required=True),
                        '끝 시': st.column_config.SelectboxColumn('끝 시', options=list(range(24)), required=True),
                        '메모': st.column_config.TextColumn('메모', max_chars=100)})
                PRESET_SHORT = [('강태윤', '연차', 24, 48, ''), ('조성민', '교육', 33, 42, '안전 교육'), ('윤재혁', '출장', 24, 48, ''),
                                ('장현우', '병가', 0, 48, ''), ('최민재', '출산휴가', 0, 72, '배우자 출산'), ('박지훈', '반차(오전)', 33, 37, '')]
                with st.container(horizontal=True):
                    go_save = st.button('저장', icon=':material/save:', type='primary', key='absence_save')
                    go_clear = st.button('전원 출근으로 되돌리기', icon=':material/restart_alt:', key='absence_clear')
                    go_short = st.button('예시 채우기: 인원 부족 (2일차 정비 2팀 대부분 부재)', icon=':material/group_off:', key='absence_preset')
                if go_save or go_clear or go_short:
                    if go_save:
                        new_rows = [{'name': r['이름'], 'kind': r['부재 종류'], 'start_h': (int(r['시작 일차']) - 1) * 24 + int(r['시작 시']),
                                     'end_h': (int(r['끝 일차']) - 1) * 24 + int(r['끝 시']), 'note': r['메모'] or ''}
                                    for r in edited.dropna(subset=['이름', '부재 종류']).to_dict('records') if r['부재 종류'] != '없음']
                    else:
                        new_rows = [] if go_clear else [{'name': n_, 'kind': k_, 'start_h': a_, 'end_h': b_, 'note': m_}
                                                        for n_, k_, a_, b_, m_ in PRESET_SHORT]
                    try:
                        n_saved = TASKS.save_absences(new_rows)
                    except (ValueError, TypeError) as err:
                        st.error(str(err), icon=':material/error:')
                    else:
                        st.session_state['absence_ver'] = st.session_state.get('absence_ver', 0) + 1   # 표를 저장된 값으로 새로 그린다
                        st.toast(f'부재 {n_saved}건을 저장했습니다' if n_saved else '전원 출근으로 바꿨습니다', icon=':material/badge:')
                        st.rerun()
                st.caption('부재 중인 사람은 작업자 고르기에서 빠지고, 팀 하루 작업 한도(출근 인원 × 8시간)와 칸별 비용(인·시)에 바로 반영됩니다.')

        st.session_state.setdefault('sid', sid)
    if sub not in MANAGER_TABS and SHOW_SUBPAGE_LIST:
        st.markdown('#### ③ 하위 페이지')
        with st.container(border=True, gap='small'):
            for key_, label_ in [('설비 상세', f'{sid} 설비 상세 — 위험 확률·네 신호·정비 이력'),
                                 ('정책·비용', '정책·비용 — 가정값, 전체 정비 결정과 근거'),
                                 ('부품 손익', '부품 손익 — 원인 부품을 얼마나 맞히면 한 부품만 바꿔도 되나'),
                                 ('모델 정보·한계', '모델 정보·한계 — 검증 성능, 한계와 대응, 온톨로지')]:
                st.button(label_, icon=':material/description:', type='tertiary', key=f'sub_{key_}', on_click=open_sub,
                          args=(key_, sid if key_ == '설비 상세' else None))
        st.caption('누르면 그 페이지로 들어갑니다. 위의 "← 정비 의사결정"으로 돌아옵니다.')


# 2. 설비 상세 (작업자 '설비 상세' 보기, 설비관리자 아래 탭)

if page == '설비 상세':
    t = status_table()
    if st.session_state.get('sid') not in t.index:
        st.session_state.pop('sid', None)
    sid = st.selectbox('설비 (지금 위험한 순서)', t.index.tolist(), key='sid',
                       format_func=lambda s: f'{nm(s)} ({s}) · {t.loc[s, "상태"]} · 24시간 내 고장 위험 {t.loc[s, "위험 확률"]:.0%}'
                       if pd.notna(t.loc[s, '위험 확률']) else f'{nm(s)} ({s}) · {t.loc[s, "상태"]}')
    st.header(nm(sid), icon=ICONS['설비 상세'])
    st.caption(f'시나리오 {sid} · {st.session_state["split_label"]} · 지금 {clock(now)} 기준')

    equipment_detail(sid, t)

    with st.expander('근거 따라가기: 경보 → 예측 → 신호 → 정비 결정 (온톨로지)', icon=':material/account_tree:'):
        tr = ontology.trace(ontology_graph(assumptions(), t), sid, with_cost=(role == '정비 책임자'))
        st.dataframe(pd.DataFrame(tr, columns=['단계', '내용']), hide_index=True, width='stretch',
                     column_config={'내용': st.column_config.TextColumn(width='large')})
        st.caption('설비·예측·신호·경보·정비 결정·정기 교체 계획을 하나의 그래프(OWL/RDF)로 잇고, 그 링크를 따라 읽은 결과입니다. '
                   '판단은 그래프가 아니라 경보 기준과 의사결정 규칙이 합니다. 센서로는 원인 부품을 구분할 수 없어 '
                   '"신호 → 부품" 링크는 두지 않았습니다.')



# 3. 의사결정 (DSS)

elif page == '의사결정':
    D = bundle.get('dss')
    st.subheader('정책·비용: 가정값, 전체 정비 결정과 근거', icon=ICONS['정책·비용'])
    if not D:
        st.warning('이 모델 파일에는 의사결정 정보가 없습니다. 터미널에서 `python analysis/05_predict_test.py`로 다시 만드세요.')
        st.stop()
    A = assumptions()
    st.caption('비용 단위: 만 원 (현장 가정값, 아래 "가정값"에서 바꿀 수 있음). 과제 공통 비용 규칙(점수)으로 낸 모델 평가는 결과보고서에 있습니다.')
    FIELD = A['mode'] == COST_MODES[1]
    U = '만 원' if FIELD else '점'

    def won(x):
        return f'{x:,.0f}{U}' if not FIELD or abs(x) < 10000 else f'{x / 10000:,.2f}억 원'

    # ② 가정값 (여기서 바꾸면 위쪽 배정할 경보와 작업자 화면도 같은 계획으로 바뀐다)
    with st.expander('가정값 (바꾸면 추천·비용·배정할 경보가 다시 계산됩니다)', expanded=False, icon=':material/tune:'):
        assume_note = st.container()          # 지금 가정 요약 (아래에서 계산 뒤 채운다)
        st.markdown('**안전·실행 가능성 (비용보다 먼저 지키는 제약)**')
        a1, a2, a3, a4 = st.columns(4)
        A['safety'] = a1.slider('안전 기준: 정비 완료 전 고장 확률 상한', 0.0, 0.5, A['safety'], 0.01)
        A['prep'] = a2.slider('부품 준비 시간 (조달 + 교체, h)', 4, 36, A['prep'], 1,
                              help='데이터셋 비용 규칙은 교체 4시간. 부품을 조달해야 하면 더 길어집니다.')
        A['crew_wait'] = a3.slider('정비팀이 시작할 수 있을 때까지 (h)', 0, 24, A['crew_wait'], 1)
        A['n_crews'] = a4.number_input('동시에 교체할 수 있는 정비팀 수', 1, 10, A['n_crews'], 1,
                                       help='경보 설비가 이보다 많으면 위험한 순서대로 차례를 기다립니다.')
        A['crew_day_h'] = a4.number_input('정비팀 1팀의 하루 작업 한도 (공수, 시간)', 4, 24, A['crew_day_h'], 1,
                                          help='기본 8공수(상주 근무: 주 5일, 하루 8시간). 교체 1번이 4시간이라 팀당 하루 2대까지이고, 넘으면 다음 날로 넘어갑니다. 연장·교대 근무면 늘리세요.')
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
        st.markdown('**부품 조달 (데이터에 없어 미리 정한 가정값)**')
        h1, h2, h3, h4, h5 = st.columns(5)
        A['stock_v'] = h1.number_input('예비 밸브 (설비당)', 0, 10, A['stock_v'], 1, help='정비 기록상 밸브 교체가 진동장치의 약 2배')
        A['stock_o'] = h2.number_input('예비 진동장치 (설비당)', 0, 10, A['stock_o'], 1)
        A['lead'] = h3.number_input('정상 조달 (h)', 1.0, 720.0, A['lead'], 12.0)
        A['rush_h'] = h4.number_input('긴급 조달 (h)', 1.0, 168.0, A['rush_h'], 1.0, help='웃돈을 내고 직접 수령')
        A['rush_mk'] = h5.number_input('긴급 조달 웃돈 (부품값 대비)', 0.0, 3.0, A['rush_mk'], 0.1)
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
    if st.session_state.get('assume') != dict(A):     # 가정값이 바뀌면 위쪽(배정할 경보·비용 막대)도 같은 값으로 다시 그린다
        st.session_state['assume'] = dict(A)
        st.rerun()
    costs, cfg = make_cfg(A)
    now_cost = 2 * costs.part + costs.unnecessary + costs.replace_fixed + costs.replace_h * costs.downtime_h
    assume_note.caption(f'지금 가정: 지금 두 부품 교체 **{won(now_cost)}** · 정기 교체 때 다른 부품 추가 **{won(costs.part)}** · '
               f'고장 시 손실 = 복구 {costs.repair:,.0f} + 복구 {costs.repair_h:.0f}시간 × (생산 차질 {costs.downtime_h:,.1f} + 인력 {costs.labor_h:,.1f}) '
               f'+ 고장 추가 손실 {costs.failure_penalty:,.0f} = **{won(costs.failure_loss)}** · 안전 기준 {cfg.safety_max:.2f} · '
               f'부품 준비 {cfg.prep_h:.0f}시간 · 정비팀 {A["n_crews"]}팀(팀당 하루 {A["crew_day_h"]}공수). 고장 확률은 학습 세트에서 경보 뒤 고장까지 걸린 시간 분포로 계산합니다.')

    t = status_table()
    W = work_plan(A, t)
    dec, need_crew, per_day, evaluate_options, cfg, costs, curve, pl = (W[k] for k in ['dec', 'need_crew', 'per_day', 'evaluate_options', 'cfg', 'costs', 'curve', 'pl'])
    n_crews, crew_day_h = A['n_crews'], A['crew_day_h']

    jobs_state = LOG.job_state(split, now_f, MNT_START)

    st.subheader(f'전체 정비 결정 · {len(dec)}건', icon=':material/event_note:')
    if dec.empty:
        st.success('지금 경보가 켜진 설비가 없습니다.', icon=':material/check_circle:')
    else:
        st.dataframe(dec.drop(columns=['_e', '_slot', '_start']).round({'기대 총비용': 0}), hide_index=True, column_config={
            '순위': st.column_config.NumberColumn(format='%d', help='지금 정비할 설비의 정비팀 배정 순서 (위험한 순서)'),
            '추천': st.column_config.TextColumn(width='medium'),
            '정비 완료까지 (h)': st.column_config.NumberColumn(format='%.0f'),
            '기대 총비용': st.column_config.NumberColumn(f'기대 총비용 ({U})', format='localized'),
            '완료 전 고장 확률': st.column_config.NumberColumn(format='%.2f'),
            '첫 경보 (h)': st.column_config.NumberColumn(format='%d')})
        st.caption(f'정비팀 {n_crews}팀이 한 번에 {n_crews}대씩 교체({costs.replace_h:.0f}시간)하고, 팀당 하루 {crew_day_h:.0f}공수(교체 {per_day}번)까지 합니다. '
                   f'하루에 {n_crews * per_day}대를 넘는 설비는 다음 날로 넘어갑니다. 차례를 기다리는 설비는 그만큼 늦게 끝나는 것으로 다시 계산합니다.')

        # ④ 설비별 근거: 선택지 비교와 산정 근거
        st.subheader('설비별 근거', icon=':material/rule:')
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
        st.caption(f'이 설비 진행: {jobs_state.get(sid, "작업 없음")} · 배정·보류·부품 발주는 설비 화면에서 합니다.')
        left, right = st.columns([1.3, 1])
        with left:
            ot = pd.DataFrame(e['opts'])
            ot['추천'] = np.where(ot['선택지'] == b['선택지'], '✔', '')
            st.dataframe(ot[['추천', '선택지', '정비 완료까지(h)', '정비 비용', '생산 차질 비용', '고장 확률', '고장 시 손실', '기대 총비용', '안전 기준 충족']]
                         .round({'정비 비용': 0, '생산 차질 비용': 0, '고장 시 손실': 0, '기대 총비용': 0}),
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
            marks = [alt.Chart(cv).mark_line(color=LINE_BLUE).encode(
                x=alt.X('h:Q', title='지금부터 시간 (h)'), y=alt.Y('p:Q', title='그 안에 고장 날 확률', scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip('h', title='시간(h)'), alt.Tooltip('p', title='고장 확률', format='.2f')]),
                alt.Chart(pd.DataFrame({'y': [cfg.safety_max]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')]
            pts_o = pd.DataFrame([{'h': o['정비 완료까지(h)'], 'p': fail_prob(curve, e['elapsed'], o['정비 완료까지(h)']), '선택지': o['선택지']}
                                  for o in e['opts']])
            marks.append(alt.Chart(pts_o).mark_point(size=90, filled=True, color=LINE_GREEN).encode(
                x='h:Q', y='p:Q', tooltip=['선택지', alt.Tooltip('h', title='정비 완료(h)'), alt.Tooltip('p', title='고장 확률', format='.2f')]))
            show_chart(alt.layer(*marks).properties(height=240), width='stretch')
            st.caption('파란 선: 지금부터 h시간 안에 고장 날 확률 · 초록 점: 선택지별 정비 완료 시점 · 빨간 점선: 안전 기준')

    # ⑤ 조치 기록과 점검 되먹임
    with st.expander('조치 기록과 점검 되먹임', icon=':material/history:'):
        fb = feedback()
        f1, f2, f3 = st.columns(3, border=True)
        f1.metric('점검 결과 (원인 확인됨)', f'{fb["n"]}건', help=f'되먹임은 {actions.MIN_CONFIRMED}건 이상부터 씁니다 (가정값)')
        f2.metric('점검으로 원인 부품을 맞힌 비율', '—' if fb['accuracy'] is None else f'{fb["accuracy"]:.2f}')
        f3.metric('교체 방법 추천', '점검 후 한 부품' if fb['ready'] else '두 부품',
                  help=f'맞힌 비율이 손익분기 {fb["breakeven"]:.2f} 이상이고 기록이 {actions.MIN_CONFIRMED}건 이상이면 "점검 후 한 부품"' if fb['breakeven'] else None)
        st.caption('센서로는 원인 부품을 구분할 수 없어(EDA E9) 지금은 두 부품을 함께 교체합니다. 현장이 점검 결과와 교체 뒤 확인된 원인을 입력하면, '
                   '점검으로 부품을 맞히는 비율이 쌓이고 이 값이 비용 계산기의 손익분기를 넘으면 추천이 바뀝니다.')
        lg = LOG.frame(split, now_f)
        if lg.empty:
            st.write('이 데이터 세트에서 지금 시각까지 기록된 조치가 없습니다.')
        else:
            st.dataframe(lg.drop(columns=['데이터 세트']).iloc[::-1], hide_index=True,
                         column_config={'시각(h)': st.column_config.NumberColumn(format='%d'),
                                        '부품': st.column_config.TextColumn(), '메모': st.column_config.TextColumn(width='medium')})

    # ⑥ 주의 설비
    cau = t.index[t['상태'] == CAU].tolist()
    if cau:
        st.subheader('주의 설비 준비', icon=':material/search_check:')
        nxt = pl[pl['SCENARIO_ID'].isin(cau) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').groupby('SCENARIO_ID').head(1)
        cd = pd.DataFrame({'설비 (시나리오)': cau})
        cd['다음 정기 교체'] = cd['설비 (시나리오)'].map(
            nxt.set_index('SCENARIO_ID').apply(lambda x: f'{x["PLAN_H"] - now:.0f}시간 뒤 ({PART_NAMES.get(x["PLAN_PART"], x["PLAN_PART"])})', axis=1)
            if len(nxt) else {}).fillna('계획 없음')
        cd['할 일'] = np.where(cd['다음 정기 교체'] != '계획 없음', '장비 점검, 정기 교체 때 두 부품 함께 교체 준비',
                              '장비 점검 (경보가 켜지면 지금 교체 대상)')
        pr = procurement(cfg, W['stock'], cau).set_index('설비')
        cd['재고'] = cd['설비 (시나리오)'].map(pr['재고'])
        cd['부품'] = cd['설비 (시나리오)'].map(pr['발주'])
        need_order = [(sid, q) for sid, qs in pr['_short'].items() for q in qs]
        if need_order:
            with st.container(border=True, horizontal=True, vertical_alignment='center'):
                st.markdown(f'**주의 설비 부품 발주 {len(need_order)}건** — 경보 전에 정상 조달({cfg.parts.lead_h:.0f}시간)로 재고를 채웁니다')
                st.space('stretch')

                def order_all():
                    for sid_, q_ in need_order:
                        LOG.record('부품 발주', split, sid_, flow_hour(), '정비 책임자', part=q_, option='정상', note='주의 단계 재고 부족')
                    st.toast(f'부품 {len(need_order)}건 발주', icon=':material/local_shipping:')
                st.button('한 번에 발주', icon=':material/local_shipping:', type='primary', on_click=order_all, key='order_all')
        st.dataframe(cd, hide_index=True)
        st.caption(f'발주 시점: 주의 단계에서 두 부품 교체에 모자라는 재고가 있을 때(정상 조달 {cfg.parts.lead_h:.0f}시간). '
                   f'경보 때 재고가 없으면 긴급 조달({cfg.parts.rush_h:.0f}시간, 웃돈 {cfg.parts.rush_markup:.0%})로 교체가 늦어집니다.')

    # ⑦ 정책 시뮬레이션 상세 (요약 카드 '없었다면 → 있으면'은 2026-10-15 사용자 요청으로 뺐다)
    sim = D.get('validation_field') if FIELD else D.get('validation')
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

    with st.expander('이 화면 보는 법', icon=':material/help:'):
        st.markdown('- 경보 설비마다 정비 선택지(지금 / N시간 뒤 / 다음 정기 교체 때)의 **기대 총비용 = 정비 비용 + 생산 차질 비용 + '
                    '고장 확률 × 고장 시 손실**을 비교해 추천합니다.\n'
                    '- 안전 기준과 실행 가능성(정비팀·부품 준비)은 비용보다 먼저 지킵니다.\n'
                    '- 판단은 검증 가능한 예측값과 계산·규칙으로만 하며, 최종 결정은 정비 책임자가 합니다.')

# 4. 모델 정보

elif page == '모델 정보':
    v = bundle['validation']
    st.subheader('모델 정보와 검증 세트 성능', icon=ICONS['모델 정보·한계'])
    st.caption(f'모델 저장: {bundle["created_at"]}' + (' · 클라우드: 미리 계산한 결과만 사용 (원자료·모델 파일 없음)' if CLOUD else ''))
    st.markdown(
        f'- **모델**: {bundle["final"]["model"]} · 특징 {len(bundle["features"])}개 ({bundle["final"]["features"]}) · {STEP}시간마다 예측\n'
        f'- **학습**: 학습 세트 216개 시나리오의 예측 시점 {bundle["n_train_points"]:,}개\n'
        f'- **경보 기준**: 24시간 안 고장 확률 ≥ {TH:.2f}. 학습 세트 교차검증에서 '
        '점진 고장을 모두 잡고 진짜 오경보가 0인 값 중 가장 낮은 값(오경보 없이 가장 일찍 경보)\n'
        '- **검증 세트**: 72개 시나리오, 학습에 쓰지 않고 평가에만 사용(모델·기준 선택은 학습 세트 교차검증)')
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
        '- 위 숫자는 정답을 알 수 있는 시점만으로 셌습니다. 관측 끝부분의 미확정 경보는 "부품 손익" 탭에 있습니다.')
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
                   '일부는 끝까지 정상인 설비였습니다. 자세한 근거는 결과보고서 7장')

    st.subheader('검증 세트 고장별 결과', icon=':material/fact_check:')
    ev = v['events'].rename(columns={'SCENARIO_ID': '시나리오', 'TYPE': '고장 종류'})
    st.dataframe(ev[['시나리오', '고장 종류', '감지', '리드타임(h)']].sort_values(['고장 종류', '리드타임(h)'], ascending=[True, False]),
                 hide_index=True, column_config={'리드타임(h)': st.column_config.NumberColumn(format='%.1f')})
    st.caption('리드타임: 고장 4시간 전까지 난 경보 중 가장 이른 경보부터 고장까지의 시간. '
               '자세한 근거는 결과보고서 5·6장')


    st.subheader('한계와 대응 방안', icon=':material/report:')
    lim = pd.DataFrame(LIMITS)
    pick = st.pills('중요도', ['높음', '중간', '낮음'], selection_mode='multi', default=['높음', '중간', '낮음'], key='lim_pick')
    lim = lim[lim['중요도'].isin(pick or [])]
    st.dataframe(lim[['중요도', '구분', '한계', '영향', '지금 대응', '추가 대응', '근거']], hide_index=True,
                 column_config={c: st.column_config.TextColumn(width='large') for c in ['영향', '지금 대응', '추가 대응']})
    st.caption('돌발 고장·원인 부품·리드타임처럼 모델만으로 풀 수 없는 한계는 운영(정기 교체 유지, 점검 결과 되먹임, 예비 재고와 주의 단계 발주)으로 메웁니다. '
               '자세한 내용은 결과보고서 8장.')

    st.subheader('온톨로지: 지금 상태를 하나의 그래프로', icon=':material/account_tree:')
    g = ontology_graph(assumptions(), status_table())
    cnt = ontology.counts(g)
    bad = ontology.check(g)
    e1, e2, e3, e4 = st.columns(4, border=True)
    e1.metric('트리플 수', f'{len(g):,}')
    e2.metric('설비 / 예측', f'{cnt.get("설비", 0)} / {cnt.get("예측", 0)}')
    e3.metric('경보 / 주의', f'{cnt.get("경보 (WARNING)", 0)} / {cnt.get("주의 (CAUTION)", 0)}')
    e4.metric('규칙 점검', '통과' if not bad else f'{len(bad)}건 어김', help='; '.join(bad) or '신호→부품 링크 없음, 주의·경보 겹침 없음, 정비 결정마다 추천 있음')
    need = ontology.parts_needed(g)
    st.markdown(
        '- **스키마**: `ontology/casting_dss.ttl` (OWL). 객체 = 설비·부품·신호·예측·경보 단계·정비 결정·선택지·정비 작업·정기 교체 계획·고장, '
        '링크 = 가진다·관측한다·근거·요청한다·추천한다·교체한다 등\n'
        '- **인스턴스**: 이 화면의 지금 시각·데이터 세트·가정값으로 만든다(예측·신호·정비 기록·DSS 결정)\n'
        f'- **질의 예 (SPARQL)**: 경보 설비가 필요로 하는 부품 = '
        + (' · '.join(f'{k} {v}개' for k, v in need.items()) if need else '없음'))
    st.download_button('지금 그래프 내려받기 (Turtle)', g.serialize(format='turtle'), file_name=f'casting_{split}_{now}h.ttl',
                       mime='text/turtle', icon=':material/download:')



# 5. 비용 계산기

elif page == '비용 계산기':
    v = bundle['validation']['event']
    n = dict(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
             n_false=v['고장과 무관한 경보 묶음: 오경보'])
    st.subheader('부품 손익: 원인 부품을 얼마나 맞히면 한 부품만 바꿔도 되나', icon=ICONS['부품 손익'])
    st.caption('원인 부품을 얼마나 맞힐 수 있으면 이득인가')
    st.markdown(
        f'검증 세트 결과(고장 {v["고장"]}건 중 {n["n_det"]}건 감지, 고장과 무관한 교체 {n["n_needed"]}번, '
        f'진짜 오경보 {n["n_false"]}번)에 현장 비용 가정(만 원, "정책·비용" 탭의 가정값)을 적용합니다.\n'
        '- 경보가 나면 바로 교체합니다(정책: 경보 즉시 교체). 한 부품만 바꾸는데 틀리면, 불필요 정비가 되고 고장도 못 막습니다.\n'
        '- 정기 교체 직전에 울린 맞는 경보도 교체 한 번으로 셉니다. 예정 교체와 묶어 처리하면 이 비용은 줄어듭니다(의사결정 지원, 다음 단계).\n'
        '- 센서로는 원인 부품을 구분할 수 없어(EDA E9), 부품을 맞힐 확률은 현장 점검 능력에 달려 있습니다.')
    p = st.slider('원인 부품을 맞힐 확률', 0.5, 1.0, 0.5, step=0.01,
                  help='0.5 = 둘 중 하나를 찍음, 1.0 = 항상 맞힘')
    costs_f, _ = make_cfg(assumptions())
    cc = cost_curve(**n, costs=costs_f)
    one = float(np.interp(p, cc['p'], cc['한 부품만 교체']))
    both, none = cc['두 부품 모두 교체'].iloc[0], cc['아무것도 안 함'].iloc[0]
    c1, c2, c3 = st.columns(3, border=True)
    c1.metric('한 부품만 교체', fmt_cost(one, costs_f), f'{fmt_cost(one - none, costs_f)} (아무것도 안 함 대비)', delta_color='inverse')
    c2.metric('두 부품 모두 교체', fmt_cost(both, costs_f), fmt_cost(both - none, costs_f), delta_color='inverse')
    c3.metric('아무것도 안 함', fmt_cost(none, costs_f))
    u = bundle['validation'].get('unlabeled')
    if u:
        two_ = 2 * costs_f.part + costs_f.unnecessary + costs_f.replace_fixed + costs_f.replace_h * costs_f.downtime_h
        st.warning(
            f'**위 비용은 정답을 알 수 있는 구간만 센 값입니다.** 관측이 끝나 결과를 알 수 없는 시점 {u["시점"]}개에서도 '
            f'경보가 {u["경보 시점"]}번(경보 묶음 {u["경보 묶음"]}개) 울렸고, 실제 운영이라면 교체가 일어납니다. '
            f'두 부품 교체 기준으로 경보 묶음마다 약 {fmt_cost(two_, costs_f)}, 모두 {fmt_cost(u["경보 묶음"] * two_, costs_f)}까지 더 들 수 있습니다. 이 경보가 관측 뒤의 고장을 막았는지, '
            '"아무것도 안 함"에서 관측 뒤에 고장이 났는지는 알 수 없어 운영 비용은 확정할 수 없습니다.')
    even = cc.loc[cc['한 부품만 교체'] <= both, 'p']
    if len(even):
        st.info(f'부품을 맞힐 확률이 **{even.iloc[0]:.2f} 이상**이면 한 부품만 바꾸는 쪽이 두 부품 모두 바꾸는 것보다 쌉니다.')
    long = cc.melt(id_vars='p', var_name='방식', value_name='비용')
    chart = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X('p:Q', title='원인 부품을 맞힐 확률'), y=alt.Y('비용:Q', title='비용 (만 원, 낮을수록 좋음)'),
        color=alt.Color('방식:N', scale=alt.Scale(domain=['한 부품만 교체', '두 부품 모두 교체', '아무것도 안 함'],
                                                range=[LINE_BLUE, LINE_GREEN, OK_GRAY])))
    now_rule = alt.Chart(pd.DataFrame({'p': [p]})).mark_rule(strokeDash=[4, 4]).encode(x='p:Q')
    show_chart((chart + now_rule).properties(height=360), width='stretch')


# 화면 위치 → 주소 (뒤로 가기·앞으로 가기용, 맨 위 apply_url과 짝)
sync_url()


# 자동 갱신: 같이 보는 저장소면 10초마다 바뀐 것이 있는지만 가볍게 확인하고, 바뀌었을 때만 화면 전체를 다시 그린다
#   (그리던 입력칸 값은 그대로 남는다). 공개 화면(세션 저장)은 혼자 보므로 하지 않는다
AUTO_REFRESH_S = 10


@st.fragment(run_every=AUTO_REFRESH_S)
def watch_changes():
    try:
        v = TASKS.version()
    except Exception as e:
        print(f'[dashboard] 자동 갱신 확인 실패: {type(e).__name__}', file=sys.stderr)
        return
    if st.session_state.pop('_fresh', False) or '_ver' not in st.session_state:
        st.session_state['_ver'] = v          # 방금 전체를 그렸다: 지금 상태를 기준으로
        return
    if v != st.session_state['_ver']:
        st.session_state['_ver'] = v
        st.rerun(scope='app')


@st.fragment(run_every=AUTO_REFRESH_S)
def watch_hour():
    """흐르는 시각이 정각을 넘으면 화면 전체를 다시 그린다 (예측 · 경보 · 비용이 1시간 요약본이라 정각마다 바뀐다)"""
    if int(flow_hour()) != now:
        st.rerun(scope='app')


if TASKS.shared:
    watch_changes()
if st.session_state.get('clock_at') and now < END_H:
    watch_hour()
