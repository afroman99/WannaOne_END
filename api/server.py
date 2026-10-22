"""CAST-PdM API (React 화면용, 2026-10-22). 계산(api/engine.py → src/)을 JSON으로 내준다. 처음 Streamlit 화면(app/dashboard.py, 2026-10-22 지움)과 같은 규칙이다.

    .venv/bin/uvicorn api.server:app --port 8000      # 프로젝트 폴더에서. web/dist가 있으면 화면도 같이 → http://localhost:8000

저장소 (옛 dashboard.py와 같은 규칙)
  공용 DB    구글 로그인한 사람 + Secrets [db] url (Supabase Postgres). 노트북·폰이 같은 작업을 본다
  로컬       actions/tasks.sqlite3 · actions/action_log.csv (커밋하지 않음). 로그인을 켜지 않았을 때
재생 시각: 데이터 세트·적용한 시각·적용한 실제 시각(clock_at)을 저장소 settings에 두고, 흐르는 시각은 요청마다 계산한다 → 모든 화면이 같은 값.
구글 로그인: .streamlit/secrets.toml의 [auth]·[users]를 읽는다. 구글 콘솔에 http://<주소>/api/auth/callback 을 승인된 리디렉션 URI로 더해야 한다.
"""
import json
import math
import os
import secrets as pysecrets
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import ENCODERS_BY_TYPE
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import engine as E                                     # noqa: E402
from src import actions, tasks                                   # noqa: E402
from src.auth import lookup                                      # noqa: E402

ENG = E.Engine()
LOCAL_TASKS = tasks.TaskStore(ROOT / 'actions' / 'tasks.sqlite3')
LOCAL_LOG = actions.ActionLog(ROOT / 'actions' / 'action_log.csv')


def read_secrets():
    """비밀값: 환경 변수 CAST_SECRETS(secrets.toml 내용 전체, 배포 서비스의 Secrets 칸) → 없으면 .streamlit/secrets.toml.
    둘 다 없거나 형식이 깨졌으면 빈 값 → 로그인·공용 DB 없이 로컬로"""
    import tomllib
    text = os.environ.get('CAST_SECRETS')
    try:
        if text:
            return tomllib.loads(text)
        path = ROOT / '.streamlit' / 'secrets.toml'
        return tomllib.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except Exception as e:
        print(f'[api] 비밀값을 읽지 못함 ({type(e).__name__}) → 로그인·공용 DB 없이 로컬 저장소로', file=sys.stderr)
        return {}


SECRETS = read_secrets()
AUTH = SECRETS.get('auth') or {}
LOGIN = bool(AUTH.get('client_id') and AUTH.get('client_secret'))
DB_URL = (SECRETS.get('db') or {}).get('url')

def _plain(x):
    """numpy 값·NaN → JSON에 넣을 수 있는 값"""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) or math.isinf(x) else float(x)
    return x


class SafeJSON(JSONResponse):
    def render(self, content):
        return json.dumps(_plain(content), ensure_ascii=False).encode('utf-8')


ENCODERS_BY_TYPE.update({np.int64: int, np.int32: int, np.float64: lambda v: _plain(v), np.float32: lambda v: _plain(v),
                         np.bool_: bool})

app = FastAPI(title='CAST-PdM API', docs_url='/api/docs', openapi_url='/api/openapi.json', default_response_class=SafeJSON)
# 로그인 유지 쿠키: 배포(https)에서는 CAST_HTTPS=1로 https에서만 보내게 한다
app.add_middleware(SessionMiddleware, secret_key=AUTH.get('cookie_secret') or pysecrets.token_hex(32), same_site='lax',
                   https_only=os.environ.get('CAST_HTTPS') == '1', max_age=60 * 60 * 12)

OAUTH = None
if LOGIN:
    from authlib.integrations.starlette_client import OAuth
    OAUTH = OAuth()
    OAUTH.register('google', client_id=AUTH['client_id'], client_secret=AUTH['client_secret'],
                   server_metadata_url=AUTH.get('server_metadata_url', 'https://accounts.google.com/.well-known/openid-configuration'),
                   client_kwargs={'scope': 'openid email profile'})

_PG = None


def pg():
    global _PG
    if _PG is None:
        from src import db
        _PG = db.Postgres(DB_URL)
    return _PG


# 요청마다: 누가 보나 · 어느 저장소 · 지금 시각

class Who:
    def __init__(self, user, store, log):
        self.user, self.store, self.log = user, store, log

    @property
    def is_manager(self):
        return self.user is None or self.user['role'] == 'manager'

    def need_manager(self):
        if not self.is_manager:
            raise HTTPException(403, '설비관리자만 할 수 있습니다')

    def role_name(self, worker=False):
        return E.WORKER if worker else E.MANAGER


def who(request: Request):
    user = request.session.get('user') if LOGIN else None
    if LOGIN and user is None:
        raise HTTPException(401, '로그인이 필요합니다')
    if user and DB_URL:
        try:
            p = pg()
        except Exception as e:
            print(f'[api] 공용 DB 연결 실패: {type(e).__name__}', file=sys.stderr)
            raise HTTPException(503, '공용 DB에 연결하지 못했습니다. 잠시 뒤 새로 고침하세요.')
        return Who(user, tasks.TaskStore(db=p), actions.ActionLog(db=p))
    return Who(user, LOCAL_TASKS, LOCAL_LOG)


def clock_state(store):
    split = store.setting('split') or 'test'
    split = split if split in E.SPLITS else 'test'
    n = store.setting('now')
    now = int(n) if n and str(n).isdigit() and 0 <= int(n) <= E.END_H else E.END_H
    try:
        at = float(store.setting('clock_at')) if store.setting('clock_at') else None
    except ValueError:
        at = None
    now_f = float(now) if not at else min(float(E.END_H), now + max(0.0, time.time() - at) / 3600)
    return split, now, at, now_f


def assume_of(store):
    try:
        return json.loads(store.setting('assume') or '{}')
    except ValueError:
        return {}


def ctx(w: Who, truth=False):
    split, _, _, now_f = clock_state(w.store)
    return E.Ctx(ENG, split, now_f, w.store, w.log, assume_of(w.store), truth)


def bad(e):
    raise HTTPException(400, str(e))


# 로그인

@app.get('/api/auth/login')
async def login(request: Request):
    if not LOGIN:
        raise HTTPException(404, '로그인이 설정되지 않았습니다')
    return await OAUTH.google.authorize_redirect(request, str(request.url_for('auth_callback')))


@app.get('/api/auth/callback', name='auth_callback')
async def auth_callback(request: Request):
    token = await OAUTH.google.authorize_access_token(request)
    info = token.get('userinfo') or {}
    email = str(info.get('email') or '').lower()
    role, team = lookup(SECRETS.get('users'), email) if info.get('email_verified') else (None, None)
    if role is None:
        request.session.clear()
        return RedirectResponse('/?login=denied')
    request.session['user'] = {'email': email, 'name': info.get('name') or email, 'role': role, 'team': team}
    return RedirectResponse('/')


@app.post('/api/auth/logout')
def logout(request: Request):
    request.session.clear()
    return {'ok': True}


@app.get('/api/session')
def session(request: Request):
    user = request.session.get('user') if LOGIN else None
    shared_db = bool(user and DB_URL)
    return {'loginEnabled': LOGIN, 'user': user, 'sharedDb': shared_db, 'cloud': ENG.cloud,
            'storeNote': '기록은 공용 DB에 남아 설비관리자·작업자가 같이 봅니다.' if shared_db else
                         '기록은 이 컴퓨터의 actions/tasks.sqlite3에 남습니다.'}


# 공통

@app.get('/api/meta')
def meta():
    return {'splits': E.SPLITS, 'endH': E.END_H, 'threshold': ENG.th, 'cautionThreshold': ENG.cth, 'caution': bool(ENG.caution),
            'step': ENG.step, 'signals': E.SIGNAL_NAMES, 'teams': tasks.TEAMS, 'leaders': tasks.LEADERS, 'teamOf': tasks.TEAM_OF,
            'priorities': tasks.PRIORITIES, 'outcomes': tasks.OUTCOMES, 'actions': tasks.ACTIONS, 'judged': tasks.JUDGED,
            'confirmed': tasks.CONFIRMED, 'statuses': tasks.STATUSES, 'absenceKinds': tasks.ABSENCE_KINDS, 'limits': tasks.LIMITS,
            'defaultGuide': tasks.DEFAULT_GUIDE, 'partNames': E.PART_NAMES,
            'checklist': {'tbm': E.TBM_ITEMS, 'safety': E.SAFETY_RULES, 'safetyKeys': E.SAFETY_KEYS, 'safetyOpts': E.SAFETY_OPTS,
                          'post': E.POST_ITEMS, 'postOrder': E.POST_ORDER, 'preKeys': E.PRE_KEYS, 'total': E.CHECK_TOTAL},
            'work': {'replace': E.WORK_REPLACE, 'repair': E.WORK_REPAIR}}


@app.get('/api/state')
def state(w: Who = Depends(who)):
    split, now, at, now_f = clock_state(w.store)
    return {'split': split, 'now': now, 'clockAt': at, 'nowF': now_f, 'version': repr(w.store.version()),
            'canSetClock': w.is_manager}


class Clock(BaseModel):
    split: str
    now: int


@app.post('/api/clock')
def set_clock(b: Clock, w: Who = Depends(who)):
    w.need_manager()
    if b.split not in E.SPLITS or not 0 <= b.now <= E.END_H:
        bad('데이터 세트는 test·validation, 시각은 0~72')
    w.store.set_setting('split', b.split)
    w.store.set_setting('clock_at', repr(time.time()))
    w.store.set_setting('now', str(b.now))
    return state(w)


# 설비관리자 홈

@app.get('/api/manager')
def manager(truth: bool = False, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w, truth)
    t, short, ost = c.status_table(), c.short(), c.order_states()
    tf = c.tf()
    q = c.queue()
    js = c.job_state()
    rows = [{'sid': s, 'name': c.nm(s), 'stage': short[s], 'risk': E.num(t.loc[s, '위험 확률']), 'orderState': ost[s],
             'truthH': E.num(t.loc[s, '다음 실제 고장까지 (h, 정답)']) if c.truth else None} for s in t.index]
    queue = [{'sid': r['설비 (시나리오)'], 'name': c.nm(r['설비 (시나리오)']), 'best': r['추천'], 'rank': E.num(r['순위']),
              'urgent': r['제약'].startswith('즉시'), 'constraint': r['제약']} for _, r in q.iterrows()] if len(q) else []
    absent = w.store.absent_at(c.now_f)
    return {
        'counts': {k: int((short == k).sum()) for k in ('경보', '주의', '정상')},
        'inMnt': int(short.isin(['정비 중', '정비 후 예측 대기']).sum()), 'rows': rows, 'queue': queue,
        'held': sum(v == '보류' for v in js.values()), 'busy': len(w.store.blocked_sids(c.split, c.now_f)), 'load': c.team_load(tf),
        'taskCounts': {'진행 중': int((tf['상태'] == '진행 중').sum()), '지연': int((tf['구분'] == '지연').sum()),
                       '완료': int((tf['상태'] == '완료').sum()), '중단': int((tf['상태'] == '중단').sum())},
        'tasks': [c.task_json(r) for r in tf.to_dict('records')],
        'staff': [{'team': k, 'on': [m for m in v if m not in absent], 'total': len(v), 'absent': {m: absent[m] for m in v if m in absent}}
                  for k, v in tasks.TEAMS.items()], 'crewDayH': c.A['crew_day_h'], 'caution': bool(ENG.caution)}


@app.get('/api/equipment/{sid}/decision')
def decision(sid: str, n_now: int | None = None, n_plan: int | None = None, n_run: int | None = None, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    if sid not in c.names:
        raise HTTPException(404, '없는 설비')
    crew = {k: v for k, v in {'now': n_now, 'plan': n_plan, 'run': n_run}.items() if v}
    return c.decision(sid, crew)


@app.get('/api/equipment/{sid}/detail')
def detail(sid: str, truth: bool = False, w: Who = Depends(who)):
    c = ctx(w, truth)
    if sid not in c.names:
        raise HTTPException(404, '없는 설비')
    return c.detail(sid)


@app.get('/api/equipment/{sid}/ontology')
def onto(sid: str, dark: bool = False, w: Who = Depends(who)):
    c = ctx(w)
    if sid not in c.names:
        raise HTTPException(404, '없는 설비')
    try:
        return {'dot': c.ontology_dot(sid, dark), 'trace': c.ontology_trace(sid, with_cost=w.is_manager)}
    except Exception as e:
        print(f'[api] 근거 사슬 실패: {type(e).__name__}: {e}', file=sys.stderr)
        return {'dot': None, 'trace': None}


# 배정 · 보류 · 발주

class Assign(BaseModel):
    sid: str
    team: str
    title: str
    priority: str
    due_h: float
    reason: str
    option: str = ''
    guide: str = ''
    members: list[str] = []
    leader: str = ''


@app.post('/api/assign')
def assign(b: Assign, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    try:
        ok = w.store.add(c.split, b.sid, b.title, b.team, b.priority, min(240.0, max(0.0, b.due_h)), c.now_f,
                         b.guide or tasks.DEFAULT_GUIDE, b.reason, b.option, b.members, b.leader)
    except ValueError as e:
        bad(e)
    if not ok:
        raise HTTPException(409, f'{b.sid}: 이미 열린 작업이 있어 배정하지 않았습니다')
    w.log.record('승인', c.split, b.sid, c.now_f, E.MANAGER, option=b.option)
    return {'ok': True, 'message': f'{b.sid}: {b.team}에 배정'}


class Act(BaseModel):
    kind: str
    sid: str
    part: str = ''
    option: str = ''


@app.post('/api/act')
def act(b: Act, w: Who = Depends(who)):
    w.need_manager()
    if b.kind not in ('보류', '부품 발주'):
        bad('보류 또는 부품 발주만')
    c = ctx(w)
    try:
        w.log.record(b.kind, c.split, b.sid, c.now_f, E.MANAGER, part=b.part, option=b.option)
    except ValueError as e:
        bad(e)
    return {'ok': True, 'message': f'{b.sid}: {b.kind}'}


@app.post('/api/order/rush/{sid}')
def order_rush(sid: str, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    coming = {q for q, _, _ in c.in_transit(c.make_cfg()[1]).get(sid, [])}
    parts = [q for q in E.PARTS if q not in coming]
    for q in parts:
        w.log.record('부품 발주', c.split, sid, c.now_f, E.MANAGER, part=q, option='긴급')
    return {'ok': True, 'message': f'{c.nm(sid)}: ' + ', '.join(E.PART_NAMES[q] for q in parts) + ' 긴급 조달'}


@app.post('/api/order/caution')
def order_caution(w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    need = c.policy()['needOrder']
    for x in need:
        w.log.record('부품 발주', c.split, x['sid'], c.now_f, E.MANAGER, part=x['part'], option='정상', note='주의 단계 재고 부족')
    return {'ok': True, 'message': f'부품 {len(need)}건 발주'}


# 작업 (작업자)

def _worker_team(w, team):
    if w.user and w.user['role'] == 'worker':
        return w.user['team']
    if team not in tasks.TEAMS:
        bad('없는 팀')
    return team


@app.get('/api/tasks')
def task_list(team: str = '정비 1팀', w: Who = Depends(who)):
    team = _worker_team(w, team)
    c = ctx(w)
    tf = c.tf()
    W = c.work_plan()
    dec_i = c.dec_i()
    cfg, costs = W['cfg'], W['costs']
    mine = tf[tf['team'] == team]
    items = []
    for r in mine.to_dict('records'):
        j = c.task_json(r)
        sid = r['sid']
        info = dec_i.loc[sid] if len(dec_i) and sid in dec_i.index else None
        plan_part = info['_e'].get('plan_part') if info is not None else None
        need = E.PARTS if not str(r['option']).startswith('다음 정기') else [q for q in E.PARTS if q != plan_part]
        j['missing'] = [E.PART_NAMES[q] for q in E.missing_parts(W['stock'][sid], need)]
        p = info['완료 전 고장 확률'] if info is not None else None
        j['riskWord'] = None if p is None or pd.isna(p) else ('높음' if p > cfg.safety_max else ('보통' if p >= 0.05 else '낮음'))
        j['endH'] = E.num(r['due_h'] + costs.replace_h)
        items.append(j)
    busy = w.store.blocked_sids(c.split, c.now_f)
    js = c.job_state()
    done_ = {x for x, v in js.items() if v == '작업 완료'}
    urgent = [{'sid': x, 'name': c.nm(x), 'option': dec_i.loc[x, '추천']} for x in dec_i.index
              if dec_i.loc[x, '제약'].startswith('즉시') and x not in busy | done_] if len(dec_i) else []
    return {'team': team, 'items': items, 'urgent': urgent, 'rushH': cfg.parts.rush_h,
            'counts': {k: int((mine['구분'] == k).sum()) for k in ('지금', '지연', '예정', '완료', '중단')}}


def _task(w, c, tid):
    tf = c.tf().set_index('id')
    if tid not in tf.index:
        raise HTTPException(404, '없는 작업입니다')
    r = tf.loc[tid]
    if w.user and w.user['role'] == 'worker' and r['team'] != w.user['team']:
        raise HTTPException(403, '다른 팀 작업입니다')
    return r


@app.post('/api/tasks/{tid}/start')
def start(tid: int, w: Who = Depends(who)):
    c = ctx(w)
    r = _task(w, c, tid)
    d = E.load_checks(r['checks'])
    if any(not d.get(k) for k in E.PRE_KEYS):
        bad('사전 절차를 모두 체크해야 시작할 수 있습니다')
    if not w.store.start(tid, c.now_f):
        raise HTTPException(409, '이미 시작했거나 끝난 작업입니다')
    w.log.record('작업 시작', c.split, r['sid'], c.now_f, E.WORKER)
    return {'ok': True, 'message': f'{r["sid"]}: 작업 시작'}


class Checks(BaseModel):
    checks: dict


@app.post('/api/tasks/{tid}/checks')
def set_checks(tid: int, b: Checks, w: Who = Depends(who)):
    c = ctx(w)
    r = _task(w, c, tid)
    saved = E.load_checks(r['checks'])
    extra = {k: v for k, v in saved.items() if k not in E.PRE_KEYS + E.SAFETY_KEYS + E.POST_KEYS}
    d = {k: bool(b.checks.get(k)) for k in E.PRE_KEYS + E.POST_KEYS}
    d.update({k: b.checks.get(k) if b.checks.get(k) in E.SAFETY_OPTS else None for k in E.SAFETY_KEYS})
    d = E.fix_checks(d)
    w.store.set_checks(tid, json.dumps({**extra, **{k: v for k, v in d.items() if v}}, ensure_ascii=False))
    return {'checks': {**extra, **{k: v for k, v in d.items() if v}}}


class Complete(BaseModel):
    outcome: str
    details: str
    judged: str = '구분 못함'
    confirmed: str = '아직 모름'
    action: str = tasks.ACTIONS[0]
    action_why: str = ''
    permit_after: bool = False


@app.post('/api/tasks/{tid}/complete')
def complete(tid: int, b: Complete, w: Who = Depends(who)):
    c = ctx(w)
    r = _task(w, c, tid)
    urgent_job = r['priority'] == '긴급' or '즉시 조치' in str(r['title'])
    if urgent_job and not b.permit_after:
        bad('즉시 조치 작업은 작업 허가서를 사후 처리한 뒤 완료 보고를 제출하세요.')
    try:
        done = w.store.complete(tid, c.now_f, b.outcome, b.details, b.judged, b.confirmed, b.action, b.action_why)
    except ValueError as e:
        bad(e)
    if not done:
        raise HTTPException(409, '이미 완료된 작업입니다')
    if urgent_job:
        w.store.set_checks(tid, json.dumps({**E.load_checks(r['checks']), 'permit_after': True}, ensure_ascii=False))
    w.log.record('작업 완료', c.split, r['sid'], c.now_f, E.WORKER, option=b.action,
                 note=b.details + (f' · 지시와 다름: {b.action_why}' if b.action != tasks.ACTIONS[0] else '')
                 + (' · 작업 허가서 사후 처리 완료' if urgent_job else ''))
    w.log.record('점검 결과', c.split, r['sid'], c.now_f, E.WORKER, judged=b.judged, confirmed=b.confirmed, note=b.outcome)
    return {'ok': True, 'message': f'{c.nm(r["sid"])}: 완료 보고'}


class Stop(BaseModel):
    why: str


@app.post('/api/tasks/{tid}/stop')
def stop(tid: int, b: Stop, w: Who = Depends(who)):
    c = ctx(w)
    r = _task(w, c, tid)
    try:
        ok = w.store.stop(tid, c.now_f, b.why)
    except ValueError as e:
        bad(e)
    if not ok:
        raise HTTPException(409, '이미 완료됐거나 중단한 작업입니다')
    w.log.record('작업 중단', c.split, r['sid'], c.now_f, E.WORKER, note=f'#{tid} 사유: {b.why.strip()}')
    return {'ok': True, 'message': f'{c.nm(r["sid"])}: 작업 중단 · 설비관리자에게 알렸습니다'}


class Urgent(BaseModel):
    sid: str
    team: str
    option: str = ''


@app.post('/api/urgent')
def urgent_create(b: Urgent, w: Who = Depends(who)):
    team = _worker_team(w, b.team)
    c = ctx(w)
    absent = w.store.absent_at(c.now_f)
    mem = [m for m in tasks.TEAMS[team] if m not in absent][:4]
    try:
        ok = w.store.add(c.split, b.sid, f'{b.sid} 즉시 조치: 두 부품 교체', team, '긴급', c.now_f, c.now_f, tasks.DEFAULT_GUIDE,
                         '지금 교체해도 완료 전 고장 확률이 안전 기준을 넘음 → 배정 전 현장이 긴급 작업을 만듦', b.option, mem)
    except ValueError as e:
        bad(e)
    if not ok:
        raise HTTPException(409, f'{c.nm(b.sid)}: 이미 열린 작업이 있습니다')
    tid = int(w.store.open_task(c.split, b.sid, c.now_f)['id'])
    return {'ok': True, 'id': tid, 'message': f'{c.nm(b.sid)}: 즉시 조치 작업을 만들었습니다 · 사전 절차를 체크하세요'}


# 작업 수정 · 배정 취소 (설비관리자)

class Edit(BaseModel):
    state: str
    team: str
    members: list[str] = []
    leader: str = ''
    priority: str
    due_h: float
    guide: str
    reason: str
    why: str
    at: float | None = None
    outcome: str = '이상 발견'
    judged: str = '구분 못함'
    confirmed: str = '아직 모름'
    details: str = ''


@app.put('/api/tasks/{tid}')
def edit(tid: int, b: Edit, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    cur = _task(w, c, tid)
    if not b.why.strip():
        bad('수정 사유를 입력해야 저장됩니다.')
    if b.state not in tasks.STATUSES:
        bad('상태는 대기·진행 중·완료')
    done_ = cur['상태'] == '완료'
    at = c.now_f
    if b.state == '완료' and not done_:
        lo = min(float(cur['started_h'] if pd.notna(cur['started_h']) else cur['created_h']), c.now_f)
        at = c.now_f if b.at is None else float(b.at)
        if not lo - 1e-6 <= at <= c.now_f + 1e-6:
            bad(f'보고 시각은 {E.clock(lo)} ~ {E.clock(c.now_f)} 사이로 고르세요.')
    try:
        w.store.edit(tid, b.team, b.priority, b.due_h, b.guide, b.reason, [m for m in b.members if tasks.TEAM_OF.get(m) == b.team], b.leader)
        if b.state != cur['상태']:
            if b.state == '완료':
                w.store.set_status(tid, '완료', at, b.outcome, b.details, b.judged, b.confirmed)
                w.log.record('작업 완료', c.split, cur['sid'], at, E.MANAGER, note=f'상태 변경 #{tid}: {b.details}')
                w.log.record('점검 결과', c.split, cur['sid'], at, E.MANAGER, judged=b.judged, confirmed=b.confirmed, note=b.outcome)
            else:
                w.store.set_status(tid, b.state, c.now_f)
                if done_:
                    w.log.record('점검 결과', c.split, cur['sid'], float(cur['completed_h']), E.MANAGER, judged='구분 못함',
                                 confirmed='아직 모름', note=f'완료 되돌림 #{tid}')
                w.log.record('작업 시작' if b.state == '진행 중' else '승인', c.split, cur['sid'], c.now_f, E.MANAGER,
                             note=f'상태 변경 #{tid}: {cur["상태"]} → {b.state}')
        elif done_:
            w.store.edit_report(tid, b.outcome, b.details, b.judged, b.confirmed)
            if (b.outcome, b.judged, b.confirmed) != (cur['outcome'], cur['judged'], cur['confirmed']):
                w.log.record('점검 결과', c.split, cur['sid'], float(cur['completed_h']), E.MANAGER, judged=b.judged,
                             confirmed=b.confirmed, note=f'완료 보고 수정 #{tid}: {b.outcome}')
        w.log.record('작업 수정', c.split, cur['sid'], c.now_f, E.MANAGER, note=f'#{tid} 사유: {b.why.strip()}')
    except ValueError as e:
        bad(e)
    return {'ok': True, 'message': f'#{tid} 수정했습니다' + (f' ({cur["상태"]} → {b.state})' if b.state != cur['상태'] else '')}


class Cancel(BaseModel):
    why: str


@app.post('/api/tasks/{tid}/cancel')
def cancel(tid: int, b: Cancel, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    cur = _task(w, c, tid)
    if not b.why.strip():
        bad('배정 취소 사유를 입력하세요')
    if not w.store.cancel(tid):
        raise HTTPException(409, '완료된 작업이거나 이미 없는 작업입니다')
    w.log.record('배정 취소', c.split, cur['sid'], c.now_f, E.MANAGER, note=f'#{tid} {cur["title"]} · 사유: {b.why.strip()}')
    return {'ok': True, 'message': f'{cur["sid"]}: 배정을 취소했습니다. 다시 배정할 수 있습니다'}


@app.get('/api/tasks.csv')
def tasks_csv(w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    data = tasks.safe_csv(c.tf().drop(columns=['split']))
    return Response(data, media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="cast_pdm_tasks_{c.split}.csv"'})


# 인원 현황

@app.get('/api/absences')
def absences(w: Who = Depends(who)):
    w.need_manager()
    ab = w.store.absences()
    return {'rows': [{'name': r['name'], 'team': r['team'], 'kind': r['kind'], 'start_h': E.num(r['start_h']), 'end_h': E.num(r['end_h']),
                      'note': r['note'] or ''} for r in ab.to_dict('records')]}


class Absences(BaseModel):
    rows: list[dict]


@app.put('/api/absences')
def save_absences(b: Absences, w: Who = Depends(who)):
    w.need_manager()
    try:
        n = w.store.save_absences(b.rows)
    except (ValueError, TypeError) as e:
        bad(e)
    return {'ok': True, 'message': f'부재 {n}건을 저장했습니다' if n else '전원 출근으로 바꿨습니다'}


@app.post('/api/absences/preset')
def absence_preset(w: Who = Depends(who)):
    w.need_manager()
    rows = [{'name': n_, 'kind': k_, 'start_h': a_, 'end_h': b_, 'note': m_} for n_, k_, a_, b_, m_ in E.ABSENCE_PRESET]
    return save_absences(Absences(rows=rows), w)


# 하위 페이지

@app.get('/api/policy')
def policy(w: Who = Depends(who)):
    w.need_manager()
    return ctx(w).policy()


@app.get('/api/policy/{sid}')
def policy_option(sid: str, override: str | None = None, w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    try:
        po = json.loads(override) if override else None
    except ValueError:
        bad('덮어쓴 확률 형식이 맞지 않습니다')
    return c.policy_option(sid, po)


@app.put('/api/assume')
def set_assume(b: dict, w: Who = Depends(who)):
    w.need_manager()
    d = ENG.default_assumptions()
    clean = {}
    for k, v in b.items():
        if k in d and k != 'mode':
            try:
                clean[k] = type(d[k])(v)
            except (TypeError, ValueError):
                bad(f'{k} 값이 맞지 않습니다')
    w.store.set_setting('assume', json.dumps(clean))
    return {'ok': True}


@app.delete('/api/assume')
def reset_assume(w: Who = Depends(who)):
    w.need_manager()
    w.store.set_setting('assume', '{}')
    return {'ok': True}


@app.get('/api/model')
def model(w: Who = Depends(who)):
    w.need_manager()
    return ctx(w).model_info()


@app.get('/api/ontology.ttl')
def ontology_ttl(w: Who = Depends(who)):
    w.need_manager()
    c = ctx(w)
    return Response(c.ontology_graph().serialize(format='turtle'), media_type='text/turtle; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="casting_{c.split}_{c.now}h.ttl"'})


@app.get('/api/parts')
def parts(p: float = Query(0.5, ge=0.5, le=1.0), w: Who = Depends(who)):
    w.need_manager()
    return ctx(w).parts_calc(p)


# 빌드한 화면(web/dist, `npm run build`)이 있으면 같은 서버가 내준다. 반드시 API 경로들 뒤에 둔다
DIST = ROOT / 'web' / 'dist'
if DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount('/', StaticFiles(directory=DIST, html=True), name='web')
