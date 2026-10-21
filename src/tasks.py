"""정비 작업 지시: 관리자가 배정하고 작업자가 시작·완료 보고한다 (팀원 설계 화면의 작업 관리 + 재생 시각).

    from src.tasks import TaskStore
    store = TaskStore(path)                  # 로컬: SQLite 파일 / TaskStore(None): 세션 안에서만 (공개 클라우드)
    store = TaskStore(db=pg)                 # 공용 DB (src/db.py, Supabase Postgres): 로그인한 설비관리자·작업자가 같이 본다
    store.add(split, sid, title, team, priority, due_h, created_h, guide, reason, option)   # 같은 설비에 열린 작업이 있으면 False
    store.start(task_id, hour); store.complete(task_id, hour, outcome, details, judged, confirmed)
    store.edit(...)  /  store.edit_report(task_id, outcome, details, judged, confirmed)   # 완료된 작업의 보고 고치기
    store.stop(task_id, hour, why)           # 작업 중단 (사유 필수, 2026-10-21)
    store.frame(split, now)                  # 지금 시각에 보이는 작업과 상태(대기·진행 중·완료·중단)
    store.setting(key) / store.set_setting(key, value)   # 함께 보는 값 (재생 시각 등)
    store.version()                          # 바뀌었는지 가볍게 확인 (자동 갱신·알림용)

재생 시각: 데이터는 시나리오 시작 후 0~72시간을 재생한다. 그래서 작업의 배정·시작·완료와 예정 시각을 시나리오 시각(h)으로
적고, 상태는 '지금 시각'으로 계산한다(슬라이더를 과거로 옮기면 그때의 작업만 보인다). 실제 기록 시각은 이력에만 남긴다.

저장 (팀 결정 2026-10-15, 공용 DB 2026-10-16)
  로컬      SQLite 파일 actions/tasks.sqlite3 (.gitignore). 연결은 쓸 때마다 열고 닫는다
  공용 DB   Supabase Postgres (castpdm.tasks). 구글 로그인을 켠 배포에서만 쓴다. 한 번 화면을 그리는 동안은 처음 읽은 표를 다시 쓴다
  클라우드  로그인 없는 공개 화면은 서버 파일·DB에 쓰지 않고 보는 사람의 세션에만 둔다

검사
  - 팀·우선순위·결과·부품은 정해진 값만, 설비 ID는 S0000 형식, 시각은 0~240시간
  - 글은 길이 제한과 제어 문자 제거(줄바꿈은 유지). 내려받는 CSV는 수식으로 읽히는 첫 글자(= + - @)를 막는다
  - 같은 설비에 열린(완료 전) 작업이 있으면 배정하지 않는다 (SQLite는 한 트랜잭션 안에서, Postgres는 설비별 잠금을 잡고 확인하고 넣는다)
"""

import contextlib
import datetime as dt
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

# 가상 정비팀 (2026-10-21): 팀당 5명, 첫 사람이 반장. 이름은 시연용 가상 이름
TEAMS = {'정비 1팀': ['김도현', '이서준', '박지훈', '최민재', '한우진'], '정비 2팀': ['강태윤', '조성민', '윤재혁', '장현우', '임동건']}
LEADERS = {k: v[0] for k, v in TEAMS.items()}
TEAM_OF = {n: k for k, v in TEAMS.items() for n in v}
# 인원 부재 (2026-10-21): 설비관리자가 '인원 현황'에서 사람마다 종류·기간을 고른다. 기간은 재생 시각(h, 0 = 1일차 00:00)
ABSENCE_KINDS = ['연차', '반차(오전)', '반차(오후)', '병가', '출산휴가', '육아휴직', '교육', '예비군', '출장']
PRIORITIES = ['긴급', '높음', '보통']
OUTCOMES = ['이상 없음', '이상 발견', '추가 점검 필요']
STATUSES = ['대기', '진행 중', '완료']      # 설비관리자가 고를 수 있는 상태 (중단은 작업자가 '작업 중단'으로만)
# 완료 보고의 실제 조치 (2026-10-21): 지시(두 부품 교체)와 다르게 했으면 이유를 적어야 한다. 재고는 실제 조치대로 뺀다
ACTIONS = ['두 부품 교체', '밸브만 교체', '진동장치만 교체', '교체 안 함 (점검만)']
ACTION_PARTS = {'두 부품 교체': ('valve', 'oscillator'), '밸브만 교체': ('valve',), '진동장치만 교체': ('oscillator',),
                '교체 안 함 (점검만)': ()}
JUDGED = ['valve', 'oscillator', '구분 못함']
CONFIRMED = ['valve', 'oscillator', '아직 모름']
LIMITS = {'title': 80, 'guide': 600, 'reason': 600, 'details': 500, 'option': 60, 'why': 200}
COLUMNS = ['id', 'split', 'sid', 'title', 'team', 'priority', 'due_h', 'created_h', 'started_h', 'completed_h', 'guide',
           'reason', 'option', 'outcome', 'details', 'judged', 'confirmed', 'created_at', 'updated_at', 'completed_at',
           'stopped_h', 'stop_why', 'action', 'action_why', 'members', 'leader', 'checks']
# 2026-10-21에 덧붙인 칸 (기존 표에 없으면 ALTER TABLE로 추가한다. 지우거나 바꾸지 않는다)
#   stopped_h 작업 중단 시각 · stop_why 중단 사유 · action 실제 조치 · action_why 지시와 다르게 한 이유
NEW_COLUMNS = {'stopped_h': 'REAL', 'stop_why': "TEXT NOT NULL DEFAULT ''", 'action': "TEXT NOT NULL DEFAULT ''",
               'action_why': "TEXT NOT NULL DEFAULT ''", 'members': "TEXT NOT NULL DEFAULT ''",   # members: 배정한 작업자 (쉼표로)
               'leader': "TEXT NOT NULL DEFAULT ''", 'checks': "TEXT NOT NULL DEFAULT ''"}   # 작업책임자 · 체크리스트 상태(JSON)
_SID = re.compile(r'^S\d{4}$')
_CTRL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')     # 줄바꿈(\n, \x0a)은 남긴다

SCHEMA = """CREATE TABLE IF NOT EXISTS tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT, split TEXT NOT NULL, sid TEXT NOT NULL, title TEXT NOT NULL,
    team TEXT NOT NULL, priority TEXT NOT NULL, due_h REAL NOT NULL, created_h REAL NOT NULL,
    started_h REAL, completed_h REAL, guide TEXT NOT NULL, reason TEXT NOT NULL, option TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '', details TEXT NOT NULL DEFAULT '', judged TEXT NOT NULL DEFAULT '',
    confirmed TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT '', stopped_h REAL, stop_why TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '', action_why TEXT NOT NULL DEFAULT '', members TEXT NOT NULL DEFAULT '',
    leader TEXT NOT NULL DEFAULT '', checks TEXT NOT NULL DEFAULT '')"""
ABSENCE_SCHEMA = """CREATE TABLE IF NOT EXISTS absences(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, team TEXT NOT NULL, kind TEXT NOT NULL,
    start_h REAL NOT NULL, end_h REAL NOT NULL, note TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)"""
ABSENCE_COLUMNS = ['id', 'name', 'team', 'kind', 'start_h', 'end_h', 'note', 'updated_at']
SETTINGS_SCHEMA = 'CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)'

DEFAULT_GUIDE = ('① 안전 절차·작업 허가 확인\n'
                 '② 설비 정지와 ILS 절차 준수 여부 확인\n'
                 '③ 진동장치·밸브 상태 점검 (센서로는 원인 부품을 구분할 수 없어 두 부품 모두 확인)\n'
                 '④ 교체 뒤 몰드 레벨·제어 지연·인발 부하·진동 스트로크 신호 확인\n'
                 '⑤ 어느 부품이 문제였는지 완료 보고에 기록')


def _now():
    return dt.datetime.now().isoformat(timespec='seconds')


def clean(text, limit):
    return _CTRL.sub(' ', str(text or '')).strip()[:limit]


def _hour(x):
    h = float(x)
    if not (0 <= h <= 240) or math.isnan(h):
        raise ValueError(f'시각 범위 밖: {x}')
    return h


def _members(team, members):
    """작업자 목록 → 저장 문자열 ('김도현,이서준'). 그 팀 사람만"""
    names = [str(m).strip() for m in (members or []) if str(m).strip()]
    bad = [m for m in names if m not in TEAMS.get(team, [])]
    if bad:
        raise ValueError(f'{team} 팀원이 아님: {", ".join(bad)}')
    return ','.join(dict.fromkeys(names))


def pick_leader(members, leader):
    """작업책임자: 작업자 중 한 명. 비었으면 작업자 중 반장, 없으면 첫 사람"""
    names = [m for m in str(members).split(',') if m]
    if leader in names:
        return leader
    return next((m for m in names if m in LEADERS.values()), names[0] if names else '')


def _missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _conflict(created_h, completed_h, new_h, stopped_h=None):
    """같은 설비의 기존 작업이 새 배정과 겹치는가: 아직 끝나기 전이거나(재생에서 미래 작업 포함), 새 배정 시각에 열려 있다.
    끝 = 완료 시각, 완료 전에 중단했으면 중단 시각 (중단한 설비는 다시 배정할 수 있다)"""
    end = completed_h if not _missing(completed_h) else (None if _missing(stopped_h) else stopped_h)
    return end is None or created_h <= new_h < end


def safe_csv(df):
    """내려받기용 CSV (표 계산기 수식 주입 방지, 한글 엑셀용 BOM)"""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@') else v)
    return out.to_csv(index=False).encode('utf-8-sig')


class TaskStore:
    """db가 있으면 공용 DB(Postgres), path가 있으면 SQLite(로컬), 둘 다 없으면 메모리(세션)"""

    def __init__(self, path=None, db=None):
        self.db = db
        self.path = Path(path) if path and not db else None
        self.rows = []
        self.settings = {}
        self._cache = None               # 공용 DB: 한 번 그리는 동안 읽은 표 (쓰면 지운다)
        self.absence_rows = []           # 세션 저장일 때 부재 기록
        self.T, self.S, self.A = ('castpdm.tasks', 'castpdm.settings', 'castpdm.absences') if db else ('tasks', 'settings', 'absences')
        if self.path:
            with self._conn():
                pass                     # 파일과 표를 미리 만든다

    @property
    def shared(self):
        """여러 사람이 같이 보는 저장소인가 (세션 메모리가 아님)"""
        return bool(self.db or self.path)

    @contextlib.contextmanager
    def _conn(self):
        """연결을 열고, 끝나면 커밋(오류면 되돌림)하고 닫는다 (sqlite3의 with는 닫지 않아 직접 닫는다)"""
        if self.db:
            with self.db.conn() as c:
                yield c
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)      # 폴더·파일이 지워져도 다시 만든다
        conn = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('PRAGMA busy_timeout=20000')
            conn.execute(SCHEMA)
            conn.execute(SETTINGS_SCHEMA)
            conn.execute(ABSENCE_SCHEMA)
            have = {r['name'] for r in conn.execute('PRAGMA table_info(tasks)').fetchall()}
            for col, typ in NEW_COLUMNS.items():       # 예전 파일에 새 칸 덧붙이기
                if col not in have:
                    conn.execute(f'ALTER TABLE tasks ADD COLUMN {col} {typ}')
            yield conn
        finally:
            conn.close()

    # 함께 보는 값 (재생 시각·데이터 세트: 설비관리자가 바꾸면 작업자 화면도 따라간다)

    def setting(self, key, default=None):
        if not self.shared:
            return self.settings.get(key, default)
        with self._conn() as c:
            r = c.execute(f'SELECT value FROM {self.S} WHERE key=?', [key]).fetchone()
        return r['value'] if r else default

    def set_setting(self, key, value):
        value = clean(value, 100)
        if not self.shared:
            self.settings[key] = value
            return
        with self._conn() as c:
            c.execute(f'INSERT INTO {self.S}(key, value, updated_at) VALUES (?, ?, ?) '
                      'ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at',
                      [key, value, _now()])

    def version(self):
        """작업 표·함께 보는 값이 바뀌었는지 가볍게 확인 (작은 쿼리 두 개). 자동 갱신이 이 값이 바뀔 때만 화면을 다시 그린다"""
        if not self.shared:
            return (len(self.rows), max([r['updated_at'] for r in self.rows], default=''), tuple(sorted(self.settings.items())),
                    len(self.absence_rows), max([r['updated_at'] for r in self.absence_rows], default=''))
        with self._conn() as c:
            r = c.execute(f"SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m, COALESCE(MAX(updated_at), '') AS u FROM {self.T}").fetchone()
            kv = tuple((x['key'], x['value']) for x in c.execute(f'SELECT key, value FROM {self.S} ORDER BY key').fetchall())
            a = c.execute(f"SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), '') AS u FROM {self.A}").fetchone()
        return (int(r['n']), int(r['m']), r['u'], kv, int(a['n']), a['u'])

    # 인원 부재 (2026-10-21)

    def absences(self):
        """부재 기록 전부 (이름·팀·종류·시작h·끝h·메모)"""
        if self.shared:
            with self._conn() as c:
                df = pd.DataFrame([dict(r) for r in c.execute(f'SELECT * FROM {self.A} ORDER BY name, start_h').fetchall()],
                                  columns=ABSENCE_COLUMNS)
        else:
            df = pd.DataFrame(self.absence_rows, columns=ABSENCE_COLUMNS)
        for k in ('start_h', 'end_h'):
            df[k] = pd.to_numeric(df[k]).astype(float)
        return df.reindex(columns=ABSENCE_COLUMNS)

    def save_absences(self, rows):
        """부재 기록을 통째로 바꾼다 (인원 현황 화면의 '저장'). rows: [{'name','kind','start_h','end_h','note'}].
        이름은 정비팀 명단에서만, 종류는 정해진 값만, 시작 < 끝. 맞지 않으면 ValueError"""
        stamp = dt.datetime.now().isoformat(timespec='microseconds')
        clean_rows = []
        for r in rows:
            name, kind = str(r.get('name', '')).strip(), str(r.get('kind', '')).strip()
            if name not in TEAM_OF or kind not in ABSENCE_KINDS:
                raise ValueError(f'명단에 없는 이름이거나 정해진 부재 종류가 아님: {name} {kind}')
            a, b = _hour(r['start_h']), _hour(r['end_h'])
            if not a < b:
                raise ValueError(f'{name}: 끝 시각이 시작 시각보다 뒤여야 합니다')
            clean_rows.append({'name': name, 'team': TEAM_OF[name], 'kind': kind, 'start_h': a, 'end_h': b,
                               'note': clean(r.get('note', ''), 100), 'updated_at': stamp})
        if not self.shared:
            self.absence_rows = [{'id': i + 1, **r} for i, r in enumerate(clean_rows)]
            return len(clean_rows)
        keys = ['name', 'team', 'kind', 'start_h', 'end_h', 'note', 'updated_at']
        with self._conn() as c:
            c.execute(f'DELETE FROM {self.A}')
            for r in clean_rows:
                c.execute(f'INSERT INTO {self.A} ({",".join(keys)}) VALUES ({",".join("?" * len(keys))})', [r[k] for k in keys])
        self.set_setting('absences_saved', stamp)    # 모두 지웠을 때도 version이 바뀌게 (자동 갱신으로 다른 화면에 반영)
        return len(clean_rows)

    def absent_at(self, hour):
        """그 시각에 부재 중인 사람 {이름: 종류}"""
        df = self.absences()
        hit = df[(df['start_h'] <= hour) & (df['end_h'] > hour)]
        return dict(zip(hit['name'], hit['kind']))

    # 읽기

    def all(self, split=None):
        if self.db and self._cache is not None:
            df = self._cache
        elif self.shared:
            with self._conn() as c:
                df = pd.DataFrame([dict(r) for r in c.execute(f'SELECT * FROM {self.T} ORDER BY id').fetchall()],
                                  columns=COLUMNS)
            for k in ('due_h', 'created_h', 'started_h', 'completed_h', 'stopped_h'):
                df[k] = pd.to_numeric(df[k]).astype(float)
            df['id'] = df['id'].astype(int)
            if self.db:
                self._cache = df
        else:
            df = pd.DataFrame(self.rows, columns=COLUMNS)
        df = df.reindex(columns=COLUMNS).copy()
        df['stopped_h'] = pd.to_numeric(df['stopped_h']).astype(float)
        for k in ('stop_why', 'action', 'action_why', 'members', 'leader', 'checks'):
            df[k] = df[k].fillna('').astype(str)
        return df[df['split'] == split].reset_index(drop=True) if split else df

    def frame(self, split, now):
        """지금 시각에 보이는 작업(배정 시각 ≤ 지금)과 그때의 상태"""
        df = self.all(split)
        df = df[df['created_h'] <= now].copy()
        done = df['completed_h'].notna() & (df['completed_h'] <= now)
        started = df['started_h'].notna() & (df['started_h'] <= now)
        stopped = ~done & df['stopped_h'].notna() & (df['stopped_h'] <= now)
        df['상태'] = '대기'
        df.loc[started, '상태'] = '진행 중'
        df.loc[done, '상태'] = '완료'
        df.loc[stopped, '상태'] = '중단'
        late = ~done & (df['due_h'] < now)
        df['구분'] = '예정'
        df.loc[~done & ((df['due_h'] <= now) | started), '구분'] = '지금'     # 예정 시각 전에 시작했어도 지금 하는 일 (2026-10-21)
        df.loc[late & ~started, '구분'] = '지연'
        df.loc[done, '구분'] = '완료'
        df.loc[stopped, '구분'] = '중단'
        return df.sort_values(['구분', 'priority', 'due_h', 'id'],
                              key=lambda s: s.map({'중단': 0, '지연': 1, '지금': 2, '예정': 3, '완료': 4, '긴급': 0, '높음': 1, '보통': 2})
                              if s.name in ('구분', 'priority') else s).reset_index(drop=True)

    def open_task(self, split, sid, now):
        """지금 시각에 열린(완료 전) 작업 한 줄 (없으면 None)"""
        df = self.frame(split, now)
        df = df[(df['sid'] == sid) & ~df['상태'].isin(['완료', '중단'])]
        return None if df.empty else df.iloc[0].to_dict()

    def blocking(self, split, sid, now):
        """now에 이 설비로 배정하면 막는 작업 (add와 같은 규칙: 완료 전이거나 now에 열려 있던 작업). 없으면 None"""
        df = self.all(split)
        df = df[df['sid'] == sid]
        hit = df[[_conflict(r.created_h, r.completed_h, now, r.stopped_h) for r in df.itertuples()]]
        return None if hit.empty else hit.iloc[-1].to_dict()

    def blocked_sids(self, split, now):
        df = self.all(split)
        return {r.sid for r in df.itertuples() if _conflict(r.created_h, r.completed_h, now, r.stopped_h)}

    # 쓰기

    def add(self, split, sid, title, team, priority, due_h, created_h, guide, reason, option='', members=(), leader=''):
        """작업 배정. 같은 설비에 지금 열린 작업이 있으면 넣지 않고 False. members: 배정한 작업자 (그 팀 사람만)"""
        if split not in ('test', 'validation') or not _SID.match(str(sid)):
            raise ValueError('데이터 세트·설비 ID 형식이 아님')
        if team not in TEAMS or priority not in PRIORITIES:
            raise ValueError('팀·우선순위는 정해진 값만')
        members = _members(team, members)
        leader = pick_leader(members, leader)
        title = clean(title, LIMITS['title'])
        if not title:
            raise ValueError('작업명을 입력하세요')
        stamp = _now()
        row = {'split': split, 'sid': sid, 'title': title, 'team': team, 'priority': priority, 'due_h': _hour(due_h),
               'created_h': _hour(created_h), 'started_h': None, 'completed_h': None,
               'guide': clean(guide, LIMITS['guide']), 'reason': clean(reason, LIMITS['reason']),
               'option': clean(option, LIMITS['option']), 'outcome': '', 'details': '', 'judged': '', 'confirmed': '',
               'created_at': stamp, 'updated_at': stamp, 'completed_at': '',
               'stopped_h': None, 'stop_why': '', 'action': '', 'action_why': '', 'members': members, 'leader': leader,
               'checks': ''}
        keys = [k for k in COLUMNS if k != 'id']
        insert = f'INSERT INTO {self.T} ({",".join(keys)}) VALUES ({",".join("?" * len(keys))})'
        select = f'SELECT created_h, completed_h, stopped_h FROM {self.T} WHERE split=? AND sid=?'
        if self.db:
            self._cache = None
            with self._conn() as c:      # 한 트랜잭션: 이 설비의 잠금을 잡고(다른 사람의 동시 배정은 기다림) 확인하고 넣는다
                c.execute('SELECT pg_advisory_xact_lock(hashtext(?))', [f'castpdm:{split}:{sid}'])
                rows = c.execute(select, (split, sid)).fetchall()
                if any(_conflict(r['created_h'], r['completed_h'], row['created_h'], r['stopped_h']) for r in rows):
                    return False
                c.execute(insert, [row[k] for k in keys])
            return True
        if self.path:
            with self._conn() as c:
                c.execute('BEGIN IMMEDIATE')
                try:
                    rows = c.execute(select, (split, sid)).fetchall()
                    if any(_conflict(r['created_h'], r['completed_h'], row['created_h'], r['stopped_h']) for r in rows):
                        c.execute('ROLLBACK')
                        return False
                    c.execute(insert, [row[k] for k in keys])
                    c.execute('COMMIT')
                except Exception:
                    c.execute('ROLLBACK')
                    raise
            return True
        if any(_conflict(r['created_h'], r['completed_h'], row['created_h'], r.get('stopped_h')) for r in self.rows
               if r['split'] == split and r['sid'] == sid):
            return False
        row['id'] = max([r['id'] for r in self.rows], default=0) + 1
        self.rows.append(row)
        return True

    def _update(self, task_id, fields, where_open=False, where_done=False):
        fields = {**fields, 'updated_at': dt.datetime.now().isoformat(timespec='microseconds')}   # 1초 안에 두 번 바뀌어도 version이 달라지게
        self._cache = None
        if self.shared:
            with self._conn() as c:
                sets = ','.join(f'{k}=?' for k in fields)
                sql = (f'UPDATE {self.T} SET {sets} WHERE id=?' + (' AND completed_h IS NULL' if where_open else '')
                       + (' AND completed_h IS NOT NULL' if where_done else ''))
                return c.execute(sql, [*fields.values(), int(task_id)]).rowcount > 0
        for r in self.rows:
            if (r['id'] == int(task_id) and not (where_open and r['completed_h'] is not None)
                    and not (where_done and r['completed_h'] is None)):
                r.update(fields)
                return True
        return False

    def start(self, task_id, hour):
        return self._update(task_id, {'started_h': _hour(hour)}, where_open=True)

    def complete(self, task_id, hour, outcome, details, judged='구분 못함', confirmed='아직 모름', action=ACTIONS[0], action_why=''):
        """완료 보고: 결과와 수행 내용이 있어야 완료. 실제 조치가 지시(두 부품 교체)와 다르면 이유도 있어야 한다.
        이미 완료된 작업이면 False"""
        details = clean(details, LIMITS['details'])
        action_why = clean(action_why, LIMITS['why'])
        if outcome not in OUTCOMES or judged not in JUDGED or confirmed not in CONFIRMED or action not in ACTIONS:
            raise ValueError('결과·부품·실제 조치는 정해진 값만')
        if not details:
            raise ValueError('수행 내용을 입력해야 완료할 수 있습니다')
        if action != ACTIONS[0] and not action_why:
            raise ValueError('지시(두 부품 교체)와 다르게 했으면 그 이유를 입력해야 완료할 수 있습니다')
        h = _hour(hour)
        return self._update(task_id, {'completed_h': h, 'outcome': outcome, 'details': details, 'judged': judged,
                                      'confirmed': confirmed, 'completed_at': _now(), 'action': action,
                                      'action_why': action_why if action != ACTIONS[0] else ''}, where_open=True)

    def set_checks(self, task_id, checks):
        """작업 상세의 체크리스트 상태 저장 (JSON 문자열, 2026-10-21). 기기를 바꿔도 남고 설비관리자가 진행률을 본다"""
        return self._update(task_id, {'checks': str(checks)[:4000]})

    def stop(self, task_id, hour, why):
        """작업 중단 (작업자, 2026-10-21): 현장 상황이 지시와 달라 끝내지 못했을 때. 사유가 있어야 한다.
        중단한 작업은 '중단'으로 남고 그 설비는 다시 배정할 수 있다. 완료됐거나 이미 중단한 작업이면 False"""
        why = clean(why, LIMITS['why'])
        if not why:
            raise ValueError('중단 사유를 입력해야 중단할 수 있습니다')
        h = _hour(hour)
        df = self.all()
        df = df[df['id'] == int(task_id)]
        if df.empty or pd.notna(df.iloc[0]['stopped_h']):
            return False
        return self._update(task_id, {'stopped_h': h, 'stop_why': why}, where_open=True)

    def set_status(self, task_id, status, hour, outcome='', details='', judged='구분 못함', confirmed='아직 모름'):
        """설비관리자가 작업 상태를 직접 바꾼다 (잘못 누른 시작·완료 되돌리기 포함). 상태는 시작·완료 시각으로 정해진다.
        대기       시작·완료 시각과 완료 보고를 지운다
        진행 중    완료 시각과 완료 보고를 지운다 (시작 시각이 없으면 hour로)
        완료       완료 보고가 있어야 한다 (complete와 같은 검사). 시작 시각이 없으면 hour로
        반환: 바뀌었으면 True"""
        h = _hour(hour)
        df = self.all()
        df = df[df['id'] == int(task_id)]
        if df.empty or status not in STATUSES:
            raise ValueError('없는 작업이거나 정해진 상태가 아님')
        started = float(df.iloc[0]['started_h']) if pd.notna(df.iloc[0]['started_h']) else None
        # 상태를 고르면 중단도 풀린다 (중단한 작업을 다시 맡기는 경우)
        clear = {'completed_h': None, 'outcome': '', 'details': '', 'judged': '', 'confirmed': '', 'completed_at': '',
                 'stopped_h': None, 'stop_why': '', 'action': '', 'action_why': ''}
        if status == '대기':
            return self._update(task_id, {'started_h': None, **clear})
        if status == '진행 중':
            return self._update(task_id, {'started_h': started if started is not None else h, **clear})
        details = clean(details, LIMITS['details'])
        if outcome not in OUTCOMES or judged not in JUDGED or confirmed not in CONFIRMED:
            raise ValueError('결과·부품은 정해진 값만')
        if not details:
            raise ValueError('수행 내용을 입력해야 완료할 수 있습니다')
        return self._update(task_id, {'started_h': started if started is not None else h, 'completed_h': max(h, started or h),
                                      'outcome': outcome, 'details': details, 'judged': judged, 'confirmed': confirmed,
                                      'completed_at': _now(), 'stopped_h': None, 'stop_why': '', 'action': ACTIONS[0],
                                      'action_why': ''})

    def cancel(self, task_id):
        """배정 취소 (설비관리자): 완료 전 작업만 지운다. 그 설비는 다시 '배정할 경보'로 돌아간다.
        취소 이력은 진행 기록(src/actions.py의 '배정 취소')에 남긴다. 완료된 작업은 먼저 상태를 되돌려야 한다"""
        self._cache = None
        if self.shared:
            with self._conn() as c:
                return c.execute(f'DELETE FROM {self.T} WHERE id=? AND completed_h IS NULL', [int(task_id)]).rowcount > 0
        n = len(self.rows)
        self.rows = [r for r in self.rows if not (r['id'] == int(task_id) and r['completed_h'] is None)]
        return len(self.rows) < n

    def edit_report(self, task_id, outcome, details, judged, confirmed):
        """완료 보고 고치기 (설비관리자): 이미 완료된 작업만. 완료 시각은 그대로 둔다"""
        details = clean(details, LIMITS['details'])
        if outcome not in OUTCOMES or judged not in JUDGED or confirmed not in CONFIRMED:
            raise ValueError('결과·부품은 정해진 값만')
        if not details:
            raise ValueError('수행 내용은 비울 수 없습니다')
        return self._update(task_id, {'outcome': outcome, 'details': details, 'judged': judged, 'confirmed': confirmed},
                            where_done=True)

    def edit(self, task_id, team, priority, due_h, guide, reason, members=None, leader=None):
        if team not in TEAMS or priority not in PRIORITIES:
            raise ValueError('팀·우선순위는 정해진 값만')
        extra = {} if members is None else {'members': _members(team, members)}
        if members is not None:
            extra['leader'] = pick_leader(extra['members'], leader or '')
        return self._update(task_id, {**extra, 'team': team, 'priority': priority, 'due_h': _hour(due_h),
                                      'guide': clean(guide, LIMITS['guide']), 'reason': clean(reason, LIMITS['reason'])})
