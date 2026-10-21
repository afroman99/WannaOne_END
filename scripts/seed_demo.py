"""시연용 목업 데이터: 작업 지시·진행 기록을 지우고 시연 장면에 맞는 목업으로 채운다 (2026-10-21).

    python scripts/seed_demo.py --target local                 # 무엇을 지우고 넣을지 보여 주기만 (기본: 실제로 바꾸지 않음)
    python scripts/seed_demo.py --target local --apply         # 로컬(actions/ 아래 파일)에 적용
    python scripts/seed_demo.py --target shared --apply        # 공용 DB(Supabase, .streamlit/secrets.toml의 [db] url)에 적용
    python scripts/seed_demo.py --target shared --empty --apply  # 작업 목업 없이 비우기만 (배정은 시연 중에 직접, 2026-10-21 시연 방식)

기준: 테스트 세트 · 2일차 02:00 (재생 시각 26h). 시연 설정도 이 값으로 맞춘다.
인원 부재는 넣지 않는다(전원 출근). 라이브 데모 때 '인원 현황'에서 바꾼다.
넣는 작업은 앱의 저장 함수(src/tasks.py · src/actions.py)를 그대로 거쳐서 앱 규칙(같은 설비 중복 배정 금지, 사유 필수 등)을 지킨다.
배정할 경보가 남아 있어야 시연에서 배정 흐름을 보여 줄 수 있어서, 경보 설비 일부는 비워 둔다.
"""

import argparse
import sqlite3
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import actions, tasks  # noqa: E402

SPLIT, NOW = 'test', 26          # 테스트 세트 · 2일차 02:00
T1, T2 = list(tasks.TEAMS)
M1, M2 = tasks.TEAMS[T1], tasks.TEAMS[T2]
M = '정비 책임자'
W = '현장 작업'

# (설비, 작업명, 팀, 작업자, 우선순위, 배정 h, 예정 h, 상태, 시작 h, 끝 h, 그 밖) — 2일차 02:00에 보이는 모습
#   완료 3 (그중 1건 지시와 다름) · 진행 중 2 · 대기 2 · 지연 1 · 중단 1
JOBS = [
    ('S0300', 'S0300 점검·부품 준비', T1, M1[:4], '보통', 12, 13, '완료', 13, 17,
     {'outcome': '이상 발견', 'details': '진동장치 베어링 마모 초기, 두 부품 교체 후 신호 정상', 'judged': 'oscillator', 'confirmed': 'oscillator'}),
    ('S0340', 'S0340 점검·부품 준비', T2, M2[:4], '보통', 14, 15, '완료', 15, 19,
     {'outcome': '이상 없음', 'details': '밸브·진동장치 상태 양호, 교체하지 않고 운전 계속', 'judged': '구분 못함', 'confirmed': '아직 모름',
      'action': '교체 안 함 (점검만)', 'action_why': '열어 보니 두 부품 모두 정상 범위, 다음 정기 정지 때 다시 점검'}),
    ('S0313', 'S0313 상태 점검', T1, M1[1:5], '보통', 16, 17, '완료', 17, 21,
     {'outcome': '이상 발견', 'details': '밸브 작동 지연 확인, 밸브만 교체 후 제어 지연 정상', 'judged': 'valve', 'confirmed': 'valve',
      'action': '밸브만 교체', 'action_why': '진동장치는 측정값 정상이라 밸브만 교체'}),
    ('S0339', 'S0339 지금 두 부품 교체', T1, M1[:4], '높음', 22, 23, '진행 중', 23, None, {}),
    ('S0345', 'S0345 지금 두 부품 교체', T2, M2[:4], '높음', 23, 24, '진행 중', 24, None, {}),
    ('S0324', 'S0324 지금 두 부품 교체', T2, M2[1:5], '높음', 25, 27, '대기', None, None, {}),
    ('S0356', 'S0356 다음 정기 교체 때 두 부품 교체', T1, [M1[0], M1[2], M1[4]], '보통', 25, 30, '대기', None, None, {}),
    ('S0323', 'S0323 지금 두 부품 교체', T2, M2[:4], '높음', 18, 21, '지연', None, None, {}),
    ('S0311', 'S0311 지금 두 부품 교체', T1, M1[1:5], '긴급', 20, 20, '중단', 21, 23,
     {'stop_why': '교체용 진동장치 규격이 맞지 않아 작업 중단, 맞는 부품 조달 필요'}),
]
EXTRA = [   # (종류, 설비, h, 그 밖) 작업과 따로 남는 진행 기록
    ('부품 발주', 'S0339', 20, {'part': 'valve', 'option': '정상'}),
    ('부품 발주', 'S0302', 25, {'part': 'valve', 'option': '긴급'}),
    ('부품 발주', 'S0302', 25, {'part': 'oscillator', 'option': '긴급'}),
    ('보류', 'S0329', 25, {'option': '지금 두 부품 교체', 'note': '정기 정지 일정 확인 뒤 다시 판단'}),
]


def stores(target):
    """(작업 저장소, 진행 기록, 지우는 함수, 남은 건수 함수)"""
    if target == 'shared':
        from src import db
        import psycopg
        url = tomllib.load(open(ROOT / '.streamlit' / 'secrets.toml', 'rb'))['db']['url']   # 주소는 출력하지 않는다
        pg = []                         # 적용할 때만 연결한다 (연결하면 표·새 칸이 없을 때 만든다. 기존 기록은 지우지 않음)

        def counts():                   # 읽기만 (표를 만들거나 바꾸지 않는다)
            n = {}
            with psycopg.connect(url, prepare_threshold=None, connect_timeout=10, autocommit=True) as c:
                c.execute('SET default_transaction_read_only = on')
                for t in ('tasks', 'action_log', 'absences'):
                    try:
                        n[t] = c.execute(f'SELECT COUNT(*) FROM castpdm.{t}').fetchone()[0]
                    except psycopg.errors.UndefinedTable:
                        n[t] = 0
            return n

        def conn():
            if not pg:
                pg.append(db.Postgres(url))
            return pg[0]

        def wipe():
            with conn().conn() as c:
                for t in ('tasks', 'action_log', 'absences'):
                    c.execute(f'DELETE FROM castpdm.{t}')
        return (lambda: tasks.TaskStore(db=conn())), (lambda: actions.ActionLog(db=conn())), wipe, counts
    path, log_path = ROOT / 'actions' / 'tasks.sqlite3', ROOT / 'actions' / 'action_log.csv'

    def counts():
        n = {'tasks': 0, 'absences': 0, 'action_log': max(0, sum(1 for _ in open(log_path, encoding='utf-8')) - 1) if log_path.exists() else 0}
        if path.exists():
            con = sqlite3.connect(path)
            for t in ('tasks', 'absences'):
                try:
                    n[t] = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                except sqlite3.OperationalError:
                    pass
            con.close()
        return n

    def wipe():
        if path.exists():
            con = sqlite3.connect(path)
            for t in ('tasks', 'absences'):
                try:
                    con.execute(f'DELETE FROM {t}')
                except sqlite3.OperationalError:
                    pass
            con.commit()
            con.close()
        log_path.unlink(missing_ok=True)
    return (lambda: tasks.TaskStore(path)), (lambda: actions.ActionLog(log_path)), wipe, counts


def seed(store, log):
    for sid, title, team, members, prio, created, due, state, start, end, extra in JOBS:
        assert store.add(SPLIT, sid, title, team, prio, due, created, tasks.DEFAULT_GUIDE,
                         f'{sid} · 시연용 목업 · {state}', title.split(' ', 1)[1], members), f'{sid} 배정 실패'
        tid = int(store.all(SPLIT).iloc[-1]['id'])
        log.record('승인', SPLIT, sid, created, M, option=title.split(' ', 1)[1])
        if start is not None:
            store.start(tid, start)
            log.record('작업 시작', SPLIT, sid, start, W)
        if state == '완료':
            act = extra.get('action', tasks.ACTIONS[0])
            store.complete(tid, end, extra['outcome'], extra['details'], extra['judged'], extra['confirmed'], act, extra.get('action_why', ''))
            log.record('작업 완료', SPLIT, sid, end, W, option=act,
                       note=extra['details'] + (f' · 지시와 다름: {extra["action_why"]}' if act != tasks.ACTIONS[0] else ''))
            log.record('점검 결과', SPLIT, sid, end, W, judged=extra['judged'], confirmed=extra['confirmed'], note=extra['outcome'])
        if state == '중단':
            store.stop(tid, end, extra['stop_why'])
            log.record('작업 중단', SPLIT, sid, end, W, note=f'#{tid} 사유: {extra["stop_why"]}')
        if sid == 'S0324':      # 작업 수정 기록 하나 (사유 필수)
            log.record('작업 수정', SPLIT, sid, 25.5, M, note=f'#{tid} 사유: 정비 2팀 반장 교육 일정으로 작업자 조정')
    for kind, sid, h, kw in EXTRA:
        log.record(kind, SPLIT, sid, h, M, **kw)
    store.set_setting('split', SPLIT)
    store.set_setting('now', str(NOW))
    store.set_setting('clock_at', '')   # 멈춘 상태로 (시연에서 '적용'을 누르면 그때부터 흐른다)
    store.save_absences([])             # 전원 출근


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--target', choices=['local', 'shared'], required=True)
    ap.add_argument('--apply', action='store_true', help='실제로 지우고 넣는다 (없으면 보여 주기만)')
    ap.add_argument('--empty', action='store_true', help='작업 목업을 넣지 않고 비우기만 (시연 시각만 맞춘다)')
    a = ap.parse_args()
    make_store, make_log, wipe, counts = stores(a.target)
    before = counts()
    print(f'[{a.target}] 지금 기록: 작업 {before["tasks"]}건 · 진행 기록 {before["action_log"]}건 · 부재 {before["absences"]}건')
    print('  → 모두 지우고 비워 둡니다 (작업·진행 기록·부재 없음), 시연 시각 테스트 세트 · 2일차 02:00' if a.empty else
          f'  → 모두 지우고 목업을 넣습니다: 작업 {len(JOBS)}건 (완료 3 · 진행 중 2 · 대기 2 · 지연 1 · 중단 1), '
          f'따로 남는 진행 기록 {len(EXTRA)}건, 부재 없음, 시연 시각 테스트 세트 · 2일차 02:00')
    if not a.apply:
        print('  (보여 주기만 했습니다. 적용하려면 --apply)')
        return
    wipe()
    store = make_store()
    if a.empty:
        store.set_setting('split', SPLIT)
        store.set_setting('now', str(NOW))
        store.set_setting('clock_at', '')   # 멈춘 상태로 (시연에서 '적용'을 누르면 그때부터 흐른다)
        store.save_absences([])
    else:
        seed(store, make_log())
    after = counts()
    print(f'  적용했습니다: 작업 {after["tasks"]}건 · 진행 기록 {after["action_log"]}건 · 부재 {after["absences"]}건')


if __name__ == '__main__':
    main()
