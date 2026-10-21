"""온톨로지 액션: 사람이 화면에서 한 일(승인·보류·부품 발주·점검 결과)을 기록하고, 지금 시각의 상태로 읽는다.

    from src.actions import ActionLog
    log = ActionLog(path)            # 로컬: CSV 파일에 쌓는다 / ActionLog(None): 세션 안에서만 (클라우드)
    log = ActionLog(db=pg)           # 공용 DB (src/db.py의 castpdm.action_log). 만들 때 한 번 읽고, 기록은 DB와 메모리에 같이 쓴다
    log.record(kind='승인', split='validation', sid='S0248', hour=16, role='정비 책임자', option='다음 정기 교체 때 두 부품 함께')
    log.decision_state('validation', 16, last_mnt_start)     # 설비마다 지금 유효한 승인·보류
    log.inspection_accuracy()                                # 점검으로 원인 부품을 맞힌 비율 (되먹임)

설계: 결과보고서 7장 (서비스·작업 흐름)
  승인 요청       현장 작업     정비 결정을 정비 책임자에게 올린다
  승인 / 보류     정비 책임자   정비 결정 → 정비 작업으로 (보류면 다시 판단)
  작업 시작·완료  현장 작업     승인된 정비 작업의 진행 (완료하면 점검 결과 입력으로)
  부품 발주       정비 책임자   부품 재고·입고 예정 (자재 담당 역할이 없어 정비 책임자 화면에 둔다)
  점검 결과       현장 작업     점검으로 판단한 부품, 교체 뒤 확인된 원인 부품 → 고장 원인 부품

되먹임: 점검 판단과 확인된 원인이 함께 있는 기록이 쌓이면 "점검으로 원인 부품을 맞힐 확률"을 잰다. 이 값이 손익분기
(비용 계산기, 1시간 모델 0.60)를 넘고 기록이 MIN_CONFIRMED건 이상이면, 교체 방법 추천을 "두 부품"에서 "점검 후 한 부품"으로 바꾼다.

보안
  - 클라우드(공개, 비밀번호 없음)에서는 파일에 쓰지 않는다. 기록은 보는 사람의 세션에만 있다(다른 사람에게 보이지 않음)
  - 공용 DB는 구글 로그인으로 허락된 사람만 쓴다 (app/dashboard.py, src/auth.py)
  - 모든 값은 정해진 선택지·형식만 받는다. 메모는 200자, 제어 문자 제거, 표 계산기 수식으로 읽히는 첫 글자(= + - @)는 막는다
  - 파일에서 읽은 줄도 같은 검사를 거쳐, 맞지 않는 줄은 버린다
"""

import csv
import datetime as dt
import re
from pathlib import Path

import pandas as pd

KINDS = ['승인 요청', '승인', '보류', '작업 시작', '작업 완료', '부품 발주', '점검 결과', '배정 취소', '작업 수정']   # 작업 수정: 설비관리자가 작업을 고친 사유 (2026-10-21, 진행 단계는 바꾸지 않음)
JOB_KINDS = ['승인 요청', '승인', '보류', '작업 시작', '작업 완료', '배정 취소']     # 정비 결정 하나의 진행 단계 (배정 취소 = 다시 배정 대기)
ROLES = ['정비 책임자', '현장 작업']
SPLITS = ['test', 'validation']
PARTS = ['valve', 'oscillator']
JUDGED = PARTS + ['구분 못함']
CONFIRMED = PARTS + ['아직 모름']
COLUMNS = ['기록 시각', '데이터 세트', '설비', '시각(h)', '종류', '누가', '선택지', '부품', '점검 판단', '확인된 원인', '메모']
DB_COLUMNS = dict(zip(COLUMNS, ['recorded_at', 'split', 'sid', 'hour', 'kind', 'who', 'option', 'part', 'judged', 'confirmed', 'note']))
MIN_CONFIRMED = 10          # 되먹임을 쓰기 시작하는 최소 기록 수 (가정값)
MAX_NOTE = 200
_SID = re.compile(r'^S\d{4}$')
_CTRL = re.compile(r'[\x00-\x1f\x7f]')


def clean_note(text):
    text = _CTRL.sub(' ', str(text or '')).strip()[:MAX_NOTE]
    return "'" + text if text[:1] in ('=', '+', '-', '@') else text     # 표 계산기 수식 주입 방지


def validate(row):
    """한 줄을 검사해 정리한 dict를 돌려준다. 맞지 않으면 ValueError"""
    r = {c: ('' if row.get(c) is None or (isinstance(row.get(c), float) and pd.isna(row.get(c))) else row.get(c)) for c in COLUMNS}
    if r['종류'] not in KINDS:
        raise ValueError(f'모르는 종류: {r["종류"]!r}')
    if r['누가'] not in ROLES:
        raise ValueError(f'모르는 역할: {r["누가"]!r}')
    if r['데이터 세트'] not in SPLITS:
        raise ValueError(f'모르는 데이터 세트: {r["데이터 세트"]!r}')
    if not _SID.match(str(r['설비'])):
        raise ValueError(f'설비 ID 형식이 아님: {r["설비"]!r}')
    h = float(r['시각(h)'])
    if not 0 <= h <= 72:
        raise ValueError(f'시각 범위 밖: {h}')
    r['시각(h)'] = h
    if r['종류'] in ('승인', '보류', '배정 취소', '작업 수정') and r['누가'] != '정비 책임자':
        raise ValueError('승인·보류·배정 취소·작업 수정은 정비 책임자만')
    if r['종류'] == '부품 발주' and r['부품'] not in PARTS:
        raise ValueError(f'모르는 부품: {r["부품"]!r}')
    if r['종류'] == '점검 결과':
        if r['점검 판단'] not in JUDGED or r['확인된 원인'] not in CONFIRMED:
            raise ValueError('점검 판단·확인된 원인은 정해진 값만')
    r['선택지'] = clean_note(r['선택지'])[:60]
    r['메모'] = clean_note(r['메모'])
    if not r['기록 시각']:
        r['기록 시각'] = dt.datetime.now().isoformat(timespec='seconds')
    return r


class ActionLog:
    """db가 있으면 공용 DB, path가 있으면 CSV 파일(로컬), 없으면 메모리(세션)에 쌓는다"""

    def __init__(self, path=None, rows=None, db=None):
        self.db = db
        self.path = Path(path) if path and not db else None
        self.rows = rows if rows is not None else []
        if self.db:
            with self.db.conn() as c:
                raws = c.execute(f'SELECT {",".join(DB_COLUMNS.values())} FROM castpdm.action_log ORDER BY id').fetchall()
            for raw in raws:
                try:
                    self.rows.append(validate({k: raw[v] for k, v in DB_COLUMNS.items()}))
                except (ValueError, TypeError):
                    continue
        elif self.path and self.path.exists():
            with open(self.path, encoding='utf-8', newline='') as f:
                for raw in csv.DictReader(f):
                    try:
                        self.rows.append(validate(raw))
                    except (ValueError, TypeError):
                        continue                 # 손상되거나 조작된 줄은 버린다

    def record(self, kind, split, sid, hour, role, option='', part='', judged='', confirmed='', note=''):
        r = validate({'종류': kind, '데이터 세트': split, '설비': sid, '시각(h)': hour, '누가': role, '선택지': option,
                      '부품': part, '점검 판단': judged, '확인된 원인': confirmed, '메모': note})
        if self.db:                              # DB에 먼저 쓴다 (실패하면 화면에도 남기지 않는다)
            with self.db.conn() as c:
                c.execute(f'INSERT INTO castpdm.action_log ({",".join(DB_COLUMNS.values())}) '
                          f'VALUES ({",".join("?" * len(DB_COLUMNS))})', [r[k] for k in DB_COLUMNS])
        self.rows.append(r)
        if self.path:
            self.path.parent.mkdir(exist_ok=True)
            new = not self.path.exists()
            with open(self.path, 'a', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(r)
        return r

    def frame(self, split=None, now=None):
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        if split is not None:
            df = df[df['데이터 세트'] == split]
        if now is not None:
            df = df[df['시각(h)'] <= now]         # 재생: 지금 시각 이후에 한 일은 아직 없다
        return df.reset_index(drop=True)

    def decision_state(self, split, now, last_mnt_start=None, kinds=('승인', '보류')):
        """설비마다 지금 유효한 마지막 기록 {sid: row} (기본: 승인·보류). 정비가 시작되면 그 전의 기록은 끝난 것으로 본다"""
        df = self.frame(split, now)
        df = df[df['종류'].isin(list(kinds))]
        out = {}
        for _, r in df.iterrows():
            start = (last_mnt_start or {}).get(r['설비'])
            if start is not None and r['시각(h)'] < start:
                continue
            out[r['설비']] = r.to_dict()              # 뒤의 기록이 앞의 기록을 덮는다
        return out

    def job_state(self, split, now, last_mnt_start=None):
        """설비마다 정비 결정의 진행 단계 {sid: '승인 요청'|'승인'|'보류'|'작업 시작'|'작업 완료'}"""
        return {k: v['종류'] for k, v in self.decision_state(split, now, last_mnt_start, JOB_KINDS).items()}

    def orders(self, split, now, last_mnt_start=None):
        """설비마다 지금 유효한 발주 부품 {sid: set(part)}"""
        df = self.frame(split, now)
        out = {}
        for _, r in df[df['종류'] == '부품 발주'].iterrows():
            start = (last_mnt_start or {}).get(r['설비'])
            if start is None or r['시각(h)'] >= start:
                out.setdefault(r['설비'], set()).add(r['부품'])
        return out

    def inspection_accuracy(self):
        """(판단과 확인이 모두 있는 기록 수, 점검으로 원인 부품을 맞힌 비율). 모든 데이터 세트·시각을 합쳐 본다(경험이 쌓이는 것)"""
        df = self.frame()
        df = df[df['종류'] == '점검 결과']
        # 완료 보고를 고치면 같은 설비·같은 완료 시각으로 다시 기록한다 → 마지막 기록만 센다 (기록 자체는 이력으로 남김)
        df = df.drop_duplicates(['데이터 세트', '설비', '시각(h)'], keep='last')
        df = df[df['확인된 원인'].isin(PARTS)]
        if df.empty:
            return 0, None
        return len(df), float((df['점검 판단'] == df['확인된 원인']).mean())


def one_part_ready(n, acc, breakeven, min_n=MIN_CONFIRMED):
    """점검 후 한 부품만 교체해도 되는가 (되먹임 조건)"""
    return acc is not None and n >= min_n and acc >= breakeven
