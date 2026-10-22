"""CAST-PdM 계산 (React 화면용, 2026-10-22). app/dashboard.py의 계산 부분을 Streamlit 없이 옮겼다(dashboard.py는 2026-10-22 지움, git 기록에 남음).

    eng = Engine()                                   # 모델·데이터를 한 번 읽고 기억한다
    c = eng.ctx(split, now_f, tasks_store, action_log, assume, truth)   # 지금 시각 하나의 계산 묶음
    c.status_table() / c.work_plan() / c.decision(sid, crew) / c.detail(sid) / c.ontology(sid) ...

대시보드와 같은 규칙: 경보 유지 · 정비 중 · 정비 후 예측 대기 · 재고(고장 복구·완료 보고·발주 도착) ·
정비팀 하루 한도(출근 인원 × 8시간 ÷ 16인·시) · 즉시 정비 / 다음 정기 교체 / 사후정비 같은 기준 비교 · 온톨로지 근거 사슬.
화면 문장은 옛 dashboard.py 그대로 두고, 화면에서 쓰는 마크다운 강조(**…**)만 뺐다.
"""
import json
import re
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import actions, app_data, ontology, tasks                                     # noqa: E402
from src.dss import PARTS, PartsPolicy, fail_prob, field_costs, missing_parts, options, plans, recommend   # noqa: E402
from src.limitations import LIMITS                                                        # noqa: E402
from src.posko_data import load_scenarios                                                 # noqa: E402
from src.serving import cost_curve, explain, failure_hours, load_bundle, score_scenarios  # noqa: E402

SIGNAL_NAMES = {'Z_MLD-AMP': '몰드 레벨 진폭', 'Z_CONTROL_DELAY': '제어 지연', 'Z_WDL': '인발 부하', 'Z_OSC-STK': '진동 스트로크'}
SPLITS = {'test': '테스트 (정답 없음)', 'validation': '검증 (정답 있음)'}
LINE_NO = {'test': 1, 'validation': 2}
LABEL = {'경보': '경보 (WARNING)', '주의': '주의 (CAUTION)', '정상': '정상 (NORMAL)'}
WARN, CAU, NORM = LABEL['경보'], LABEL['주의'], LABEL['정상']
WORKER, MANAGER = '현장 작업', '정비 책임자'          # 기록(src/actions.py)에 쓰는 역할 이름
PART_NAMES = {'valve': '밸브', 'oscillator': '진동장치'}
END_H = 72
COST_MODES = ['데이터셋 점수', '현장 가정 (만 원)']
WORK_REPLACE, WORK_REPAIR = 16.0, 32.0     # 교체 · 고장 복구 (인·시)
SHIFT = (9, 17)                            # 정규 근무 9시~17시
OVERTIME = 1.5                             # 정규 근무 밖 인건비 배율
PLAN_GUESS_H = 12                          # 정기 교체 계획이 없을 때 비교에만 쓰는 '계획 정비' 시점

# 작업 상세 체크리스트 (옛 dashboard.py와 같다)
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
ABSENCE_PRESET = [('강태윤', '연차', 24, 48, ''), ('조성민', '교육', 33, 42, '안전 교육'), ('윤재혁', '출장', 24, 48, ''),
                  ('장현우', '병가', 0, 48, ''), ('최민재', '출산휴가', 0, 72, '배우자 출산'), ('박지훈', '반차(오전)', 33, 37, '')]


def num(x):
    """JSON에 넣을 수 있는 숫자 (NaN·None → None)"""
    if x is None:
        return None
    try:
        return None if pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return None


def clock(h):
    if h is None or pd.isna(h):
        return '—'
    m = int(round(float(h) * 60))
    d, m = divmod(m, 24 * 60)
    return f'{d + 1}일차 {m // 60:02d}:{m % 60:02d}'


def hms(h):
    sec = int(round(h * 3600))
    return f'{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def load_checks(raw):
    try:
        d = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        d = {}
    return d if isinstance(d, dict) else {}


def check_count(d):
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


def overtime_share(start_h, dur):
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


def remaining_median(curve, elapsed, cap=240.0):
    for h in np.arange(0.25, cap + 0.25, 0.25):
        if fail_prob(curve, elapsed, h) >= 0.5:
            return float(h)
    return None


def stock_text(st_):
    return f'밸브 {st_["valve"]} · 진동장치 {st_["oscillator"]}'


def people(r):
    lead = r['leader'] or tasks.pick_leader(r['members'], '')
    return ' · '.join(m + (' (작업책임자)' if m == lead else '') for m in str(r['members']).split(',') if m)


class Engine:
    """모델·데이터 (서버가 켜져 있는 동안 한 번만 읽는다)"""

    def __init__(self):
        self.cloud = not (ROOT / 'data' / 'posko').exists() and app_data.available()
        self.bundle = self._app_data()[0] if self.cloud else load_bundle()
        b = self.bundle
        self.th = b['threshold']
        self.step = b.get('step_h', 6)
        self.caution = b.get('caution')
        self.cth = self.caution['threshold'] if self.caution else None

    @lru_cache(maxsize=1)
    def _app_data(self):
        return app_data.load()

    @lru_cache(maxsize=2)
    def data(self, split):
        if self.cloud:
            pts, hr, contrib, unit, mnt, _ = self._app_data()[1][split]
            return pts, hr, contrib, unit, mnt
        ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
        pts, hr = score_scenarios(ids, self.bundle)
        contrib, unit = explain(self.bundle, pts)
        hr = hr.assign(DONE_H=hr['HOUR'] + 1)
        return pts, hr[['SCENARIO_ID', 'DONE_H', 'SCORE', *SIGNAL_NAMES]], contrib, unit, failure_hours(ids)

    @lru_cache(maxsize=2)
    def names(self, split):
        ids = sorted(self.data(split)[0]['SCENARIO_ID'].unique())
        names = {s: f'{LINE_NO[split]}연주 #{i:02d}' for i, s in enumerate(ids, 1)}
        f = ROOT / 'app' / 'equipment_names.csv'
        if f.exists():
            m = pd.read_csv(f, dtype=str).dropna()
            names.update({s: n.strip()[:30] for s, n in zip(m['SCENARIO_ID'], m['NAME']) if s in names and n.strip()})
        return names

    @lru_cache(maxsize=2)
    def plans(self, split):
        if self.cloud:
            return self._app_data()[1][split][5]
        ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
        start = load_scenarios().set_index('SCENARIO_ID')['START_AT']
        pl = plans(ids)
        pl['PLAN_H'] = (pl['PLAN_START'] - pl['SCENARIO_ID'].map(start)).dt.total_seconds() / 3600
        return pl

    def default_assumptions(self):
        D = self.bundle.get('dss')
        base, bc = D['config'], D['config'].costs
        bp = getattr(base, 'parts', PartsPolicy())
        return {'safety': float(base.safety_max), 'prep': int(base.prep_h), 'crew_wait': int(base.crew_wait_h),
                'n_crews': 2, 'crew_day_h': 8,
                'down_h': float(bc.downtime_h), 'penalty': float(bc.failure_penalty),
                'p_other': float(base.p_other_needed), 'max_wait': int(base.max_wait_h), 'delay_h': 0,
                'delay_rate': float(bc.downtime_h),
                'mode': COST_MODES[1], 'f_cons': 100.0, 'f_part': 200.0, 'f_labor_part': 20.0, 'f_labor_ph': 3.0,
                'f_crew': 4, 'f_down': 500.0, 'f_fail': 300.0, 'f_delay_rate': 500.0,
                'stock_v': int(bp.stock_valve), 'stock_o': int(bp.stock_oscillator), 'lead': float(bp.lead_h),
                'rush_h': float(bp.rush_h), 'rush_mk': float(bp.rush_markup)}

    def ctx(self, split, now_f, store, log, assume=None, truth=False):
        return Ctx(self, split, now_f, store, log, assume or {}, truth)


class Ctx:
    """지금 시각 하나의 계산 (요청마다 새로 만든다. 값은 옛 dashboard.py의 한 번 그리기와 같다)"""

    def __init__(self, eng, split, now_f, store, log, assume, truth):
        self.eng, self.split, self.store, self.log = eng, split, store, log
        self.now_f = float(min(END_H, max(0.0, now_f)))
        self.now = int(self.now_f)
        self.truth = bool(truth) and split == 'validation'
        self.pts, self.hr, self.contrib, self.contrib_unit, self.mnt = eng.data(split)
        now = self.now
        self.pts_now = self.pts[self.pts['T_H'] <= now]
        self.hr_now = self.hr[self.hr['DONE_H'] <= now]
        self.mnt_now = self.mnt[self.mnt['START_H'] <= now].assign(
            END_SHOWN=lambda d: d['END_H'].clip(upper=now))
        self.last_mnt_start = self.mnt_now.groupby('SCENARIO_ID')['START_H'].max()
        self.MNT_START = self.last_mnt_start.to_dict()
        self.names = eng.names(split)
        A = {**eng.default_assumptions(), **{k: v for k, v in assume.items() if k in eng.default_assumptions()}}
        A['mode'] = COST_MODES[1]
        self.A = A
        self._t = self._W = None

    def nm(self, sid):
        return self.names.get(sid, sid)

    # 상태

    def after_mnt(self, df, col):
        start = df['SCENARIO_ID'].map(self.last_mnt_start).fillna(-np.inf)
        return df[df[col] >= start]

    def status_table(self):
        if self._t is not None:
            return self._t
        pts_now, hr_now, mnt, now = self.pts_now, self.hr_now, self.mnt, self.now
        CAUTION = self.eng.caution
        last = pts_now.sort_values('T_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
        first_alarm = pts_now[pts_now['ALARM']].groupby('SCENARIO_ID')['T_H'].min()
        first_caution = pts_now[pts_now['STAGE'] != '정상'].groupby('SCENARIO_ID')['T_H'].min()
        cur = self.after_mnt(hr_now.assign(HOUR_START=hr_now['DONE_H'] - 1), 'HOUR_START')
        last_hr = cur.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1)
        worst = last_hr.set_index('SCENARIO_ID')[list(SIGNAL_NAMES)].idxmax(axis=1).map(SIGNAL_NAMES)
        in_mnt = set(mnt[(mnt['START_H'] <= now) & (mnt['END_H'] > now)]['SCENARIO_ID'])
        lms = self.last_mnt_start
        stale = set(lms[lms > last['T_H'].reindex(lms.index).fillna(-1)].index) - in_mnt
        t = pd.DataFrame({'설비 (시나리오)': sorted(self.pts['SCENARIO_ID'].unique())}).set_index('설비 (시나리오)', drop=False)
        t['위험 확률'] = last['PROB']
        t['열화 확률'] = last['DEGRADED_PROB'] if CAUTION else np.nan
        t['마지막 예측 (h)'] = last['T_H']
        since = self.after_mnt(pts_now, 'T_H')
        latched = set(since.loc[since['ALARM'], 'SCENARIO_ID']) - set(last.index[last['ALARM']])
        stage = last['STAGE'].reindex(t.index).fillna('정상').where(~t.index.isin(latched), '경보')
        t['상태'] = np.select([t.index.isin(in_mnt), t.index.isin(stale)], ['정비 중', '정비 후 예측 대기'], default=stage)
        t['경보 유지'] = np.where(t.index.isin(latched) & (t['상태'] == '경보'), '유지 (확률 내려감)', '')
        t.loc[t['상태'].isin(['정비 중', '정비 후 예측 대기']), ['위험 확률', '열화 확률']] = np.nan
        t['첫 주의 (h)'] = first_caution if CAUTION else np.nan
        t['첫 경보 (h)'] = first_alarm
        t['가장 많이 벗어난 신호'] = worst
        if self.truth:
            fails = mnt.dropna(subset=['FAIL_H'])
            nxt = fails[fails['FAIL_H'] > now].groupby('SCENARIO_ID')['FAIL_H'].min()
            t['다음 실제 고장까지 (h, 정답)'] = (nxt - now).round(1)
        order = t['상태'].map({'경보': 0, '주의': 1, '정상': 2}).fillna(3)
        t['상태'] = t['상태'].replace(LABEL)
        self._t = t.assign(_o=order).sort_values(['_o', '위험 확률'], ascending=[True, False]).drop(columns='_o')
        return self._t

    def short(self):
        return self.status_table()['상태'].str.split(' (', regex=False).str[0]

    # 비용 가정·재고

    def make_cfg(self):
        A = self.A
        base = self.eng.bundle['dss']['config']
        costs = field_costs(consumables=A['f_cons'], major_part=A['f_part'], labor_per_part=A['f_labor_part'],
                            labor_per_person_h=A['f_labor_ph'], crew_size=A['f_crew'], downtime_h=A['f_down'],
                            failure_extra=A['f_fail'])
        parts = PartsPolicy(stock_valve=A['stock_v'], stock_oscillator=A['stock_o'], lead_h=A['lead'], rush_h=A['rush_h'],
                            rush_markup=A['rush_mk'])
        cfg = replace(base, safety_max=A['safety'], prep_h=A['prep'], crew_wait_h=A['crew_wait'], p_other_needed=A['p_other'],
                      max_wait_h=A['max_wait'], costs=costs, parts=parts)
        return costs, cfg

    def lg(self):
        return self.log.frame(self.split, self.now_f)

    def stock_now(self, cfg):
        out = {sid: cfg.parts.stock() for sid in self.pts['SCENARIO_ID'].unique()}
        for r in self.mnt_now[self.mnt_now['종류'] == '고장 후 복구'].itertuples():
            if r.PART_CD in PARTS:
                out[r.SCENARIO_ID][r.PART_CD] = max(0, out[r.SCENARIO_ID][r.PART_CD] - 1)
        lg = self.lg()
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
            if arrive <= self.now and r['부품'] in PARTS:
                out[r['설비']][r['부품']] += 1
        return out

    def in_transit(self, cfg):
        lg = self.lg()
        out = {}
        for _, r in lg[lg['종류'] == '부품 발주'].iterrows():
            arrive = r['시각(h)'] + (cfg.parts.rush_h if r['선택지'] == '긴급' else cfg.parts.lead_h)
            if arrive > self.now:
                out.setdefault(r['설비'], []).append((r['부품'], arrive, r['선택지'] == '긴급'))
        return out

    def procurement(self, cfg, stock, ids):
        trans = self.in_transit(cfg)
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

    def part_breakeven(self):
        v = self.eng.bundle['validation']['event']
        cc = cost_curve(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
                        n_false=v['고장과 무관한 경보 묶음: 오경보'], costs=self.make_cfg()[0])
        even = cc.loc[cc['한 부품만 교체'] <= cc['두 부품 모두 교체'].iloc[0], 'p']
        return float(even.iloc[0]) if len(even) else None

    def feedback(self):
        n, acc = self.log.inspection_accuracy()
        be = self.part_breakeven()
        return {'n': n, 'accuracy': acc, 'breakeven': be, 'ready': be is not None and actions.one_part_ready(n, acc, be)}

    # 정비 계획 (경보 설비마다 추천과 정비팀 순서)

    def work_plan(self):
        if self._W is not None:
            return self._W
        A, t, now = self.A, self.status_table(), self.now
        costs, cfg = self.make_cfg()
        STOCK = self.stock_now(cfg)
        crew_day_h = A['crew_day_h']
        delay_h = A['delay_h']
        delay_rate = A['f_delay_rate']
        curve = self.eng.bundle['dss']['curve']
        pl = self.eng.plans(self.split)
        since = self.after_mnt(self.pts_now, 'T_H')
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
            return {'t_a': t_a, 'wait_h': wait_h, 'part': part, 'plan_part': plan_part, 'elapsed': elapsed, 'opts': opts,
                    'best': best, 'urgent': urgent}

        alarm_ids = t.index[t['상태'] == WARN].tolist()
        first = {sid: evaluate_options(sid) for sid in alarm_ids}
        need_crew = sorted([s for s in alarm_ids if not first[s]['best']['선택지'].startswith('다음 정기')],
                           key=lambda s: -fail_prob(curve, first[s]['elapsed'], first[s]['opts'][0]['정비 완료까지(h)'] + costs.replace_h))
        absent_now = self.store.absent_at(self.now_f)
        team_jobs = {k: int(sum(m not in absent_now for m in v) * crew_day_h // WORK_REPLACE) for k, v in tasks.TEAMS.items()}
        per_day = max(1, max(team_jobs.values()))
        slots = sorted(d * 24.0 + j * costs.replace_h for d in range(30) for n_ in team_jobs.values() for j in range(n_))

        def crew_delay(k):
            return slots[k] if k < len(slots) else 24.0 * 30

        rows = []
        for k, sid in enumerate(need_crew + [s for s in alarm_ids if s not in need_crew]):
            wait_crew = crew_delay(k) if sid in need_crew else 0.0
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
                         '_e': e, '_slot': wait_crew,
                         '_start': (e['wait_h'] if b['선택지'].startswith('다음 정기') else b['정비 완료까지(h)'] - costs.replace_h)})
        dec = pd.DataFrame(rows)
        if len(dec):
            state = self.log.job_state(self.split, self.now_f, self.MNT_START)
            dec.insert(3, '방법', '점검 후 한 부품' if self.feedback()['ready'] else '두 부품')
            dec.insert(1, '조치', dec['설비 (시나리오)'].map(lambda x: state.get(x, '대기')))
            dec.insert(4, '재고', dec['설비 (시나리오)'].map(lambda x: stock_text(STOCK[x])))
        self._W = {'stock': STOCK, 'dec': dec, 'need_crew': need_crew, 'per_day': per_day, 'evaluate_options': evaluate_options,
                   'cfg': cfg, 'costs': costs, 'curve': curve, 'pl': pl}
        return self._W

    def dec_i(self):
        dec = self.work_plan()['dec']
        return dec.set_index('설비 (시나리오)') if len(dec) else pd.DataFrame()

    # 작업 지시

    def tf(self):
        return self.store.frame(self.split, self.now_f)

    def job_state(self):
        return self.log.job_state(self.split, self.now_f, self.MNT_START)

    def team_load(self, tf=None):
        tf = self.tf() if tf is None else tf
        o = tf[tf['상태'] != '완료']
        return {k: int((o['team'] == k).sum()) for k in tasks.TEAMS}

    def queue(self):
        """배정할 경보: 열린 작업이 없고, 보류하지 않았고, 이번 운전 구간에서 작업을 마치지 않은 경보 설비 (즉시 조치 먼저)"""
        dec = self.work_plan()['dec']
        if not len(dec):
            return dec
        busy = self.store.blocked_sids(self.split, self.now_f)
        js = self.job_state()
        q = dec[~dec['설비 (시나리오)'].isin(busy) & ~dec['설비 (시나리오)'].map(lambda x: js.get(x)).isin(['보류', '작업 완료'])]
        if len(q):
            q = q.assign(_u=(~q['제약'].str.startswith('즉시')).astype(int)).sort_values(['_u', '순위'], na_position='last')
        return q

    def order_states(self):
        tf = self.tf()
        js = self.job_state()
        short = self.short()
        tf_last = tf.sort_values('id').groupby('sid').tail(1).set_index('sid') if len(tf) else pd.DataFrame()
        out = {}
        for x in short.index:
            if len(tf_last) and x in tf_last.index:
                out[x] = tf_last.loc[x, '상태']
            elif js.get(x) == '보류':
                out[x] = '보류'
            else:
                out[x] = '배정 전' if short[x] == '경보' else '작업 없음'
        return out

    def task_json(self, r):
        keys = ['id', 'sid', 'title', 'team', 'priority', 'due_h', 'created_h', 'started_h', 'completed_h', 'stopped_h', 'guide',
                'reason', 'option', 'outcome', 'details', 'judged', 'confirmed', 'action', 'action_why', 'stop_why', 'members',
                'leader', 'completed_at']
        out = {k: (num(r.get(k)) if k.endswith('_h') else (int(r[k]) if k == 'id' else ('' if pd.isna(r.get(k)) else str(r.get(k)))))
               for k in keys}
        d = load_checks(r.get('checks'))
        out.update({'status': r['상태'], 'bucket': r['구분'], 'name': self.nm(r['sid']), 'checks': d,
                    'checkCount': check_count(d), 'people': people(r),
                    'leaderName': r['leader'] or tasks.pick_leader(r['members'], ''),
                    'urgentJob': r['priority'] == '긴급' or '즉시 조치' in str(r['title'])})
        return out

    # 설비 화면

    def guide_content(self, row):
        """상태별 조치 안내: (색, 제목, 본문 줄)"""
        st_, CAUTION = row['상태'], self.eng.caution
        if st_ == WARN:
            return ('red', '조치 안내: 교체', ['24시간 안에 고장 날 위험이 높습니다.',
                    '교체에 4시간이 걸리므로, 늦어도 고장 예상 4시간 전에는 교체를 시작해야 합니다.',
                    '센서로는 원인 부품(진동장치·밸브)을 구분할 수 없습니다. 두 부품을 모두 점검하세요.',
                    '고장 원인과 상관없이 같은 신호가 오르므로, 분석 근거의 신호 흐름으로 열화가 계속되는지 확인하세요.'])
        if st_ == CAU:
            return ('orange', '조치 안내: 점검', ['열화가 시작된 것으로 보입니다. 아직 24시간 안 고장 위험은 낮습니다.',
                    '장비 상태를 확인하세요. 설비가 정상적으로 돌아가면 다음 정비 때 이 부분까지 함께 정비합니다.',
                    '두 부품(진동장치·밸브)의 재고를 확인하고, 없으면 지금 발주를 시작하세요. 경보까지 보통 반나절 정도 남습니다.',
                    f'주의 중 일부는 열화가 아닌 헛걸음입니다(검증 세트 {CAUTION["validation"]["주의 묶음"]}번 중 '
                    f'{CAUTION["validation"]["열화 아닌 설비의 주의"]}번). 점검에서 이상이 없으면 그대로 운전합니다.'])
        if st_ in ('정비 중', '정비 후 예측 대기'):
            return ('orange', '조치 안내: 정비 중',
                    [f'정비가 시작돼 정비 전 예측은 더 이상 쓰지 않습니다. 재가동 뒤 다음 예측 시점({self.eng.step}시간마다)에 다시 예측합니다.'])
        return ('green', '조치 안내: 운전 계속',
                ['지금은 열화 신호가 없고, 24시간 안 고장 위험이 낮습니다.' if CAUTION else '지금은 24시간 안 고장 위험이 낮습니다.'])

    def detail(self, sid):
        """설비 하나의 분석 근거: 지표 · 입력 감시 · 위험/열화 확률 흐름 · 네 신호 흐름 · 지금 신호 · 기여도"""
        t = self.status_table()
        row = t.loc[sid]
        now = self.now
        out = {'sid': sid, 'name': self.nm(sid), 'stage': row['상태'], 'risk': num(row['위험 확률']), 'degraded': num(row['열화 확률']),
               'lastPredH': num(row['마지막 예측 (h)']), 'firstCautionH': num(row['첫 주의 (h)']), 'firstAlarmH': num(row['첫 경보 (h)']),
               'latched': bool(row['경보 유지']), 'oor': None}
        last_p = self.after_mnt(self.pts_now[self.pts_now['SCENARIO_ID'] == sid], 'T_H').tail(1)
        if 'OOR_N' in last_p and len(last_p) and last_p['OOR_N'].iloc[0] > 0:
            out['oor'] = (f'마지막 예측({last_p["T_H"].iloc[0]:.0f}시간)의 특징 {int(last_p["OOR_N"].iloc[0])}개가 '
                          f'학습 때 본 범위(0.1~99.9%) 밖입니다: {last_p["OOR_FEATS"].iloc[0]}. '
                          '아주 심한 열화이거나 센서 이상일 수 있어, 모델 확률만 믿지 말고 현장에서 센서와 설비를 확인하세요.')
        p = self.pts_now[self.pts_now['SCENARIO_ID'] == sid].sort_values('T_H')
        has_deg = 'DEGRADED_PROB' in p.columns
        out['risk_series'] = [{'h': num(r['T_H']), 'risk': num(r['PROB']), 'degraded': num(r['DEGRADED_PROB']) if has_deg else None,
                               'stage': LABEL.get(r['STAGE'], r['STAGE'])} for _, r in p.iterrows()]
        h = self.hr_now[self.hr_now['SCENARIO_ID'] == sid].sort_values('DONE_H')
        out['signal_series'] = [{'h': num(r['DONE_H']), '평균': num(r['SCORE']), **{SIGNAL_NAMES[k]: num(r[k]) for k in SIGNAL_NAMES}}
                                for _, r in h.iterrows()]
        m = self.mnt_now[self.mnt_now['SCENARIO_ID'] == sid]
        out['maintenance'] = [{'start': num(r['START_H']), 'end': num(r['END_H']) if r['END_H'] <= now else None,
                               'endShown': num(r['END_SHOWN']), 'kind': r.get('종류', ''), 'part': PART_NAMES.get(r.get('PART_CD', ''), r.get('PART_CD', '')),
                               'fail': num(r['FAIL_H'])} for _, r in m.iterrows()]
        out['truthFails'] = ([num(x) for x in self.mnt[(self.mnt['SCENARIO_ID'] == sid) & (self.mnt['FAIL_H'] > now)]['FAIL_H'].dropna()]
                             if self.truth else [])
        cur = self.after_mnt(h.assign(HOUR_START=h['DONE_H'] - 1), 'HOUR_START')
        last = cur.dropna(subset=list(SIGNAL_NAMES)).tail(1)
        if row['상태'] == '정비 중':
            out['signalsNow'], out['signalsNote'] = None, '정비 중이라 신호가 없습니다.'
        elif last.empty:
            out['signalsNow'], out['signalsNote'] = None, ('재가동 뒤 아직 계산된 신호가 없습니다. 재가동 후 처음 2시간은 기준을 잡는 시간입니다. '
                                                           '(정비 전 신호는 지금 상태가 아니라서 보여 주지 않습니다.)')
        else:
            out['signalsNow'] = sorted([{'name': SIGNAL_NAMES[k], 'value': num(last[k].iloc[0])} for k in SIGNAL_NAMES],
                                       key=lambda x: -(x['value'] or 0))
            out['signalsNote'] = f'{last["DONE_H"].iloc[0]:.0f}시간까지의 1시간 값. 1을 넘으면 정상 시나리오의 95%보다 많이 벗어난 것입니다.'
        cp = self.after_mnt(self.pts_now[self.pts_now['SCENARIO_ID'] == sid], 'T_H')
        if self.contrib is None:
            out['contrib'], out['contribNote'] = None, (f'최종 모델({self.eng.bundle["final"]["model"]})은 특징별 기여도를 계산할 수 없는 종류입니다. '
                                                        '신호 상태와 신호 흐름으로 경보 이유를 확인하세요.')
        elif cp.empty:
            out['contrib'], out['contribNote'] = None, (f'아직 예측이 없습니다. 정비가 시작되면 정비 전 예측은 쓰지 않고, '
                                                        f'재가동 뒤 다음 예측 시점({self.eng.step}시간마다)을 기다립니다.')
        else:
            cr = self.contrib.loc[cp.index[-1]]
            out['contrib'] = sorted([{'name': str(k), 'value': num(v)} for k, v in cr.items()], key=lambda x: -(x['value'] or 0))
            unit = self.contrib_unit
            out['contribUnit'] = unit
            out['contribNote'] = (f'{cp["T_H"].iloc[-1]:.0f}시간 예측 기준. 양수는 위험을 올린 쪽, 음수는 내린 쪽입니다. '
                                  + ('나무마다 결정 경로를 따라 확률이 바뀐 양을 그 분기에 쓰인 특징 묶음에 나눠 준 값으로, '
                                     '기준값에 모두 더하면 위의 위험 확률이 됩니다. ' if str(unit).startswith('확률') else '0은 학습 세트 평균 시점입니다. ')
                                  + '네 신호와 결합 점수는 서로 비슷한 정보라 모델 안에서 서로 상쇄될 수 있습니다. '
                                  '묶음 하나의 크기보다 신호 상태를 먼저 보세요.')
        return out

    # 정비 판단 (설비관리자 설비 화면)

    def decision(self, sid, crew=None):
        """설비 하나의 정비 판단: 조치 안내 · 고장 예상까지 · 세 칸(즉시 정비 · 다음 정기 교체 · 사후정비) 비용 · 설명 · 접힌 칸.
        crew: 칸마다 고른 작업자 수 {'now': n, 'plan': n, 'run': n} (없으면 출근한 사람 4명)"""
        t = self.status_table()
        srow = t.loc[sid]
        sstage = srow['상태'].split(' (')[0]
        W = self.work_plan()
        cfg, costs = W['cfg'], W['costs']
        dec_i = self.dec_i()
        now, now_f = self.now, self.now_f
        busy = self.store.blocked_sids(self.split, now_f)
        js = self.job_state()
        job_ = '작업 있음' if sid in busy else js.get(sid)
        tone, g_title, g_body = self.guide_content(srow)
        alarm = len(dec_i) > 0 and sid in dec_i.index
        tf = self.tf()
        load = self.team_load(tf)
        existing = self.store.blocking(self.split, sid, now_f)
        out = {'sid': sid, 'name': self.nm(sid), 'stage': sstage, 'stageLabel': srow['상태'], 'risk': num(srow['위험 확률']),
               'orderState': self.order_states()[sid], 'job': job_, 'guide': {'tone': tone, 'title': g_title, 'body': g_body},
               'alarm': alarm, 'load': load, 'unit': costs.unit,
               'existing': None if existing is None else {'id': int(existing['id']), 'title': existing['title'], 'team': existing['team'],
                                                          'createdH': num(existing['created_h'])},
               'transit': [{'part': q, 'partName': PART_NAMES[q], 'arrive': num(a), 'rush': bool(r)}
                           for q, a, r in self.in_transit(cfg).get(sid, [])],
               'lead': cfg.parts.lead_h, 'rushH': cfg.parts.rush_h, 'rushMk': cfg.parts.rush_markup,
               'stock': W['stock'][sid], 'safety': cfg.safety_max}
        if sstage == '주의':
            pr = self.procurement(cfg, W['stock'], [sid]).iloc[0]
            out['procurement'] = f'부품: {pr["발주"]} (재고 {pr["재고"]})'
        absent_now = self.store.absent_at(now_f)
        open_ = tf[~tf['상태'].isin(['완료', '중단'])]
        out['absent'] = absent_now
        out['busyPeople'] = sorted({m for v in open_['members'] for m in str(v).split(',') if m})
        out['team0'] = min(load, key=load.get)
        out['recommendFallback'] = None
        if not alarm:
            out['recommendFallback'] = '점검·부품 준비' if sstage == '주의' else '상태 점검'
            out['defaultPriority'] = {'경보': '높음', '주의': '높음'}.get(sstage, '보통')
            return out
        r_ = dec_i.loc[sid]
        e = r_['_e']
        b = e['best']
        curve = W['curve']
        p24 = fail_prob(curve, e['elapsed'], 24)
        now_opt = next(o for o in e['opts'] if o['선택지'].startswith('지금'))
        remain = remaining_median(curve, e['elapsed'])
        urgent = r_['제약'].startswith('즉시')
        base_wait = cfg.crew_wait_h + r_['_slot']
        now_start = 0.0 if urgent else max(0.0, now_opt['정비 완료까지(h)'] - costs.replace_h)
        horizon = float(e['wait_h']) if pd.notna(e['wait_h']) else max(1.0, 72.0 - now)
        ph = self.A['f_labor_ph']

        def card_values(kind, n):
            if kind == 'run':
                c_n = crew_costs(costs, n, now, ph)
                p_h = fail_prob(curve, e['elapsed'], horizon)
                return {'cost': p_h * c_n.failure_loss, 'p': p_h, 'start': 0.0,
                        'sub': f'{horizon:.0f}시간 안 고장 확률 {p_h:.0%} · 그 뒤에도 고장 위험 남음 ({1 - p_h:.0%})', 'note': ''}
            if kind == 'now':
                c_n = crew_costs(costs, n, now + now_start, ph)
                o = next(o for o in options(e['wait_h'], curve, replace(cfg, crew_wait_h=base_wait, costs=c_n), e['elapsed'],
                                            stock=W['stock'][sid], plan_part=e['plan_part']) if o['선택지'].startswith('지금'))
                return {'cost': o['기대 총비용'], 'p': o['고장 확률'], 'start': now_start, 'sub': f'{o["선택지"]} · 교체 {c_n.replace_h:.1f}시간',
                        'note': ''}
            has_plan = pd.notna(e['wait_h'])
            c_n = crew_costs(costs, n, now + (e['wait_h'] if has_plan else PLAN_GUESS_H), ph)
            cfg_n = replace(cfg, crew_wait_h=base_wait, costs=c_n)
            o = next((o for o in options(e['wait_h'], curve, cfg_n, e['elapsed'], stock=W['stock'][sid],
                                         plan_part=e['plan_part']) if not o['선택지'].startswith('지금')), None)
            if o is not None:
                return {'cost': o['기대 총비용'], 'p': o['고장 확률'], 'start': float(e['wait_h']), 'sub': o['선택지'], 'note': ''}
            loose = replace(cfg_n, max_wait_h=1e9, prep_h=0.0)
            guesses = tuple(d for d in (PLAN_GUESS_H, 2 * PLAN_GUESS_H) if d > base_wait) or (int(np.ceil(base_wait)) + PLAN_GUESS_H,)
            for d_ in ((None,) if has_plan else ()) + guesses:
                po_ = [o for o in options(e['wait_h'] if d_ is None else np.nan, curve, loose, e['elapsed'], delay_h=d_,
                                          stock=W['stock'][sid], plan_part=e['plan_part']) if not o['선택지'].startswith('지금')]
                if po_:
                    note = (f'다음 정기 교체({e["wait_h"]:.0f}시간 뒤)는 최대 대기·부품·정비팀 조건 밖이라 비교만 합니다. ' if d_ is None else
                            (f'다음 정기 교체({e["wait_h"]:.0f}시간 뒤)에는 부품을 맞출 수 없어' if has_plan else '정기 교체 계획이 없어')
                            + f' 다음 정기 교체 칸은 {d_}시간 뒤 정지를 가정해 비교만 합니다. ')
                    return {'cost': po_[0]['기대 총비용'], 'p': po_[0]['고장 확률'], 'note': note,
                            'start': float(e['wait_h']) if d_ is None else float(d_),
                            'sub': (f'{e["wait_h"]:.0f}시간 뒤 · 조건 밖이라 비교만' if d_ is None else
                                    ('정기 교체 때 부품을 맞출 수 없음' if has_plan else '정기 교체 계획 없음') + f' · {d_}시간 뒤 정지 가정')}
            return {'cost': None, 'p': None, 'start': None, 'sub': '계산할 수 없음', 'note': ''}

        KIND = [('now', '즉시 정비', '경보를 보고 지금 두 부품을 교체'),
                ('plan', '다음 정기 교체', '정해 둔 정기 교체(계획 정지) 때 두 부품을 함께 교체'),
                ('run', '사후정비', '지금은 운전을 계속하고, 고장 나면 복구 (비교 기간 안 고장 확률 × 고장 손실)')]
        team0 = out['team0']
        avail0 = [m for m in tasks.TEAMS[team0] if m not in absent_now]
        crew = crew or {}
        vals = {k: card_values(k, max(1, int(crew.get(k, len(avail0[:4])) or 1))) for k, _, _ in KIND}
        priced = {k: v['cost'] for k, v in vals.items() if v['cost'] is not None}
        rec_kind = min(priced, key=priced.get) if priced else None
        cards = []
        for kind, label, help_ in KIND:
            v = vals[kind]
            cards.append({'kind': kind, 'label': label, 'help': help_, 'cost': num(v['cost']), 'p': num(v['p']), 'sub': v['sub'],
                          'start': num(v['start']), 'note': v.get('note', ''),
                          'overSafety': v['p'] is not None and pd.notna(v['p']) and v['p'] > cfg.safety_max,
                          'title': {'now': f'{sid} 지금 두 부품 교체', 'plan': f'{sid} 다음 정기 교체 때 두 부품 교체',
                                    'run': f'{sid} 사후정비: 운전 계속 · 고장 나면 복구'}[kind],
                          'priority': {'now': '긴급' if urgent else '높음', 'plan': '높음', 'run': '보통'}[kind],
                          'workH': (WORK_REPAIR if kind == 'run' else WORK_REPLACE)})
        lab = {'now': '즉시 정비', 'plan': '다음 정기 교체', 'run': '사후정비'}
        U = (lambda x: f'{x:,.0f}만 원') if costs.unit == '만 원' else (lambda x: f'{x:,.0f}점')
        lines = []
        if rec_kind is not None:
            rv, run_v = vals[rec_kind], vals['run']
            lines = [f'고장 나면 드는 돈(정지·생산 차질·복구, 사후정비 칸 작업자 기준): {U(run_v["cost"] / run_v["p"]) if run_v["p"] else "—"}',
                     f'추천({lab[rec_kind]})의 기대 비용: {U(rv["cost"])} · '
                     + (f'비교 기간 안 고장 확률 {rv["p"]:.0%}' if rec_kind == 'run' else f'끝나기 전에 고장 날 확률 {rv["p"]:.0%}')
                     + f' (안전 기준 {cfg.safety_max:.0%})']
            if rec_kind != 'run' and run_v['cost'] > rv['cost']:
                lines.append(f'사후정비(기대 비용)보다 {U(run_v["cost"] - rv["cost"])} 적게 듭니다')
            if rec_kind == 'run':
                lines.append('이 설비는 지금 세우는 것보다 운전을 계속하고 고장 나면 복구하는 쪽의 기대 비용이 낮습니다. '
                             '다만 비교 기간 뒤에도 고장 위험이 남고, 안전 기준을 넘으면 현장 판단이 먼저입니다.')
        basis = [f'{o["선택지"]}: 고장 확률 P {o["고장 확률"]:.2f} · {o["근거"]}' for o in e['opts']]
        basis.append(f'24시간 그냥 두면: 앞으로 24시간 안 고장 확률 {p24:.2f} × 고장 시 손실 = {U(p24 * costs.failure_loss)}')
        basis.append(f'경보가 켜진 뒤 실제 고장까지 걸린 시간(학습 세트)으로 보면, 이 설비가 앞으로 24시간 안에 고장 날 확률은 '
                     f'{p24:.0%}입니다. "24시간 내 고장 위험"(모델 예측)과는 계산 방법이 다릅니다.')
        if remain is not None:
            basis.append(f'고장 예상까지 {hms(remain)}: 같은 분포에서 경보 뒤 {e["elapsed"]:.0f}시간 동안 고장이 없었다는 조건으로 '
                         '고장 확률이 절반이 되는 시간(중앙값)입니다. 확정된 고장 시각이 아니라 추정입니다.')
        basis.append(f'P는 학습 세트에서 경보 뒤 실제 고장까지 걸린 시간 분포(경보 뒤 {e["elapsed"]:.0f}시간 동안 고장 없음 조건).')
        out.update({
            'remainH': remain, 'remainDeadline': None if remain is None else now + remain, 'urgent': urgent,
            'constraint': r_['제약'], 'best': b['선택지'], 'cards': cards, 'recKind': rec_kind, 'horizonH': horizon,
            'horizonLabel': f'지금 ~ {clock(now + horizon)} ({"다음 정기 정지" if pd.notna(e["wait_h"]) else "정기 정지 계획이 없어 시연 끝"}까지)',
            'canAssign': existing is None and job_ != '작업 완료', 'lines': lines,
            'assumption': [f'지금 두 부품 교체 비용: {U(now_opt["정비 비용"])} (교체 {costs.replace_h:.0f}시간)',
                           f'계획 밖 정지 1시간 생산 차질: {U(costs.downtime_h)}',
                           f'고장 시 손실: {U(costs.failure_loss)}',
                           f'안전 기준(완료 전 고장 확률): {cfg.safety_max:.2f} · 다음 정기 교체: {r_["다음 정기 교체"]}',
                           f'예비 재고: {stock_text(W["stock"][sid])} · 첫 경보 뒤 {e["elapsed"]:.0f}시간 지남'],
            'basis': basis,
            'basisNote': (f'기대 총비용 = (1 − P) × (정비 비용 + 생산 차질) + P × 고장 시 손실 · P = 정비를 마치기 전에 고장 날 확률. '
                          f'추천 = 완료 전 고장 확률이 안전 기준({cfg.safety_max:.2f}) 이하인 선택지 중 기대 총비용이 가장 낮은 것. '
                          + ('어떤 선택지도 안전 기준을 못 지켜 가장 빨리 할 수 있는 조치를 권합니다. ' if e['urgent'] else '')
                          + vals['plan'].get('note', '') + '최소 비용과 실제 권장 조치는 다를 수 있습니다. 안전·현장 절차 우선.'),
        })
        return out

    # 온톨로지

    def ontology_graph(self, with_decisions=True):
        t = self.status_table()
        status = pd.DataFrame({
            'STAGE': t['상태'].str.split(' (', regex=False).str[0], 'PROB': t['위험 확률'],
            'DEGRADED_PROB': t['열화 확률'], 'PRED_H': t['마지막 예측 (h)'],
            'FIRST_ALARM_H': t['첫 경보 (h)'], 'FIRST_CAUTION_H': t['첫 주의 (h)']}, index=t.index)
        cur = self.after_mnt(self.hr_now.assign(HOUR_START=self.hr_now['DONE_H'] - 1), 'HOUR_START')
        signals = cur.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
        decisions, unit, safety = [], None, None
        if with_decisions and self.eng.bundle.get('dss'):
            W = self.work_plan()
            unit, safety = W['costs'].unit, W['cfg'].safety_max
            decisions = [{'sid': r['설비 (시나리오)'], 'rank': r['순위'], 'constraint': r['제약'], 'best': r['추천'],
                          'options': r['_e']['opts']} for _, r in W['dec'].iterrows()]
        return ontology.build(self.now, status, signals, self.mnt_now, self.eng.plans(self.split), decisions, cost_unit=unit,
                              safety_max=safety, th=self.eng.th, cth=self.eng.cth, actions=self.lg(), feedback=self.feedback())

    def ontology_dot(self, sid, dark=False):
        """설비 하나의 연결 그림 (graphviz DOT). 색은 화면 테마에 맞춘다"""
        t = self.status_table()
        srow = t.loc[sid]
        dec_i = self.dec_i()
        e = dec_i.loc[sid, '_e'] if len(dec_i) and sid in dec_i.index else None
        costs = self.work_plan()['costs']

        def q(x):
            return '"' + str(x).replace('"', "'") + '"'
        ink, line, base = ('#FAFAFA', '#A1A1A1', '#262626') if dark else ('#0A0A0A', '#737373', '#F5F5F5')
        tint = {'경보': '#5C1A1D', '주의': '#4A3410', '정상': '#0F3B2E'} if dark else {'경보': '#FFD6DC', '주의': '#FCE9C6', '정상': '#DDF3EA'}
        blue, purple, rec = ('#0B2E59', '#2E2350', '#003A7A') if dark else ('#D9E6FB', '#E6E0FA', '#CFE0FF')
        stage = srow['상태'].split(' (')[0]
        out = ['digraph G {', 'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;',
               f'node [shape=box, style="rounded,filled", fillcolor="{base}", color="{line}", fontcolor="{ink}", '
               'fontname="Pretendard Variable", fontsize=11];',
               f'edge [fontname="Pretendard Variable", fontsize=10, color="{line}", fontcolor="{line}"];',
               f'E [label={q(self.nm(sid) + chr(10) + sid)}, fillcolor="{blue}"];',
               f'S [label={q("단계" + chr(10) + stage)}, fillcolor="{tint.get(stage, base)}"];', 'E -> S [label="상태"];']
        if pd.notna(srow['위험 확률']):
            pred_h, prob = srow['마지막 예측 (h)'], srow['위험 확률']
            out += [f'P [label={q(f"{pred_h:.0f}h 예측" + chr(10) + f"24시간 내 고장 위험 {prob:.0%}")}];',
                    'E -> P [label="관측"]; S -> P [label="근거"];']
            cur = self.after_mnt(self.hr_now[self.hr_now['SCENARIO_ID'] == sid].assign(HOUR_START=lambda d: d['DONE_H'] - 1), 'HOUR_START')
            last = cur.dropna(subset=list(SIGNAL_NAMES)).tail(1)
            for k, (col, name) in enumerate(SIGNAL_NAMES.items()):
                if len(last):
                    v = float(last[col].iloc[0])
                    out += [f'G{k} [label={q(f"{name}" + chr(10) + f"{v:.2f}" + (" (벗어남)" if v > 1 else ""))}, '
                            f'fillcolor="{tint["경보"] if v > 1 else base}"];', f'P -> G{k} [label="근거"];']
        if e is not None:
            out += [f'D [label="정비 결정", fillcolor="{purple}"];', 'S -> D [label="요청"];']
            for k, o in enumerate(e['opts']):
                is_rec = o is e['best']
                cost = f"{chr(10)}{o['기대 총비용']:,.0f}{costs.unit}"
                out += [f'O{k} [label={q(o["선택지"] + cost)}, fillcolor="{rec if is_rec else base}", penwidth={2 if is_rec else 1}];',
                        f'D -> O{k} [label="{"추천" if is_rec else "비교"}"];']
                if is_rec:
                    out += ['V [label="밸브"]; X [label="진동장치"];', f'O{k} -> V [label="교체"]; O{k} -> X [label="교체"];']
        return '\n'.join(out + ['}'])

    def ontology_trace(self, sid, with_cost=True):
        tr = ontology.trace(self.ontology_graph(), sid, with_cost=with_cost)
        unit_ = '만 원' if self.work_plan()['costs'].unit == '만 원' else '점'
        return [(k, re.sub(r'(기대 총비용 [\d,]+)(?![\d,]|만 원|점)', rf'\1{unit_}', v)) for k, v in tr]

    # 하위 페이지

    def assume_note(self):
        costs, cfg = self.make_cfg()
        A = self.A
        won = lambda x: f'{x:,.0f}만 원' if abs(x) < 10000 else f'{x / 10000:,.2f}억 원'
        now_cost = 2 * costs.part + costs.unnecessary + costs.replace_fixed + costs.replace_h * costs.downtime_h
        return (f'지금 가정: 지금 두 부품 교체 {won(now_cost)} · 정기 교체 때 다른 부품 추가 {won(costs.part)} · '
                f'고장 시 손실 = 복구 {costs.repair:,.0f} + 복구 {costs.repair_h:.0f}시간 × (생산 차질 {costs.downtime_h:,.1f} + 인력 {costs.labor_h:,.1f}) '
                f'+ 고장 추가 손실 {costs.failure_penalty:,.0f} = {won(costs.failure_loss)} · 안전 기준 {cfg.safety_max:.2f} · '
                f'부품 준비 {cfg.prep_h:.0f}시간 · 정비팀 {A["n_crews"]}팀(팀당 하루 {A["crew_day_h"]}공수). '
                '고장 확률은 학습 세트에서 경보 뒤 고장까지 걸린 시간 분포로 계산합니다.')

    def policy(self):
        W = self.work_plan()
        dec, costs, cfg = W['dec'], W['costs'], W['cfg']
        t = self.status_table()
        rows = []
        for _, r in dec.iterrows():
            rows.append({'rank': num(r['순위']), 'sid': r['설비 (시나리오)'], 'name': self.nm(r['설비 (시나리오)']), 'state': r['조치'],
                         'best': r['추천'], 'method': r['방법'], 'stock': r['재고'], 'doneH': num(r['정비 완료까지 (h)']),
                         'cost': num(r['기대 총비용']), 'p': num(r['완료 전 고장 확률']), 'constraint': r['제약'],
                         'firstAlarmH': num(r['첫 경보 (h)']), 'nextPlan': r['다음 정기 교체']})
        fb = self.feedback()
        lg = self.lg()
        cau = t.index[t['상태'] == CAU].tolist()
        caution, need_order = [], []
        if cau:
            pl, now = W['pl'], self.now
            nxt = pl[pl['SCENARIO_ID'].isin(cau) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').groupby('SCENARIO_ID').head(1).set_index('SCENARIO_ID')
            pr = self.procurement(cfg, W['stock'], cau).set_index('설비')
            for sid in cau:
                plan_ = (f'{nxt.loc[sid, "PLAN_H"] - now:.0f}시간 뒤 ({PART_NAMES.get(nxt.loc[sid, "PLAN_PART"], nxt.loc[sid, "PLAN_PART"])})'
                         if sid in nxt.index else '계획 없음')
                caution.append({'sid': sid, 'name': self.nm(sid), 'nextPlan': plan_,
                                'todo': '장비 점검, 정기 교체 때 두 부품 함께 교체 준비' if plan_ != '계획 없음' else '장비 점검 (경보가 켜지면 지금 교체 대상)',
                                'stock': pr.loc[sid, '재고'], 'parts': pr.loc[sid, '발주']})
            need_order = [{'sid': sid, 'part': q} for sid, qs in pr['_short'].items() for q in qs]
        D = self.eng.bundle['dss']
        sim = D.get('validation_field')
        sim_rows = []
        if sim:
            for k, v in sim.items():
                sim_rows.append({**{kk: (num(vv) if isinstance(vv, (int, float, np.number)) else vv) for kk, vv in v.items()},
                                 '예상 절감액 (① 대비)': num(sim['regular']['비용'] - v['비용'])})
        return {'assumptions': self.A, 'defaults': self.eng.default_assumptions(), 'note': self.assume_note(),
                'unit': costs.unit, 'rows': rows, 'perDay': W['per_day'], 'replaceH': costs.replace_h,
                'feedback': fb, 'minConfirmed': actions.MIN_CONFIRMED,
                'log': [{k: (num(v) if k == '시각(h)' else ('' if pd.isna(v) else str(v))) for k, v in r.items() if k != '데이터 세트'}
                        for r in lg.iloc[::-1].to_dict('records')],
                'caution': caution, 'needOrder': need_order, 'lead': cfg.parts.lead_h, 'rushH': cfg.parts.rush_h,
                'rushMk': cfg.parts.rush_markup, 'sim': sim_rows}

    def policy_option(self, sid, p_override=None):
        """정책·비용의 설비별 근거: 선택지 표 · 산정 근거 · 경계 · 고장 확률 곡선"""
        W = self.work_plan()
        dec = W['dec'].set_index('설비 (시나리오)')
        slot = dec.loc[sid, '_slot']
        e = W['evaluate_options'](sid, crew_extra=slot, p_override=p_override) if p_override else dec.loc[sid, '_e']
        b = e['best']
        costs, cfg, curve = W['costs'], W['cfg'], W['curve']
        opts = [{'name': o['선택지'], 'doneH': num(o['정비 완료까지(h)']), 'maint': num(o['정비 비용']), 'down': num(o['생산 차질 비용']),
                 'p': num(o['고장 확률']), 'loss': num(o['고장 시 손실']), 'total': num(o['기대 총비용']), 'safe': bool(o['안전 기준 충족']),
                 'basis': o['근거'], 'rec': o['선택지'] == b['선택지']} for o in e['opts']]
        boundary = ''
        wait = next((o for o in e['opts'] if o['선택지'].startswith('다음 정기')), None)
        if wait is not None:
            c_wait = wait['정비 비용'] + wait['생산 차질 비용']
            even = (e['opts'][0]['기대 총비용'] - c_wait) / (costs.failure_loss - c_wait)
            boundary = ((f'추천이 바뀌는 경계: 안전 기준을 {wait["고장 확률"]:.2f}보다 낮추면 "지금 교체". ' if wait['고장 확률'] >= 0.005 else
                         '정기 교체 전까지 고장 확률이 거의 0이라 안전 기준과 상관없이 기다릴 수 있습니다. ')
                        + f'비용만 보면 정기 교체 전 고장 확률이 {even:.2f}를 넘을 때 지금 교체가 더 쌉니다. '
                        '부품 준비 시간이 정기 교체보다 길어지면 "지금 교체"로 바뀝니다.')
        elif pd.isna(e['wait_h']):
            boundary = '다음 정기 교체 계획이 없어 지금(또는 N시간 뒤) 교체만 가능합니다.'
        return {'sid': sid, 'name': self.nm(sid), 'best': b['선택지'], 'urgent': e['urgent'], 'total': num(b['기대 총비용']),
                'p': num(b['고장 확률']), 'maint': num(b['정비 비용']), 'down': num(b['생산 차질 비용']), 'loss': num(b['고장 시 손실']),
                'opts': opts, 'elapsed': e['elapsed'], 'boundary': boundary, 'jobState': self.job_state().get(sid, '작업 없음'),
                'safety': cfg.safety_max, 'unit': costs.unit,
                'curve': [{'h': int(h), 'p': num(fail_prob(curve, e['elapsed'], h))} for h in range(49)],
                'points': [{'h': num(o['정비 완료까지(h)']), 'p': num(fail_prob(curve, e['elapsed'], o['정비 완료까지(h)'])), 'name': o['선택지']}
                           for o in e['opts']]}

    def model_info(self):
        b = self.eng.bundle
        v = b['validation']
        C = self.eng.caution
        g = self.ontology_graph()
        cnt = ontology.counts(g)
        bad = ontology.check(g)
        ev = v['events'].rename(columns={'SCENARIO_ID': '시나리오', 'TYPE': '고장 종류'})
        ev = ev[['시나리오', '고장 종류', '감지', '리드타임(h)']].sort_values(['고장 종류', '리드타임(h)'], ascending=[True, False])
        return {
            'createdAt': str(b['created_at']), 'cloud': self.eng.cloud, 'model': b['final']['model'], 'nFeatures': len(b['features']),
            'features': b['final']['features'], 'step': self.eng.step, 'nTrain': int(b['n_train_points']), 'th': self.eng.th,
            'point': {k: num(x) if isinstance(x, (int, float, np.number)) else x for k, x in v['point'].items()},
            'event': {k: num(x) if isinstance(x, (int, float, np.number)) else x for k, x in v['event'].items()},
            'caution': None if not C else {
                'model': C['final']['model'], 'features': C['final']['features'], 'nFeatures': len(C['features']),
                'th': self.eng.cth, 'minPrecision': C['min_precision'],
                'v': {k: num(x) if isinstance(x, (int, float, np.number)) else x for k, x in C['validation'].items()}},
            'events': [{k: (num(x) if isinstance(x, (int, float, np.number)) and not isinstance(x, bool) else
                            (bool(x) if isinstance(x, (bool, np.bool_)) else str(x))) for k, x in r.items()} for r in ev.to_dict('records')],
            'limits': LIMITS,
            'ontology': {'triples': len(g), 'counts': cnt, 'bad': bad, 'parts': ontology.parts_needed(g)},
        }

    def parts_calc(self, p):
        v = self.eng.bundle['validation']['event']
        n = dict(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'], n_false=v['고장과 무관한 경보 묶음: 오경보'])
        costs_f, _ = self.make_cfg()
        cc = cost_curve(**n, costs=costs_f)
        one = float(np.interp(p, cc['p'], cc['한 부품만 교체']))
        both, none = float(cc['두 부품 모두 교체'].iloc[0]), float(cc['아무것도 안 함'].iloc[0])
        u = self.eng.bundle['validation'].get('unlabeled')
        two_ = 2 * costs_f.part + costs_f.unnecessary + costs_f.replace_fixed + costs_f.replace_h * costs_f.downtime_h
        even = cc.loc[cc['한 부품만 교체'] <= both, 'p']
        return {'event': {k: num(x) if isinstance(x, (int, float, np.number)) else x for k, x in v.items()}, 'n': n,
                'one': one, 'both': both, 'none': none, 'unit': costs_f.unit,
                'unlabeled': u and {k: num(x) if isinstance(x, (int, float, np.number)) else x for k, x in u.items()},
                'two': two_, 'even': num(even.iloc[0]) if len(even) else None,
                'curve': [{k: num(x) for k, x in r.items()} for r in cc.to_dict('records')]}
