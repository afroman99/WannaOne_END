"""정비 의사결정 지원(DSS): 경보가 켜진 설비마다 정비 선택지(지금 / N시간 뒤 / 다음 정기 교체 때)의
기대 총비용을 비교해 추천하고, 정책별 비용을 같은 데이터로 시뮬레이션한다.

    from src.dss import DSSConfig, failure_curve, options, decide, simulate

    curve = failure_curve(train_episodes)                    # 학습 세트: 경보 뒤 고장까지 시간 분포
    opts = options(wait_h=3, curve=curve, cfg=DSSConfig())   # 선택지별 비용 항목과 기대 총비용
    res = simulate(pts, prob, th, curve, DSSConfig(), policy='dss')

설계: reports/연주_DSS_설계초안.md

기대 총비용 (강사 안내의 틀) = 정비 비용 + 생산 차질 비용 + P(정비가 끝나기 전 고장) × 고장 시 손실
  - 정비가 끝나기 전에 고장 나면 정비 비용·생산 차질 대신 고장 시 손실이 든다고 본다
  - 확률은 선택한 정비 시점까지의 기간 값이다(failure_curve, 경보 뒤 지난 시간을 조건으로 함)
  - 비용 값은 데이터셋 비용 규칙(점수)이 기본값이고 화면에서 바꿀 수 있는 가정값이다(Costs)
제약 (비용보다 먼저)
  - 안전: 정비가 끝나기 전 고장 확률이 safety_max를 넘는 선택지는 고르지 않는다. 모든 선택지가 넘으면 지금 교체(즉시 조치)
  - 실행 가능성: 정비팀이 가능한 시각(crew_wait_h), 부품 준비 시간(prep_h)보다 이른 정비는 할 수 없다

정책 (시뮬레이션)
  ① regular  정기 교체만 (경보를 쓰지 않음, 주기 기반 정비)
  ② now      경보가 켜지면 즉시 두 부품 교체 (예측 임계값을 넘으면 즉시 정비)
  ③ dss      위 기대 총비용·제약으로 지금 / 다음 정기 교체 때 중 고른다 (N시간 뒤 선택지는 대시보드에서 사용자가 정할 때만)

판단에 쓰는 정보 (그 시각에 알 수 있는 것만)
  - 경보가 처음 켜진 시각 (운전 구간마다 한 번 판단. 교체하면 운전 구간이 끝나므로 다음 경보는 없다)
  - 정기 교체 계획(calendar 표, 미리 안다고 가정)과 계획된 부품(정비 기록의 같은 시작 시각 정기 교체)
  - 경보 뒤 고장까지 시간 분포(학습 세트에서 추정한 failure_curve)
채점에만 쓰는 정보 (정답): 정비가 없었다면 고장 날 시각(features.state_outcomes), 고장 원인 부품(진단 정답)

근사 (설계초안 7절의 한계)
  - 고장을 막았는지는 운전 구간 안에서만 본다(evaluate.py와 같음). 다음 운전 구간의 고장은 그 구간의 경보로 따로 센다
  - 관측 끝이 가까워 결과를 알 수 없는 경보(고장 예정 없음 + 남은 관측 24시간 미만)는 '미확정'으로 따로 센다
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluate import MIN_LEAD_H
from src.posko_data import load_answer, load_source
from src.preprocess import EQP

PARTS = ['valve', 'oscillator']


@dataclass
class Costs:
    """비용 가정값 (기본값 = 데이터셋 비용 규칙, toolkit/targets.md. 점수이며 실제 돈이 아니다)"""
    part: float = 12.0              # 부품 하나 교체 비용
    downtime_h: float = 2.0         # 계획 밖 정지 1시간의 생산 차질 비용
    replace_h: float = 4.0          # 교체 시간
    repair: float = 30.0            # 고장 후 복구 비용
    repair_h: float = 8.0           # 복구 시간 (계획 밖 정지)
    failure_penalty: float = 20.0   # 고장 벌점 (안전·품질 영향의 대리값)
    unnecessary: float = 8.0        # 멀쩡한 부품을 바꾼 불필요 정비 벌점

    @property
    def failure_loss(self):         # 고장 시 손실 = 복구 + 복구 동안 생산 차질 + 고장 벌점 (기본 66)
        return self.repair + self.repair_h * self.downtime_h + self.failure_penalty


@dataclass
class DSSConfig:
    """정책의 바꿀 수 있는 값 (팀이 정할 것: 설계초안 9절)"""
    safety_max: float = 0.2      # 정비가 끝나기 전 고장 확률이 이보다 크면 그 선택지는 고르지 않는다 (안전 규칙)
    max_wait_h: float = 48.0     # 정기 교체 계획이 이보다 멀면 기다리지 않는다
    p_other_needed: float = 0.5  # 정기 교체 때 함께 바꾸는 다른 부품이 실제 원인일 확률 (부품을 모르므로 0.5)
    prep_h: float = MIN_LEAD_H   # 경보 뒤 교체를 끝내기까지 필요한 시간 (부품 조달 + 교체 4시간). 데이터셋 규칙은 4시간
    crew_wait_h: float = 0.0     # 정비팀이 일을 시작할 수 있을 때까지 시간 (실행 가능성)
    costs: Costs = field(default_factory=Costs)


# 1. 판단에 쓰는 정보

def plans(scenario_ids):
    """정기 교체 계획: SCENARIO_ID, PLAN_START, PLAN_PART (계획된 부품은 같은 시각에 시작한 정기 교체 기록의 부품)"""
    cal = load_source('calendar').query('EQP_TAG == @EQP and SCENARIO_ID in @scenario_ids')
    rep = load_source('maintenance').query('EQP_TAG == @EQP and MNT_TYPE == "replace" and SCENARIO_ID in @scenario_ids')
    out = cal.merge(rep[['SCENARIO_ID', 'STRT_DT', 'PART_CD']], on=['SCENARIO_ID', 'STRT_DT'], how='left')
    return out.rename(columns={'STRT_DT': 'PLAN_START', 'PART_CD': 'PLAN_PART'})[['SCENARIO_ID', 'PLAN_START', 'PLAN_PART']]


def episodes(pts, prob, th):
    """운전 구간(정비 상태)마다 경보가 처음 켜진 시점 한 줄. pts는 features.state_outcomes 결과(+ 특징), prob는 같은 순서."""
    d = pts.assign(PROB=np.asarray(prob, dtype=float)).sort_values(['SCENARIO_ID', 'OBSERVED_AT'])
    first = d[d['PROB'] >= th].drop_duplicates(['SCENARIO_ID', 'STATE'], keep='first')
    pl = plans(first['SCENARIO_ID'].unique().tolist())
    nxt = pd.merge_asof(first.sort_values('OBSERVED_AT'), pl.sort_values('PLAN_START'), left_on='OBSERVED_AT',
                        right_on='PLAN_START', by='SCENARIO_ID', direction='forward', allow_exact_matches=False)
    nxt['WAIT_H'] = (nxt['PLAN_START'] - nxt['OBSERVED_AT']).dt.total_seconds() / 3600
    return nxt.sort_values(['SCENARIO_ID', 'OBSERVED_AT']).reset_index(drop=True)


def failure_curve(ep, grid=np.arange(0, 73)):
    """경보가 처음 켜진 뒤 '정비가 없었다면' 고장 날 때까지 시간의 누적 분포 F(h) (카플란-마이어, 학습 세트 경보로 추정).

    고장 예정이 없으면 관측 끝까지의 시간에서 중도 절단으로 본다. 결과를 알 수 없는 경보(EV NaN)는 뺀다.
    반환: 시간(h) → 그 시간 안에 고장 날 확률 (pd.Series)
    """
    e = ep[ep['EV'].notna()]
    t = np.where(e['EV'] == 1, (e['TSTAR'] - e['OBSERVED_AT']).dt.total_seconds() / 3600, e['FOLLOW_H'])
    event = (e['EV'] == 1).values
    times = np.unique(t[event])
    factor = np.array([1 - ((t == u) & event).sum() / (t >= u).sum() for u in times])
    surv = [factor[times <= h].prod() for h in grid]
    return pd.Series(1 - np.array(surv), index=grid, name='F')


def _f(curve, h):
    return float(np.interp(h, curve.index, curve.values))


def fail_prob(curve, elapsed, horizon):
    """경보가 켜진 지 elapsed시간 동안 고장이 없었을 때, 앞으로 horizon시간 안에 고장 날 확률 (조건부)"""
    alive = 1 - _f(curve, elapsed)
    return (_f(curve, elapsed + horizon) - _f(curve, elapsed)) / alive if alive > 1e-9 else 1.0


# 2. 선택지와 판단

def _option(name, done_h, maint, down, curve, cfg, elapsed, p_override=None, note=''):
    """선택지 하나: 정비가 done_h시간 뒤에 끝날 때 비용 항목과 기대 총비용"""
    p = fail_prob(curve, elapsed, done_h) if p_override is None else p_override
    c = cfg.costs
    expected = (1 - p) * (maint + down) + p * c.failure_loss
    return {'선택지': name, '정비 완료까지(h)': done_h, '정비 비용': maint, '생산 차질 비용': down,
            '고장 확률': p, '고장 시 손실': c.failure_loss, '기대 총비용': expected,
            '안전 기준 충족': p <= cfg.safety_max, '근거': note}


def options(wait_h, curve, cfg, elapsed=0.0, delay_h=None, delay_downtime_h=None, p_override=None):
    """선택지별 비용 항목과 기대 총비용 (대시보드 의사결정 화면과 decide가 같이 쓴다).

    wait_h            지금부터 다음 정기 교체까지 시간 (계획이 없으면 NaN)
    elapsed           경보가 처음 켜진 뒤 지난 시간 (시뮬레이션은 경보가 켜진 순간에 판단하므로 0)
    delay_h           'N시간 뒤 교체' 선택지 (예: 생산 영향이 적은 시간대). None이면 넣지 않는다
    delay_downtime_h  그 시간대의 정지 1시간 생산 차질 비용 (None이면 평소와 같음)
    p_override        선택지마다 고장 확률을 직접 줄 때 {'지금 교체': p, ...} (현장 판단으로 덮어쓰기)
    반환              선택지 목록. 각 선택지는 비용 항목, 고장 확률, 기대 총비용, 안전 기준 충족 여부, 근거
    """
    c = cfg.costs
    po = p_override or {}
    two_parts = 2 * c.part + c.unnecessary                       # 두 부품 교체 (하나는 멀쩡하다고 본다): 24 + 8 = 32
    start_now = max(cfg.crew_wait_h, cfg.prep_h - c.replace_h)    # 정비팀·부품이 준비되는 시각
    out = [_option('지금 두 부품 교체', start_now + c.replace_h, two_parts, c.replace_h * c.downtime_h, curve, cfg, elapsed,
                   po.get('지금 두 부품 교체'),
                   f'부품 {2 * c.part:.0f} + 불필요 정비 {c.unnecessary:.0f}(두 부품 중 하나는 멀쩡) · 정지 {c.replace_h:.0f}시간 × {c.downtime_h:.0f}'
                   + (f' · 정비팀·부품 준비까지 {start_now:.0f}시간' if start_now > 0 else ''))]
    if delay_h is not None and delay_h > start_now:
        rate = c.downtime_h if delay_downtime_h is None else delay_downtime_h
        out.append(_option(f'{delay_h:.0f}시간 뒤 두 부품 교체', delay_h + c.replace_h, two_parts, c.replace_h * rate, curve, cfg,
                           elapsed, po.get('N시간 뒤 교체'),
                           f'정지 {c.replace_h:.0f}시간 × 그 시간대 생산 차질 {rate:.0f} (생산 영향이 적은 시간대에 정비)'))
    if pd.notna(wait_h) and wait_h <= cfg.max_wait_h and wait_h + c.replace_h >= cfg.prep_h and wait_h >= cfg.crew_wait_h:
        extra = c.part + (1 - cfg.p_other_needed) * c.unnecessary
        out.append(_option('다음 정기 교체 때 두 부품 함께', wait_h, extra, 0.0, curve, cfg, elapsed, po.get('정기 교체 때'),
                           f'계획된 부품은 어차피 교체 → 다른 부품 {c.part:.0f} + 멀쩡할 확률 {1 - cfg.p_other_needed:.1f} × {c.unnecessary:.0f}'
                           ' · 계획 정지를 써서 생산 차질 없음 · 정기 교체가 시작되면 설비가 서므로 그 전까지의 고장만 위험'))
    return out


def recommend(opts):
    """안전 기준을 지키는 선택지 중 기대 총비용이 가장 낮은 것. 모두 넘으면 '지금 두 부품 교체'(즉시 조치)"""
    ok = [o for o in opts if o['안전 기준 충족']]
    if not ok:
        return opts[0], '모든 선택지가 안전 기준을 넘음 → 즉시 조치'
    best = min(ok, key=lambda o: o['기대 총비용'])
    return best, ''


def decide(row, curve, cfg, policy, elapsed=0.0):
    """경보 한 건의 행동: ('now' | 'wait' | 'none', 근거 문장, 정기 교체까지 고장 확률). row에는 WAIT_H"""
    if policy == 'regular':
        return 'none', '정기 교체만 (경보를 쓰지 않음)', np.nan
    if policy == 'now':
        return 'now', '경보 즉시 교체', np.nan
    opts = options(row['WAIT_H'], curve, cfg, elapsed)
    best, urgent = recommend(opts)
    wait = next((o for o in opts if o['선택지'].startswith('다음 정기')), None)
    p = wait['고장 확률'] if wait else np.nan
    if wait is None:
        why = ('다음 정기 교체 계획 없음' if pd.isna(row['WAIT_H']) else
               f'정기 교체({row["WAIT_H"]:.0f}시간 뒤)를 기다릴 수 없음(최대 대기·부품 준비·정비팀)') + ' → 지금 교체'
        return 'now', why, p
    if best is wait:
        return 'wait', (f'정기 교체 {row["WAIT_H"]:.0f}시간 뒤, 그때까지 고장 확률 {p:.2f} → 기대 총비용 {wait["기대 총비용"]:.0f}점 '
                        f'< 지금 교체 {opts[0]["기대 총비용"]:.0f}점 → 정기 교체 때 두 부품 함께'), p
    if not wait['안전 기준 충족']:
        return 'now', f'정기 교체까지 {row["WAIT_H"]:.0f}시간 동안 고장 확률 {p:.2f} > 안전 기준 {cfg.safety_max:.2f} → 지금 교체', p
    return 'now', f'기다리는 기대 총비용 {wait["기대 총비용"]:.0f}점 ≥ 지금 교체 {opts[0]["기대 총비용"]:.0f}점 → 지금 교체', p


# 3. 채점 (정답 사용)

def _cause_part(row, dg):
    """경보 시각 이후 이 운전 구간의 열화 원인 부품 (고장 예정이 없으면 None)"""
    if row['EV'] != 1:
        return None
    g = dg[(dg['SCENARIO_ID'] == row['SCENARIO_ID']) & (dg['END_AT'] > row['OBSERVED_AT'])
           & (dg['START_AT'] <= row['TSTAR'])]
    return g['CAUSE_PART'].iloc[0] if len(g) else None


def score(row, action, cause, cfg=None):
    """행동의 결과와 비용 (정답 사용): (결과, 비용, 계획 밖 정지 시간)"""
    cfg = cfg or DSSConfig()
    c = cfg.costs
    unknown = pd.isna(row['EV']) or (row['EV'] == 0 and row['FOLLOW_H'] < 24)
    lead = (row['TSTAR'] - row['OBSERVED_AT']).total_seconds() / 3600 if row['EV'] == 1 else np.inf
    if action == 'none':
        return ('고장' if row['EV'] == 1 else '해당 없음'), (c.failure_loss if row['EV'] == 1 else 0), (c.repair_h if row['EV'] == 1 else 0)
    if unknown:
        return '미확정', np.nan, np.nan
    if action == 'now':
        done = max(cfg.crew_wait_h, cfg.prep_h - c.replace_h) + c.replace_h
        if lead < done:
            return '늦은 경보 → 고장', c.failure_loss, c.repair_h
        n_bad = 1 if row['EV'] == 1 else 2                      # 고장 예정이면 하나는 원인 부품, 아니면 둘 다 멀쩡
        return (('교체로 고장 막음' if row['EV'] == 1 else '불필요 교체'),
                2 * c.part + c.replace_h * c.downtime_h + n_bad * c.unnecessary, c.replace_h)
    # 기다림: 정기 교체가 시작되기 전에 고장 나면 실패
    if lead <= row['WAIT_H']:
        return '기다리다 고장', c.failure_loss, c.repair_h
    other = next(p for p in PARTS if p != row['PLAN_PART'])
    needed = cause == other
    return ('정기 교체 때 함께 교체' + ('' if row['EV'] == 1 else ' (고장 예정 없음)'),
            c.part + (0 if needed else c.unnecessary), 0)


def simulate(pts, prob, th, curve, cfg=None, policy='dss'):
    """정책 하나를 시뮬레이션한다.

    pts    features.state_outcomes 결과 (정답을 알 수 없는 시점 포함, 모든 예측 시점)
    반환   {'summary': 요약, 'episodes': 경보별 판단·결과 표}
           비용 = 경보 판단 비용 + 경보가 없던 운전 구간의 고장 손실 (정답을 알 수 있는 경보만, 미확정은 따로)
    """
    cfg = cfg or DSSConfig()
    c = cfg.costs
    fails = _actual_failures(pts)
    names = {'regular': '① 정기 교체만', 'now': '② 경보 즉시 교체', 'dss': '③ DSS'}
    if policy == 'regular':
        n = len(fails)
        summary = {'정책': names[policy], '비용': float(n * c.failure_loss), '고장': n, '고장 손실': float(n * c.failure_loss),
                   '정비 횟수': 0, '지금 교체': 0, '정기 교체 때': 0, '기다리다 고장': 0,
                   '고장 전 조치 완료율': 0.0, '계획 밖 정지(h)': float(n * c.repair_h), '미확정 경보': 0}
        return {'summary': summary, 'episodes': pd.DataFrame()}
    ep = episodes(pts, prob, th)
    dg = load_answer('diagnostics').query('EQP_CD == @EQP and HEALTH_STATE == "degraded"')
    rows = []
    for _, r in ep.iterrows():
        action, why, p = decide(r, curve, cfg, policy)
        outcome, cost, down = score(r, action, _cause_part(r, dg), cfg)
        rows.append({'SCENARIO_ID': r['SCENARIO_ID'], '경보 시각(h)': r['T_H'], '정기 교체까지(h)': r['WAIT_H'],
                     '판단': {'now': '지금 교체', 'wait': '정기 교체 때'}[action], '근거': why,
                     '기다릴 때 고장 확률': p, '결과': outcome, '비용': cost, '계획 밖 정지(h)': down,
                     'EV': r['EV'], 'STATE': r['STATE']})
    out = pd.DataFrame(rows)
    alarmed = set(zip(out['SCENARIO_ID'], out['STATE']))
    missed = [k for k in fails if k not in alarmed]               # 경보가 없던 운전 구간의 실제 고장
    lab = out[out['결과'] != '미확정']
    failed = lab['결과'].isin(['늦은 경보 → 고장', '기다리다 고장'])
    n_fail = len(missed) + int(failed.sum())
    at_risk = lab['EV'] == 1                                      # 고장 예정이었던 경보
    summary = {
        '정책': names[policy],
        '비용': float(len(missed) * c.failure_loss + lab['비용'].sum()),
        '고장': n_fail, '고장 손실': float(n_fail * c.failure_loss),
        '정비 횟수': int((~failed).sum()),
        '지금 교체': int(((lab['판단'] == '지금 교체') & ~failed).sum()),
        '정기 교체 때': int(((lab['판단'] == '정기 교체 때') & ~failed).sum()),
        '기다리다 고장': int((lab['결과'] == '기다리다 고장').sum()),
        '고장 전 조치 완료율': float((at_risk & ~failed).sum() / (at_risk.sum() + len(missed))) if at_risk.sum() + len(missed) else np.nan,
        '계획 밖 정지(h)': float(len(missed) * c.repair_h + lab['계획 밖 정지(h)'].sum()),
        '미확정 경보': int((out['결과'] == '미확정').sum()),
    }
    return {'summary': summary, 'episodes': out}


def savings(summaries):
    """정책별 요약 {policy: summary}에서 '시뮬레이션에 따른 예상 절감액' (① 정기 교체만 대비)"""
    base = summaries['regular']['비용']
    return {p: {'예상 절감액': base - s['비용'], '절감률': (base - s['비용']) / base if base else np.nan}
            for p, s in summaries.items()}


def _actual_failures(pts):
    """예측 시점이 있는 시나리오의 실제 고장이 난 운전 구간 (SCENARIO_ID, STATE) 목록"""
    ids = pts['SCENARIO_ID'].unique().tolist()
    mnt = load_source('maintenance').query('EQP_TAG == @EQP and SCENARIO_ID in @ids')
    f = mnt[mnt['FAILURE_DT'].notna()]
    keys = []
    for r in f.itertuples():
        state = int((mnt.loc[mnt['SCENARIO_ID'] == r.SCENARIO_ID, 'STRT_DT'] < r.FAILURE_DT).sum())
        keys.append((r.SCENARIO_ID, state))
    return keys
