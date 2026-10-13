"""연주 조기경보 평가: 시점 단위 혼동행렬, 고장 단위 감지, 비용.

평가 결정(reports/연주_조기경보_모델_결과.md 3절의 A~G)은 C만 확정이고 나머지는 잠정이라, 선택지를 모두 함께 계산한다.

    from src.evaluate import evaluate, print_report, choose_threshold

    th = choose_threshold(train_df, train_oof_prob)                  # 잠정 결정: 점진 고장 모두 감지 + 진짜 오경보 0 중 가장 낮은 임계값
    res = evaluate(valid_df, valid_prob, th)
    print_report(res)

df에는 make_features 결과의 SCENARIO_ID, OBSERVED_AT, risk24, RUL_EVENT, RUL_HOURS 열이 있어야 한다.
prob은 df와 같은 순서의 24시간 고장 확률(또는 점수)이다. 경보 = prob ≥ th.

정의
- 이른 경보: 정답은 0(24시간 안 고장 없음)이지만, 관측 기간 안에 고장 예정(RUL_EVENT=1)인 시점의 경보
- 돌발 고장: 진단 정답에 고장 직전 열화 구간이 없는 고장 (EDA 7-1). 센서 신호가 없어 미리 잡기 어렵다
- 고장 단위 감지: 고장 전 같은 운전 구간 안에서, 고장 min_lead시간(교체 4시간) 이전에 경보가 한 번이라도 났는지
- 비용: 데이터셋 비용 규칙(toolkit/targets.md). 경보가 나면 바로 교체한다고 가정한다
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.posko_data import load_answer, load_scenarios, load_source
from src.preprocess import EQP

# 비용 규칙 (점수): 교체 12 + 정지 4시간×2, 복구 30 + 정지 8시간×2 + 고장 벌점 20, 불필요 정비 벌점 8
COST_REPLACE = 12 + 4 * 2            # 20
COST_FAILURE = 30 + 8 * 2 + 20       # 66
COST_UNNECESSARY = 8
MIN_LEAD_H = 4                       # 교체(4시간)가 고장 전에 끝나야 고장을 막는다

# 부품 선택 가정(논의 항목 C). 센서로는 원인 부품(진동장치·밸브)을 구분할 수 없다(EDA 10절)
#   맞힘       원인 부품을 맞혀 교체한다고 가정 (상한)
#   두 부품    경보마다 두 부품을 모두 교체. 고장은 막지만 멀쩡한 부품 하나는 불필요 정비
#   반반       둘 중 하나를 골라 교체. 절반은 틀린 부품이라 불필요 정비 + 고장 (기댓값)
PART_RULES = {
    # prevent: 고장 전에 경보해서 교체할 때 / needed: 고장 예정인 부품을 교체할 때(실제로는 정기 교체로 막힌 경우)
    # false: 고장 예정이 없는데 교체할 때(오경보)
    '맞힘': {'prevent': COST_REPLACE,                                                         # 20
             'needed': COST_REPLACE,                                                          # 20
             'false': COST_REPLACE + COST_UNNECESSARY},                                       # 28
    '두 부품': {'prevent': 2 * 12 + 4 * 2 + COST_UNNECESSARY,                                  # 40
                'needed': 2 * 12 + 4 * 2 + COST_UNNECESSARY,                                  # 40
                'false': 2 * 12 + 4 * 2 + 2 * COST_UNNECESSARY},                              # 48
    '반반': {'prevent': 0.5 * COST_REPLACE + 0.5 * (COST_REPLACE + COST_UNNECESSARY + COST_FAILURE),   # 57
             'needed': 0.5 * COST_REPLACE + 0.5 * (COST_REPLACE + COST_UNNECESSARY),                    # 24
             'false': COST_REPLACE + COST_UNNECESSARY},                                                 # 28
}


# 1. 고장 사건과 시점별 고장 정보 (평가용, 특징 아님)

def failure_events(scenario_ids):
    """평가 대상 시나리오의 실제 고장 목록.

    TYPE       점진 고장 / 돌발 고장 (고장 직전 열화 구간이 있으면 점진)
    SEG_START  고장이 난 운전 구간의 시작 (직전 정비가 끝난 시각, 없으면 시나리오 시작)
    """
    scen = load_scenarios().set_index('SCENARIO_ID')
    mnt = load_source('maintenance').query('EQP_TAG == @EQP and SCENARIO_ID in @scenario_ids')
    fails = mnt[mnt['FAILURE_DT'].notna()][['SCENARIO_ID', 'WO_NO', 'FAILURE_DT']].copy()
    dg = load_answer('diagnostics').query('EQP_CD == @EQP and HEALTH_STATE == "degraded"')
    x = fails.merge(dg[['SCENARIO_ID', 'END_AT']], on='SCENARIO_ID')
    gradual = set(x.loc[(x['END_AT'] - x['FAILURE_DT']).abs() <= pd.Timedelta(seconds=1), 'WO_NO'])
    fails['TYPE'] = np.where(fails['WO_NO'].isin(gradual), '점진 고장', '돌발 고장')

    def seg_start(r):
        prev = mnt[(mnt['SCENARIO_ID'] == r['SCENARIO_ID']) & (mnt['END_DT'] <= r['FAILURE_DT'])]['END_DT']
        return prev.max() if len(prev) else scen.loc[r['SCENARIO_ID'], 'START_AT']
    fails['SEG_START'] = fails.apply(seg_start, axis=1)
    return fails.reset_index(drop=True)


def attach_failure_info(df, events):
    """시점마다 같은 운전 구간 안의 다음 실제 고장을 붙인다.

    FAIL_WO    다음 고장의 작업 번호 (같은 운전 구간에 고장이 없으면 NaN)
    HTF        고장까지 남은 시간(h)
    FAIL_TYPE  점진 고장 / 돌발 고장
    """
    out = df.copy()
    out['_ord'] = np.arange(len(out))
    ev = events.sort_values('FAILURE_DT')
    m = pd.merge_asof(out.sort_values('OBSERVED_AT'), ev[['SCENARIO_ID', 'WO_NO', 'FAILURE_DT', 'TYPE', 'SEG_START']],
                      left_on='OBSERVED_AT', right_on='FAILURE_DT', by='SCENARIO_ID',
                      direction='forward', allow_exact_matches=False)
    same_seg = m['SEG_START'] <= m['OBSERVED_AT']          # 사이에 정비가 끼면 다른 운전 구간
    m.loc[~same_seg, ['WO_NO', 'FAILURE_DT', 'TYPE']] = np.nan
    m['HTF'] = (m['FAILURE_DT'] - m['OBSERVED_AT']).dt.total_seconds() / 3600
    m = m.sort_values('_ord')
    out['FAIL_WO'] = m['WO_NO'].values
    out['HTF'] = m['HTF'].values
    out['FAIL_TYPE'] = m['TYPE'].values
    return out.drop(columns='_ord')


# 2. 시점 단위 평가 (혼동행렬)

def point_metrics(d, prob, th):
    """시점 단위 혼동행렬과 지표. 오경보는 이른 경보/진짜 오경보로, 놓친 고장은 돌발/점진으로 나눈다."""
    y = d['risk24'].values.astype(int)
    alarm = prob >= th
    tp, fp = alarm & (y == 1), alarm & (y == 0)
    fn, tn = ~alarm & (y == 1), ~alarm & (y == 0)
    early = fp & (d['RUL_EVENT'].values == 1)
    sudden = d['FAIL_TYPE'].values == '돌발 고장'
    n_tp, n_fp, n_fn = tp.sum(), fp.sum(), fn.sum()
    prec = n_tp / (n_tp + n_fp) if n_tp + n_fp else np.nan
    rec = n_tp / (n_tp + n_fn) if n_tp + n_fn else np.nan
    g_pos = (y == 1) & ~sudden

    def fbeta(b):
        return (1 + b * b) * prec * rec / (b * b * prec + rec) if prec + rec else 0.0

    return {
        '시점 수': len(y), '양성': int(y.sum()), '임계값': th,
        'TP': int(n_tp), 'FP': int(n_fp), 'FN': int(n_fn), 'TN': int(tn.sum()),
        'FP 중 이른 경보': int(early.sum()), 'FP 중 진짜 오경보': int((fp & ~early).sum()),
        'FN 중 돌발 고장': int((fn & sudden).sum()), 'FN 중 점진 고장': int((fn & ~sudden).sum()),
        'AP': average_precision_score(y, prob), 'ROC AUC': roc_auc_score(y, prob),
        '정밀도': prec, '재현율': rec, 'F1': fbeta(1), 'F2': fbeta(2),
        '정밀도 (이른 경보 = 맞음)': (n_tp + early.sum()) / (n_tp + n_fp) if n_tp + n_fp else np.nan,
        '재현율 (점진 고장만)': (tp & g_pos).sum() / g_pos.sum() if g_pos.sum() else np.nan,
    }


# 3. 고장 단위 평가와 비용

def _alarm_episodes(d, alarm):
    """같은 시나리오·같은 운전 구간에서 연속으로 경보가 난 시점들을 하나의 경보 묶음으로 센다."""
    s = d.assign(ALARM=alarm).sort_values(['SCENARIO_ID', 'OBSERVED_AT'])
    seg = s['FAIL_WO'].fillna('none')
    new = s['ALARM'] & ~((s['ALARM'].shift(fill_value=False)) & (s['SCENARIO_ID'] == s['SCENARIO_ID'].shift())
                         & (seg == seg.shift()))
    return s[s['ALARM']].assign(EP=new.cumsum()[s['ALARM']])


def event_and_cost(d, prob, th, events):
    """고장마다 미리 경보했는지와 리드타임, 그리고 부품 선택 가정별 비용.

    - 고장: 고장 MIN_LEAD_H시간 이전에 경보가 있으면 '감지'. 감지하면 교체 비용, 놓치면 고장 비용
    - 고장과 상관없는 경보 묶음: 고장 예정(RUL_EVENT=1)이면 필요한 교체(고장은 실제 데이터에서 정기 교체로 막혔음),
      아니면 오경보(불필요 정비)
    """
    alarm = prob >= th
    ev = events[events['SCENARIO_ID'].isin(d['SCENARIO_ID'])].copy()
    a = d[alarm & (d['HTF'] >= MIN_LEAD_H)]
    first = a.groupby('FAIL_WO')['HTF'].max()            # 가장 이른 경보의 리드타임
    ev['감지'] = ev['WO_NO'].isin(first.index)
    ev['리드타임(h)'] = ev['WO_NO'].map(first)

    ep = _alarm_episodes(d, alarm)
    other = ep[ep['FAIL_WO'].isna()].groupby('EP').agg(RUL_EVENT=('RUL_EVENT', 'first'))
    n_needed, n_false = int((other['RUL_EVENT'] == 1).sum()), int((other['RUL_EVENT'] != 1).sum())

    n_det = int(ev['감지'].sum())
    n_miss = len(ev) - n_det
    cost = {}
    for rule, c in PART_RULES.items():
        cost[rule] = n_det * c['prevent'] + n_miss * COST_FAILURE + n_needed * c['needed'] + n_false * c['false']
    by_type = ev.groupby('TYPE').agg(고장=('WO_NO', 'size'), 감지=('감지', 'sum'),
                                     리드타임_중앙값=('리드타임(h)', 'median'))
    summary = {
        '고장': len(ev), '감지': n_det, '놓침': n_miss,
        '고장 감지율': n_det / len(ev) if len(ev) else np.nan,
        '리드타임 중앙값(h)': ev['리드타임(h)'].median(),
        '고장과 무관한 경보 묶음: 필요한 교체': n_needed, '고장과 무관한 경보 묶음: 오경보': n_false,
    }
    refs = {
        '아무것도 안 함': len(ev) * COST_FAILURE,
        '완벽한 경보 (부품 맞힘)': len(ev) * COST_REPLACE,
        '점진 고장만 완벽 (돌발은 놓침, 부품 맞힘)': ((ev['TYPE'] == '점진 고장').sum() * COST_REPLACE
                                              + (ev['TYPE'] == '돌발 고장').sum() * COST_FAILURE),
    }
    return summary, by_type, cost, refs, ev


# 4. 한 번에 평가하기

def evaluate(df, prob, th, events=None):
    """시점 단위·고장 단위·비용을 함께 계산한다. events를 주지 않으면 df의 시나리오로 만든다."""
    prob = np.asarray(prob, dtype=float)
    if events is None:
        events = failure_events(df['SCENARIO_ID'].unique().tolist())
    d = attach_failure_info(df.reset_index(drop=True), events)
    summary, by_type, cost, refs, ev = event_and_cost(d, prob, th, events)
    return {'point': point_metrics(d, prob, th), 'event': summary, 'event_by_type': by_type,
            'cost': cost, 'cost_reference': refs, 'events': ev}


def choose_threshold(df, prob, criterion='earliest', part_rule='두 부품', all_gradual=True,
                     grid=np.round(np.arange(0.01, 1.0, 0.01), 2), events=None):
    """임계값을 고른다 (논의 항목 D). 학습 세트의 교차검증 예측에서 고르고 검증 세트에 그대로 쓴다.

    criterion
      'earliest' (잠정 결정) 점진 고장을 모두 잡고 진짜 오경보(고장 예정이 없는 경보 묶음)가 0인 임계값 중 가장 낮은 것.
                 오경보 없이 가장 일찍 경보한다. 그런 임계값이 없으면 진짜 오경보가 가장 적은 것 중 가장 낮은 것
      'cost'     part_rule 가정의 비용 최소 (같으면 높은 임계값). 비용 규칙은 교체할 4시간만 있으면
                 막았다고 보므로 리드타임을 반영하지 못해 늦은 경보를 고르는 경향이 있다(참고용)
      'f1' / 'f2'  시점 단위 F1 / F2(재현율 중시) 최대
    all_gradual 'earliest'·'cost'에서 점진 고장을 하나도 놓치지 않는 임계값 중에서만 고른다
    """
    prob = np.asarray(prob, dtype=float)
    if criterion in ('earliest', 'cost'):
        if events is None:
            events = failure_events(df['SCENARIO_ID'].unique().tolist())
        d = attach_failure_info(df.reset_index(drop=True), events)
        rows = []
        for t in grid:
            summary, by_type, cost, _, _ = event_and_cost(d, prob, t, events)
            g = by_type.loc['점진 고장'] if '점진 고장' in by_type.index else None
            all_g = g is None or int(g['감지']) == int(g['고장'])
            rows.append({'t': t, 'cost': cost[part_rule], 'all_g': all_g,
                         'false': summary['고장과 무관한 경보 묶음: 오경보']})
        cand = [r for r in rows if r['all_g']] if all_gradual else rows
        if all_gradual and not cand:
            print('  알림: 점진 고장을 모두 잡는 임계값이 없어 조건 없이 고른다')
            cand = rows
        if criterion == 'earliest':
            fewest = min(r['false'] for r in cand)
            return float(min(r['t'] for r in cand if r['false'] == fewest))
        best = min(r['cost'] for r in cand)
        return float(max(r['t'] for r in cand if r['cost'] == best))
    y = df['risk24'].values.astype(int)
    b = 1 if criterion == 'f1' else 2

    def fb(t):
        a = prob >= t
        tp, fp, fn = (a & (y == 1)).sum(), (a & (y == 0)).sum(), (~a & (y == 1)).sum()
        return (1 + b * b) * tp / ((1 + b * b) * tp + b * b * fn + fp) if tp else 0.0
    scores = {t: fb(t) for t in grid}
    return float(max(scores, key=scores.get))


def print_report(res, title=''):
    if title:
        print(f'\n■ {title}')
    p = res['point']
    print(f'  [시점 단위] 임계값 {p["임계값"]:.2f} · 시점 {p["시점 수"]}개 (양성 {p["양성"]})')
    print(f'    TP {p["TP"]}  FP {p["FP"]} (이른 경보 {p["FP 중 이른 경보"]}, 진짜 오경보 {p["FP 중 진짜 오경보"]})')
    print(f'    FN {p["FN"]} (돌발 고장 {p["FN 중 돌발 고장"]}, 점진 고장 {p["FN 중 점진 고장"]})  TN {p["TN"]}')
    print(f'    AP {p["AP"]:.3f} · ROC AUC {p["ROC AUC"]:.3f} · 정밀도 {p["정밀도"]:.3f} · 재현율 {p["재현율"]:.3f}'
          f' · F1 {p["F1"]:.3f} · F2 {p["F2"]:.3f}')
    print(f'    정밀도 (이른 경보 = 맞음) {p["정밀도 (이른 경보 = 맞음)"]:.3f} · 재현율 (점진 고장만) {p["재현율 (점진 고장만)"]:.3f}')
    e = res['event']
    print(f'  [고장 단위] 고장 {e["고장"]}건 중 {e["감지"]}건 감지 ({e["고장 감지율"]:.3f}), '
          f'리드타임 중앙값 {e["리드타임 중앙값(h)"]:.0f}h · 고장과 무관한 경보: 필요한 교체 '
          f'{e["고장과 무관한 경보 묶음: 필요한 교체"]}, 오경보 {e["고장과 무관한 경보 묶음: 오경보"]}')
    for t, r in res['event_by_type'].iterrows():
        print(f'    {t}: {int(r["감지"])}/{int(r["고장"])} 감지, 리드타임 중앙값 {r["리드타임_중앙값"]:.0f}h')
    print('  [비용, 점수] ' + ' · '.join(f'{k} {v:.0f}' for k, v in res['cost'].items()))
    print('    참고: ' + ' · '.join(f'{k} {v:.0f}' for k, v in res['cost_reference'].items()))
