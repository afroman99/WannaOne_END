"""결과보고서 피드백(2026-10-19) 확인 분석: 감지 기준·비교 조건·특징 기여·미확정 경보 비용.

    python analysis/12_report_checks.py      # 약 10분

최종 모델·기준·정책 값은 바꾸지 않는다. 이미 정한 설정이 어떤 조건에서 나온 숫자인지 다시 보여 주는 분석이다.
  1. 고장 단위 감지: 점진 고장마다 첫 경보 시각, 4시간·24시간 기준 각각 몇 건인지
  2. 시점 단위 지표의 분모: TP·FP·FN과 지표마다 몇 개 시점·몇 건 고장 기준인지
  3. 같은 조건 비교: 1시간 예측 시점, 같은 표본(기준 예제와 우리 특징이 모두 계산되는 시점),
     같은 학습·검증 나눔에서 특징 묶음 × 모델을 바꿔 본다 (특징과 모델의 몫을 따로 본다)
       특징: 기준 예제(원래 값 12시간 요약 31개) / 우리 특징(센서 + 정비 이력 28개)
       모델: 로지스틱 회귀 / 랜덤포레스트 (src/model.py와 같은 설정)
       참고: 학습 없는 표준화 점수(SCORE) 하나, 기준 예제 원래 설정(랜덤포레스트 180그루)
     임계값은 모든 줄에서 같은 규칙(팀 결정 D)으로 학습 세트 교차검증 예측에서 고른다.
  4. 미확정 경보의 비용: DSS 시뮬레이션에서 결과를 알 수 없어 합계에서 뺀 경보를,
     '고장 예정이 없었다'고 보고 조치 비용을 모두 넣은 상한과 함께 본다 (src/dss.py DSSConfig.charge_unknown)
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dss import Costs, DSSConfig, PartsPolicy, episodes, failure_curve, field_costs, simulate
from src.evaluate import _alarm_episodes, attach_failure_info, choose_threshold, evaluate, failure_events
from src.features import TARGET_COLS, make_features, make_targets_hourly, normal_scenarios, state_outcomes
from src.model import FEATURES, FINAL, MODELS, SEED, fold_features, oof_predict
from src.posko_data import load_scenarios
from src.preprocess import apply_params, build_hourly, fit_params

spec = importlib.util.spec_from_file_location('baseline', ROOT / 'analysis' / '02_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

TH = 0.56          # 최종 경보 기준 (analysis/09_hourly.py)
KEY = ['SCENARIO_ID', 'OBSERVED_AT']


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


def event_row(d, prob, events, th):
    """고장 단위 결과 한 줄 (점진 고장 기준 감지·리드타임, 진짜 오경보 묶음)"""
    r = evaluate(d, prob, th, events)
    ev = r['events']
    g = ev[ev['TYPE'] == '점진 고장']
    p = r['point']
    return r, {
        'AP': round(average_precision_score(d['risk24'], prob), 3), '임계값': th,
        '점진 감지(4h 전)': f'{int(g["감지"].sum())}/{len(g)}',
        '24h 이상 전': f'{int((g["리드타임(h)"] >= 24).sum())}/{len(g)}',
        '리드타임 중앙값(h)': round(g['리드타임(h)'].median(), 1),
        '진짜 오경보 묶음': r['event']['고장과 무관한 경보 묶음: 오경보'],
        'TP·FP·FN': f'{p["TP"]}·{p["FP"]}·{p["FN"]}',
    }


# 0. 데이터

section('0. 데이터 (1시간 예측 시점)')
scen = load_scenarios()
train_ids = scen.loc[scen['SPLIT'] == 'train', 'SCENARIO_ID'].tolist()
valid_ids = scen.loc[scen['SPLIT'] == 'validation', 'SCENARIO_ID'].tolist()
hr = build_hourly(train_ids + valid_ids)
tg = make_targets_hourly(train_ids + valid_ids, step_h=1)
params = fit_params(hr[hr['SCENARIO_ID'].isin(train_ids)], normal_scenarios(train_ids))
hr_fit = apply_params(hr, params)
feat = make_features(hr_fit, tg)
ftr = feat[feat['SPLIT'] == 'train'].reset_index(drop=True)
fva = feat[feat['SPLIT'] == 'validation'].reset_index(drop=True)
ev_train, ev_valid = failure_events(train_ids), failure_events(valid_ids)
cols, make = FEATURES[FINAL['features']], MODELS[FINAL['model']]
model = make().fit(ftr[cols].values.astype(float), ftr['risk24'].values)
pv = model.predict_proba(fva[cols].values.astype(float))[:, 1]
print(f'학습 {len(ftr)}개 시점 · 검증 {len(fva)}개 시점 (양성 {int(fva["risk24"].sum())}) · 경보 기준 {TH}')


# 1. 고장 단위 감지 (검증 세트, 최종 모델)

section('1. 고장 단위 감지: 4시간 전 감지와 24시간 이상 전 감지는 다른 기준이다')
r_final, row_final = event_row(fva, pv, ev_valid, TH)
ev = r_final['events'].sort_values('리드타임(h)')
print(ev[['SCENARIO_ID', 'TYPE', '감지', '리드타임(h)']].to_string(index=False))
g = ev[ev['TYPE'] == '점진 고장']
print(f'\n점진 고장 {len(g)}건: 4시간 전 감지 {int(g["감지"].sum())}건, 24시간 이상 전 {int((g["리드타임(h)"] >= 24).sum())}건, '
      f'24시간 미만 {int((g["리드타임(h)"] < 24).sum())}건 ({g.loc[g["리드타임(h)"] < 24, "리드타임(h)"].min():.2f}~'
      f'{g.loc[g["리드타임(h)"] < 24, "리드타임(h)"].max():.2f}시간). 첫 경보 중앙값 {g["리드타임(h)"].median():.3f}시간. '
      f'돌발 고장 {int((ev["TYPE"] == "돌발 고장").sum())}건은 감지 {int(ev.loc[ev["TYPE"] == "돌발 고장", "감지"].sum())}건')


# 2. 시점 단위 지표의 분모

section('2. 시점 단위 지표와 분모 (검증 세트, 최종 모델)')
p = r_final['point']
gpos = int(p['TP'] + p['FN'] - p['FN 중 돌발 고장'])
print(f'예측 시점 {p["시점 수"]}개 (양성 = 24시간 안 고장 {p["양성"]}개, 그중 점진 고장 {gpos}개 · 돌발 고장 {p["FN 중 돌발 고장"]}개 이상)')
print(f'TP {p["TP"]} · FP {p["FP"]} (이른 경보 {p["FP 중 이른 경보"]} · 진짜 오경보 {p["FP 중 진짜 오경보"]}) · '
      f'FN {p["FN"]} (돌발 {p["FN 중 돌발 고장"]} · 점진 {p["FN 중 점진 고장"]}) · TN {p["TN"]}')
print(f'AP {p["AP"]:.3f} (전체 {p["시점 수"]}개 시점) · 정밀도 {p["정밀도"]:.3f} (경보 {p["TP"] + p["FP"]}개 시점) · '
      f'재현율 돌발 포함 {p["재현율"]:.3f} (양성 {p["양성"]}개 시점) · 점진만 {p["재현율 (점진 고장만)"]:.3f} · F1 {p["F1"]:.3f}')
e = r_final['event']
print(f'고장 단위 {len(ev)}건 (점진 {len(g)} · 돌발 {len(ev) - len(g)}) · 진짜 오경보 묶음 {e["고장과 무관한 경보 묶음: 오경보"]}')
d = attach_failure_info(fva, ev_valid)
eps = _alarm_episodes(d, pv >= TH)
print(f'경보 묶음 {eps["EP"].nunique()}개 (같은 운전 구간에서 이어진 경보 시점을 한 번으로 묶음)')


# 3. 같은 조건 비교: 특징 × 모델

section('3. 같은 조건 비교 (1시간 예측, 같은 표본, 같은 나눔, 같은 임계값 규칙)')
scen_tv = scen[scen['SPLIT'] != 'test']
obs = baseline.load_observations(scen_tv)
rec = baseline.make_records(obs, tg.rename(columns={'T_H': 'h'}).assign(SPLIT=tg['SPLIT']))
common = rec[KEY].merge(feat[KEY], on=KEY)
print(f'기준 예제 특징이 계산되는 시점 {len(rec)}개 · 우리 특징 {len(feat)}개 · 공통 {len(common)}개 '
      f'(예측 시점에 정지 중인 시점은 기준 예제가 뺀다)')
F = feat.merge(rec[KEY + ['temporal']], on=KEY)
tr = F[F['SPLIT'] == 'train'].reset_index(drop=True)
va = F[F['SPLIT'] == 'validation'].reset_index(drop=True)
y = tr['risk24'].values.astype(int)
Xb_tr, Xb_va = np.stack(tr['temporal']), np.stack(va['temporal'])
folds = fold_features(hr, tr[TARGET_COLS])

base_rf = lambda: make_pipeline(SimpleImputer(add_indicator=True),
                                RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=5,
                                                       class_weight='balanced', random_state=SEED, n_jobs=-1))
arms = {}
for fname in ['기준 예제 특징', '우리 특징']:
    for mname in ['로지스틱 회귀', '랜덤포레스트']:
        mk = MODELS[mname]
        if fname == '기준 예제 특징':
            oof = oof_predict(mk, Xb_tr, y, tr['SCENARIO_ID'].values)
            prob = mk().fit(Xb_tr, y).predict_proba(Xb_va)[:, 1]
        else:
            oof = np.zeros(len(tr))
            for tr_idx, te_idx, Ff in folds:
                X = Ff[cols].values.astype(float)
                oof[te_idx] = mk().fit(X[tr_idx], y[tr_idx]).predict_proba(X[te_idx])[:, 1]
            prob = mk().fit(tr[cols].values.astype(float), y).predict_proba(va[cols].values.astype(float))[:, 1]
        arms[(fname, mname)] = (oof, prob)
oof = oof_predict(base_rf, Xb_tr, y, tr['SCENARIO_ID'].values)
arms[('기준 예제 특징', '기준 예제 원래 설정 (RF 180그루)')] = (oof, base_rf().fit(Xb_tr, y).predict_proba(Xb_va)[:, 1])
# 학습 없는 점수: 표준화 평균(SCORE)을 학습 세트 분포의 백분위로 바꿔 0~1 확률처럼 쓴다 (순서만 쓰므로 AP는 SCORE와 같다)
ref = np.sort(np.nan_to_num(ftr['SCORE'].values, nan=-np.inf))
pct = lambda s: np.searchsorted(ref, np.nan_to_num(s, nan=-np.inf), side='right') / len(ref)
arms[('표준화 점수 하나', '학습 없음')] = (pct(tr['SCORE'].values), pct(va['SCORE'].values))

rows_cv, rows_va = {}, {}
for k, (oof, prob) in arms.items():
    th = choose_threshold(tr, oof, criterion='earliest', events=ev_train)
    rows_cv[k] = {'교차검증 AP': round(average_precision_score(y, oof), 3), '임계값': th}
    rows_va[k] = event_row(va, prob, ev_valid, th)[1]
out = pd.concat([pd.DataFrame(rows_cv).T, pd.DataFrame(rows_va).T.drop(columns='임계값')], axis=1)
out.index.names = ['특징', '모델']
print(out.to_string())
a = {k: v['AP'] for k, v in rows_va.items()}
print('\n검증 AP 차이')
print(f'  특징만 바꿈 (로지스틱 회귀): {a[("우리 특징", "로지스틱 회귀")] - a[("기준 예제 특징", "로지스틱 회귀")]:+.3f} · '
      f'(랜덤포레스트): {a[("우리 특징", "랜덤포레스트")] - a[("기준 예제 특징", "랜덤포레스트")]:+.3f}')
print(f'  모델만 바꿈 (기준 예제 특징): {a[("기준 예제 특징", "랜덤포레스트")] - a[("기준 예제 특징", "로지스틱 회귀")]:+.3f} · '
      f'(우리 특징): {a[("우리 특징", "랜덤포레스트")] - a[("우리 특징", "로지스틱 회귀")]:+.3f}')


# 4. 미확정 경보의 비용

section('4. DSS 비용: 미확정 경보를 뺀 값(보고서)과 조치 비용을 넣은 상한')


def with_labels(ids):
    pts = state_outcomes(ids, step_h=1)
    lab = make_targets_hourly(ids, step_h=1)[['SCENARIO_ID', 'OBSERVED_AT', 'risk24', 'RUL_EVENT', 'RUL_HOURS']]
    pts = pts.merge(lab, on=KEY, how='left')
    pts['T_H'] = pts['T_H'].astype(float)
    return pts


pts_tr = with_labels(train_ids)
oof = np.full(len(pts_tr), np.nan)
lab = pts_tr['risk24'].notna().values
for tr_idx, te_idx, Ff in fold_features(hr, pts_tr[TARGET_COLS].assign(risk24=pts_tr['risk24'].fillna(-1))):
    X = Ff[cols].values.astype(float)
    fit = tr_idx[lab[tr_idx]]
    oof[te_idx] = make().fit(X[fit], pts_tr['risk24'].values[fit].astype(int)).predict_proba(X[te_idx])[:, 1]
curve = failure_curve(episodes(pts_tr, oof, TH))
pts_va = with_labels(valid_ids)
prob_va = model.predict_proba(make_features(hr_fit, pts_va[TARGET_COLS])[cols].values.astype(float))[:, 1]

POL = {'regular': '① 정기 교체만', 'now': '② 경보 즉시 교체', 'dss': '③ DSS'}
rows = {}
for cname, costs in [('점수', Costs()), ('만 원', field_costs())]:
    for charge in [False, True]:
        cfg = DSSConfig(costs=costs, charge_unknown=charge)
        s = {p: simulate(pts_va, prob_va, TH, curve, cfg, policy=p)['summary'] for p in POL}
        for p, r in s.items():
            rows[(cname, '포함 (상한)' if charge else '제외 (보고서)', POL[p])] = {
                '비용': r['비용'], '① 대비': f'{(r["비용"] - s["regular"]["비용"]) / s["regular"]["비용"]:+.1%}',
                '고장': r['고장'], '미확정 경보': r['미확정 경보']}
print(pd.DataFrame(rows).T.rename_axis(['단위', '미확정 경보 비용', '정책']).to_string())

e = simulate(pts_va, prob_va, TH, curve, DSSConfig(costs=field_costs(), charge_unknown=True), policy='dss')['episodes']
print('\n미확정 경보 (③ DSS, 만 원, 조치 비용 포함)')
print(e[e['결과'].str.startswith('미확정')][['SCENARIO_ID', '경보 시각(h)', '판단', '비용']].to_string(index=False))

print('\n민감도: 정지 1시간 생산 차질 · 예비 재고 (만 원, ③ DSS의 ① 대비 절감률)')
sens = {}
for dt in [0, 100, 500, 1000]:
    for sname, parts in [('기본 재고', PartsPolicy()), ('재고 없음', PartsPolicy(0, 0))]:
        row = {}
        for charge in [False, True]:
            cfg = DSSConfig(costs=field_costs(downtime_h=dt), parts=parts, charge_unknown=charge)
            s = {p: simulate(pts_va, prob_va, TH, curve, cfg, policy=p)['summary'] for p in ['regular', 'dss']}
            row['포함 (상한)' if charge else '제외'] = f'{(s["regular"]["비용"] - s["dss"]["비용"]) / s["regular"]["비용"]:+.1%} ' \
                                                   f'({s["regular"]["비용"] - s["dss"]["비용"]:,.0f})'
        sens[(dt, sname)] = row
print(pd.DataFrame(sens).T.rename_axis(['생산 차질 (만 원/시간)', '재고']).to_string())
