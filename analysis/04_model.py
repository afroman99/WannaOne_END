"""연주 24시간 조기경보 모델: 학습 세트 교차검증으로 고르고, 검증 세트에서 한 번 평가한다.

    python analysis/04_model.py

잠정 결정(보수적 기준, reports/연주_조기경보_EDA_정리.md 6-1)
- 모델·특징 묶음·임계값은 모두 학습 세트의 시나리오 단위 5겹 교차검증으로 정한다. 검증 세트는 마지막에 한 번만 본다.
- 임계값: 교차검증 예측에서 점진 고장을 모두 잡고 진짜 오경보가 0인 임계값 중 가장 낮은 것 (오경보 없이 가장 일찍 경보)
  처음에는 "두 부품 교체" 비용 최소로 골랐으나, 비용 규칙이 리드타임을 반영하지 못해 늦은 경보를 골라서 바꿨다
- 모델 선택: 교차검증 AP가 가장 높은 조합 (임계값과 상관없는 순위 능력). 비용은 함께 보고한다
- 특징: 센서 특징만 / 센서 + 정비 이력을 비교한다. 경과 시간은 쓰지 않는다.
- 기준 예제(02_baseline.py)도 같은 규칙으로 다시 평가해 공정하게 비교한다.
- 대표 숫자는 이른 경보도 오경보로, 돌발 고장도 놓친 것으로 센다. 유리한 해석은 참고로 함께 출력한다.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import choose_threshold, evaluate, failure_events, print_report
from src.features import FEATURE_SETS, build_dataset
from src.posko_data import load_scenarios

N_FOLDS = 5
SEED = 31
PART_RULE = '두 부품'


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


# 1. 데이터

section('1. 데이터')
X, params, hr = build_dataset()
train = X[X['SPLIT'] == 'train'].reset_index(drop=True)
valid = X[X['SPLIT'] == 'validation'].reset_index(drop=True)
ev_train = failure_events(train['SCENARIO_ID'].unique().tolist())
ev_valid = failure_events(valid['SCENARIO_ID'].unique().tolist())
print(f'학습 {len(train)}개 시점 (양성 {int(train["risk24"].sum())}, 고장 {len(ev_train)}건) · '
      f'검증 {len(valid)}개 시점 (양성 {int(valid["risk24"].sum())}, 고장 {len(ev_valid)}건)')

FEATURES = {
    '센서': FEATURE_SETS['sensor'],
    '센서 + 정비 이력': FEATURE_SETS['sensor'] + FEATURE_SETS['maintenance'],
}
MODELS = {
    '로지스틱 회귀': lambda: make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
                                       LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)),
    '랜덤포레스트': lambda: make_pipeline(SimpleImputer(add_indicator=True),
                                     RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=5,
                                                            class_weight='balanced', random_state=SEED, n_jobs=-1)),
    '그래디언트 부스팅': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                          l2_regularization=1.0, class_weight='balanced',
                                                          random_state=SEED),
}


def oof_predict(make_model, Xtr, y, groups):
    """시나리오 단위 5겹 교차검증의 겹 밖 예측 확률"""
    oof = np.zeros(len(y))
    for tr_idx, te_idx in GroupKFold(n_splits=N_FOLDS).split(Xtr, y, groups):
        m = make_model().fit(Xtr[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict_proba(Xtr[te_idx])[:, 1]
    return oof


def cv_summary(df, oof, events):
    """교차검증 예측으로 임계값을 고르고, 그 임계값에서의 교차검증 성능"""
    th = choose_threshold(df, oof, criterion='earliest', events=events)
    res = evaluate(df, oof, th, events)
    g = res['event_by_type']
    return th, {
        'CV AP': average_precision_score(df['risk24'], oof),
        '임계값': th,
        f'CV 비용 ({PART_RULE})': res['cost'][PART_RULE],
        'CV 정밀도': res['point']['정밀도'], 'CV 재현율': res['point']['재현율'],
        'CV 점진 고장 감지': f'{int(g.loc["점진 고장", "감지"])}/{int(g.loc["점진 고장", "고장"])}',
        'CV 리드타임 중앙값(h)': res['event']['리드타임 중앙값(h)'],
    }


# 2. 학습 세트 교차검증으로 모델·특징 묶음 고르기

section('2. 학습 세트 교차검증 (시나리오 단위 5겹)')
y_tr, groups = train['risk24'].values, train['SCENARIO_ID'].values
rows, oofs = [], {}
for fname, cols in FEATURES.items():
    Xtr = train[cols].values.astype(float)
    for mname, make_model in MODELS.items():
        oof = oof_predict(make_model, Xtr, y_tr, groups)
        th, summ = cv_summary(train, oof, ev_train)
        oofs[(mname, fname)] = (oof, th)
        rows.append({'모델': mname, '특징': fname, **summ})
cv = pd.DataFrame(rows).set_index(['모델', '특징'])
print(cv.round(3).to_string())

best = cv['CV AP'].idxmax()
best_model, best_feat = best
print(f'\n→ 선택: {best_model} · {best_feat} (교차검증 AP 최대)')


# 3. 기준 예제를 같은 규칙으로 다시 평가

section('3. 기준 예제 (같은 규칙: 학습 세트 교차검증으로 임계값)')
spec = importlib.util.spec_from_file_location('baseline', ROOT / 'analysis' / '02_baseline.py')
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)
scen = load_scenarios().query('SPLIT != "test"')
rec = bl.make_records(bl.load_observations(scen), bl.load_targets(scen))
rec_tr = rec[rec['SPLIT'] == 'train'].reset_index(drop=True)
rec_va = rec[rec['SPLIT'] == 'validation'].reset_index(drop=True)


def make_baseline():
    return make_pipeline(SimpleImputer(add_indicator=True),
                         RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=5,
                                                class_weight='balanced', random_state=31, n_jobs=2))


Btr = np.stack(rec_tr['temporal'])
b_oof = oof_predict(make_baseline, Btr, rec_tr['risk24'].values, rec_tr['SCENARIO_ID'].values)
b_th, b_summ = cv_summary(rec_tr, b_oof, ev_train)
print(pd.Series(b_summ).to_string())


# 4. 검증 세트 평가 (한 번)

section('4. 검증 세트 평가')
cols = FEATURES[best_feat]
final = MODELS[best_model]().fit(train[cols].values.astype(float), y_tr)
p_val = final.predict_proba(valid[cols].values.astype(float))[:, 1]
th = oofs[best][1]
res = evaluate(valid, p_val, th, ev_valid)
print_report(res, f'우리 모델: {best_model} · {best_feat} (임계값 {th:.2f}, 학습 세트 교차검증으로 결정)')

b_final = make_baseline().fit(Btr, rec_tr['risk24'].values)
b_val = b_final.predict_proba(np.stack(rec_va['temporal']))[:, 1]
b_res = evaluate(rec_va, b_val, b_th, ev_valid)
print_report(b_res, f'기준 예제 (임계값 {b_th:.2f}, 같은 규칙)')

# 참고: 다른 임계값 기준 (모두 학습 세트 교차검증에서 고름)
for crit, label in [('cost', f'{PART_RULE} 비용 최소'), ('f1', 'F1 최대')]:
    th_ref = choose_threshold(train, oofs[best][0], criterion=crit, events=ev_train)
    print_report(evaluate(valid, p_val, th_ref, ev_valid), f'참고: 우리 모델, {label} 임계값 {th_ref:.2f}')

section('5. 요약 (검증 세트, 대표 숫자)')
summary = pd.DataFrame({
    name: {'AP': r['point']['AP'], '정밀도': r['point']['정밀도'], '재현율': r['point']['재현율'],
           'TP': r['point']['TP'], 'FP': r['point']['FP'], 'FN': r['point']['FN'],
           '고장 감지': f'{r["event"]["감지"]}/{r["event"]["고장"]}',
           '리드타임 중앙값(h)': r['event']['리드타임 중앙값(h)'],
           f'비용 ({PART_RULE})': r['cost'][PART_RULE], '비용 (반반, 최악)': r['cost']['반반'],
           '비용 (맞힘, 상한)': r['cost']['맞힘']}
    for name, r in {'우리 모델': res, '기준 예제 (같은 규칙)': b_res}.items()
})
summary['아무것도 안 함'] = pd.Series({f'비용 ({PART_RULE})': res['cost_reference']['아무것도 안 함'],
                                  '비용 (반반, 최악)': res['cost_reference']['아무것도 안 함'],
                                  '비용 (맞힘, 상한)': res['cost_reference']['아무것도 안 함']})
print(summary.to_string())
