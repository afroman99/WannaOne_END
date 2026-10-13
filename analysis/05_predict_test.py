"""테스트 세트(정답 없음) 24시간 고장위험 예측.

04_model.py에서 검증 세트로 평가한 모델을 그대로 쓴다 (보수적 기준: 평가하지 않은 모델은 내보내지 않는다).
- 모델: src/model.py의 FINAL (로지스틱 회귀 · 센서 + 정비 이력), 학습 세트로 학습
- 보정값: 학습 세트 정상 시나리오에서 정한 값 (preprocess.fit_params)
- 임계값: 학습 세트 교차검증에서 점진 고장을 모두 잡고 진짜 오경보가 0인 임계값 중 가장 낮은 것

    python analysis/05_predict_test.py

결과 (둘 다 저장소에는 올리지 않음)
  predictions/test_risk24.csv    테스트 세트 예측
  models/risk24.joblib           학습한 모델·임계값·보정값·검증 세트 결과 (대시보드 app/dashboard.py가 불러온다)

predictions/test_risk24.csv 열
  SCENARIO_ID, OBSERVED_AT, T_H  예측 시점
  RISK24_PROB                    24시간 안 고장 확률
  ALARM                          경보 여부 (RISK24_PROB ≥ 임계값)
  SCORE, SCORE_F                 네 신호 평균 점수 (재가동 기준, 처음 수준 기준) — 경보 근거 확인용

참고: 데이터셋 앱용 예측 JSON(posko.export_predictions)은 '관측 교환본'의 해시 값이 필요한데,
받은 데이터 폴더에는 관측 교환본이 없어 CSV로 저장한다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import choose_threshold, evaluate, failure_events
from src.features import build_dataset, make_features, make_prediction_points
from src.model import FEATURES, FINAL, MODELS, oof_predict
from src.posko_data import load_answer, load_scenarios
from src.preprocess import EQP, apply_params, build_hourly
from src.serving import save_bundle

OUT = ROOT / 'predictions' / 'test_risk24.csv'


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


# 1. 예측 시점 규칙 확인: 학습·검증 세트에서 정답 파일의 AT_RISK와 같은가

section('1. 예측 시점 규칙 확인 (학습·검증 세트)')
scen = load_scenarios()
known = scen.loc[scen['SPLIT'] != 'test', 'SCENARIO_ID'].tolist()
test_ids = scen.loc[scen['SPLIT'] == 'test', 'SCENARIO_ID'].tolist()

mine = make_prediction_points(known).set_index(['SCENARIO_ID', 'OBSERVED_AT']).index
tg = load_answer('targets').query('EQP_CD == @EQP and SCENARIO_ID in @known')
tg = tg[(tg['OBSERVED_AT'] - tg['SCENARIO_ID'].map(scen.set_index('SCENARIO_ID')['START_AT'])) < pd.Timedelta(hours=72)]
at_risk = tg[tg['AT_RISK'] == 1].set_index(['SCENARIO_ID', 'OBSERVED_AT']).index
same = mine.sort_values().equals(at_risk.sort_values())
print(f'만든 시점 {len(mine)}개 vs AT_RISK=1 {len(at_risk)}개 → {"일치" if same else "불일치"}')
if not same:
    sys.exit('예측 시점 규칙이 정답 파일과 다르다. make_prediction_points를 확인할 것')


# 2. 학습 세트로 최종 모델 학습, 교차검증으로 임계값

section(f'2. 최종 모델: {FINAL["model"]} · {FINAL["features"]}')
X, params, _ = build_dataset()
train = X[X['SPLIT'] == 'train'].reset_index(drop=True)
cols = FEATURES[FINAL['features']]
make_model = MODELS[FINAL['model']]
Xtr, ytr = train[cols].values.astype(float), train['risk24'].values

oof = oof_predict(make_model, Xtr, ytr, train['SCENARIO_ID'].values)
th = choose_threshold(train, oof, criterion='earliest', events=failure_events(train['SCENARIO_ID'].unique().tolist()))
model = make_model().fit(Xtr, ytr)
print(f'학습 시점 {len(train)}개로 학습 · 임계값 {th:.2f} (학습 세트 교차검증)')


# 3. 테스트 세트 특징과 예측

section('3. 테스트 세트 예측')
hr_test = apply_params(build_hourly(test_ids), params)          # 보정값은 학습 세트에서 정한 그대로
pts = make_prediction_points(test_ids)
Xte = make_features(hr_test, pts)
prob = model.predict_proba(Xte[cols].values.astype(float))[:, 1]

pred = Xte[['SCENARIO_ID', 'OBSERVED_AT', 'T_H']].copy()
pred['RISK24_PROB'] = prob.round(4)
pred['ALARM'] = (prob >= th).astype(int)
pred['SCORE'] = Xte['SCORE'].round(3)
pred['SCORE_F'] = Xte['SCORE_F'].round(3)
OUT.parent.mkdir(exist_ok=True)
pred.to_csv(OUT, index=False, encoding='utf-8-sig')

print(f'시나리오 {pred["SCENARIO_ID"].nunique()}개, 예측 시점 {len(pred)}개 → {OUT.relative_to(ROOT)}')
print(f'경보 {int(pred["ALARM"].sum())}개 시점 ({pred["ALARM"].mean():.1%}), '
      f'경보가 한 번이라도 난 시나리오 {pred.loc[pred["ALARM"] == 1, "SCENARIO_ID"].nunique()}개')
print('센서 특징이 비어 있는 시점(시작 시점·정비 직후):', int(Xte['SCORE'].isna().sum()))

# 검증 세트와 비교: 경보 비율이 크게 다르면 데이터가 달라졌을 가능성을 확인한다
valid = X[X['SPLIT'] == 'validation']
v_alarm = (model.predict_proba(valid[cols].values.astype(float))[:, 1] >= th).mean()
print(f'참고: 같은 모델의 검증 세트 경보 비율 {v_alarm:.1%} (정답 시점 기준이라 정확히 같은 조건은 아님)')
print('\n확률 분포 (테스트 세트)')
print(pd.Series(prob).describe(percentiles=[0.5, 0.9, 0.95]).round(3).to_string())


# 4. 모델 저장 (대시보드용)

section('4. 모델 저장')
res = evaluate(valid.reset_index(drop=True), model.predict_proba(valid[cols].values.astype(float))[:, 1], th)
ev = res['events'][['SCENARIO_ID', 'WO_NO', 'FAILURE_DT', 'TYPE', '감지', '리드타임(h)']]
bundle = {
    'model': model, 'threshold': th, 'features': cols, 'params': params, 'final': FINAL,
    'created_at': pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M'),
    'n_train_points': len(train),
    'validation': {'point': res['point'], 'event': res['event'], 'cost': res['cost'],
                   'cost_reference': res['cost_reference'], 'events': ev},
}
path = save_bundle(bundle)
print(f'{path.relative_to(ROOT)} 저장 · 검증 세트 AP {res["point"]["AP"]:.3f}, '
      f'고장 감지 {res["event"]["감지"]}/{res["event"]["고장"]}, 비용(두 부품) {res["cost"]["두 부품"]:.0f}')
