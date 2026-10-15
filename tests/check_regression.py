"""회귀 점검: 코드를 고친 뒤에도 보고서·발표의 숫자와 서비스 동작이 그대로인지 확인한다.

    python analysis/05_predict_test.py      # 모델을 다시 만든 경우
    python tests/check_regression.py        # 약 30초

점검 항목 (하나라도 실패하면 종료 코드 1)
  1. 모델 파일: 해시·필수 값·만든 환경(meta)이 맞는가
  2. 대표 숫자: 경보 기준 0.67, 주의 기준 0.38, 검증 세트 AP 0.860, 고장 감지 14/16, 비용(두 부품) 860
  3. 서비스 예측: 대시보드 경로(serving.score_scenarios)의 테스트 세트 예측이 predictions/test_risk24.csv와 같은가
     (테스트 경보 83개 시점, 경보 시나리오 18개)
  4. 경보 이유: 랜덤포레스트 기여도의 기준값 + 합이 위험 확률과 같은가
  5. 비용 곡선: 대시보드 비용 계산기가 평가 함수(src/evaluate.py)와 같은 비용을 내는가 (손익분기 0.67)
  6. 안전장치: 모르는 정비 유형, 빠진 특징, 바뀐 모델 파일을 조용히 넘기지 않고 멈추는가

숫자를 일부러 바꾼 경우(모델 개선 등)에는 보고서를 고친 뒤 EXPECTED도 함께 고친다.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import _maintenance_features
from src.posko_data import load_scenarios
from src.serving import MODEL_PATH, _forest_contributions, _matrix, cost_curve, explain, load_bundle, score_scenarios

# 결과보고서·발표자료에 나오는 숫자 (2026-10-13 확정)
EXPECTED = {
    'threshold': 0.67, 'caution_threshold': 0.38, 'AP': 0.860, '감지': 14, '고장': 16, '비용 두 부품': 860,
    'test_points': 833, 'test_alarms': 83, 'test_alarm_scenarios': 18, 'breakeven_p': 0.67,
}

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print(f'  [{"통과" if ok else "실패"}] {name}' + (f' — {detail}' if detail else ''))


def raises(fn, error):
    try:
        fn()
    except error:
        return True
    return False


print('1. 모델 파일')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('ignore')                 # 라이브러리 내부 경고(joblib 불러오기 중 NumPy 사용 중단 안내)는 빼고
    warnings.simplefilter('always', UserWarning)    # load_bundle이 내는 환경 경고만 본다
    bundle = load_bundle()
    w = [x for x in w if issubclass(x.category, UserWarning) and not issubclass(x.category, DeprecationWarning)]
check('해시·필수 값 확인을 통과해 불러온다', True)
check('만든 환경(meta)이 있고 지금 환경과 같다 (경고 없음)', 'meta' in bundle and not w,
      '; '.join(str(x.message) for x in w) or str(bundle.get('meta')))

print('\n2. 대표 숫자')
v = bundle['validation']
check('경보 기준', round(bundle['threshold'], 2) == EXPECTED['threshold'], f'{bundle["threshold"]:.2f}')
check('주의 기준', round(bundle['caution']['threshold'], 2) == EXPECTED['caution_threshold'],
      f'{bundle["caution"]["threshold"]:.2f}')
check('검증 세트 AP', round(v['point']['AP'], 3) == EXPECTED['AP'], f'{v["point"]["AP"]:.3f}')
check('고장 감지', (v['event']['감지'], v['event']['고장']) == (EXPECTED['감지'], EXPECTED['고장']),
      f'{v["event"]["감지"]}/{v["event"]["고장"]}')
check('비용 (두 부품)', round(v['cost']['두 부품']) == EXPECTED['비용 두 부품'], f'{v["cost"]["두 부품"]:.0f}')

print('\n3. 서비스 예측 (테스트 세트)')
test_ids = load_scenarios().query('SPLIT == "test"')['SCENARIO_ID'].tolist()
pts, _ = score_scenarios(test_ids, bundle)
saved = pd.read_csv(ROOT / 'predictions' / 'test_risk24.csv')
same_keys = (pts['SCENARIO_ID'].tolist() == saved['SCENARIO_ID'].tolist()) and np.array_equal(pts['T_H'], saved['T_H'])
check('예측 시점이 저장된 예측 CSV와 같다', same_keys, f'{len(pts)}개')
if same_keys:
    check('위험 확률이 같다 (소수 4자리)', np.array_equal(pts['PROB'].round(4).values, saved['RISK24_PROB'].values))
    check('열화 확률이 같다 (소수 4자리)', np.array_equal(pts['DEGRADED_PROB'].round(4).values, saved['DEGRADED_PROB'].values))
    check('단계(경보·주의·정상)가 같다', (pts['STAGE'].values == saved['STAGE'].values).all())
n_alarm, n_scen = int(pts['ALARM'].sum()), pts.loc[pts['ALARM'], 'SCENARIO_ID'].nunique()
check('테스트 예측 시점·경보 수', (len(pts), n_alarm, n_scen) == (EXPECTED['test_points'], EXPECTED['test_alarms'],
                                                        EXPECTED['test_alarm_scenarios']),
      f'시점 {len(pts)}, 경보 {n_alarm}, 경보 시나리오 {n_scen}')

print('\n4. 경보 이유 (기여도)')
contrib, unit = explain(bundle, pts)
check('최종 모델의 기여도를 계산한다', contrib is not None, unit or '')
if contrib is not None and unit.startswith('확률'):
    *prep, (_, forest) = bundle['model'].steps
    X = _matrix(pts, bundle['features'])
    for _, step in prep:
        X = step.transform(X)
    raw, base = _forest_contributions(forest, X)
    gap = np.abs(raw.sum(axis=1) + base - pts['PROB'].values).max()
    check('기준값 + 기여도 합 = 위험 확률', gap < 1e-9, f'최대 차이 {gap:.1e}, 기준값 {base:.3f}')
    check('신호 묶음으로 더해도 합이 같다', np.allclose(contrib.sum(axis=1).values, raw.sum(axis=1)))

print('\n5. 비용 곡선')
e = v['event']
n = dict(n_det=e['감지'], n_miss=e['놓침'], n_needed=e['고장과 무관한 경보 묶음: 필요한 교체'],
         n_false=e['고장과 무관한 경보 묶음: 오경보'])
cc = cost_curve(**n)
check('두 부품 교체 비용 = 평가 함수 비용', cc['두 부품 모두 교체'].iloc[0] == v['cost']['두 부품'],
      f'{cc["두 부품 모두 교체"].iloc[0]:.0f} vs {v["cost"]["두 부품"]:.0f}')
check('p=1 한 부품 비용 = 평가 함수 "맞힘"', np.isclose(cc['한 부품만 교체'].iloc[-1], v['cost']['맞힘']))
check('p=0.5 한 부품 비용 = 평가 함수 "반반"', np.isclose(cc['한 부품만 교체'].iloc[0], v['cost']['반반']))
even = cc.loc[cc['한 부품만 교체'] <= cc['두 부품 모두 교체'], 'p']
check('손익분기', len(even) and round(even.iloc[0], 2) == EXPECTED['breakeven_p'],
      f'{even.iloc[0]:.2f}' if len(even) else '없음')

print('\n6. 안전장치')
odd = pd.DataFrame({'END_DT': [pd.Timestamp('2025-01-01', tz='UTC')], 'MNT_TYPE': ['inspect']})
check('모르는 정비 유형이면 멈춘다', raises(lambda: _maintenance_features(odd, pd.Timestamp('2025-01-02', tz='UTC')),
                                   ValueError))
check('모델이 쓰는 특징이 빠지면 멈춘다', raises(lambda: _matrix(pts.drop(columns=bundle['features'][0]),
                                                     bundle['features']), KeyError))
bad = pts.copy()
bad.loc[bad.index[0], bundle['features'][0]] = np.inf
check('특징에 무한대 값이 있으면 멈춘다', raises(lambda: _matrix(bad, bundle['features']), ValueError))
with tempfile.TemporaryDirectory() as d:
    copy = Path(d) / MODEL_PATH.name
    shutil.copy(MODEL_PATH, copy)
    shutil.copy(MODEL_PATH.with_name(MODEL_PATH.name + '.sha256'), Path(d))
    with open(copy, 'ab') as f:
        f.write(b'0')
    check('저장 뒤 바뀐 모델 파일은 불러오지 않는다', raises(lambda: load_bundle(copy), ValueError))

n_fail = sum(not ok for _, ok in results)
print(f'\n결과: {len(results) - n_fail}/{len(results)}개 통과')
sys.exit(1 if n_fail else 0)
