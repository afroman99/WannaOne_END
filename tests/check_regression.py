"""회귀 점검: 코드를 고친 뒤에도 보고서·발표의 숫자와 서비스 동작이 그대로인지 확인한다.

    python analysis/05_predict_test.py      # 모델을 다시 만든 경우
    python tests/check_regression.py        # 약 1분

점검 항목 (하나라도 실패하면 종료 코드 1)
  1. 모델 파일: 해시·필수 값·만든 환경(meta)이 맞는가
  2. 대표 숫자: 예측 간격 1시간, 경보 기준 0.56, 주의 기준 0.57, 검증 세트 AP 0.866, 고장 감지 14/16, 비용(두 부품) 1020
  3. 서비스 예측: 대시보드 경로(serving.score_scenarios)의 테스트 세트 예측이 predictions/test_risk24.csv와 같은가
     (테스트 5,028개 시점 중 경보 554개, 경보 시나리오 18개)
  4. 경보 이유: 랜덤포레스트 기여도의 기준값 + 합이 위험 확률과 같은가
  5. 비용 곡선: 대시보드 비용 계산기가 평가 함수(src/evaluate.py)와 같은 비용을 내는가 (손익분기 0.60)
  6. 안전장치: 모르는 정비 유형, 빠진 특징, 바뀐 모델 파일을 조용히 넘기지 않고 멈추는가
  7. 1시간 정답: make_targets_hourly가 6시간 시점에서 정답 파일 기준(make_targets)과 같은가
  8. 의사결정 지원(DSS): 모델 파일의 정책 비용(검증 세트 ① 1056 ② 980 ③ 900)과, 정기 교체 직전이면 기다리고 준비가 안 되면 지금 교체하는가
  9. 보안 (reports/보안_점검.md): 클라우드용 데이터(app_data)가 서비스 예측과 같고 원자료·모델 객체가 없는가, 바뀐 파일을 막는가,
     비밀번호 확인이 맞는가, 입력 감시 범위가 모델 파일에 있는가

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
from src.features import _maintenance_features, make_targets, make_targets_hourly
from src.posko_data import load_scenarios
from src.serving import MODEL_PATH, _forest_contributions, _matrix, cost_curve, explain, load_bundle, score_scenarios

# 대표 숫자. 중간발표(git 태그 midterm, 6시간 예측): 경보 0.67, 주의 0.38, AP 0.860, 비용 860, 손익분기 0.67, 테스트 833시점·경보 83
# 개선 1(2026-10-15, 1시간 예측, analysis/09_hourly.py)부터 아래 값
EXPECTED = {
    'threshold': 0.56, 'caution_threshold': 0.57, 'AP': 0.866, '감지': 14, '고장': 16, '비용 두 부품': 1020,
    'test_points': 5028, 'test_alarms': 554, 'test_alarm_scenarios': 18, 'breakeven_p': 0.60, 'step_h': 1,
    'dss_cost': {'regular': 1056, 'now': 980, 'dss': 900},
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
check('예측 간격', bundle.get('step_h', 6) == EXPECTED['step_h'], f'{bundle.get("step_h", 6)}시간')
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

print('\n7. 1시간 정답')
known = load_scenarios().query('SPLIT != "test"')['SCENARIO_ID'].tolist()
ref, hourly = make_targets(known), make_targets_hourly(known, step_h=1)
m = ref.merge(hourly, on=['SCENARIO_ID', 'OBSERVED_AT'], how='left', suffixes=('', '_h'))
check('6시간 시점이 모두 1시간 정답에 있다', m['risk24_h'].notna().all(), f'{len(ref)}개')
check('정답·잔여수명이 같다', (m['risk24'] == m['risk24_h']).all() and (m['RUL_EVENT'] == m['RUL_EVENT_h']).all()
      and np.allclose(m['RUL_HOURS'].fillna(-1), m['RUL_HOURS_h'].fillna(-1)), f'1시간 정답 {len(hourly)}개')
h6 = make_targets_hourly(known, step_h=6)
check('6시간 간격으로 만들면 정답 파일 기준과 똑같다 (잔여수명은 부동소수 오차 안)',
      h6.drop(columns='RUL_HOURS').equals(ref.drop(columns='RUL_HOURS'))
      and np.allclose(h6['RUL_HOURS'].fillna(-1), ref['RUL_HOURS'].fillna(-1)))

print('\n8. 의사결정 지원 (DSS)')
from src.dss import DSSConfig, decide
D = bundle.get('dss')
check('모델 파일에 DSS 정보가 있다', D is not None)
if D:
    got = {p: round(r['비용']) for p, r in D['validation'].items()}
    check('검증 세트 정책 비용', got == EXPECTED['dss_cost'], str(got))
    check('정기 교체 3시간 전 경보면 기다린다', decide({'WAIT_H': 3}, D['curve'], DSSConfig(), 'dss')[0] == 'wait')
    check('계획이 없으면 지금 교체', decide({'WAIT_H': np.nan}, D['curve'], DSSConfig(), 'dss')[0] == 'now')
    check('준비 시간 24시간이면 3시간 뒤 정기 교체를 기다리지 않는다',
          decide({'WAIT_H': 3}, D['curve'], DSSConfig(prep_h=24), 'dss')[0] == 'now')
    check('안전 기준을 넘으면 지금 교체 (30시간 기다림)',
          decide({'WAIT_H': 30}, D['curve'], DSSConfig(), 'dss')[0] == 'now')

print('\n9. 보안')
import json
import tempfile as _tf
from src import app_data
from src.auth import hash_password, verify
check('입력 감시 범위가 모델 파일에 있다', 'feature_range' in bundle and len(bundle['feature_range']['lo']) == len(bundle['features']))
check('서비스 예측에 입력 감시 열이 있다', {'OOR_N', 'OOR_FEATS'} <= set(pts.columns))
salt, dig, it = hash_password('test-only-password', iterations=1000)
check('비밀번호 확인: 맞으면 통과, 틀리면 거부', verify('test-only-password', salt, dig, it) and not verify('wrong', salt, dig, it))
with _tf.TemporaryDirectory() as d:
    app_data.export(d)
    meta, data = app_data.load(d)
    p_app = data['test'][0]
    check('클라우드용 데이터의 테스트 예측이 서비스 예측과 같다',
          np.allclose(p_app['PROB'].values, pts['PROB'].values) and (p_app['STAGE'].values == pts['STAGE'].values).all())
    text = (Path(d) / 'meta.json').read_text(encoding='utf-8')
    names = sorted(f.name for f in Path(d).iterdir())
    check('클라우드용 데이터에 모델 객체·원자료 특징이 없다 (Parquet·JSON만, 1시간 점수까지만)',
          all(n.endswith(('.parquet', '.json')) for n in names) and 'RandomForest' not in text
          and not any(c in data['test'][1].columns for c in ['LOAD', 'WD-LOD', 'MLD-LVL']))
    check('클라우드용 정책 비용이 모델 파일과 같다', {k: round(v['비용']) for k, v in meta['dss']['validation'].items()} == EXPECTED['dss_cost'])
    with open(Path(d) / 'test_pts.parquet', 'ab') as f:
        f.write(b'0')
    check('바뀐 클라우드용 파일은 불러오지 않는다', raises(lambda: app_data.load(d), ValueError))
from src.posko_data import _check_manifest
check('manifest의 경로 조작 값(../)을 막는다',
      raises(lambda: _check_manifest({'files': [{'id': '../x', 'scenarioId': 'S0001', 'chunks': []}], 'scenarios': []}), ValueError)
      and raises(lambda: _check_manifest({'files': [{'id': 'casting', 'scenarioId': 'S0001',
                                                     'chunks': [{'sha256': '../../etc/passwd'}]}], 'scenarios': []}), ValueError))

n_fail = sum(not ok for _, ok in results)
print(f'\n결과: {len(results) - n_fail}/{len(results)}개 통과')
sys.exit(1 if n_fail else 0)
