"""조기경보 서비스용 함수: 저장한 모델로 예측하고, 경보 이유를 계산한다.

analysis/05_predict_test.py가 학습한 모델을 models/risk24.joblib에 저장하고,
대시보드(app/dashboard.py)는 저장된 모델을 불러와 예측만 한다(다시 학습하지 않는다).

    from src.serving import load_bundle, score_scenarios, explain

    bundle = load_bundle()
    pts, hr = score_scenarios(['S0289', 'S0290'], bundle)   # 예측 시점별 확률·경보·주의, 1시간 요약
    contrib = explain(bundle, pts)                           # 시점별·신호 묶음별 기여도

실시간 데이터가 생기면 score_scenarios의 원자료 읽기(build_hourly)만 새로 들어온 데이터로 바꾸면 된다.
특징은 예측 시점까지 끝난 정보만으로 계산하므로(03_check_features.py 누설 검사) 같은 결과가 나온다.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import make_features, make_prediction_points
from src.preprocess import apply_params, build_hourly

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'risk24.joblib'

# 특징 이름 → 화면에 보여 줄 묶음 이름
GROUPS = [
    ('MLD-AMP', '몰드 레벨 진폭'),
    ('CONTROL_DELAY', '제어 지연'),
    ('WDL', '인발 부하'),
    ('OSC-STK', '진동 스트로크'),
    ('SCORE', '네 신호 결합 점수'),
    ('PERSIST', '네 신호 결합 점수'),
    ('N_MNT', '정비 이력'),
    ('H_SINCE_MNT', '정비 이력'),
    ('LAST_', '정비 이력'),
    ('SEG_HOURS', '운전 상태'),
    ('GAP_H', '운전 상태'),
]


def save_bundle(bundle, path=MODEL_PATH):
    """모델과 함께 쓰는 값(임계값, 특징 목록, 보정값, 검증 결과)을 한 파일로 저장한다."""
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f'{path} 가 없다. 먼저 python analysis/05_predict_test.py 를 실행할 것')
    return joblib.load(path)


def score_scenarios(scenario_ids, bundle):
    """시나리오들의 예측 시점(6시간마다)별 24시간 고장 확률과 경보.

    반환
      pts  예측 시점 표: SCENARIO_ID, OBSERVED_AT, T_H, 특징들, PROB(24시간 안 고장 확률), ALARM(경보 여부)
           주의 모델이 있으면 DEGRADED_PROB(지금 열화 중일 확률)과 STAGE(경보 / 주의 / 정상)도 붙는다
      hr   1시간 요약 표: 신호별 표준화 점수(Z_), 평균 점수(SCORE) — 화면의 신호 흐름용
    """
    hr = apply_params(build_hourly(list(scenario_ids)), bundle['params'])   # 보정값은 학습 세트에서 정한 그대로
    pts = make_features(hr, make_prediction_points(list(scenario_ids)))
    pts['PROB'] = bundle['model'].predict_proba(pts[bundle['features']].values.astype(float))[:, 1]
    pts['ALARM'] = pts['PROB'] >= bundle['threshold']
    c = bundle.get('caution')
    if c:   # 2단계 경보: 경보가 아니어도 열화 확률이 주의 기준 이상이면 주의
        pts['DEGRADED_PROB'] = c['model'].predict_proba(pts[c['features']].values.astype(float))[:, 1]
        pts['STAGE'] = np.where(pts['ALARM'], '경보', np.where(pts['DEGRADED_PROB'] >= c['threshold'], '주의', '정상'))
    else:
        pts['STAGE'] = np.where(pts['ALARM'], '경보', '정상')
    return pts, hr


def _group(name):
    base = name.removeprefix('missingindicator_')
    group = next((g for key, g in GROUPS if base.startswith(key)), '기타')
    return '결측 표시 (정비 직후 등)' if name.startswith('missingindicator_') else group


def explain(bundle, pts):
    """선형 모델(로지스틱 회귀)의 특징별 기여도를 신호 묶음별로 더한다.

    기여도 = 계수 × 표준화한 특징값. 0은 학습 세트 평균 시점과 같다는 뜻이고,
    양수면 위험을 올린 쪽, 음수면 내린 쪽이다. 모두 더하고 절편을 더하면 로그 오즈(확률로 바꾸기 전 값)가 된다.
    계수가 없는 모델(랜덤포레스트 등)이면 None을 돌려준다.
    """
    model = bundle['model']
    *prep, (_, clf) = model.steps
    if not hasattr(clf, 'coef_'):
        return None
    X = pts[bundle['features']].values.astype(float)
    for _, step in prep:
        X = step.transform(X)
    names = model.steps[0][1].get_feature_names_out(bundle['features'])
    contrib = pd.DataFrame(X * clf.coef_[0], columns=names, index=pts.index)
    return contrib.T.groupby([_group(n) for n in names]).sum().T


def failure_hours(scenario_ids):
    """시나리오별 정비 기록(정비 시작·끝, 유형, 고장 여부)을 시작 후 경과 시간(h)으로 바꾼다."""
    from src.posko_data import load_scenarios, load_source
    from src.preprocess import EQP

    start = load_scenarios().set_index('SCENARIO_ID')['START_AT']
    m = load_source('maintenance').query('EQP_TAG == @EQP and SCENARIO_ID in @scenario_ids').copy()
    s0 = m['SCENARIO_ID'].map(start)
    m['START_H'] = (m['STRT_DT'] - s0).dt.total_seconds() / 3600
    m['END_H'] = ((m['END_DT'] - s0).dt.total_seconds() / 3600).fillna(np.inf)   # 관측 끝까지 진행 중인 정비
    m['FAIL_H'] = (m['FAILURE_DT'] - s0).dt.total_seconds() / 3600
    m['종류'] = m['MNT_TYPE'].map({'repair': '고장 후 복구', 'replace': '정기 교체', 'partial': '부분 정비'})
    return m[['SCENARIO_ID', 'PART_CD', '종류', 'START_H', 'END_H', 'FAIL_H']].reset_index(drop=True)


def cost_curve(n_det, n_miss, n_needed, n_false, p_grid=np.linspace(0.5, 1.0, 51)):
    """부품을 맞힐 확률 p에 따른 비용 (평가 함수 src/evaluate.py와 같은 비용 규칙).

    한 부품만 교체: 맞히면 교체 20, 틀리면 교체 20 + 불필요 정비 8 + 고장 66 = 94
    두 부품 모두 교체: 고장 전 40, 오경보 48 (p와 상관없음)
    """
    rows = []
    for p in p_grid:
        one = (n_det * (20 * p + 94 * (1 - p)) + n_miss * 66
               + n_needed * (20 * p + 28 * (1 - p)) + n_false * 28)
        rows.append({'p': p, '한 부품만 교체': one})
    out = pd.DataFrame(rows)
    out['두 부품 모두 교체'] = n_det * 40 + n_miss * 66 + n_needed * 40 + n_false * 48
    out['아무것도 안 함'] = (n_det + n_miss) * 66
    return out
