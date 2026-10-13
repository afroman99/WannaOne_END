"""연주 24시간 조기경보 모델 정의와 교차검증 도우미.

analysis/04_model.py(모델 선택·검증 평가)와 analysis/05_predict_test.py(테스트 세트 예측)가 같은 정의를 쓴다.

    from src.model import MODELS, FEATURES, FINAL, oof_predict

    make = MODELS[FINAL['model']]
    cols = FEATURES[FINAL['features']]
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS

N_FOLDS = 5
SEED = 31

# 비교하는 특징 묶음. 경과 시간(FEATURE_SETS['elapsed'])은 쓰지 않는다(잠정 결정 E)
FEATURES = {
    '센서': FEATURE_SETS['sensor'],
    '센서 + 정비 이력': FEATURE_SETS['sensor'] + FEATURE_SETS['maintenance'],
}

# 비교하는 모델. 불균형(양성 약 11%)이라 클래스 가중치를 맞춘다
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

# 04_model.py에서 학습 세트 교차검증 AP로 고르고 검증 세트에서 평가한 최종 조합 (2026-10-13)
FINAL = {'model': '로지스틱 회귀', 'features': '센서 + 정비 이력'}


def oof_predict(make_model, X, y, groups, n_folds=N_FOLDS):
    """시나리오 단위 n겹 교차검증의 겹 밖 예측 확률 (같은 시나리오는 한 겹에만 들어간다)"""
    oof = np.zeros(len(y))
    for tr_idx, te_idx in GroupKFold(n_splits=n_folds).split(X, y, groups):
        m = make_model().fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict_proba(X[te_idx])[:, 1]
    return oof
