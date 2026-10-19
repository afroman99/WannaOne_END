"""검증 세트 예측 확률과 정답을 한 파일로 저장한다: 원자료 없이 AP·혼동행렬·고장 단위 결과를 다시 계산할 수 있게.

    python analysis/13_export_validation.py      # 약 1분 (원자료 data/posko/ 필요)

저장: predictions/validation_risk24.csv
  SCENARIO_ID, T_H, OBSERVED_AT  예측 시점 (1시간마다)
  PROB, ALARM                    최종 모델의 24시간 고장 확률과 경보(≥ 0.56). app_data/validation_pts.parquet과 같은 값
  risk24                         정답 (24시간 안 고장 1 / 0 / 관측 종료로 알 수 없음은 빈칸)
  RUL_HOURS                      고장까지 남은 시간 (고장 예정이 있을 때만)

원자료 없이 확인하는 방법
  import pandas as pd; from sklearn.metrics import average_precision_score
  d = pd.read_csv('predictions/validation_risk24.csv').dropna(subset=['risk24'])
  average_precision_score(d['risk24'], d['PROB'])      # 0.866
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import make_targets_hourly
from src.posko_data import load_scenarios

scen = load_scenarios()
valid_ids = scen.loc[scen['SPLIT'] == 'validation', 'SCENARIO_ID'].tolist()
pts = pd.read_parquet(ROOT / 'app_data' / 'validation_pts.parquet', columns=['SCENARIO_ID', 'T_H', 'PROB', 'ALARM'])
lab = make_targets_hourly(valid_ids, step_h=1)[['SCENARIO_ID', 'T_H', 'OBSERVED_AT', 'risk24', 'RUL_HOURS']]
out = pts.merge(lab, on=['SCENARIO_ID', 'T_H'], how='left')
start = out['SCENARIO_ID'].map(scen.set_index('SCENARIO_ID')['START_AT'])
out['OBSERVED_AT'] = out['OBSERVED_AT'].fillna(start + pd.to_timedelta(out['T_H'], unit='h'))
out = out[['SCENARIO_ID', 'T_H', 'OBSERVED_AT', 'PROB', 'ALARM', 'risk24', 'RUL_HOURS']]
path = ROOT / 'predictions' / 'validation_risk24.csv'
out.to_csv(path, index=False)

d = out.dropna(subset=['risk24'])
y, a = d['risk24'].astype(int).values, d['ALARM'].astype(bool).values
print(f'저장: {path.relative_to(ROOT)} · 시점 {len(out)}개 중 정답이 있는 시점 {len(d)}개 (양성 {int(y.sum())})')
print(f'AP {average_precision_score(y, d["PROB"]):.3f} · TP {int((a & (y == 1)).sum())} · FP {int((a & (y == 0)).sum())} · '
      f'FN {int((~a & (y == 1)).sum())} · TN {int((~a & (y == 0)).sum())}')
