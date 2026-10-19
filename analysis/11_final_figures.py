"""최종 모델(1시간 예측) 기준 보고서·발표용 그림. 원자료 없이 app_data/(저장된 결과)만으로 그린다.

    python analysis/11_final_figures.py

그림 (figures/)
  22_final_검증_고장별_경보타임라인.png   검증 세트 고장 16건: 1시간마다 예측한 정상·주의·경보와 실제 고장 시각
  23_final_검증_혼동행렬.png             검증 세트 시점 단위 혼동행렬 (FP·FN 안쪽 구성까지)
  24_dss_정책비교.png                   정비 정책 세 가지의 기대 총비용·고장·계획 밖 정지 (현장 가정, 만 원)

숫자는 app_data/meta.json의 검증 결과와 결과보고서 6·7장을 그대로 쓴다(04_model.py의 그림 12~15는 모델 선택 단계인 6시간 시점 기준).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.plotting import save_figure, setup_korean_font

setup_korean_font()
APP = ROOT / 'app_data'
INK, MUTED, GRID = '#1f2937', '#6b7280', '#e5e7eb'
RED, ORANGE, BLUE, GRAY = '#e5484d', '#f5a524', '#2563eb', '#cbd5e1'
plt.rcParams.update({'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
meta = json.loads((APP / 'meta.json').read_text(encoding='utf-8'))


def timeline():
    """고장마다 같은 운전 구간(직전 정비 뒤 ~ 고장)의 1시간 예측을 점으로"""
    pts = pd.read_parquet(APP / 'validation_pts.parquet')
    mnt = pd.read_parquet(APP / 'validation_mnt.parquet')
    ev = pd.read_parquet(APP / 'validation_events.parquet').set_index('SCENARIO_ID')
    rows = []
    for _, f in mnt[mnt['종류'] == '고장 후 복구'].iterrows():
        sid, fh = f['SCENARIO_ID'], f['FAIL_H']
        prev = mnt[(mnt.SCENARIO_ID == sid) & (mnt.END_H <= fh) & (mnt['종류'] != '고장 후 복구')]['END_H']
        p = pts[(pts.SCENARIO_ID == sid) & (pts.T_H >= (prev.max() if len(prev) else 0)) & (pts.T_H < fh)]
        lead = fh - p[p.ALARM].T_H.min() if p.ALARM.any() else None
        rows.append((sid, ev.loc[sid, 'TYPE'], fh, p, lead))
    rows.sort(key=lambda r: (r[4] is None, -(r[4] or 0)))
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    ax.axvspan(-24, 0, color='#fde8e8', zorder=0)
    for i, (sid, typ, fh, p, lead) in enumerate(rows):
        y = len(rows) - 1 - i
        c = np.where(p.ALARM, RED, np.where(p.STAGE == '주의', ORANGE, GRAY))
        ax.scatter(p.T_H - fh, [y] * len(p), c=c, s=13, zorder=2, linewidths=0)
        ax.scatter([0], [y], marker='x', c=INK, s=22, zorder=3)
        ax.text(1.5, y, f'{lead:.0f}h 전' if lead is not None else '감지 못함', va='center', fontsize=8,
                color=RED if lead is not None else MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{r[0]}{" (돌발)" if r[1] != "점진 고장" else ""}' for r in rows][::-1], fontsize=8)
    ax.set_xlim(-62, 9)
    ax.set_xlabel('고장까지 남은 시간 (시간)')
    for lab, col in [('정상', GRAY), ('주의 (열화 감지)', ORANGE), ('경보 (24시간 안 고장 위험)', RED)]:
        ax.scatter([], [], c=col, s=20, label=lab)
    ax.scatter([], [], marker='x', c=INK, s=22, label='실제 고장')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.13), fontsize=8, frameon=False, ncol=4)
    ax.set_title('검증 세트 고장 16건: 1시간마다 예측한 결과 (분홍 = 고장 24시간 안)', fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    save_figure(fig, '22_final_검증_고장별_경보타임라인.png', dpi=200)


def confusion():
    v = meta['validation']['point']
    cells = [[('TP (맞힌 경보)', v['TP'], ''), ('FN (놓침)', v['FN'], f'돌발 {v["FN 중 돌발 고장"]} · 점진 {v["FN 중 점진 고장"]}')],
             [('FP (오경보)', v['FP'], f'이른 경보 {v["FP 중 이른 경보"]} · 진짜 {v["FP 중 진짜 오경보"]}'), ('TN (맞힌 정상)', v['TN'], '')]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    colors = [['#dbeafe', '#fee2e2'], ['#ffedd5', '#f1f5f9']]
    for r in range(2):
        for c in range(2):
            name, n, sub = cells[r][c]
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=colors[r][c], ec='white', lw=3))
            ax.text(c + 0.5, 1 - r + 0.62, f'{n:,}', ha='center', va='center', fontsize=20, fontweight='bold', color=INK)
            ax.text(c + 0.5, 1 - r + 0.36, name, ha='center', va='center', fontsize=10, color=INK)
            if sub:
                ax.text(c + 0.5, 1 - r + 0.17, sub, ha='center', va='center', fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['예측: 경보', '예측: 정상'])
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(['실제: 24시간 안 고장', '실제: 고장 없음'])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'검증 세트 시점 {v["시점 수"]:,}개 (기준 {v["임계값"]}) · 정밀도 {v["정밀도"]:.3f} · '
                 f'재현율(점진) {v["재현율 (점진 고장만)"]:.3f}', fontsize=9.5, color=INK, loc='left')
    fig.tight_layout()
    save_figure(fig, '23_final_검증_혼동행렬.png', dpi=200)


def policy():
    """결과보고서 7장 정책 비교 (검증 세트, 기본 재고 가정, 현장 가정 만 원)"""
    pol = ['① 정기 교체만\n(시스템 없음)', '② 경보 즉시 교체', '③ 경보 + 의사결정 지원']
    cost, fail, stop = [75456, 64080, 54608], [16, 2, 2], [128, 100, 84]
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=200)
    bars = ax.barh(pol[::-1], [c / 10000 for c in cost[::-1]], color=[BLUE, '#93b4f5', GRAY], height=0.55)
    for b, c, f, s in zip(bars, cost[::-1], fail[::-1], stop[::-1]):
        ax.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2, f'{c / 10000:.2f}억 원 · 고장 {f}건 · 계획 밖 정지 {s}시간',
                va='center', fontsize=9, color=INK)
    ax.set_xlim(0, 13.5)
    ax.set_xlabel('기대 총비용 (억 원, 현장 가정값)')
    ax.set_title('정비 정책 비교 (검증 세트 72개 시나리오, 현장 가정 단가, 미확정 경보 8건 제외): ③이 ①보다 27.6% 적음\n'
                 '미확정 8건을 모두 불필요 교체로 보고 조치 비용을 넣으면 ③ 7.61억 원으로 ①보다 0.9% 많음 (analysis/12_report_checks.py)',
                 fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    save_figure(fig, '24_dss_정책비교.png', dpi=200)


if __name__ == '__main__':
    timeline()
    confusion()
    policy()
    print('그림 22~24 저장')
