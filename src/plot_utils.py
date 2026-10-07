"""그래프 공통 설정."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 프로젝트 루트의 figures 폴더
FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'

# 한글 글꼴 후보 (앞에 있을수록 우선)
# AppleGothic은 °C가 어색하게 나와서 Apple SD Gothic Neo를 먼저 쓴다
KOREAN_FONTS = [
    'Apple SD Gothic Neo',  # macOS
    'AppleGothic',          # macOS
    'Malgun Gothic',        # Windows
    'NanumGothic',          # Linux (설치된 경우)
    'Noto Sans CJK KR',     # Linux (설치된 경우)
    'Noto Sans KR',
]


def set_korean_font():
    """사용 가능한 한글 글꼴을 찾아 matplotlib 기본 글꼴로 정한다.

    한글 글꼴에는 유니코드 마이너스 기호가 없어서 일반 하이픈을 쓰게 한다.
    로그 축 눈금은 각 그림에서 일반 숫자로 표시한다.
    """
    available = {font.name for font in fm.fontManager.ttflist}
    plt.rcParams['axes.unicode_minus'] = False

    for font in KOREAN_FONTS:
        if font in available:
            plt.rcParams['font.family'] = font
            return font

    print('사용 가능한 한글 글꼴을 찾지 못했습니다.')
    return None


def save_figure(fig, name, dpi=120):
    """figures/<name>.png로 저장하고 그림을 닫는다.

    name에는 번호와 내용을 넣는다 (예: '03_lag_correlation').
    """
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f'{name}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'그림 저장: figures/{path.name}')
    return path
