# 프로젝트 규칙

## 디자인 시스템 (COMPASS)

사용자가 '디자인 시스템 써', 'COMPASS 써', 'use the design system'라고 하면 COMPASS Design System 0.1.0을 적용한다. docs/compass.md를 읽고 설치된 컴포넌트와 토큰을 재사용한다. 프레임워크를 유지하고 누락된 컴포넌트만 https://compass.designcompass.org/r/0.1.0/에서 추가한다. COMPASS를 자동 업그레이드하거나 프로젝트의 수정 내용을 덮어쓰지 않는다.

이 프로젝트에서의 위치 (React 화면은 `web/`):

- COMPASS 문서: `web/docs/compass.md` (레지스트리가 설치한 원문). 도입 기록: `docs/compass-adoption.md`.
- 설치된 컴포넌트: `web/src/components/ui/` (button, card, dialog, alert-dialog, tabs, input, select, slider, sheet, pagination, toggle, toggle-group, badge, avatar, progress), 테마: `web/src/components/theme-provider.tsx`.
- 토큰: `web/src/index.css`. 프로젝트 조정 두 가지만 있다: `--radius` 1.25rem(COMPASS 기본 2.5rem보다 덜 둥글게), 상태색 `--warning`·`--success`.
- 컴포넌트를 추가할 때는 `web/`에서 `npx shadcn@4.21.0 add https://compass.designcompass.org/r/0.1.0/<이름>.json`. `--overwrite` 금지.
- 화면 코드는 `web/src/components/app/`. 비슷한 모양을 새로 만들지 말고 `components/ui`를 쓴다. 아이콘은 `lucide-react`, 선 두께 2.
- 분석·모델·작업 저장소는 Python `src/`를 그대로 쓰고, 화면은 `api/server.py`(FastAPI)를 거쳐서만 읽고 쓴다.
