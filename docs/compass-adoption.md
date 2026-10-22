# COMPASS 도입 기록 (React 사본)

- 설치 버전: **0.1.0** (고정)
- 레지스트리: https://compass.designcompass.org
- 설치 명령 (`web/`에서, 패키지 관리자 npm):
  `npx shadcn@4.21.0 add https://compass.designcompass.org/r/0.1.0/foundations.json https://compass.designcompass.org/r/0.1.0/compass.json -y`
- 적용일: 2026-10-22. 대상: `WannaOne_END_react/web` (원본 `WannaOne_END`의 사본, 원본은 바꾸지 않음)

## 설치 전 준비

| 항목 | 값 |
|---|---|
| 프레임워크 | Vite 7 + React 19 + TypeScript 5.8 (새로 만듦: `npm create vite@7.1.3 web -- --template react-ts`) |
| Tailwind | v4.3 (`@tailwindcss/vite`) |
| shadcn | `npx shadcn@4.21.0 init -t vite -b radix -p nova` → style `radix-nova`, baseColor `neutral`, cssVariables, iconLibrary `lucide` |
| 경로 별칭 | `@/*` → `web/src/*` (tsconfig, vite.config.ts) |

- shadcn 초기화가 만든 `button.tsx`는 사용자 수정이 없는 생성본이라 지우고 COMPASS 버전을 설치했다 (덮어쓰기 옵션 없이).
- 초기화가 넣은 Geist 글꼴은 지웠다. 글꼴은 COMPASS의 Pretendard Variable 한 번만 (`index.css`의 `pretendard/dist/web/variable/pretendardvariable.css`).

## 설치된 것

- 토큰: `web/src/index.css` (색, Pretendard, 반지름, 그림자, squircle, primary-hover)
- 컴포넌트 15개: `web/src/components/ui/` alert-dialog, avatar, badge, button, card, dialog, input, pagination, progress, select, sheet, slider, tabs, toggle, toggle-group
- `web/src/components/theme-provider.tsx` → `web/src/main.tsx`에서 앱 최상위에 연결 (기본 system, 라이트·다크, `d` 키로 전환)
- 문서: `web/docs/compass.md`
- 의존성: class-variance-authority, cn(shadcn-ui/cn), lucide-react, next-themes, pretendard, radix-ui, tw-animate-css, shadcn

## 프로젝트 조정 (COMPASS 원본과 다른 점)

- `--radius` 2.5rem → **1.25rem**: Streamlit 사본에서 "모서리가 너무 둥글다"는 의견(2026-10-22)을 반영. 컴포넌트 파일은 고치지 않고 토큰 하나만 바꿔서 버튼 11px · 입력 7px · 카드 20px. 비율과 중첩 규칙(`rounded-container`)은 그대로.
- 상태색 `--warning`(주의)·`--success`(정상)를 `index.css` 끝에 더함. 경보는 COMPASS `--destructive`.
- eslint: `src/components/ui/**`만 `react-refresh/only-export-components`를 끔 (COMPASS 파일이 변형 함수를 같이 내보내는 원래 구조).

## 검증 (2026-10-22)

- `npx tsc -b`, `npm run lint`, `npm run build` 통과 (빌드 경고: JS 묶음 794kB > 500kB, Pretendard 전체 글꼴 2MB)
- 라이트·다크 화면 확인: 배경 #ffffff / oklch(0.15 0 0), 글꼴 Pretendard Variable, Primary #007AFF
