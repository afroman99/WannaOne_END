# React + FastAPI 전환 (2026-10-22)

Streamlit 화면(`app/dashboard.py`)과 같은 계산을 쓰는 새 화면을 옆에 만들었다. 옮긴 뒤 Streamlit 화면·설정(`app/`, `.streamlit/config.toml`, `deploy.toml`, 접속 비밀번호)과 패키지(streamlit·altair·tornado)는 지웠다(2026-10-22, git 기록에 남음). 아래 표의 '원본'은 지운 Streamlit 화면이다.

```
web/ (React + Tailwind v4 + shadcn + COMPASS)  ──/api──▶  api/server.py (FastAPI)  ──▶  api/engine.py ──▶ src/ · models/
                                                                                    └─▶  actions/ (로컬) 또는 Supabase (로그인 시)
```

- `api/engine.py`: dashboard.py의 계산 부분(상태표, 재고, 정비 계획, 정비 판단 세 칸, 조치 안내, 분석 근거, 온톨로지, 정책·비용, 모델 정보, 부품 손익)을 Streamlit 없이 옮겼다. 규칙과 문장은 그대로다.
- `api/server.py`: 엔드포인트, 저장소 고르기(로그인 + Secrets [db] → 공용 DB, 아니면 로컬), 재생 시각, 구글 로그인.
- `web/src/components/app/`: 화면. 모양은 COMPASS 컴포넌트(`web/src/components/ui`)로만.

## 실행

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r api/requirements.txt   # 처음 한 번
cd web && npm install && npm run build && cd ..                                             # 화면 빌드
.venv/bin/uvicorn api.server:app --port 8000                                                # http://localhost:8000
```

### 배포 (Render 무료 웹 서비스)

Hugging Face Spaces는 2026-10 기준 Docker가 유료(PRO)라 Render를 쓴다. 저장소 맨 위 `Dockerfile`이 화면을 빌드하고 서버를 켠다
(원자료·모델·기록·비밀값은 `.dockerignore`와 `.gitignore`로 빠진다. 화면은 `app_data/`만 읽는다). 서버 최대 메모리는 약 263MB로 무료 512MB 안이다.

**배포 주소: https://cast-pdm.onrender.com** (2026-10-22, 포크 afroman99/WannaOne_END의 `react` 가지, Virginia, Free).
설비관리자·작업자 계정 로그인과 공용 DB 저장을 이 주소에서 확인했다. Streamlit Cloud 앱(`cast-pdm.streamlit.app`)은 같은 날 지웠다.

1. Render → New → **Web Service** → 깃허브 저장소, Branch **react**.
2. Language **Docker** (기본값 Python 3이면 Build/Start Command를 물어본다). Root Directory는 **비움**, Dockerfile Path `./Dockerfile`.
3. Region **Virginia (US East)**: 공용 DB(Supabase)가 us-east-1이라 같은 지역에 둔다. Singapore에 두었을 때는 로그인 뒤 요청마다 DB 왕복이 길어 느렸다. 지역은 만든 뒤 바꿀 수 없어 서비스를 새로 만들어야 한다.
4. Instance Type **Free** ($0, 0.1 CPU, 512MB). 아래 막대가 `$0 / month`인지 확인.
5. Environment Variables → KEY `CAST_SECRETS`(**밑줄**. `CAST-SECRETS`처럼 하이픈이면 못 읽고 로그인 없이 뜬다), VALUE = `.streamlit/secrets.toml` 내용 전체(`pbcopy < .streamlit/secrets.toml` 뒤 붙여넣기). Secret Files로 올리면 `/etc/secrets/`에 들어가 서버가 찾지 못한다.
6. 배포가 끝나면 구글 클라우드 콘솔 OAuth 클라이언트의 승인된 리디렉션 URI에 `https://<이름>.onrender.com/api/auth/callback`을 더한다. 서버는 `secrets.toml`의 `redirect_uri`(Streamlit용)를 쓰지 않고 접속한 주소로 만든다.
7. 확인: `https://<이름>.onrender.com/api/session`이 `"loginEnabled": true`면 비밀값이 들어간 것이다.

- 무료는 15분 동안 접속이 없으면 잠들고, 다음 접속 때 약 1분 걸려 깬다 (발표 전에 한 번 열어 둘 것).
- 무료 CPU(0.1개)라 첫 계산이 노트북보다 느리다(설비 목록 계산 약 2초, 로컬 0.06초).
- 로그인 뒤 `mismatching_state` 오류(`Internal Server Error`)가 나면, 로그인을 시작한 창이 아닌 곳에서 돌아온 주소를 열었거나 남은 쿠키 탓이다 → 시크릿 창에서 배포 주소를 새로 열고 로그인한다. 같은 콜백 주소를 나중에 다시 열면 `invalid_grant`(일회용 코드 만료)가 난다.
- 로그에 나오는 `AuthlibDeprecationWarning`(httpx)과 Vite의 `chunkSizeWarningLimit` 경고는 동작과 상관없다.

화면을 고치면서 볼 때는 위 서버를 켠 채로 `cd web && npm run dev` → http://localhost:5173 (/api는 8000번으로 넘김).
같은 Wi-Fi의 폰에서 보려면 `--host 0.0.0.0`으로 켜고 `http://<노트북 IP>:8000`.

## 옮긴 것 (원본과 같은 동작)

| 원본 (Streamlit) | React 주소 |
|---|---|
| 시연 설정: 데이터 세트·지금 시각·적용, 적용한 순간부터 실제 속도로 흐르는 시계, 정답 보기 | 머리말 |
| 보는 사람 (설비관리자·작업자), 라이트·다크·시스템 | 머리말 |
| 설비 상태: 경보·주의·정상 칸과 칸 안 목록, 상태별 넓은 목록 | `#/m`, `#/m/stage/경보` |
| 설비 화면: 조치 안내 · 고장 예상 타이머 · 분석 근거 · 온톨로지 그림과 근거 사슬 | `#/m/eq/S0309` |
| 정비 판단 세 칸(즉시 정비 · 다음 정기 교체 · 사후정비): 칸마다 팀 · 작업자 · 작업책임자, 인원에 따라 비용 다시 계산, 인원 부족·안전 기준 초과 표시, 배정 | 같은 화면 |
| 보류 · 밸브/진동장치 보충 발주 · 두 부품 긴급 조달 · 작업 내용 바꾸기 · 비용 가정 · 비용 계산 근거 | 같은 화면 |
| 경보가 아닌 설비 배정(작업명·우선순위·시작 예정·점검 가이드·추천 이유) | 같은 화면 |
| 작업 지시 현황: 지표 6칸 · 먼저 볼 설비 · 작업 목록(구분 필터) · 작업 이력 CSV | `#/m/orders` |
| 작업 수정하기: 상태(대기·진행 중·완료, 완료 보고·시각) · 팀 · 작업자 · 작업책임자 · 우선순위 · 시작 예정 · 가이드 · 이유 · 수정 사유, 배정 취소(사유) | `#/m/orders` |
| 배정할 시나리오 목록 → 근거·비용 → 배정 | `#/m/assign/S0309` |
| 인원 현황: 팀별 출근 · 하루 작업 한도, 부재 표(줄 더하기·지우기) · 저장 · 전원 출근 · 인원 부족 예시 | `#/m/staff` |
| 설비 상세 하위 페이지 (설비 고르기 · 분석 근거 · 근거 따라가기) | `#/m/page/설비 상세/S0309` |
| 정책·비용: 가정값(저장·기본값) · 전체 정비 결정 · 설비별 근거(확률 덮어쓰기, 선택지 표, 경계, 고장 확률 곡선) · 조치 기록과 점검 되먹임 · 주의 설비 준비(한 번에 발주) · 정책 시뮬레이션 | `#/m/page/정책·비용` |
| 모델 정보·한계: 검증 성능 · 주의 단계 · 고장별 결과 · 한계(중요도 필터) · 온톨로지 지표 · Turtle 내려받기 | `#/m/page/모델 정보·한계` |
| 부품 손익: 맞힐 확률 슬라이더 · 세 방식 비용 · 미확정 경보 · 손익분기 · 그래프 | `#/m/page/부품 손익` |
| 작업자 홈: 팀 · 배정 전 즉시 조치(작업 만들기) · 지금/지연/예정/완료/중단 칸 | `#/w` |
| 구분별 목록, 작업 상세: 배지(우선순위·상태·지연·즉시 조치·작업자·체크 수·위험·재고) · 점검 가이드 | `#/w/list/지금`, `#/w/task/6` |
| 체크리스트: 사전 절차(①→②→③ 순서, TBM 6개 → TBM 완료) → 작업 시작 → 10대 안전철칙(준수/해당 없음) → 정비 후 절차(①~④ 순서) · 바로 저장 | `#/w/task/6` |
| 완료 보고(결과 · 판단 부품 · 확인된 원인 · 실제 조치 · 다르게 한 이유 · 수행 내용 · 즉시 조치 허가서 사후 처리) / 작업 중단(사유) | `#/w/task/6/report`, `/stop` |
| 자동 갱신(10초마다 바뀐 것·정각 확인)과 알림(관리자 ← 완료 보고·중단, 작업자 ← 새 작업) | 전체 |
| 뒤로 가기·앞으로 가기 (주소에 화면 위치) | 전체 |
| 구글 로그인 · 역할 고정(설비관리자는 관리자 화면만, 작업자는 자기 팀만) · 로그인 시 공용 DB | `/api/auth/login` |
| 폰 홈 화면 설치(manifest·아이콘) | `web/public` |

## 원본과 다른 점

- 가정값(정책·비용)은 원본에서는 보는 사람 세션마다였지만, 여기서는 저장소 settings에 저장해 모든 화면이 같은 값을 쓴다.
- 작업자 화면의 단계(홈 → 목록 → 상세)는 같고, 버튼·배지 모양은 COMPASS다. 선택된 토글은 Primary 테두리·글자로 표시한다(`web/src/lib/styles.ts`).
- 분석 근거의 차트는 Altair 대신 recharts, 온톨로지 그림은 graphviz 대신 viz.js(같은 DOT)로 그린다.

## 검증하지 못한 것 / 제한

- **구글 로그인·공용 DB**: `.streamlit/secrets.toml` 11번째 줄 따옴표 오류를 고친 뒤 로컬(`http://localhost:8000`, 리디렉션 URI `http://localhost:8000/api/auth/callback`)에서 설비관리자 계정, 배포 주소에서 설비관리자·작업자 계정 로그인을 확인했다(2026-10-22).
- **클라우드 배포**: Streamlit Community Cloud는 FastAPI+React를 올릴 수 없어 Render에 올렸다(위 '배포').
- 원본과 같은 시각·같은 기록에서 숫자를 화면끼리 하나하나 맞대어 보지는 않았다(계산 코드는 줄 단위로 옮김).
- 첫 요청 때 예측을 계산해서 10초쯤 걸린다.
- 빌드 경고: JS 묶음 약 1MB(차트·graphviz), Pretendard 전체 글꼴 2MB.

## 시험 중 남긴 기록 (사본의 `actions/`에만)

작업 #5 배정·시작·수정, #4 완료(예전 API라 진행 기록 없음), #6 배정(정비 2팀, 사후정비) → 체크리스트 → 시작 → 완료 보고(밸브만 교체), 재생 시각 2일차 16:00 적용.
