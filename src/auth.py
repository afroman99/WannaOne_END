"""로그인 계정의 역할 찾기 (구글 로그인, 2차 2026-10-16 → React 서버 api/server.py가 씀, 2026-10-22).

구글 로그인 자체(OpenID Connect)는 api/server.py가 Authlib로 한다. 비밀값은 .streamlit/secrets.toml
(배포에서는 환경 변수 CAST_SECRETS에 같은 내용)에 둔다:

    [auth]
    cookie_secret = "..."                                 # 긴 무작위 문자열 (로그인 유지 쿠키 서명)
    client_id = "....apps.googleusercontent.com"
    client_secret = "..."
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

    [users]                                               # 이메일별 역할·팀 (여기 없는 계정은 로그인해도 못 들어온다)
    managers = ["관리자@gmail.com"]
    [users.teams]
    "정비 1팀" = ["작업자1@gmail.com"]
    "정비 2팀" = ["작업자2@gmail.com"]

  - 구글이 확인한 이메일(email_verified)만 받는다. 비교는 소문자로
  - 설비관리자는 설비관리자 화면만, 작업자는 자기 팀의 작업자 화면만 본다
  - 로그인해야 공용 DB(Secrets [db])에 쓴다
"""


def lookup(users, email):
    """이메일 → (역할 'manager'|'worker'|None, 팀). users: Secrets [users]"""
    email = str(email or '').strip().lower()
    if not email or not users:
        return None, None
    if email in [str(x).strip().lower() for x in users.get('managers', [])]:
        return 'manager', None
    for team, emails in (users.get('teams') or {}).items():
        if email in [str(x).strip().lower() for x in emails]:
            return 'worker', team
    return None, None
