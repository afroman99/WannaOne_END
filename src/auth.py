"""대시보드 접속 관리: 구글 로그인(2차, 2026-10-16) 또는 접속 비밀번호.

구글 로그인 (Streamlit st.login, OpenID Connect). Secrets에 아래가 있으면 켜진다:

    [auth]
    redirect_uri = "https://<앱주소>/oauth2callback"      # 로컬 시험은 http://localhost:8501/oauth2callback
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
  - 설비관리자는 작업자 화면도 볼 수 있고, 작업자는 자기 팀의 내 작업만 본다
  - 로그인을 켜야 공용 DB(Secrets [db])에 쓴다. 로그인 없는 공개 화면은 보는 사람의 세션에만 쓴다

접속 비밀번호 (구글 로그인이 없을 때). 비밀번호 자체는 어디에도 저장하지 않고, 솔트를 넣은 해시만 둔다.

비밀번호 해시 만들기 (터미널에서, 입력한 비밀번호는 화면에 보이지 않는다):

    python -m src.auth

나온 세 줄을 Streamlit Community Cloud 앱 설정의 Secrets(또는 로컬 .streamlit/secrets.toml, 커밋하지 않음)에 넣는다
([auth]는 구글 로그인이 쓰므로 비밀번호는 [password], 2026-10-16):

    [password]
    salt = "..."
    password_pbkdf2 = "..."
    iterations = 600000

동작
- Secrets에 [password]가 있으면 비밀번호를 맞혀야 화면이 열린다 (세션마다, 비교는 상수 시간 hmac.compare_digest)
- 틀릴 때마다 대기 시간이 늘고(1·2·4·8·16초), 한 세션에서 5번 틀리면 그 세션은 잠긴다.
  새로 고침하면 새 세션이라 횟수가 다시 시작된다 → 무차별 대입을 늦출 뿐 막지는 못하므로 긴 비밀번호와 아래 비공개 앱 설정이 필요하다
- 클라우드 모드(app_data로 실행)인데 구글 로그인도 [password]도 없으면 열지 않는다 (설정을 빠뜨려 공개되는 것을 막는 기본 거부)
  일부러 공개하려면 저장소의 deploy.toml에 [access] public = true 를 커밋하거나(팀 결정 2026-10-15: 공개),
  Secrets에 [password] disabled = true 를 넣는다
- 이 비밀번호는 1차 방어선이다. Streamlit Community Cloud의 '비공개 앱 + 볼 수 있는 사람 이메일 지정'을 함께 쓴다
"""

import getpass
import hashlib
import hmac
import secrets as pysecrets
import time
import tomllib
from pathlib import Path

DEPLOY_FILE = Path(__file__).resolve().parent.parent / 'deploy.toml'

ITERATIONS = 600_000      # OWASP 권장 PBKDF2-HMAC-SHA256 반복 수 (2023~)
MAX_FAILS = 5


def hash_password(password, salt=None, iterations=ITERATIONS):
    salt = salt or pysecrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), iterations).hex()
    return salt, digest, iterations


def verify(password, salt, expected, iterations=ITERATIONS):
    _, digest, _ = hash_password(password, salt, iterations)
    return hmac.compare_digest(digest, expected)


def public_by_decision(path=DEPLOY_FILE):
    """저장소의 deploy.toml에 [access] public = true가 명시돼 있으면 공개 (팀 결정을 커밋으로 남긴다)"""
    try:
        return bool(tomllib.loads(Path(path).read_text(encoding='utf-8')).get('access', {}).get('public', False))
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return False


def _secret(st, key):
    try:
        return st.secrets[key] if key in st.secrets else None
    except Exception:          # Secrets 파일이 아예 없을 때
        return None


def google_configured(st):
    conf = _secret(st, 'auth')
    return bool(conf and conf.get('client_id'))


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


def google_user(st, title='CAST-PdM'):
    """구글 로그인을 요구한다. 통과하면 {'email', 'name', 'role', 'team'}, 아니면 로그인 화면을 그리고 st.stop()"""
    if not st.user.is_logged_in:
        with st.container(horizontal_alignment='center'):
            st.space('large')
            st.markdown(f'## :material/shield: {title}', text_alignment='center')
            st.caption('정비 의사결정 지원 솔루션 · 등록된 구글 계정으로 들어갑니다', text_alignment='center')
            st.button('구글 계정으로 로그인', icon=':material/login:', type='primary', on_click=st.login)
        st.stop()
    email = str(st.user.get('email') or '').lower()
    role, team = lookup(_secret(st, 'users'), email) if st.user.get('email_verified', False) else (None, None)
    if role is None:
        st.error('등록되지 않은 계정입니다. 설비관리자에게 이메일 등록을 요청하세요.', icon=':material/block:')
        st.caption(email)
        st.button('로그아웃', icon=':material/logout:', on_click=st.logout)
        st.stop()
    return {'email': email, 'name': str(st.user.get('name') or email), 'role': role, 'team': team}


def require_login(st, cloud_mode):
    """구글 로그인이 설정돼 있으면 로그인한 사람(dict)을 돌려준다. 아니면 비밀번호 검사 뒤 None (막으면 st.stop())"""
    if google_configured(st):
        return google_user(st)
    conf = _secret(st, 'password')
    if conf is None:
        if cloud_mode and public_by_decision():
            return             # 팀이 공개로 정함 (deploy.toml)
        if cloud_mode:
            st.error('접속 비밀번호가 설정되지 않아 화면을 열 수 없습니다. 관리자에게 문의하세요.')
            st.stop()
        return                 # 로컬 개발 모드: 내 컴퓨터 안에서만 열리므로 비밀번호 없이
    if conf.get('disabled', False):
        return
    if st.session_state.get('auth_ok'):
        return
    fails = st.session_state.get('auth_fails', 0)
    if fails >= MAX_FAILS:
        st.error('비밀번호를 여러 번 틀려 이 세션이 잠겼습니다. 잠시 뒤 다시 시도하세요.')
        st.stop()
    st.title('연주 설비 조기경보')
    with st.form('login'):
        pw = st.text_input('접속 비밀번호', type='password')
        ok = st.form_submit_button('들어가기')
    if ok:
        if verify(pw, conf['salt'], conf['password_pbkdf2'], int(conf.get('iterations', ITERATIONS))):
            st.session_state['auth_ok'] = True
            st.session_state['auth_fails'] = 0
            st.rerun()
        st.session_state['auth_fails'] = fails + 1
        time.sleep(min(2 ** fails, 16))     # 틀릴수록 오래 기다리게 해 무차별 대입을 늦춘다
        st.error(f'비밀번호가 맞지 않습니다 ({fails + 1}/{MAX_FAILS}).')
    st.stop()


if __name__ == '__main__':
    pw = getpass.getpass('대시보드 접속 비밀번호 (12자 이상 권장): ')
    if len(pw) < 12:
        print('경고: 12자 이상을 권장합니다.')
    if pw != getpass.getpass('한 번 더: '):
        raise SystemExit('두 비밀번호가 다릅니다.')
    salt, digest, it = hash_password(pw)
    print('\n아래를 Secrets에 붙여 넣으세요 (비밀번호 자체는 어디에도 저장하지 않습니다):\n')
    print(f'[password]\nsalt = "{salt}"\npassword_pbkdf2 = "{digest}"\niterations = {it}')
