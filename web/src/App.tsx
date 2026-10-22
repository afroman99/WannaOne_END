import { LogIn, ShieldCheck } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Notice, ToastProvider } from '@/components/app/common'
import { useToast } from '@/lib/feedback'
import { Header, type Role } from '@/components/app/header'
import { Manager } from '@/components/app/manager'
import { EquipPage, ModelPage, PartsPage, PolicyPage, SubPage } from '@/components/app/pages'
import { Worker } from '@/components/app/worker'
import { Button } from '@/components/ui/button'
import { api, ApiError, type ClockState, type ManagerRow, type Meta, type Session, type Task } from '@/lib/api'
import { useRoute } from '@/lib/route'

// 이 브라우저에만 기억하는 편의 값 (없어도 기본값으로 동작)
function remembered<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v === null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
function remember(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* 저장 못 해도 동작 */ }
}

const AUTO_REFRESH_MS = 10_000

function Login({ denied }: { denied: boolean }) {
  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col items-center justify-center gap-4 p-4 text-center">
      <ShieldCheck className="size-10" strokeWidth={2} aria-hidden />
      <h1 className="text-3xl font-semibold">CAST-<span className="text-primary">PdM</span></h1>
      <p className="text-muted-foreground">정비 의사결정 지원 솔루션 · 등록된 구글 계정으로 들어갑니다</p>
      {denied && <Notice tone="red">등록되지 않은 계정입니다. 설비관리자에게 이메일 등록을 요청하세요.</Notice>}
      <Button asChild><a href="/api/auth/login"><LogIn strokeWidth={2} aria-hidden />구글 계정으로 로그인</a></Button>
    </main>
  )
}

function Main({ meta, session }: { meta: Meta; session: Session }) {
  const toast = useToast()
  const [route, go] = useRoute()
  const [st, setSt] = useState<ClockState | null>(null)
  const [version, setVersion] = useState(0)
  const [truth, setTruth] = useState<boolean>(() => remembered('truth', false))
  const [team, setTeam] = useState<string>(() => session.user?.team ?? remembered('team', Object.keys(meta.teams)[0]))
  const [rows, setRows] = useState<ManagerRow[]>([])
  const seen = useRef<{ key: string; done: Set<number>; stop: Set<number>; mine: Set<number> } | null>(null)
  const lockedRole: Role | null = session.user ? (session.user.role === 'manager' ? 'manager' : 'worker') : null
  const role: Role = lockedRole ?? (route[0] === 'w' ? 'worker' : 'manager')
  const refresh = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => { remember('truth', truth); remember('team', team) }, [truth, team])
  // 첫 화면: 주소가 없으면 기억한 역할 (로그인한 사람은 그 역할만)
  useEffect(() => {
    if (!route.length || (lockedRole && route[0] !== lockedRole[0])) go(lockedRole === 'worker' ? 'w' : lockedRole === 'manager' ? 'm' : remembered('role', 'm'))
  }, [route, lockedRole, go])
  useEffect(() => { if (route[0]) remember('role', route[0]) }, [route])

  // 흐르는 시각 = 적용한 시연 시각 + 적용 뒤 흐른 실제 시간 (최신 값은 ref로 읽어 함수가 바뀌지 않게 한다)
  const stRef = useRef(st)
  useEffect(() => { stRef.current = st }, [st])
  const nowF = useCallback(() => {
    const s = stRef.current
    if (!s) return meta.endH
    return s.clockAt ? Math.min(meta.endH, s.now + Math.max(0, Date.now() / 1000 - s.clockAt) / 3600) : s.now
  }, [meta.endH])

  // 자동 갱신: 10초마다 바뀐 것(작업·설정 version)이 있거나 흐르는 시각이 정각을 넘으면 화면을 다시 읽는다
  const lastHour = useRef<number | null>(null)
  useEffect(() => {
    let live = true
    const poll = async () => {
      try {
        const s = await api.state()
        if (!live) return
        const old = stRef.current
        stRef.current = s
        setSt(s)
        if (old && old.version !== s.version) setVersion((v) => v + 1)
      } catch { /* 다음에 다시 */ }
    }
    poll()
    const id = setInterval(() => {
      poll()
      const h = Math.floor(nowF())
      if (lastHour.current !== null && h !== lastHour.current) setVersion((v) => v + 1)
      lastHour.current = h
    }, AUTO_REFRESH_MS)
    return () => { live = false; clearInterval(id) }
  }, [nowF])

  // 알림: 다른 사람이 한 일 (설비관리자 ← 완료 보고·작업 중단, 작업자 ← 새 배정). 처음 볼 때는 기록만 한다
  const split = st?.split
  useEffect(() => {
    if (!split) return
    let live = true
    const key = `${role}|${split}|${team}`
    const notify = (items: Task[]) => {
      const done = new Set(items.filter((t) => t.completed_at).map((t) => t.id))
      const stop = new Set(items.filter((t) => t.stopped_h !== null).map((t) => t.id))
      const mine = new Set(items.map((t) => t.id))
      const s = seen.current
      if (s && s.key === key) {
        for (const t of items) {
          if (role === 'manager' && done.has(t.id) && !s.done.has(t.id))
            toast(`완료 보고 · ${t.name} ${t.outcome} (${t.team}, 작업 #${t.id})${t.action_why ? ' · 지시와 다름' : ''}`, 'green', true)
          if (role === 'manager' && stop.has(t.id) && !s.stop.has(t.id))
            toast(`작업 중단 · ${t.name} (${t.team}, 작업 #${t.id}) · 사유: ${t.stop_why}`, 'orange', true)
          if (role === 'worker' && !s.mine.has(t.id)) toast(`새 작업 · ${t.title} (${t.priority})`, 'blue', true)
        }
      }
      seen.current = { key, done, stop, mine }
    }
    if (role === 'manager') api.manager(false).then((m) => { if (live) { notify(m.tasks); setRows(m.rows) } }).catch(() => {})
    else api.tasks(team).then((x) => live && notify(x.items)).catch(() => {})
    return () => { live = false }
  }, [version, role, team, split, toast])

  if (!st) return <main className="mx-auto max-w-6xl p-4 sm:p-8"><p className="text-muted-foreground">불러오는 중...</p></main>

  const apply = async (s: string, h: number) => {
    try {
      setSt(await api.setClock(s as ClockState['split'], h))
      refresh()
    } catch (e) {
      toast((e as Error).message, 'red')
    }
  }
  const [r0, r1, r2, r3] = route
  let body
  if (role === 'worker') {
    body = <Worker meta={meta} route={route} go={go} team={team} setTeam={setTeam} locked={!!session.user} version={version} nowF={nowF}
      storeNote={session.storeNote} onChanged={refresh} />
  } else if (r1 === 'page' && r2) {
    const page = r2 === '설비 상세'
      ? <EquipPage meta={meta} sid={r3} rows={rows} onSid={(s) => go(`m/page/설비 상세/${s}`)} truth={truth && st.split === 'validation'} version={version} />
      : r2 === '정책·비용' ? <PolicyPage version={version} onChanged={refresh} />
        : r2 === '부품 손익' ? <PartsPage version={version} />
          : r2 === '모델 정보·한계' ? <ModelPage version={version} /> : null
    body = <SubPage name={r2 === '설비 상세' ? (rows.find((x) => x.sid === r3)?.name ?? r2) : r2} onBack={() => history.length > 1 ? history.back() : go('m')}>{page}</SubPage>
  } else {
    body = <Manager meta={meta} route={route} go={go} now={Math.floor(nowF())} nowF={nowF} version={version} truth={truth && st.split === 'validation'} onChanged={refresh} />
  }
  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-8 p-4 sm:p-8">
      <Header meta={meta} session={session} st={st} nowF={nowF} role={role} truth={truth} setTruth={setTruth}
        onRole={(r) => go(r === 'worker' ? 'w' : 'm')} onApply={apply}
        onLogout={() => api.logout().finally(() => window.location.assign('/'))} />
      {r0 && body}
    </main>
  )
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    Promise.all([api.session(), api.meta()]).then(([s, m]) => { setSession(s); setMeta(m) })
      .catch((e: Error) => setError(e instanceof ApiError ? e.message : '서버에 연결하지 못했습니다'))
  }, [])
  if (error) return <main className="mx-auto max-w-6xl p-4 sm:p-8"><Notice tone="red">{error}</Notice></main>
  if (!meta || !session) return <main className="mx-auto max-w-6xl p-4 sm:p-8"><p className="text-muted-foreground">불러오는 중...</p></main>
  if (session.loginEnabled && !session.user) return <Login denied={new URLSearchParams(window.location.search).get('login') === 'denied'} />
  return <ToastProvider><Main meta={meta} session={session} /></ToastProvider>
}
