import { CircleUserRound, HardHat, LogOut, Monitor, Moon, ShieldCheck, SlidersHorizontal, Sun, UserCog } from 'lucide-react'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

import { Check } from '@/components/app/common'
import { Button } from '@/components/ui/button'
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger,
} from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { picked } from '@/lib/styles'
import { clock, clockS, type ClockState, type Meta, type Session, type Split } from '@/lib/api'

export type Role = 'manager' | 'worker'

const THEMES = [
  { value: 'system', label: '시스템 설정', Icon: Monitor },
  { value: 'light', label: '라이트', Icon: Sun },
  { value: 'dark', label: '다크', Icon: Moon },
] as const

function ThemeToggle() {
  const { theme = 'system', setTheme } = useTheme()
  const i = Math.max(0, THEMES.findIndex((t) => t.value === theme))
  const cur = THEMES[i]
  const next = THEMES[(i + 1) % THEMES.length]
  return (
    <Button variant="ghost" size="icon" aria-label={`테마: ${cur.label} (누르면 ${next.label})`} title={`테마: ${cur.label}`}
      onClick={() => setTheme(next.value)}>
      <cur.Icon className="size-5" strokeWidth={2} aria-hidden />
    </Button>
  )
}

/** 1초씩 흐르는 시계 (적용한 시연 시각 + 적용 뒤 흐른 실제 시간) */
function LiveClock({ nowF, running }: { nowF: () => number; running: boolean }) {
  const [, tick] = useState(0)
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => tick((x) => x + 1), 1000)
    return () => clearInterval(id)
  }, [running])
  return <span className="text-lg font-semibold tabular-nums">{clockS(nowF())}</span>
}

/** 시연 설정: 데이터 세트와 지금 시각 (적용을 눌러야 바뀌고, 적용한 순간부터 실제 속도로 흐른다) */
function ClockDialog({ meta, st, truth, setTruth, onApply, shared }: {
  meta: Meta; st: ClockState; truth: boolean; setTruth: (v: boolean) => void; onApply: (s: Split, h: number) => void; shared: boolean
}) {
  const [open, setOpen] = useState(false)
  const [s, setS] = useState<Split>(st.split)
  const [h, setH] = useState(st.now)
  return (
    <Dialog open={open} onOpenChange={(o) => { setOpen(o); if (o) { setS(st.split); setH(Math.floor(st.nowF)) } }}>
      <DialogTrigger asChild>
        <Button variant="outline"><SlidersHorizontal strokeWidth={2} aria-hidden />{meta.splits[st.split].split(' ')[0]} · 시연 설정</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>테스트 · 시연 설정</DialogTitle>
          <DialogDescription>
            옮긴 뒤 "적용"을 눌러야 바뀝니다. 적용한 순간부터 실제 시간과 같은 속도로 흐르고(4일차 00:00에서 멈춤), 예측 · 경보 · 비용은 1시간 요약본이라 정각마다 바뀝니다.
            독립된 합성 시나리오를 재생합니다 · 실시간 설비 연결 아님{shared ? ' · 데이터 세트와 시각은 모든 화면이 함께 씁니다' : ''}{st.canSetClock ? '' : ' (설비관리자가 바꿈)'}
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-6">
          <div className="grid gap-2">
            <label htmlFor="split" className="text-sm font-medium">데이터 세트</label>
            <Select value={s} onValueChange={(v) => setS(v as Split)} disabled={!st.canSetClock}>
              <SelectTrigger id="split" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{Object.entries(meta.splits).map(([k, v]) => <SelectItem key={k} value={k}>{v}</SelectItem>)}</SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">테스트: 정답이 없는 실제 운영과 같은 상황 · 검증: 정답이 있어 실제 고장과 비교할 수 있음</p>
          </div>
          <div className="grid gap-3">
            <div className="flex items-baseline justify-between">
              <span id="now-label" className="text-sm font-medium">지금 시각</span>
              <span className="font-medium text-primary tabular-nums">{clock(h)}</span>
            </div>
            <Slider aria-labelledby="now-label" min={0} max={meta.endH} step={1} value={[h]} disabled={!st.canSetClock} onValueChange={([v]) => setH(v)} />
            <div className="flex justify-between text-sm text-muted-foreground"><span>{clock(0)}</span><span>{clock(meta.endH)}</span></div>
          </div>
          <Check id="truth" checked={truth && st.split === 'validation'} disabled={st.split !== 'validation'} onChange={setTruth}
            label={<span>정답 보기 (검증 세트)<br /><span className="text-sm text-muted-foreground">앞으로 일어날 실제 고장 시각(정답)을 겹쳐 봅니다. 모델은 이 정보를 보지 않습니다.</span></span>} />
        </div>
        <DialogFooter>
          <Button disabled={!st.canSetClock} onClick={() => { onApply(s, h); setOpen(false) }}>적용</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function UserMenu({ s, onLogout }: { s: Session; onLogout: () => void }) {
  const u = s.user!
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost"><CircleUserRound strokeWidth={2} aria-hidden />{u.name.split('@')[0]}</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{u.name}</DialogTitle>
          <DialogDescription>{u.email} · {u.role === 'manager' ? '설비관리자' : `작업자 · ${u.team}`}</DialogDescription>
        </DialogHeader>
        <DialogFooter><Button variant="outline" onClick={onLogout}><LogOut strokeWidth={2} aria-hidden />로그아웃</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function Header({ meta, session, st, nowF, role, onRole, onApply, truth, setTruth, onLogout }: {
  meta: Meta; session: Session; st: ClockState; nowF: () => number; role: Role; onRole: (r: Role) => void
  onApply: (s: Split, h: number) => void; truth: boolean; setTruth: (v: boolean) => void; onLogout: () => void
}) {
  const running = !!st.clockAt && st.nowF < meta.endH
  return (
    <header className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-6" strokeWidth={2} aria-hidden />
          <h1 className="text-2xl font-semibold">CAST-<span className="text-primary">PdM</span></h1>
          <span className="hidden text-sm text-muted-foreground sm:inline">정비 의사결정 지원 솔루션 · 시연</span>
        </div>
        <div className="flex items-center gap-1">
          {session.user && <UserMenu s={session} onLogout={onLogout} />}
          <ThemeToggle />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <LiveClock nowF={nowF} running={running} />
        {!st.clockAt && <span className="text-sm text-muted-foreground">(멈춤 · 적용하면 흐름)</span>}
        <ClockDialog meta={meta} st={st} truth={truth} setTruth={setTruth} onApply={onApply} shared />
      </div>
      {!session.user && (
        <ToggleGroup type="single" variant="outline" value={role} aria-label="보는 사람" onValueChange={(v) => v && onRole(v as Role)} className="w-fit">
          <ToggleGroupItem className={picked} value="manager"><UserCog strokeWidth={2} aria-hidden />설비관리자</ToggleGroupItem>
          <ToggleGroupItem className={picked} value="worker"><HardHat strokeWidth={2} aria-hidden />작업자</ToggleGroupItem>
        </ToggleGroup>
      )}
    </header>
  )
}
