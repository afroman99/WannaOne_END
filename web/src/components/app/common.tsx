import { ArrowLeft, ChevronDown, CircleAlert, CircleCheck, Info, TriangleAlert, X } from 'lucide-react'
import { useCallback, useState, type ComponentProps, type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
import { ToastCtx, type Tone } from '@/lib/feedback'
import { cn } from '@/lib/utils'

// 화면 공통 조각. COMPASS에 없는 것(접는 칸 · 여러 줄 입력 · 체크 · 알림)만 여기서 COMPASS 토큰으로 만든다

/** 접는 칸 (<details>): 펼칠 때만 안의 내용을 그린다 → 무거운 그래프는 열어야 계산 */
export function Disclosure({ title, icon, children, defaultOpen = false, className }: {
  title: ReactNode; icon?: ReactNode; children: ReactNode; defaultOpen?: boolean; className?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <details open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className={cn('group rounded-2xl border border-border', className)}>
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-2xl px-4 py-3 font-medium outline-none select-none hover:bg-black/5 focus-visible:ring-3 focus-visible:ring-ring/50 dark:hover:bg-white/5 [&::-webkit-details-marker]:hidden">
        {icon}
        <span className="flex-1">{title}</span>
        <ChevronDown className="size-4 text-muted-foreground transition-transform group-open:rotate-180" strokeWidth={2} aria-hidden />
      </summary>
      {open && <div className="flex flex-col gap-4 px-4 pt-1 pb-4">{children}</div>}
    </details>
  )
}

/** 여러 줄 입력 (Input과 같은 토큰) */
export function TextArea({ className, ...props }: ComponentProps<'textarea'>) {
  return (
    <textarea {...props}
      className={cn('min-h-24 w-full rounded-xl border border-input bg-transparent px-[calc(.625rem+4pt)] py-[calc(.25rem+4pt)] text-base outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30', className)} />
  )
}

/** 체크 칸 (체크리스트) */
export function Check({ id, label, checked, disabled, onChange }: {
  id: string; label: ReactNode; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label htmlFor={id} className={cn('flex items-start gap-3 rounded-xl px-2 py-2', disabled ? 'opacity-50' : 'cursor-pointer hover:bg-black/5 dark:hover:bg-white/5')}>
      <input id={id} type="checkbox" className="mt-0.5 size-5 shrink-0 accent-primary" checked={checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)} />
      <span className="leading-6">{label}</span>
    </label>
  )
}

export function Field({ label, htmlFor, children, hint }: { label: string; htmlFor?: string; children: ReactNode; hint?: string }) {
  return (
    <div className="grid gap-2">
      <label htmlFor={htmlFor} className="text-sm font-medium">{label}</label>
      {children}
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
  )
}

export const TONE = {
  red: { box: 'border-destructive/40 bg-destructive/10', text: 'text-destructive', Icon: CircleAlert },
  orange: { box: 'border-warning/50 bg-warning/10', text: 'text-warning', Icon: TriangleAlert },
  green: { box: 'border-success/45 bg-success/10', text: 'text-success', Icon: CircleCheck },
  blue: { box: 'border-primary/40 bg-primary/10', text: 'text-primary', Icon: Info },
} as const

/** 알림 상자 (경보·주의·정상 조치 안내, 오류·안내) */
export function Notice({ tone, title, children, className }: { tone: Tone; title?: ReactNode; children?: ReactNode; className?: string }) {
  const t = TONE[tone]
  return (
    <div role={tone === 'red' ? 'alert' : 'status'} className={cn('flex gap-3 rounded-2xl border p-4', t.box, className)}>
      <t.Icon className={cn('mt-0.5 size-5 shrink-0', t.text)} strokeWidth={2} aria-hidden />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {title && <p className={cn('font-semibold', t.text)}>{title}</p>}
        {children && <div className="text-foreground">{children}</div>}
      </div>
    </div>
  )
}

/** 지표 칸 (st.metric) */
export function Metric({ label, value, sub, onClick, className }: {
  label: ReactNode; value: ReactNode; sub?: ReactNode; onClick?: () => void; className?: string
}) {
  const body = (
    <>
      <CardHeader><CardDescription className="flex items-center gap-1">{label}</CardDescription></CardHeader>
      <CardContent className="flex flex-col gap-1">
        <p className="text-3xl font-semibold tabular-nums">{value}</p>
        {sub && <p className="text-sm text-muted-foreground">{sub}</p>}
      </CardContent>
    </>
  )
  if (onClick) {
    return (
      <Card size="sm" className={cn('cursor-pointer transition-colors hover:bg-black/5 dark:hover:bg-white/5', className)}>
        <button type="button" onClick={onClick} className="flex flex-col gap-(--card-spacing) rounded-container text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
          {body}
        </button>
      </Card>
    )
  }
  return <Card size="sm" className={className}>{body}</Card>
}

/** 경로 (← 위 단계 / 지금 화면) */
export function Crumb({ back, onBack, here }: { back: string; onBack: () => void; here?: ReactNode }) {
  return (
    <nav aria-label="경로" className="flex flex-wrap items-center gap-1">
      <Button variant="ghost" size="sm" className="-ml-2" onClick={onBack}>
        <ArrowLeft strokeWidth={2} aria-hidden />{back}
      </Button>
      {here && <><span className="text-muted-foreground" aria-hidden>/</span><span className="font-semibold">{here}</span></>}
    </nav>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-muted-foreground">{children}</p>
}

// 알림 (st.toast): 화면 오른쪽 아래에 잠깐

interface Toast { id: number; text: string; tone: Tone; long?: boolean }

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])
  const push = useCallback((text: string, tone: Tone = 'green', long = false) => {
    const id = Date.now() + Math.random()
    setItems((x) => [...x, { id, text, tone, long }])
    setTimeout(() => setItems((x) => x.filter((t) => t.id !== id)), long ? 10000 : 4000)
  }, [])
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed right-4 bottom-4 left-4 z-50 flex flex-col items-end gap-2 sm:left-auto sm:w-96">
        {items.map((t) => {
          const s = TONE[t.tone]
          return (
            <div key={t.id} role="status" className="pointer-events-auto flex w-full items-start gap-3 rounded-2xl border border-border bg-popover p-4 text-popover-foreground shadow-lg">
              <s.Icon className={cn('mt-0.5 size-5 shrink-0', s.text)} strokeWidth={2} aria-hidden />
              <p className="flex-1">{t.text}</p>
              <Button variant="ghost" size="icon-xs" aria-label="알림 닫기" onClick={() => setItems((x) => x.filter((y) => y.id !== t.id))}>
                <X strokeWidth={2} aria-hidden />
              </Button>
            </div>
          )
        })}
      </div>
    </ToastCtx.Provider>
  )
}
