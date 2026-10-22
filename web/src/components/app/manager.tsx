import {
  ArrowRight, Badge as BadgeIcon, CircleCheck, ClipboardList, Download, FileText, ListChecks, Pencil, RotateCcw, Save, Scale, Siren, Users,
  UserX, Undo2, X, Plus, Trash2, Radio,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'

import { Crumb, Empty, Field, Metric, Notice, TextArea } from '@/components/app/common'
import { useAction } from '@/lib/feedback'
import { DecisionPanel } from '@/components/app/decision'
import { StageDot } from '@/components/app/stage'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { picked } from '@/lib/styles'
import { api, clock, type Absence, type ManagerData, type ManagerRow, type Meta, type Task } from '@/lib/api'

const ORDER_VARIANT: Record<string, 'destructive' | 'secondary' | 'outline' | 'default'> = {
  '배정 전': 'destructive', 대기: 'default', '진행 중': 'default', 완료: 'secondary', 보류: 'outline', 중단: 'outline', '작업 없음': 'outline',
}
const STAGE_HELP = { 경보: '24시간 안 고장 위험 · 교체', 주의: '열화 감지 · 점검·부품 준비', 정상: '운전 계속' } as const

function EquipLine({ r, onPick, compact }: { r: ManagerRow; onPick: (sid: string) => void; compact?: boolean }) {
  return (
    <Button variant="ghost" onClick={() => onPick(r.sid)} className="h-auto w-full justify-start gap-3 py-2 text-left whitespace-normal">
      <StageDot stage={r.stage} />
      <span className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="font-semibold">{r.name}{!compact && <span className="ml-2 font-normal text-muted-foreground">{r.sid}</span>}</span>
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
          {compact && <span>{r.sid}</span>}<span>{r.stage}</span>
          {r.risk !== null && <span>24시간 내 고장 위험 {Math.round(r.risk * 100)}%</span>}
          {r.truthH !== null && <span>실제 고장까지 {r.truthH}h (정답)</span>}
          <Badge variant={ORDER_VARIANT[r.orderState] ?? 'outline'}>작업 지시: {r.orderState}</Badge>
        </span>
      </span>
    </Button>
  )
}

/** 설비 상태: 경보·주의·정상 칸마다 그 상태의 설비 전부 (칸 제목 → 넓은 목록) */
function EquipStatus({ m, onStage, onPick }: { m: ManagerData; onStage: (s: string) => void; onPick: (sid: string) => void }) {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {(['경보', '주의', '정상'] as const).map((s) => {
        const rows = m.rows.filter((r) => r.stage === s)
        return (
          <Card key={s}>
            <CardHeader>
              <button type="button" onClick={() => onStage(s)} className="flex flex-col items-start gap-1 rounded-xl text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
                <CardTitle className="flex items-center gap-2 text-base"><StageDot stage={s} />{s}<ArrowRight className="size-4" strokeWidth={2} aria-hidden /></CardTitle>
                <span className="text-4xl font-semibold tabular-nums">{s !== '주의' || m.caution ? m.counts[s] : '—'}</span>
                <CardDescription>{STAGE_HELP[s]} · 제목을 누르면 넓은 목록</CardDescription>
              </button>
            </CardHeader>
            <CardContent>
              <ul className="flex max-h-[28rem] flex-col gap-1 overflow-y-auto" aria-label={`${s} 설비 목록`}>
                {rows.length === 0 && <li className="py-2 text-sm text-muted-foreground">해당하는 설비가 없습니다.</li>}
                {rows.map((r) => <li key={r.sid}><EquipLine r={r} onPick={onPick} compact /></li>)}
              </ul>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

export function StageList({ m, stage, onBack, onOrders, onPick }: {
  m: ManagerData; stage: string; onBack: () => void; onOrders: () => void; onPick: (sid: string) => void
}) {
  const rows = m.rows.filter((r) => r.stage === stage)
  return (
    <section className="flex flex-col gap-4">
      <Crumb back="설비 상태" onBack={onBack} />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-2xl font-semibold">{stage} 설비 {rows.length}대</h2>
        <Button variant="outline" onClick={onOrders}><ClipboardList strokeWidth={2} aria-hidden />작업 지시 현황 보기</Button>
      </div>
      <p className="text-sm text-muted-foreground">
        {stage === '경보' ? `위험한 순서 · 작업 지시가 아직 없는 설비 ${rows.filter((r) => r.orderState === '배정 전').length}대 · 누르면 그 설비의 근거·비용과 배정이 열립니다`
          : '위험한 순서 · 누르면 그 설비 화면이 열립니다'}
      </p>
      <Card><CardContent className="flex flex-col gap-1">{rows.length ? rows.map((r) => <EquipLine key={r.sid} r={r} onPick={onPick} />) : <Empty>해당하는 설비가 없습니다.</Empty>}</CardContent></Card>
    </section>
  )
}

/** 배정할 시나리오 목록 → 고르면 바로 아래에 근거·비용 → 배정·보류 */
export function AssignList({ m, sid, onBack, onPick, panel }: {
  m: ManagerData; sid: string | null; onBack: () => void; onPick: (sid: string) => void; panel: ReactNode
}) {
  return (
    <section className="flex flex-col gap-4">
      <Crumb back="작업 지시 현황" onBack={onBack} />
      <h2 className="flex items-center gap-2 text-2xl font-semibold"><Siren className="size-6" strokeWidth={2} aria-hidden />배정할 경보 {m.queue.length}건 · 배정할 시나리오 고르기</h2>
      {m.queue.length === 0 ? <Notice tone="green">지금 배정할 경보가 없습니다.</Notice> : (
        <>
          <p className="text-sm text-muted-foreground">즉시 조치 → 정비팀 배정 순서 · 누르면 바로 아래에 그 설비의 근거와 비용이 열립니다</p>
          <Card><CardContent className="flex flex-col gap-1">
            {m.queue.map((q) => (
              <Button key={q.sid} variant={q.sid === sid ? 'secondary' : 'ghost'} onClick={() => onPick(q.sid)}
                aria-current={q.sid === sid ? 'true' : undefined} className="h-auto w-full justify-start gap-3 py-2 text-left whitespace-normal">
                <StageDot stage="경보" />
                <span className="flex flex-1 flex-wrap items-center gap-2">
                  <b>{q.name}</b>
                  {q.urgent && <Badge variant="destructive">즉시 조치</Badge>}
                  <span>추천: {q.best}</span>
                  <span className="text-muted-foreground">{q.rank !== null ? `정비팀 순서 ${q.rank}번` : '정기 교체 때'} · {q.sid}</span>
                </span>
              </Button>
            ))}
          </CardContent></Card>
        </>
      )}
      {sid && <div className="border-t border-border pt-6">{panel}</div>}
    </section>
  )
}

/** 작업 수정하기: 지시된 작업을 골라 수정하거나 배정을 취소한다. 수정·취소 모두 사유를 적어야 저장된다 */
function TaskEditor({ meta, tasks, tid, setTid, now, nowF, onClose, onDone }: {
  meta: Meta; tasks: Task[]; tid: number | null; setTid: (i: number) => void; now: number; nowF: () => number; onClose: () => void; onDone: () => void
}) {
  const cur = tasks.find((t) => t.id === tid) ?? tasks[0]
  const allPeople = Object.values(meta.teams).flat()
  const done = cur.status === '완료'
  const [st, setSt] = useState<string>(cur.status === '중단' ? '대기' : cur.status)
  const [team, setTeam] = useState(cur.team)
  const [members, setMembers] = useState(cur.members.split(',').filter((m) => meta.teams[cur.team].includes(m)))
  const [leader, setLeader] = useState(cur.leaderName || allPeople[0])
  const [priority, setPriority] = useState(cur.priority)
  const [due, setDue] = useState(String(Math.min(240, Math.round(cur.due_h))))
  const [guide, setGuide] = useState(cur.guide)
  const [reason, setReason] = useState(cur.reason)
  const lo = Math.min(Math.ceil(cur.started_h ?? cur.created_h), now)
  const [atDay, setAtDay] = useState(String(Math.floor(now / 24) + 1))
  const [atHour, setAtHour] = useState(String(now % 24))
  const pick = (opts: string[], v: string, d: string) => (opts.includes(v) ? v : d)
  const [outcome, setOutcome] = useState(pick(meta.outcomes, cur.outcome, '이상 발견'))
  const [judged, setJudged] = useState(pick(meta.judged, cur.judged, '구분 못함'))
  const [confirmed, setConfirmed] = useState(pick(meta.confirmed, cur.confirmed, '아직 모름'))
  const [details, setDetails] = useState(done ? cur.details : '')
  const [why, setWhy] = useState('')
  const [cancelWhy, setCancelWhy] = useState('')
  const { busy, run } = useAction(onDone)
  const pn = (x: string) => meta.partNames[x] ?? x
  const at = (Number(atDay) - 1) * 24 + Number(atHour)
  const save = () => run(() => api.edit(cur.id, {
    state: st, team, members, leader, priority, due_h: Number(due), guide, reason, why,
    at: st === '완료' && !done ? (at === now ? nowF() : at) : null, outcome, judged, confirmed, details,
  }))
  const days = Array.from({ length: Math.floor(now / 24) - Math.floor(lo / 24) + 1 }, (_, i) => Math.floor(lo / 24) + 1 + i)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg"><Pencil className="size-5" strokeWidth={2} aria-hidden />작업 수정</CardTitle>
        <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end">
          <Button variant="ghost" size="sm" onClick={onClose}><X strokeWidth={2} aria-hidden />닫기</Button>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Field label="수정할 작업" htmlFor="e-task">
          <Select value={String(cur.id)} onValueChange={(v) => setTid(Number(v))}>
            <SelectTrigger id="e-task" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{tasks.map((t) => <SelectItem key={t.id} value={String(t.id)}>#{t.id} · {t.name} · {t.title} · {t.status}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        {cur.status === '중단' ? (
          <Notice tone="orange" title={`작업자가 ${clock(cur.stopped_h)}에 중단한 작업입니다 · 사유: ${cur.stop_why}`}>
            다시 맡기려면 상태를 대기·진행 중으로 바꿔 저장하고, 맡기지 않으려면 아래에서 배정을 취소하세요.
          </Notice>
        ) : cur.action_why && <Notice tone="blue">실제 조치: {cur.action} · 지시와 다름: {cur.action_why}</Notice>}
        <div className="grid gap-2">
          <span id="e-state" className="text-sm font-medium">작업 상태</span>
          <ToggleGroup type="single" variant="outline" aria-labelledby="e-state" value={st} onValueChange={(v) => v && setSt(v)} className="w-fit">
            {meta.statuses.map((s) => <ToggleGroupItem className={picked} key={s} value={s}>{s}</ToggleGroupItem>)}
          </ToggleGroup>
          <p className="text-sm text-muted-foreground">대기·진행 중으로 되돌리면 완료 보고·중단이 지워집니다. 완료로 바꾸려면 완료 보고가 필요합니다.</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="담당 팀" htmlFor="e-team">
            <Select value={team} onValueChange={setTeam}>
              <SelectTrigger id="e-team" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{Object.keys(meta.teams).map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="우선순위" htmlFor="e-prio">
            <Select value={priority} onValueChange={setPriority}>
              <SelectTrigger id="e-prio" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{meta.priorities.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="시작 예정" htmlFor="e-due">
            <Select value={due} onValueChange={setDue}>
              <SelectTrigger id="e-due" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{Array.from({ length: 241 }, (_, h) => <SelectItem key={h} value={String(h)}>{clock(h)}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
        </div>
        <div className="grid gap-2">
          <span id="e-mem" className="text-sm font-medium">작업자 (팀을 바꾸면 저장할 때 새 팀원만 남깁니다)</span>
          <ToggleGroup type="multiple" variant="outline" size="sm" spacing={4} aria-labelledby="e-mem" value={members} onValueChange={setMembers} className="flex-wrap justify-start">
            {allPeople.map((m) => <ToggleGroupItem className={picked} key={m} value={m}>{m} · {meta.teamOf[m]}</ToggleGroupItem>)}
          </ToggleGroup>
        </div>
        <Field label="작업책임자 (작업자 중 한 명, 아니면 저장할 때 반장 · 첫 사람으로 정해집니다)" htmlFor="e-lead">
          <Select value={leader} onValueChange={setLeader}>
            <SelectTrigger id="e-lead" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{allPeople.map((m) => <SelectItem key={m} value={m}>{m} · {meta.teamOf[m]}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <Field label="점검 가이드" htmlFor="e-guide"><TextArea id="e-guide" rows={5} value={guide} maxLength={meta.limits.guide} onChange={(e) => setGuide(e.target.value)} /></Field>
        <Field label="추천 이유" htmlFor="e-reason"><TextArea id="e-reason" value={reason} maxLength={meta.limits.reason} onChange={(e) => setReason(e.target.value)} /></Field>
        {st === '완료' && (
          <div className="flex flex-col gap-3 rounded-2xl border border-border p-4">
            <p className="font-semibold">완료 보고{done ? ` (완료 ${clock(cur.completed_h)})` : ''}</p>
            {!done && (
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="완료 시각 · 일차" htmlFor="e-at-d">
                  <Select value={atDay} onValueChange={setAtDay}>
                    <SelectTrigger id="e-at-d" className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>{days.map((x) => <SelectItem key={x} value={String(x)}>{x}일차</SelectItem>)}</SelectContent>
                  </Select>
                </Field>
                <Field label="시" htmlFor="e-at-h">
                  <Select value={atHour} onValueChange={setAtHour}>
                    <SelectTrigger id="e-at-h" className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>{Array.from({ length: 24 }, (_, h) => <SelectItem key={h} value={String(h)}>{String(h).padStart(2, '0')}:00</SelectItem>)}</SelectContent>
                  </Select>
                </Field>
              </div>
            )}
            <div className="grid gap-3 sm:grid-cols-3">
              {([['작업 결과', meta.outcomes, outcome, setOutcome], ['점검으로 판단한 부품', meta.judged, judged, setJudged],
                ['교체 뒤 확인된 원인 부품', meta.confirmed, confirmed, setConfirmed]] as const).map(([label, opts, v, set], i) => (
                <Field key={label} label={label} htmlFor={`e-r${i}`}>
                  <Select value={v} onValueChange={set}>
                    <SelectTrigger id={`e-r${i}`} className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>{opts.map((o) => <SelectItem key={o} value={o}>{pn(o)}</SelectItem>)}</SelectContent>
                  </Select>
                </Field>
              ))}
            </div>
            <Field label="수행 내용" htmlFor="e-det"><TextArea id="e-det" value={details} maxLength={meta.limits.details} onChange={(e) => setDetails(e.target.value)} /></Field>
          </div>
        )}
        <Field label="수정 사유 (필수)" htmlFor="e-why">
          <Input id="e-why" value={why} maxLength={150} placeholder="예: 정비 1팀 인원이 부족해 2팀으로 넘김" onChange={(e) => setWhy(e.target.value)} />
        </Field>
        <Button className="w-fit" disabled={busy || !why.trim()} onClick={save}><Save strokeWidth={2} aria-hidden />수정 저장</Button>
        <div className="border-t border-border pt-4">
          {done ? <p className="text-sm text-muted-foreground">완료된 작업은 배정을 취소할 수 없습니다. 위에서 상태를 대기·진행 중으로 되돌린 뒤 취소하세요.</p> : (
            <div className="flex flex-col gap-3">
              <Field label={`#${cur.id} 배정 취소 사유 (필수)`} htmlFor="e-cancel">
                <Input id="e-cancel" value={cancelWhy} maxLength={150} placeholder="예: 정기 교체 일정과 겹쳐 그때 함께 하기로 함" onChange={(e) => setCancelWhy(e.target.value)} />
              </Field>
              <Button variant="destructive" className="w-fit" disabled={busy || !cancelWhy.trim()} title="작업 목록에서 빠지고, 설비는 다시 배정할 경보로 돌아갑니다"
                onClick={() => run(() => api.cancel(cur.id, cancelWhy))}>
                <Undo2 strokeWidth={2} aria-hidden />배정 취소
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

const FILTERS = ['전체', '중단', '지금', '지연', '예정', '완료'] as const

/** 작업 지시 현황: 지표 · 먼저 볼 설비 · 작업 목록 · 작업 수정 */
function Orders({ meta, m, onAssign, onPickAssign, editTid, setEditTid, now, nowF, onChanged }: {
  meta: Meta; m: ManagerData; onAssign: () => void; onPickAssign: (sid: string) => void
  editTid: number | null | undefined; setEditTid: (t: number | null | undefined) => void; now: number; nowF: () => number; onChanged: () => void
}) {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('전체')
  const shown = filter === '전체' ? m.tasks : m.tasks.filter((t) => t.bucket === filter)
  const first = m.queue[0]
  const pn = (x: string) => meta.partNames[x] ?? x
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Metric label={<>배정할 경보<ArrowRight className="size-4" strokeWidth={2} aria-hidden /></>} value={m.queue.length} onClick={onAssign}
          sub="경보인데 아직 작업이 없는 설비" />
        <Metric label="진행 중" value={m.taskCounts['진행 중']} />
        <Metric label="지연" value={m.taskCounts['지연']} sub="예정 시각이 지났는데 시작 전" />
        <Metric label="완료" value={m.taskCounts['완료']} />
        <Metric label="중단" value={m.taskCounts['중단']} sub="현장에서 중단한 작업" />
        <Metric label="정비 중" value={m.inMnt} sub="정비 중 · 예측 대기" />
      </div>
      <Card>
        <CardContent className="flex flex-col gap-2">
          {first ? (
            <>
              <p className="text-sm text-muted-foreground">지금 먼저 볼 설비 · 배정할 경보 {m.queue.length}건{m.held ? ` · 보류 ${m.held}건` : ''}</p>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h3 className="text-2xl font-semibold">{first.name} — {first.best}</h3>
                <Button onClick={() => onPickAssign(first.sid)}><ArrowRight strokeWidth={2} aria-hidden />이 설비 보기</Button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {first.urgent && <Badge variant="destructive"><Siren strokeWidth={2} aria-hidden />즉시 조치</Badge>}
                <Badge variant="secondary">{first.rank !== null ? `정비팀 순서 ${first.rank}번` : '정기 교체 때'}</Badge>
                <span className="text-sm text-muted-foreground">시나리오 {first.sid}</span>
              </div>
            </>
          ) : <p className="flex items-center gap-2 font-semibold"><CircleCheck className="size-5 text-success" strokeWidth={2} aria-hidden />지금 배정할 경보가 없습니다</p>}
          <p className="text-sm text-muted-foreground">열린 작업 {m.busy}건 · 팀별 {Object.entries(m.load).map(([k, v]) => `${k} ${v}건`).join(', ')}</p>
        </CardContent>
      </Card>
      {m.tasks.length === 0 ? <p className="text-sm text-muted-foreground">아직 지시된 작업이 없습니다. "배정할 경보"를 눌러 배정하세요.</p> : (
        <>
          <ToggleGroup type="single" variant="outline" value={filter} aria-label="작업 구분" onValueChange={(v) => v && setFilter(v as typeof filter)} className="flex-wrap">
            {FILTERS.map((f) => <ToggleGroupItem className={picked} key={f} value={f}>{f}</ToggleGroupItem>)}
          </ToggleGroup>
          <div className="overflow-x-auto rounded-2xl border border-border">
            <table className="w-full min-w-[64rem] text-sm">
              <thead className="bg-muted text-left"><tr>
                {['번호', '설비', '작업', '팀', '작업자', '작업책임자', '우선순위', '상태', '구분', '체크', '예정', '실제 조치', '완료 결과', '확인된 원인', '사유'].map((h) => (
                  <th key={h} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {shown.map((t) => (
                  <tr key={t.id} className="border-t border-border align-top">
                    <td className="px-3 py-2 tabular-nums">{t.id}</td><td className="px-3 py-2 whitespace-nowrap">{t.name}</td>
                    <td className="px-3 py-2">{t.title}</td><td className="px-3 py-2 whitespace-nowrap">{t.team}</td>
                    <td className="px-3 py-2">{t.members.split(',').filter(Boolean).join(', ')}</td><td className="px-3 py-2 whitespace-nowrap">{t.leaderName}</td>
                    <td className="px-3 py-2">{t.priority}</td><td className="px-3 py-2 whitespace-nowrap">{t.status}</td><td className="px-3 py-2">{t.bucket}</td>
                    <td className="px-3 py-2 tabular-nums">{t.checkCount}/{meta.checklist.total}</td><td className="px-3 py-2 whitespace-nowrap">{clock(t.due_h)}</td>
                    <td className="px-3 py-2">{t.status === '완료' ? (t.action || meta.actions[0]) : ''}</td><td className="px-3 py-2">{t.outcome}</td>
                    <td className="px-3 py-2">{pn(t.confirmed)}</td>
                    <td className="px-3 py-2">{t.status === '중단' ? `중단: ${t.stop_why}` : t.action_why ? `지시와 다름: ${t.action_why}` : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap gap-2">
            {editTid === undefined && <Button variant="outline" onClick={() => setEditTid(null)}><Pencil strokeWidth={2} aria-hidden />작업 수정하기</Button>}
            <Button variant="outline" asChild><a href="/api/tasks.csv" download><Download strokeWidth={2} aria-hidden />작업 이력 CSV</a></Button>
          </div>
          {editTid !== undefined && (
            <TaskEditor key={editTid ?? m.tasks[0].id} meta={meta} tasks={m.tasks} tid={editTid} setTid={setEditTid} now={now} nowF={nowF}
              onClose={() => setEditTid(undefined)} onDone={onChanged} />
          )}
        </>
      )}
    </div>
  )
}

/** 인원 현황: 팀마다 지금 출근 · 사람마다 부재 종류·기간 */
function Staff({ meta, m, now, version, onChanged }: { meta: Meta; m: ManagerData; now: number; version: number; onChanged: () => void }) {
  const [rows, setRows] = useState<Absence[] | null>(null)
  const { busy, run } = useAction(onChanged)
  useEffect(() => { api.absences().then((x) => setRows(x.rows)).catch(() => setRows([])) }, [version])
  const people = Object.values(meta.teams).flat()
  const dh = (h: number) => [Math.floor(h / 24) + 1, h % 24] as const
  const set = (i: number, patch: Partial<Absence>) => setRows((r) => r!.map((x, j) => (j === i ? { ...x, ...patch } : x)))
  const sel = (id: string, v: string, opts: [string, string][], on: (v: string) => void, label: string) => (
    <Select value={v} onValueChange={on}>
      <SelectTrigger id={id} aria-label={label} size="sm" className="w-full min-w-24"><SelectValue /></SelectTrigger>
      <SelectContent>{opts.map(([k, l]) => <SelectItem key={k} value={k}>{l}</SelectItem>)}</SelectContent>
    </Select>
  )
  const hours: [string, string][] = Array.from({ length: 24 }, (_, h) => [String(h), `${String(h).padStart(2, '0')}시`])
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {m.staff.map((s) => {
          const cap = s.on.length * m.crewDayH
          return (
            <Metric key={s.team} label={`${s.team} 지금 출근`} value={`${s.on.length} / ${s.total}명`}
              sub={`하루 작업 한도 ${cap}인·시 (교체 ${Math.floor(cap / meta.work.replace)}번) · ` +
                (Object.keys(s.absent).length ? '부재: ' + Object.entries(s.absent).map(([k, v]) => `${k}(${v})`).join(', ') : '전원 출근')} />
          )
        })}
      </div>
      <h3 className="flex items-center gap-2 text-lg font-semibold"><BadgeIcon className="size-5" strokeWidth={2} aria-hidden />인원별 부재</h3>
      <p className="text-sm text-muted-foreground">부재 종류를 고르고 기간(시연 시각 기준 일차 · 시)을 정한 뒤 저장하세요. 끝 시각은 그 시각부터 출근입니다. 한 사람이 두 번 이상 빠지면 줄을 더합니다. 지금 {clock(now)} 기준</p>
      {rows === null ? <p className="text-muted-foreground">불러오는 중...</p> : (
        <div className="overflow-x-auto rounded-2xl border border-border">
          <table className="w-full min-w-[56rem] text-sm">
            <thead className="bg-muted text-left"><tr>
              {['이름', '팀', '부재 종류', '시작 일차', '시작 시', '끝 일차', '끝 시', '메모', ''].map((h, i) => <th key={i} scope="col" className="px-3 py-2 font-medium">{h}</th>)}
            </tr></thead>
            <tbody>
              {rows.length === 0 && <tr><td colSpan={9} className="px-3 py-4 text-center text-muted-foreground">부재 없음 (전원 출근)</td></tr>}
              {rows.map((r, i) => {
                const [d1, h1] = dh(r.start_h)
                const [d2, h2] = dh(r.end_h)
                return (
                  <tr key={i} className="border-t border-border">
                    <td className="px-2 py-2">{sel(`ab-n${i}`, r.name, people.map((p) => [p, p]), (v) => set(i, { name: v }), '이름')}</td>
                    <td className="px-2 py-2 whitespace-nowrap">{meta.teamOf[r.name]}</td>
                    <td className="px-2 py-2">{sel(`ab-k${i}`, r.kind, meta.absenceKinds.map((k) => [k, k]), (v) => set(i, { kind: v }), '부재 종류')}</td>
                    <td className="px-2 py-2">{sel(`ab-d1${i}`, String(d1), [1, 2, 3, 4].map((d) => [String(d), `${d}일차`]), (v) => set(i, { start_h: (Number(v) - 1) * 24 + h1 }), '시작 일차')}</td>
                    <td className="px-2 py-2">{sel(`ab-h1${i}`, String(h1), hours, (v) => set(i, { start_h: (d1 - 1) * 24 + Number(v) }), '시작 시')}</td>
                    <td className="px-2 py-2">{sel(`ab-d2${i}`, String(d2), [1, 2, 3, 4, 5].map((d) => [String(d), `${d}일차`]), (v) => set(i, { end_h: (Number(v) - 1) * 24 + h2 }), '끝 일차')}</td>
                    <td className="px-2 py-2">{sel(`ab-h2${i}`, String(h2), hours, (v) => set(i, { end_h: (d2 - 1) * 24 + Number(v) }), '끝 시')}</td>
                    <td className="px-2 py-2"><Input aria-label="메모" value={r.note} maxLength={100} onChange={(e) => set(i, { note: e.target.value })} /></td>
                    <td className="px-2 py-2"><Button variant="ghost" size="icon-sm" aria-label={`${r.name} 부재 줄 지우기`} onClick={() => setRows(rows.filter((_, j) => j !== i))}><Trash2 strokeWidth={2} aria-hidden /></Button></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" onClick={() => setRows([...(rows ?? []), { name: people[0], kind: meta.absenceKinds[0], start_h: Math.floor(now / 24) * 24 + 9, end_h: Math.floor(now / 24) * 24 + 17, note: '' }])}>
          <Plus strokeWidth={2} aria-hidden />줄 더하기
        </Button>
        <Button disabled={busy || rows === null} onClick={() => run(() => api.saveAbsences(rows!))}><Save strokeWidth={2} aria-hidden />저장</Button>
        <Button variant="outline" disabled={busy} onClick={() => run(() => api.saveAbsences([]))}><RotateCcw strokeWidth={2} aria-hidden />전원 출근으로 되돌리기</Button>
        <Button variant="outline" disabled={busy} onClick={() => run(() => api.absencePreset())}><UserX strokeWidth={2} aria-hidden />예시 채우기: 인원 부족 (2일차 정비 2팀 대부분 부재)</Button>
      </div>
      <p className="text-sm text-muted-foreground">부재 중인 사람은 작업자 고르기에서 빠지고, 팀 하루 작업 한도(출근 인원 × 8시간)와 칸별 비용(인·시)에 바로 반영됩니다.</p>
    </div>
  )
}

const HOME = [
  { key: 'equip', label: '설비 상태', Icon: Radio },
  { key: 'orders', label: '작업 지시 현황', Icon: ClipboardList },
  { key: 'staff', label: '인원 현황', Icon: Users },
] as const

/** 설비관리자 화면 (주소 #/m/...) */
export function Manager({ meta, route, go, now, nowF, version, truth, onChanged }: {
  meta: Meta; route: string[]; go: (p: string) => void; now: number; nowF: () => number; version: number; truth: boolean; onChanged: () => void
}) {
  const [m, setM] = useState<ManagerData | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [editTid, setEditTid] = useState<number | null | undefined>(undefined)
  useEffect(() => {
    let live = true
    api.manager(truth).then((x) => live && setM(x)).catch((e: Error) => live && setErr(e.message))
    return () => { live = false }
  }, [version, truth])
  const [, a, b] = route
  const openEdit = (tid: number) => { setEditTid(tid); go('m/orders') }
  const panel = (sid: string) => (
    <DecisionPanel meta={meta} sid={sid} now={now} nowF={nowF} version={version} truth={truth} onChanged={onChanged}
      onEdit={openEdit} onDetail={(s) => go(`m/page/설비 상세/${s}`)} />
  )
  if (err) return <Notice tone="red">{err}</Notice>
  if (!m) return <p className="text-muted-foreground">불러오는 중...</p>
  if (a === 'stage' && b) return <StageList m={m} stage={b} onBack={() => go('m')} onOrders={() => go('m/orders')} onPick={(s) => go(`m/eq/${s}/${b}`)} />
  if (a === 'assign') return <AssignList m={m} sid={b ?? null} onBack={() => go('m/orders')} onPick={(s) => go(`m/assign/${s}`)} panel={b ? panel(b) : null} />
  if (a === 'eq' && b) {
    const from = route[3]
    return (
      <div className="flex flex-col gap-4">
        <Crumb back={from ? `${from} 설비 목록` : '정비 의사결정'} onBack={() => go(from ? `m/stage/${from}` : 'm')} />
        {panel(b)}
      </div>
    )
  }
  const home = a === 'orders' ? 'orders' : a === 'staff' ? 'staff' : 'equip'
  return (
    <section className="flex flex-col gap-4" aria-labelledby="mh-title">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="mh-title" className="flex items-center gap-2 text-2xl font-semibold"><Scale className="size-6" strokeWidth={2} aria-hidden />정비 의사결정</h2>
        <div className="flex flex-wrap gap-1">
          {[['정책·비용', Scale], ['부품 손익', FileText], ['모델 정보·한계', ListChecks]].map(([k, Icon]) => {
            const I = Icon as typeof Scale
            return <Button key={k as string} variant="ghost" size="sm" onClick={() => go(`m/page/${k}`)}><I strokeWidth={2} aria-hidden />{k as string}</Button>
          })}
        </div>
      </div>
      <ToggleGroup type="single" variant="outline" value={home} aria-label="홈 범주" onValueChange={(v) => v && go(v === 'equip' ? 'm' : `m/${v}`)} className="w-fit flex-wrap">
        {HOME.map(({ key, label, Icon }) => <ToggleGroupItem className={picked} key={key} value={key}><Icon strokeWidth={2} aria-hidden />{label}</ToggleGroupItem>)}
      </ToggleGroup>
      {home === 'equip' && <EquipStatus m={m} onStage={(s) => go(`m/stage/${s}`)} onPick={(s) => go(`m/eq/${s}`)} />}
      {home === 'orders' && <Orders meta={meta} m={m} onAssign={() => go('m/assign')} onPickAssign={(s) => go(`m/assign/${s}`)}
        editTid={editTid} setEditTid={setEditTid} now={now} nowF={nowF} onChanged={onChanged} />}
      {home === 'staff' && <Staff meta={meta} m={m} now={now} version={version} onChanged={onChanged} />}
    </section>
  )
}
