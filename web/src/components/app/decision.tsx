import { Calculator, Factory, Info, Monitor, Pause, Pencil, Send, SlidersHorizontal, Truck, TriangleAlert, Users, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { Analysis, OntologyDisclosure } from '@/components/app/analysis'
import { Disclosure, Field, Notice, TextArea } from '@/components/app/common'
import { useAction, type Tone } from '@/lib/feedback'
import { StageBadge } from '@/components/app/stage'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { picked } from '@/lib/styles'
import { api, clock, hms, money, type Card as Opt, type Decision, type Meta } from '@/lib/api'

const ORDER_TONE: Record<string, 'destructive' | 'secondary' | 'outline'> = { '배정 전': 'destructive', 보류: 'outline', 중단: 'outline' }
const JOB_NOTE: Record<string, [Tone, string]> = {
  '작업 완료': ['green', '작업 완료 · 재가동 뒤 새 예측이 나올 때까지 경보 표시가 남습니다'],
  '작업 있음': ['blue', '작업이 이미 배정돼 있습니다 · 작업 지시 현황에서 확인'],
  '작업 시작': ['blue', '작업 진행 중'],
  보류: ['blue', '보류한 설비 · 다시 배정하려면 아래에서 배정'],
  승인: ['blue', '승인됨 · 작업 배정 전'],
  '작업 중단': ['orange', '작업자가 중단한 설비 · 사유는 작업 지시 현황에서 확인, 다시 배정할 수 있습니다'],
}

/** 1초씩 줄어드는 고장 예상 타이머 (정각의 추정값에서 흐르는 시각만큼 뺀다) */
function Countdown({ deadline, nowF }: { deadline: number; nowF: () => number }) {
  const [, tick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => tick((x) => x + 1), 1000)
    return () => clearInterval(id)
  }, [])
  return <span className="tabular-nums">{hms(Math.max(0, deadline - nowF()))}</span>
}

/** 칸 하나에서 고르는 팀·작업자·작업책임자 */
interface Pick { team: string; members: string[]; leader: string }

function CrewPicker({ meta, d, pick, onChange, kind }: { meta: Meta; d: Decision; pick: Pick; onChange: (p: Pick) => void; kind: string }) {
  const avail = meta.teams[pick.team].filter((m) => !(m in d.absent))
  const off = meta.teams[pick.team].filter((m) => m in d.absent)
  const leaders = new Set(Object.values(meta.leaders))
  const setTeam = (team: string) => {
    const a = meta.teams[team].filter((m) => !(m in d.absent)).slice(0, 4)
    onChange({ team, members: a, leader: a.find((m) => leaders.has(m)) ?? a[0] ?? '' })
  }
  const setMembers = (members: string[]) =>
    onChange({ ...pick, members, leader: members.includes(pick.leader) ? pick.leader : (members.find((m) => leaders.has(m)) ?? members[0] ?? '') })
  return (
    <div className="flex flex-col gap-3">
      <Field label="담당 팀" htmlFor={`team-${kind}`}>
        <Select value={pick.team} onValueChange={setTeam}>
          <SelectTrigger id={`team-${kind}`} className="w-full"><SelectValue /></SelectTrigger>
          <SelectContent>{Object.keys(meta.teams).map((t) => <SelectItem key={t} value={t}>{t} · 열린 작업 {d.load[t]}건</SelectItem>)}</SelectContent>
        </Select>
      </Field>
      <div className="grid gap-2">
        <span id={`mem-${kind}`} className="text-sm font-medium">작업자</span>
        <ToggleGroup type="multiple" variant="outline" size="sm" spacing={4} aria-labelledby={`mem-${kind}`} value={pick.members}
          onValueChange={setMembers} className="flex-wrap justify-start">
          {avail.map((m) => (
            <ToggleGroupItem className={picked} key={m} value={m} aria-label={m}>
              {m}{leaders.has(m) ? ' (반장)' : ''}{d.busyPeople.includes(m) ? ' (작업 중)' : ''}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        {off.length > 0 && <p className="text-sm text-muted-foreground">부재: {off.map((m) => `${m}(${d.absent[m]})`).join(', ')}</p>}
      </div>
      {pick.members.length > 0 && (
        <Field label="작업책임자" htmlFor={`lead-${kind}`} hint="정비 중 10대 안전철칙을 확인·체크하는 사람 (고른 작업자 중)">
          <Select value={pick.leader} onValueChange={(leader) => onChange({ ...pick, leader })}>
            <SelectTrigger id={`lead-${kind}`} className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{pick.members.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
      )}
    </div>
  )
}

/** 보류 · 부품 보충 발주 · 두 부품 긴급 조달 */
function OrderButtons({ d, best, onDone }: { d: Decision; best: string; onDone: () => void }) {
  const { busy, run } = useAction(onDone)
  if (!['경보', '주의'].includes(d.stage)) return null
  const coming = new Set(d.transit.map((x) => x.part))
  const rush = ['valve', 'oscillator'].filter((q) => !coming.has(q))
  return (
    <div className="flex flex-wrap gap-2">
      {d.alarm && (
        <Button variant="outline" disabled={busy} title="지금은 배정하지 않고 지켜봅니다. 배정할 경보 수에서 빠집니다."
          onClick={() => run(() => api.act('보류', d.sid, { option: best }))}>
          <Pause strokeWidth={2} aria-hidden />보류
        </Button>
      )}
      {(['valve', 'oscillator'] as const).map((q) => (
        <Button key={q} variant="outline" disabled={busy || coming.has(q)}
          onClick={() => run(() => api.act('부품 발주', d.sid, { part: q, option: '정상' }))}>
          <Truck strokeWidth={2} aria-hidden />{q === 'valve' ? '밸브' : '진동장치'} {coming.has(q) ? '조달 중' : `보충 발주 (${d.lead.toFixed(0)}시간)`}
        </Button>
      ))}
      <Button variant="outline" disabled={busy || rush.length === 0} title={`${d.rushH.toFixed(0)}시간 안에 도착, 웃돈 부품값의 ${Math.round(d.rushMk * 100)}%`}
        onClick={() => run(() => api.orderRush(d.sid))}>
        <Zap strokeWidth={2} aria-hidden />두 부품 긴급 조달{rush.length ? ` (${d.rushH.toFixed(0)}시간)` : ' · 조달 중'}
      </Button>
    </div>
  )
}

/** 경보가 아닌 설비의 배정 칸 (작업명·우선순위·시작 시각·점검 가이드) */
function SimpleAssign({ meta, d, now, onDone }: { meta: Meta; d: Decision; now: number; onDone: () => void }) {
  const rec = d.recommendFallback ?? '상태 점검'
  const [team, setTeam] = useState(d.team0)
  const [title, setTitle] = useState(`${d.sid} ${rec}`)
  const [priority, setPriority] = useState(d.defaultPriority ?? '보통')
  const [due, setDue] = useState(String(now))
  const [guide, setGuide] = useState(meta.defaultGuide)
  const [reason, setReason] = useState(`${d.sid} · ${d.stageLabel} · 추천 ${rec}` + (d.risk !== null ? ` · 24시간 안 고장 확률 ${d.risk.toFixed(2)}` : ''))
  const { busy, run } = useAction(onDone)
  return (
    <div className="flex flex-col gap-4">
      <p className="font-semibold">배정: {d.name} {rec}</p>
      <div className="grid gap-3 sm:grid-cols-[1.3fr_1fr] sm:items-end">
        <Field label="담당 팀" htmlFor="s-team">
          <Select value={team} onValueChange={setTeam}>
            <SelectTrigger id="s-team" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{Object.keys(meta.teams).map((t) => <SelectItem key={t} value={t}>{t} · 열린 작업 {d.load[t]}건</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <Button disabled={busy} onClick={() => run(() => api.assign({ sid: d.sid, team, title, priority, due_h: Number(due), reason, option: rec, guide }))}>
          <Send strokeWidth={2} aria-hidden />작업자 화면으로 배정
        </Button>
      </div>
      <Disclosure title="작업 내용 바꾸기 (작업명 · 우선순위 · 시작 시각 · 점검 가이드)">
        <Field label="작업명" htmlFor="s-title"><Input id="s-title" value={title} maxLength={meta.limits.title} onChange={(e) => setTitle(e.target.value)} /></Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="우선순위" htmlFor="s-prio">
            <Select value={priority} onValueChange={setPriority}>
              <SelectTrigger id="s-prio" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{meta.priorities.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="시작 예정" htmlFor="s-due">
            <Select value={due} onValueChange={setDue}>
              <SelectTrigger id="s-due" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{Array.from({ length: 241 }, (_, h) => <SelectItem key={h} value={String(h)}>{clock(h)}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
        </div>
        <Field label="점검 가이드" htmlFor="s-guide"><TextArea id="s-guide" rows={6} value={guide} maxLength={meta.limits.guide} onChange={(e) => setGuide(e.target.value)} /></Field>
        <Field label="추천 이유" htmlFor="s-reason"><TextArea id="s-reason" value={reason} maxLength={meta.limits.reason} onChange={(e) => setReason(e.target.value)} /></Field>
      </Disclosure>
      <p className="text-sm text-muted-foreground">배정하면 그 팀 작업자 화면의 "지금 할 일"에 바로 나타납니다.</p>
    </div>
  )
}

/** 설비 하나의 정비 판단 (dashboard.py decision_panel) */
export function DecisionPanel({ meta, sid, now, nowF, version, truth, onChanged, onEdit, onDetail }: {
  meta: Meta; sid: string; now: number; nowF: () => number; version: number; truth: boolean
  onChanged: () => void; onEdit: (tid: number) => void; onDetail: (sid: string) => void
}) {
  const [d, setD] = useState<Decision | null>(null)
  const [picks, setPicks] = useState<Record<Opt['kind'], Pick> | null>(null)
  const [guide, setGuide] = useState(meta.defaultGuide)
  const [note, setNote] = useState('')
  const { busy, run } = useAction(onChanged)
  const crew = useMemo(() => picks ? { now: picks.now.members.length, plan: picks.plan.members.length, run: picks.run.members.length } : {}, [picks])

  useEffect(() => { setPicks(null); setGuide(meta.defaultGuide); setNote('') }, [sid, meta.defaultGuide])
  useEffect(() => {
    let live = true
    api.decision(sid, crew).then((x) => {
      if (!live) return
      setD(x)
      setPicks((p) => {
        if (p) return p
        const a = meta.teams[x.team0].filter((m) => !(m in x.absent)).slice(0, 4)
        const lead = a.find((m) => Object.values(meta.leaders).includes(m)) ?? a[0] ?? ''
        const one = { team: x.team0, members: a, leader: lead }
        return { now: one, plan: one, run: one }
      })
    }).catch(() => {})
    return () => { live = false }
  }, [sid, crew, version, meta.teams, meta.leaders])

  if (!d || d.sid !== sid || !picks) return <p className="text-muted-foreground">불러오는 중...</p>
  const note_ = d.job ? JOB_NOTE[d.job] : undefined
  const cards = d.cards ?? []
  const assignCard = (c: Opt) => {
    const p = picks[c.kind]
    return run(() => api.assign({
      sid, team: p.team, members: p.members, leader: p.leader, title: c.title, priority: c.priority,
      due_h: Math.min(240, Math.max(0, now + (c.start ?? 0))), option: c.label, guide,
      reason: `${sid} · ${d.stageLabel} · ${c.label} 선택 · 기대 비용 ${money(c.cost, d.unit)}` + (note.trim() ? ` · ${note.trim()}` : ''),
    }))
  }

  return (
    <section className="flex flex-col gap-6" aria-labelledby="dp-title">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <Factory className="size-6" strokeWidth={2} aria-hidden />
          <h2 id="dp-title" className="text-3xl font-semibold">{d.name}</h2>
          <Button variant="ghost" size="sm" onClick={() => onDetail(sid)}><Monitor strokeWidth={2} aria-hidden />설비 페이지</Button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StageBadge stage={d.stage} />
          <Badge variant={ORDER_TONE[d.orderState] ?? 'secondary'}>작업 지시: {d.orderState}</Badge>
          {d.risk !== null && <span>24시간 내 고장 위험 <b>{Math.round(d.risk * 100)}%</b></span>}
          <span className="text-sm text-muted-foreground">시나리오 {sid} · {clock(now)} 기준</span>
        </div>
        {note_ && <Notice tone={note_[0]}>{note_[1]}</Notice>}
      </div>

      <Notice tone={d.guide.tone} title={d.guide.title}>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
          <ul className="flex flex-1 list-disc flex-col gap-1 pl-5">
            {d.guide.body.map((x, i) => <li key={i} className={i === 0 ? 'list-none -ml-5' : ''}>{x}</li>)}
          </ul>
          {d.alarm && (
            <div className="shrink-0 sm:w-48 sm:text-right">
              <p className="text-sm">고장 예상까지</p>
              <p className="text-4xl font-semibold">{d.remainDeadline != null ? <Countdown deadline={d.remainDeadline} nowF={nowF} /> : '—'}</p>
              <p className="text-sm text-muted-foreground">추정 · 정각마다 다시 계산</p>
            </div>
          )}
        </div>
        {d.procurement && <p className="mt-2 text-sm text-muted-foreground">{d.procurement}</p>}
      </Notice>
      <Disclosure title="분석 근거" icon={<Info className="size-4" strokeWidth={2} aria-hidden />}>
        <Analysis meta={meta} sid={sid} truth={truth} version={version} />
      </Disclosure>
      <OntologyDisclosure sid={sid} version={version} />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{d.name} · 정비 판단</CardTitle>
          {d.alarm && <CardDescription>비교 기간: {d.horizonLabel}</CardDescription>}
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {d.alarm ? (
            <>
              <div className="grid gap-4 lg:grid-cols-3">
                {cards.map((c) => (
                  <Card key={c.kind} size="sm" className={c.kind === d.recKind ? 'ring-2 ring-primary' : ''}>
                    <CardHeader>
                      <CardDescription title={c.help}>{c.label}{c.kind === d.recKind ? ' · 추천' : ''}</CardDescription>
                      <p className="text-3xl font-semibold tabular-nums">{c.cost !== null ? money(c.cost, d.unit) : '—'}</p>
                    </CardHeader>
                    <CardContent className="flex flex-col gap-3">
                      <p className="text-sm text-muted-foreground">{c.sub}</p>
                      {c.overSafety && c.p !== null && (
                        <Badge variant="outline" className="whitespace-normal"><TriangleAlert className="text-warning" strokeWidth={2} aria-hidden />
                          안전 기준 초과: 고장 확률 {Math.round(c.p * 100)}% &gt; {Math.round(d.safety * 100)}%</Badge>
                      )}
                      {d.canAssign && c.cost !== null && (
                        <>
                          <CrewPicker meta={meta} d={d} kind={c.kind} pick={picks[c.kind]} onChange={(p) => setPicks({ ...picks, [c.kind]: p })} />
                          {picks[c.kind].members.length > 0 && picks[c.kind].members.length < 4 && (
                            <Badge variant="outline" className="whitespace-normal"><Users className="text-warning" strokeWidth={2} aria-hidden />
                              인원 부족 (4명 기준) · {c.kind === 'run' ? '고장 복구' : '교체'} {(c.workH / picks[c.kind].members.length).toFixed(1)}시간</Badge>
                          )}
                        </>
                      )}
                    </CardContent>
                    {d.canAssign && c.cost !== null && (
                      <CardFooter>
                        <Button className="w-full" variant={c.kind === d.recKind ? 'default' : 'outline'} disabled={busy || picks[c.kind].members.length === 0}
                          onClick={() => assignCard(c)}>
                          <Send strokeWidth={2} aria-hidden />작업자 화면으로 배정
                        </Button>
                      </CardFooter>
                    )}
                  </Card>
                ))}
              </div>
              {d.existing ? (
                <Notice tone="orange" title={`이미 작업 #${d.existing.id} · ${d.existing.title} (${d.existing.team})${d.existing.createdH > nowF() ? ` (${clock(d.existing.createdH)}에 만든 작업)` : ''}이 있어 다시 배정하지 않습니다.`}>
                  <div className="flex flex-col items-start gap-2">
                    <span>"작업 수정하기"에서 고치거나 배정을 취소할 수 있습니다.</span>
                    <Button variant="outline" size="sm" onClick={() => onEdit(d.existing!.id)}><Pencil strokeWidth={2} aria-hidden />작업 수정하기</Button>
                  </div>
                </Notice>
              ) : d.canAssign && (
                <>
                  <Disclosure title="작업 내용 바꾸기 (점검 가이드 · 추천 이유 덧붙이기)">
                    <Field label="점검 가이드" htmlFor="d-guide"><TextArea id="d-guide" rows={6} value={guide} maxLength={meta.limits.guide} onChange={(e) => setGuide(e.target.value)} /></Field>
                    <Field label="추천 이유에 덧붙일 말" htmlFor="d-note"><Input id="d-note" value={note} maxLength={200} onChange={(e) => setNote(e.target.value)} /></Field>
                  </Disclosure>
                  <p className="text-sm text-muted-foreground">칸에서 배정하면 그 선택지와 작업자로 작업이 만들어지고, 그 팀 작업자 화면의 "지금 할 일"에 바로 나타납니다.</p>
                </>
              )}
            </>
          ) : d.existing ? (
            <Notice tone="orange" title={`이미 작업 #${d.existing.id} · ${d.existing.title} (${d.existing.team})이 있어 다시 배정하지 않습니다.`}>
              <Button variant="outline" size="sm" onClick={() => onEdit(d.existing!.id)}><Pencil strokeWidth={2} aria-hidden />작업 수정하기</Button>
            </Notice>
          ) : d.job !== '작업 완료' && <SimpleAssign meta={meta} d={d} now={now} onDone={onChanged} />}
          <OrderButtons d={d} best={d.best ?? d.recommendFallback ?? ''} onDone={onChanged} />
        </CardContent>
      </Card>

      {d.alarm && (
        <>
          {d.lines && d.lines.length > 0 && <ul className="flex list-disc flex-col gap-1 pl-5">{d.lines.map((x, i) => <li key={i}>{x}</li>)}</ul>}
          <Disclosure title="이 설비 비용 가정" icon={<SlidersHorizontal className="size-4" strokeWidth={2} aria-hidden />}>
            <ul className="flex list-disc flex-col gap-1 pl-5">{d.assumption?.map((x, i) => <li key={i}>{x}</li>)}</ul>
            <p className="text-sm text-muted-foreground">지금 가정값으로 계산했습니다. 가정값은 정책·비용 페이지에서 바꿉니다.</p>
          </Disclosure>
          <Disclosure title="비용 계산 근거" icon={<Calculator className="size-4" strokeWidth={2} aria-hidden />}>
            <ul className="flex list-disc flex-col gap-1 pl-5">{d.basis?.map((x, i) => <li key={i}>{x}</li>)}</ul>
            <p className="text-sm text-muted-foreground">{d.basisNote}</p>
          </Disclosure>
        </>
      )}
    </section>
  )
}
