import {
  ArrowRight, CalendarClock, CircleCheck, CircleStop, ClipboardCheck, Clock, FileCheck, HardHat, NotebookPen, Package, Play, PlayCircle,
  ShieldCheck, Siren, Sparkles, TriangleAlert, Users,
} from 'lucide-react'
import { useEffect, useState } from 'react'

import { Check, Crumb, Disclosure, Empty, Field, Notice, TextArea } from '@/components/app/common'
import { useAction } from '@/lib/feedback'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { picked } from '@/lib/styles'
import { api, clock, clockS, type Bucket, type Checks, type Meta, type Task, type TaskList } from '@/lib/api'

const LISTS: { key: Bucket; Icon: typeof PlayCircle; help: string }[] = [
  { key: '지금', Icon: PlayCircle, help: '예정 시각이 됐거나 이미 시작한 작업' },
  { key: '지연', Icon: Clock, help: '예정 시각이 지났는데 아직 시작하지 않은 작업' },
  { key: '예정', Icon: CalendarClock, help: '아직 예정 시각 전인 작업 (보기만)' },
  { key: '완료', Icon: CircleCheck, help: '완료 보고한 작업' },
  { key: '중단', Icon: CircleStop, help: '현장에서 중단한 작업' },
]
const PRIORITY_VARIANT: Record<string, 'destructive' | 'secondary' | 'outline'> = { 긴급: 'destructive', 높음: 'secondary', 보통: 'outline' }
const RISK = { 높음: ['destructive', '끝나기 전 고장 위험 높음'], 보통: ['secondary', '끝나기 전 고장 위험 보통'], 낮음: ['outline', '끝나기 전 고장 위험 낮음'] } as const

function TaskBadges({ t, meta, list }: { t: Task; meta: Meta; list?: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge variant={PRIORITY_VARIANT[t.priority] ?? 'outline'}>{t.priority}</Badge>
      <Badge variant={t.status === '진행 중' ? 'default' : 'outline'}>{t.status}</Badge>
      {t.bucket === '지연' && <Badge variant="destructive"><Clock strokeWidth={2} aria-hidden />예정 시각 지남</Badge>}
      {t.urgentJob && !list && <Badge variant="destructive"><Siren strokeWidth={2} aria-hidden />즉시 조치</Badge>}
      {list && <Badge variant="outline">{t.team}</Badge>}
      {t.members && <Badge variant="outline"><Users strokeWidth={2} aria-hidden />{t.people}</Badge>}
      <Badge variant="secondary">{t.name}</Badge>
      {(t.status === '대기' || t.status === '진행 중') && t.checkCount > 0 && <Badge variant="outline"><ClipboardCheck strokeWidth={2} aria-hidden />체크 {t.checkCount}/{meta.checklist.total}</Badge>}
      {!list && t.riskWord && <Badge variant={RISK[t.riskWord as keyof typeof RISK][0]}>{RISK[t.riskWord as keyof typeof RISK][1]}</Badge>}
      {!list && t.missing && (t.missing.length
        ? <Badge variant="outline"><Package className="text-warning" strokeWidth={2} aria-hidden />{t.missing.join(', ')} 재고 없음 · 긴급 조달</Badge>
        : <Badge variant="outline"><Package className="text-success" strokeWidth={2} aria-hidden />부품 준비됨</Badge>)}
    </div>
  )
}

function TaskLine({ t, meta, onOpen }: { t: Task; meta: Meta; onOpen: () => void }) {
  const pn = (x: string) => meta.partNames[x] ?? x
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>#{t.id} {t.title.startsWith(t.sid) ? t.title : `${t.sid} — ${t.title}`}</CardTitle>
        <CardDescription>
          예정 {clock(t.due_h)} · 배정 {clock(t.created_h)}{t.status === '완료' && t.completed_h !== null ? ` · 완료 ${clock(t.completed_h)}` : ''}{t.reason ? ` · ${t.reason}` : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <TaskBadges t={t} meta={meta} list />
        {t.status === '완료' && (
          <p>완료 보고: <b>{t.outcome}</b> · 실제 조치 {t.action || meta.actions[0]} · 점검 판단 {pn(t.judged)} · 확인된 원인 {pn(t.confirmed)}<br />{t.details}
            {t.action_why && <><br /><span className="text-warning">지시와 다름 · {t.action_why}</span></>}</p>
        )}
        {t.status === '중단' && <p className="text-warning">작업 중단 {clock(t.stopped_h)} · 사유: {t.stop_why}</p>}
      </CardContent>
      <CardFooter>
        <Button variant={t.bucket === '지금' || t.bucket === '지연' ? 'default' : 'outline'} onClick={onOpen}>
          배정된 작업 보기<ArrowRight strokeWidth={2} aria-hidden />
        </Button>
      </CardFooter>
    </Card>
  )
}

/** 작업 상세: 사전 절차 → 작업 시작 → 정비 중 10대 안전철칙 → 정비 후 절차 → 완료 보고 / 작업 중단 */
function TaskDetail({ meta, t, storeNote, go, onChanged }: { meta: Meta; t: Task; storeNote: string; go: (p: string) => void; onChanged: () => void }) {
  const [c, setC] = useState<Checks>(t.checks)
  useEffect(() => setC(t.checks), [t.id, t.checks])
  const { busy, run } = useAction(onChanged)
  const cl = meta.checklist
  const v = (k: string) => c[k]
  const canPre = t.status === '대기' && (t.bucket === '지금' || t.bucket === '지연')
  const running = t.status === '진행 중'
  const save = async (k: string, val: boolean | string | null) => {
    const next = { ...c, [k]: val }
    setC(next)
    try {
      const r = await api.checks(t.id, next)
      setC(r.checks)
    } catch {
      setC(c)
    }
  }
  const preLeft = cl.preKeys.filter((k) => !v(k)).length
  const safetyLeft = cl.safetyKeys.filter((k) => !cl.safetyOpts.includes(String(v(k)))).length
  const postLeft = cl.post.filter(([k]) => !v(k)).length
  const postOk = running && safetyLeft === 0
  const tbmAll = cl.tbm.every(([k]) => v(k))
  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">{t.title.startsWith(t.sid) ? t.title : `${t.sid} — ${t.title}`}</CardTitle>
          <CardDescription>시나리오 {t.sid} · 배정 {clock(t.created_h)}{t.started_h !== null ? ` · 시작 ${clock(t.started_h)}` : ''}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <TaskBadges t={t} meta={meta} />
          <p><b>예정 {clock(t.due_h)} → {clock(t.endH)}</b> · {t.reason}</p>
          <Disclosure title="점검 가이드 (설비관리자 메모)" icon={<FileCheck className="size-4" strokeWidth={2} aria-hidden />}>
            <p className="whitespace-pre-line">{t.guide}</p>
          </Disclosure>
        </CardContent>
      </Card>
      {t.status === '완료' && (
        <Notice tone="green" title={`완료 보고 ${clock(t.completed_h)}: ${t.outcome}`}>
          실제 조치 {t.action || meta.actions[0]} · 확인된 원인 {meta.partNames[t.confirmed] ?? t.confirmed}<br />{t.details}
          {t.action_why && <><br />지시와 다름 · {t.action_why}</>}
        </Notice>
      )}
      {t.status === '중단' && <Notice tone="orange" title={`작업 중단 ${clock(t.stopped_h)} · 사유: ${t.stop_why}`} />}
      {t.status === '대기' && t.bucket === '예정' && (
        <Notice tone="blue">예정 시각({clock(t.due_h)}) 전이라 보기만 합니다. 그 시각이 되면 "지금"에서 사전 절차를 체크할 수 있습니다.</Notice>
      )}

      <Card size="sm">
        <CardHeader><CardTitle className="flex items-center gap-2"><ClipboardCheck className="size-5" strokeWidth={2} aria-hidden />사전 절차{preLeft ? ` · ${preLeft}개 남음` : ' · 완료'}</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-1">
          <Check id="ck-permit" checked={!!v('permit')} disabled={!canPre} onChange={(x) => save('permit', x)}
            label={t.urgentJob ? '① 관리자 구두 승인 확인 (작업 허가서는 사후 처리)' : '① 작업 허가서 출력·확인 (전날 정비 범위 미팅에서 전산 입력한 것)'} />
          <Check id="ck-halt" checked={!!v('halt')} disabled={!canPre || !v('permit')} onChange={(x) => save('halt', x)} label="② 설비 정지 확인" />
          <div className="ml-2 flex flex-col gap-1 rounded-2xl border border-border p-3">
            <p className="px-2">③ <b>TBM</b> (Tool Box Meeting) · 작업자 전원, 아래 6개는 순서 상관없음</p>
            {cl.tbm.map(([k, label]) => <Check key={k} id={`ck-${k}`} checked={!!v(k)} disabled={!canPre || !v('halt')} onChange={(x) => save(k, x)} label={label} />)}
            <Check id="ck-tbm_done" checked={!!v('tbm_done')} disabled={!canPre || !tbmAll} onChange={(x) => save('tbm_done', x)} label={<b>TBM 완료</b>} />
          </div>
          <Check id="ck-parts" checked={!!v('parts')} disabled={!canPre} onChange={(x) => save('parts', x)} label="④ 교체 부품·공구 준비" />
        </CardContent>
        {t.status === '대기' && (
          <CardFooter className="flex flex-col items-start gap-2">
            <Button disabled={busy || !canPre || preLeft > 0} onClick={() => run(() => api.start(t.id))}
              title={preLeft ? '사전 절차를 모두 체크해야 켜집니다' : undefined}>
              <Play strokeWidth={2} aria-hidden />작업 시작
            </Button>
            {canPre && preLeft > 0 && <p className="text-sm text-muted-foreground">사전 절차 {preLeft}개 남음 · ①→②→③ 순서대로 켜집니다</p>}
          </CardFooter>
        )}
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><ShieldCheck className="size-5" strokeWidth={2} aria-hidden />정비 중 · 10대 안전철칙{safetyLeft ? ` · ${safetyLeft}개 남음` : ' · 완료'}</CardTitle>
          <CardDescription>작업책임자({t.leaderName || '미지정'})가 항목마다 준수 / 해당 없음을 고릅니다.{running || ['완료', '중단'].includes(t.status) ? '' : ' 작업 시작 뒤에 열립니다.'}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {cl.safety.map(([kind, text], i) => (
            <div key={i} className="flex flex-col gap-2">
              {(i === 0 || kind !== cl.safety[i - 1][0]) && <p className="font-semibold">{kind} 수칙</p>}
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <span id={`sr-${i}`}>{i + 1}. {text}</span>
                <ToggleGroup type="single" variant="outline" size="sm" aria-labelledby={`sr-${i}`} disabled={!running}
                  value={typeof v(cl.safetyKeys[i]) === 'string' ? String(v(cl.safetyKeys[i])) : ''}
                  onValueChange={(x) => save(cl.safetyKeys[i], x || null)} className="shrink-0">
                  {cl.safetyOpts.map((o) => <ToggleGroupItem className={picked} key={o} value={o}>{o}</ToggleGroupItem>)}
                </ToggleGroup>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Sparkles className="size-5" strokeWidth={2} aria-hidden />정비 후 절차{postLeft ? ` · ${postLeft}개 남음` : ' · 완료'}</CardTitle>
          {running && safetyLeft > 0 && <CardDescription>정비 중 10대 안전철칙을 모두 고르면 열립니다. ①~④는 순서대로 켜집니다.</CardDescription>}
        </CardHeader>
        <CardContent className="flex flex-col gap-1">
          {cl.post.map(([k, label], j) => {
            const first = j === 0 || j >= cl.postOrder.length
            return <Check key={k} id={`ck-${k}`} checked={!!v(k)} disabled={!postOk || !(first || v(cl.postOrder[j - 1]))} onChange={(x) => save(k, x)} label={label} />
          })}
        </CardContent>
      </Card>

      {(t.status === '대기' || running) && (
        <div className="flex flex-wrap gap-2">
          {running && (
            <Button disabled={safetyLeft > 0 || postLeft > 0} onClick={() => go(`w/task/${t.id}/report`)}
              title={safetyLeft || postLeft ? '정비 중 · 정비 후 항목을 모두 체크해야 켜집니다' : undefined}>
              <NotebookPen strokeWidth={2} aria-hidden />완료 보고 작성
            </Button>
          )}
          <Button variant="outline" onClick={() => go(`w/task/${t.id}/stop`)} title="현장 상황이 지시와 달라 끝낼 수 없을 때 (언제든)">
            <CircleStop strokeWidth={2} aria-hidden />작업 중단
          </Button>
        </div>
      )}
      <p className="text-sm text-muted-foreground">현장 안전 규정과 작업 허가가 우선입니다. 체크는 바로 저장되어 설비관리자가 진행률을 봅니다. {storeNote}</p>
    </div>
  )
}

function Seg({ id, label, opts, value, onChange, fmt }: { id: string; label: string; opts: string[]; value: string; onChange: (v: string) => void; fmt?: (x: string) => string }) {
  return (
    <div className="grid gap-2">
      <span id={id} className="text-sm font-medium">{label}</span>
      <ToggleGroup type="single" variant="outline" aria-labelledby={id} value={value} onValueChange={(x) => x && onChange(x)} className="w-fit flex-wrap">
        {opts.map((o) => <ToggleGroupItem className={picked} key={o} value={o}>{fmt ? fmt(o) : o}</ToggleGroupItem>)}
      </ToggleGroup>
    </div>
  )
}

function Report({ meta, t, nowF, onDone }: { meta: Meta; t: Task; nowF: () => number; onDone: () => void }) {
  const [outcome, setOutcome] = useState('이상 발견')
  const [judged, setJudged] = useState('구분 못함')
  const [confirmed, setConfirmed] = useState('아직 모름')
  const [action, setAction] = useState(meta.actions[0])
  const [why, setWhy] = useState('')
  const [details, setDetails] = useState('')
  const [permit, setPermit] = useState(false)
  const { busy, run } = useAction(onDone)
  const pn = (x: string) => meta.partNames[x] ?? x
  if (t.status !== '진행 중') return <Notice tone="blue">진행 중인 작업만 완료 보고를 쓸 수 있습니다.</Notice>
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">{t.name} · 완료 보고</CardTitle>
        <CardDescription>작업 #{t.id} · {t.title} · 결과와 수행 내용을 남겨야 완료됩니다.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Seg id="r-out" label="작업 결과" opts={meta.outcomes} value={outcome} onChange={setOutcome} />
        <Seg id="r-jud" label="점검으로 판단한 부품" opts={meta.judged} value={judged} onChange={setJudged} fmt={pn} />
        <Seg id="r-con" label="교체 뒤 확인된 원인 부품" opts={meta.confirmed} value={confirmed} onChange={setConfirmed} fmt={pn} />
        <Seg id="r-act" label="실제 조치 (지시: 두 부품 교체)" opts={meta.actions} value={action} onChange={setAction} />
        <Field label="지시와 다르게 한 이유 (실제 조치가 두 부품 교체가 아니면 필수)" htmlFor="r-why">
          <Input id="r-why" value={why} maxLength={meta.limits.why} placeholder="예: 열어 보니 밸브만 마모, 진동장치는 정상" onChange={(e) => setWhy(e.target.value)} />
        </Field>
        <Field label="수행 내용 (필수)" htmlFor="r-det">
          <TextArea id="r-det" value={details} maxLength={meta.limits.details} placeholder="예: 진동장치 베어링 마모 확인, 두 부품 교체 후 신호 정상" onChange={(e) => setDetails(e.target.value)} />
        </Field>
        <p className="text-sm text-muted-foreground">완료 시각은 제출하는 순간의 시각으로 남습니다 (지금 {clockS(nowF())})</p>
        {t.urgentJob && <Check id="r-permit" checked={permit} onChange={setPermit} label="작업 허가서 사후 처리 완료 (즉시 조치는 필수)" />}
      </CardContent>
      <CardFooter>
        <Button disabled={busy} onClick={() => run(() => api.complete(t.id, { outcome, details, judged, confirmed, action, action_why: why, permit_after: permit }))}>
          <CircleCheck strokeWidth={2} aria-hidden />완료 보고 제출
        </Button>
      </CardFooter>
    </Card>
  )
}

function Stop({ t, onDone }: { t: Task; onDone: () => void }) {
  const [why, setWhy] = useState('')
  const { busy, run } = useAction(onDone)
  if (t.status !== '대기' && t.status !== '진행 중') return <Notice tone="blue">이미 완료됐거나 중단한 작업입니다.</Notice>
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">{t.name} · 작업 중단</CardTitle>
        <CardDescription>작업 #{t.id} · {t.title} · 중단하면 설비관리자에게 알리고, 그 설비는 다시 배정할 수 있게 됩니다.</CardDescription>
      </CardHeader>
      <CardContent>
        <Field label="중단 사유 (필수)" htmlFor="s-why">
          <TextArea id="s-why" value={why} maxLength={200} placeholder="예: 부품이 맞지 않아 교체 못 함, 안전 문제로 작업 허가 보류" onChange={(e) => setWhy(e.target.value)} />
        </Field>
      </CardContent>
      <CardFooter>
        <Button disabled={busy || !why.trim()} onClick={() => run(() => api.stop(t.id, why))}><CircleStop strokeWidth={2} aria-hidden />작업 중단 제출</Button>
      </CardFooter>
    </Card>
  )
}

/** 작업자 화면 (주소 #/w/...) */
export function Worker({ meta, route, go, team, setTeam, locked, version, nowF, storeNote, onChanged }: {
  meta: Meta; route: string[]; go: (p: string) => void; team: string; setTeam: (t: string) => void; locked: boolean
  version: number; nowF: () => number; storeNote: string; onChanged: () => void
}) {
  const [data, setData] = useState<TaskList | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const { busy, run } = useAction(onChanged)
  useEffect(() => {
    let live = true
    api.tasks(team).then((x) => live && setData(x)).catch((e: Error) => live && setErr(e.message))
    return () => { live = false }
  }, [team, version])
  if (err) return <Notice tone="red">{err}</Notice>
  if (!data) return <p className="text-muted-foreground">불러오는 중...</p>
  const [, a, b, c] = route
  const listKey = LISTS.find((l) => l.key === b)
  const task = a === 'task' ? data.items.find((x) => x.id === Number(b)) : undefined
  const back = () => go(listKey ? 'w' : 'w')

  const head = (
    <div className="flex flex-wrap items-center justify-between gap-3">
      {a === 'list' && listKey ? <Crumb back="내 작업" onBack={() => go('w')} here={<span className="flex items-center gap-1"><listKey.Icon className="size-4" strokeWidth={2} aria-hidden />{listKey.key}</span>} />
        : a === 'task' && task ? (c
          ? <Crumb back={`작업 #${task.id}`} onBack={() => go(`w/task/${task.id}`)} here={c === 'report' ? '완료 보고' : '작업 중단'} />
          : <Crumb back={task.bucket} onBack={() => go(`w/list/${task.bucket}`)} here={`작업 #${task.id}`} />)
          : <h2 className="flex items-center gap-2 text-2xl font-semibold"><HardHat className="size-6" strokeWidth={2} aria-hidden />내 작업</h2>}
      {locked || a ? <Badge variant="secondary"><Users strokeWidth={2} aria-hidden />{data.team}</Badge> : (
        <ToggleGroup type="single" variant="outline" value={team} aria-label="내 팀" onValueChange={(v) => v && setTeam(v)}
          title={'팀원: ' + Object.entries(meta.teams).map(([k, v]) => `${k} ${v.join('·')}`).join(' / ')}>
          {Object.keys(meta.teams).map((t) => <ToggleGroupItem className={picked} key={t} value={t}>{t}</ToggleGroupItem>)}
        </ToggleGroup>
      )}
    </div>
  )

  if (a === 'task') {
    if (!task) return <div className="flex flex-col gap-4">{head}<Notice tone="blue">작업이 없어졌거나(배정 취소) 다른 팀 작업입니다.</Notice><Button className="w-fit" onClick={back}>내 작업으로</Button></div>
    return (
      <div className="flex flex-col gap-4">
        {head}
        {c === 'report' ? <Report meta={meta} t={task} nowF={nowF} onDone={() => { onChanged(); go('w') }} />
          : c === 'stop' ? <Stop t={task} onDone={() => { onChanged(); go('w') }} />
            : <TaskDetail meta={meta} t={task} storeNote={storeNote} go={go} onChanged={onChanged} />}
      </div>
    )
  }
  if (a === 'list' && listKey) {
    let rows = data.items.filter((x) => x.bucket === listKey.key)
    if (listKey.key === '지금') rows = [...rows].sort((x, y) => Number(x.status !== '진행 중') - Number(y.status !== '진행 중'))
    return (
      <div className="flex flex-col gap-4">
        {head}
        <p className="text-sm text-muted-foreground">{listKey.help} · {rows.length}건</p>
        {rows.length === 0 ? <Empty>없습니다{listKey.key === '지금' || listKey.key === '예정' ? ' · 설비관리자가 배정하면 지금·예정에 나타납니다.' : ''}</Empty>
          : rows.map((t) => <TaskLine key={t.id} t={t} meta={meta} onOpen={() => go(`w/task/${t.id}`)} />)}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-4">
      {head}
      {data.urgent.length > 0 && (
        <Notice tone="red" title={`배정 전 즉시 조치 ${data.urgent.length}대`}>
          <div className="flex flex-col gap-3">
            <p>지금 교체해도 완료 전 고장 확률이 안전 기준을 넘습니다. 안전이 먼저라 설비관리자 배정을 기다리지 않고 작업을 만들 수 있습니다. 관리자 구두 승인을 받은 뒤 사전 절차를 체크하세요 (작업 허가서는 사후 처리).</p>
            <div className="flex flex-wrap gap-2">
              {data.urgent.map((u) => (
                <Button key={u.sid} variant="destructive" disabled={busy}
                  onClick={async () => {
                    let id = 0
                    const ok = await run(async () => { const r = await api.urgent(u.sid, data.team, u.option); id = r.id; return r })
                    if (ok && id) go(`w/task/${id}`)
                  }}>
                  <Siren strokeWidth={2} aria-hidden />{u.name} 즉시 조치 작업 만들기
                </Button>
              ))}
            </div>
          </div>
        </Notice>
      )}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {LISTS.map(({ key, Icon, help }) => (
          <Card key={key} size="sm" className="transition-colors hover:bg-black/5 dark:hover:bg-white/5">
            <button type="button" onClick={() => go(`w/list/${key}`)} title={help}
              className="flex min-h-32 flex-col gap-3 rounded-container px-(--card-spacing) text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
              <span className="flex items-center gap-1 text-muted-foreground"><Icon className="size-4" strokeWidth={2} aria-hidden />{key}<ArrowRight className="size-4" strokeWidth={2} aria-hidden /></span>
              <span className="text-4xl font-semibold tabular-nums">{data.counts[key]}</span>
            </button>
          </Card>
        ))}
      </div>
      <p className="text-sm text-muted-foreground">재생 시각 기준: 지금 = 예정 시각이 됐거나 이미 시작함 · 지연 = 예정 시각이 지났는데 시작 전 · 예정 = 아직 · {storeNote}</p>
      {data.items.some((x) => x.status === '진행 중' && x.missing?.length) && (
        <p className="flex items-center gap-2 text-sm text-warning"><TriangleAlert className="size-4" strokeWidth={2} aria-hidden />진행 중인 작업 중 부품 재고가 없는 작업이 있습니다 · 긴급 조달 {data.rushH.toFixed(0)}시간</p>
      )}
    </div>
  )
}
