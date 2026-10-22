// FastAPI(api/server.py) 응답 타입과 호출. 개발 서버는 /api를 8000번으로 넘긴다 (vite.config.ts)

export type Split = 'test' | 'validation'
export type ShortStage = '경보' | '주의' | '정상' | '정비 중' | '정비 후 예측 대기'
export type Bucket = '지금' | '지연' | '예정' | '완료' | '중단'
export type TaskStatus = '대기' | '진행 중' | '완료' | '중단'
export type Checks = Record<string, boolean | string | null>

export interface User { email: string; name: string; role: 'manager' | 'worker'; team: string | null }
export interface Session { loginEnabled: boolean; user: User | null; sharedDb: boolean; cloud: boolean; storeNote: string }

export interface Meta {
  splits: Record<Split, string>
  endH: number
  threshold: number
  cautionThreshold: number | null
  caution: boolean
  step: number
  signals: Record<string, string>
  teams: Record<string, string[]>
  leaders: Record<string, string>
  teamOf: Record<string, string>
  priorities: string[]
  outcomes: string[]
  actions: string[]
  judged: string[]
  confirmed: string[]
  statuses: string[]
  absenceKinds: string[]
  limits: Record<string, number>
  defaultGuide: string
  partNames: Record<string, string>
  checklist: {
    tbm: [string, string][]
    safety: [string, string][]
    safetyKeys: string[]
    safetyOpts: string[]
    post: [string, string][]
    postOrder: string[]
    preKeys: string[]
    total: number
  }
  work: { replace: number; repair: number }
}

export interface ClockState { split: Split; now: number; clockAt: number | null; nowF: number; version: string; canSetClock: boolean }

export interface Task {
  id: number
  sid: string
  name: string
  title: string
  team: string
  priority: string
  due_h: number
  created_h: number
  started_h: number | null
  completed_h: number | null
  stopped_h: number | null
  guide: string
  reason: string
  option: string
  outcome: string
  details: string
  judged: string
  confirmed: string
  action: string
  action_why: string
  stop_why: string
  members: string
  leader: string
  leaderName: string
  completed_at: string
  status: TaskStatus
  bucket: Bucket
  checks: Checks
  checkCount: number
  people: string
  urgentJob: boolean
  missing?: string[]
  riskWord?: string | null
  endH?: number
}

export interface ManagerRow { sid: string; name: string; stage: ShortStage; risk: number | null; orderState: string; truthH: number | null }
export interface QueueRow { sid: string; name: string; best: string; rank: number | null; urgent: boolean; constraint: string }
export interface ManagerData {
  counts: Record<'경보' | '주의' | '정상', number>
  inMnt: number
  rows: ManagerRow[]
  queue: QueueRow[]
  held: number
  busy: number
  load: Record<string, number>
  taskCounts: Record<'진행 중' | '지연' | '완료' | '중단', number>
  tasks: Task[]
  staff: { team: string; on: string[]; total: number; absent: Record<string, string> }[]
  crewDayH: number
  caution: boolean
}

export interface Card {
  kind: 'now' | 'plan' | 'run'
  label: string
  help: string
  cost: number | null
  p: number | null
  sub: string
  start: number | null
  note: string
  overSafety: boolean
  title: string
  priority: string
  workH: number
}

export interface Decision {
  sid: string
  name: string
  stage: ShortStage
  stageLabel: string
  risk: number | null
  orderState: string
  job: string | null
  guide: { tone: 'red' | 'orange' | 'green'; title: string; body: string[] }
  alarm: boolean
  load: Record<string, number>
  unit: string
  existing: { id: number; title: string; team: string; createdH: number } | null
  transit: { part: string; partName: string; arrive: number; rush: boolean }[]
  lead: number
  rushH: number
  rushMk: number
  stock: Record<string, number>
  safety: number
  procurement?: string
  absent: Record<string, string>
  busyPeople: string[]
  team0: string
  recommendFallback: string | null
  defaultPriority?: string
  remainH?: number | null
  remainDeadline?: number | null
  urgent?: boolean
  constraint?: string
  best?: string
  cards?: Card[]
  recKind?: Card['kind'] | null
  horizonH?: number
  horizonLabel?: string
  canAssign?: boolean
  lines?: string[]
  assumption?: string[]
  basis?: string[]
  basisNote?: string
}

export interface Detail {
  sid: string
  name: string
  stage: string
  risk: number | null
  degraded: number | null
  lastPredH: number | null
  firstCautionH: number | null
  firstAlarmH: number | null
  latched: boolean
  oor: string | null
  risk_series: { h: number; risk: number | null; degraded: number | null; stage: string }[]
  signal_series: ({ h: number } & Record<string, number | null>)[]
  maintenance: { start: number; end: number | null; endShown: number; kind: string; part: string; fail: number | null }[]
  truthFails: number[]
  signalsNow: { name: string; value: number }[] | null
  signalsNote: string
  contrib: { name: string; value: number }[] | null
  contribNote: string
  contribUnit?: string
}

export interface TaskList { team: string; items: Task[]; urgent: { sid: string; name: string; option: string }[]; rushH: number; counts: Record<Bucket, number> }

export interface Absence { name: string; team?: string; kind: string; start_h: number; end_h: number; note: string }

export type Assumptions = Record<string, number | string>

export interface PolicyRow {
  rank: number | null
  sid: string
  name: string
  state: string
  best: string
  method: string
  stock: string
  doneH: number | null
  cost: number | null
  p: number | null
  constraint: string
  firstAlarmH: number | null
  nextPlan: string
}

export interface Policy {
  assumptions: Assumptions
  defaults: Assumptions
  note: string
  unit: string
  rows: PolicyRow[]
  perDay: number
  replaceH: number
  feedback: { n: number; accuracy: number | null; breakeven: number | null; ready: boolean }
  minConfirmed: number
  log: Record<string, string | number | null>[]
  caution: { sid: string; name: string; nextPlan: string; todo: string; stock: string; parts: string }[]
  needOrder: { sid: string; part: string }[]
  lead: number
  rushH: number
  rushMk: number
  sim: Record<string, string | number | null>[]
}

export interface PolicyOption {
  sid: string
  name: string
  best: string
  urgent: string
  total: number
  p: number
  maint: number
  down: number
  loss: number
  opts: { name: string; doneH: number; maint: number; down: number; p: number; loss: number; total: number; safe: boolean; basis: string; rec: boolean }[]
  elapsed: number
  boundary: string
  jobState: string
  safety: number
  unit: string
  curve: { h: number; p: number }[]
  points: { h: number; p: number; name: string }[]
}

type Num = number | null
export interface ModelInfo {
  createdAt: string
  cloud: boolean
  model: string
  nFeatures: number
  features: string
  step: number
  nTrain: number
  th: number
  point: Record<string, Num | string>
  event: Record<string, Num | string>
  caution: null | { model: string; features: string; nFeatures: number; th: number; minPrecision: number; v: Record<string, Num | string> }
  events: Record<string, string | number | boolean | null>[]
  limits: Record<string, string>[]
  ontology: { triples: number; counts: Record<string, number>; bad: string[]; parts: Record<string, number> }
}

export interface PartsCalc {
  event: Record<string, Num | string>
  n: { n_det: number; n_miss: number; n_needed: number; n_false: number }
  one: number
  both: number
  none: number
  unit: string
  unlabeled: null | Record<string, Num | string>
  two: number
  even: number | null
  curve: { p: number; '한 부품만 교체': number; '두 부품 모두 교체': number; '아무것도 안 함': number }[]
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, ...init })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(typeof body.detail === 'string' ? body.detail : `요청 실패 (${res.status})`, res.status)
  }
  return res.json() as Promise<T>
}

const q = (p: Record<string, string | number | boolean | undefined | null>) =>
  new URLSearchParams(Object.entries(p).filter(([, v]) => v !== undefined && v !== null && v !== false)
    .map(([k, v]) => [k, String(v)])).toString()
const post = <T = { ok: boolean; message?: string }>(path: string, body?: unknown, method = 'POST') =>
  call<T>(path, { method, body: body === undefined ? undefined : JSON.stringify(body) })

export const api = {
  session: () => call<Session>('/api/session'),
  logout: () => post('/api/auth/logout'),
  meta: () => call<Meta>('/api/meta'),
  state: () => call<ClockState>('/api/state'),
  setClock: (split: Split, now: number) => post<ClockState>('/api/clock', { split, now }),
  manager: (truth: boolean) => call<ManagerData>(`/api/manager?${q({ truth })}`),
  decision: (sid: string, crew: Partial<Record<Card['kind'], number>>) =>
    call<Decision>(`/api/equipment/${sid}/decision?${q({ n_now: crew.now, n_plan: crew.plan, n_run: crew.run })}`),
  detail: (sid: string, truth: boolean) => call<Detail>(`/api/equipment/${sid}/detail?${q({ truth })}`),
  ontology: (sid: string, dark: boolean) => call<{ dot: string | null; trace: [string, string][] | null }>(`/api/equipment/${sid}/ontology?${q({ dark })}`),
  assign: (body: { sid: string; team: string; title: string; priority: string; due_h: number; reason: string; option: string; guide?: string; members?: string[]; leader?: string }) =>
    post('/api/assign', body),
  act: (kind: '보류' | '부품 발주', sid: string, extra: { part?: string; option?: string } = {}) => post('/api/act', { kind, sid, ...extra }),
  orderRush: (sid: string) => post(`/api/order/rush/${sid}`),
  orderCaution: () => post('/api/order/caution'),
  tasks: (team: string) => call<TaskList>(`/api/tasks?${q({ team })}`),
  start: (id: number) => post(`/api/tasks/${id}/start`),
  checks: (id: number, checks: Checks) => post<{ checks: Checks }>(`/api/tasks/${id}/checks`, { checks }),
  complete: (id: number, body: { outcome: string; details: string; judged: string; confirmed: string; action: string; action_why: string; permit_after: boolean }) =>
    post(`/api/tasks/${id}/complete`, body),
  stop: (id: number, why: string) => post(`/api/tasks/${id}/stop`, { why }),
  urgent: (sid: string, team: string, option: string) => post<{ ok: boolean; id: number; message: string }>('/api/urgent', { sid, team, option }),
  edit: (id: number, body: Record<string, unknown>) => post(`/api/tasks/${id}`, body, 'PUT'),
  cancel: (id: number, why: string) => post(`/api/tasks/${id}/cancel`, { why }),
  absences: () => call<{ rows: Absence[] }>('/api/absences'),
  saveAbsences: (rows: Absence[]) => post('/api/absences', { rows }, 'PUT'),
  absencePreset: () => post('/api/absences/preset'),
  policy: () => call<Policy>('/api/policy'),
  policyOption: (sid: string, override?: Record<string, number>) =>
    call<PolicyOption>(`/api/policy/${sid}?${q({ override: override ? JSON.stringify(override) : undefined })}`),
  setAssume: (a: Assumptions) => post('/api/assume', a, 'PUT'),
  resetAssume: () => post('/api/assume', undefined, 'DELETE'),
  model: () => call<ModelInfo>('/api/model'),
  parts: (p: number) => call<PartsCalc>(`/api/parts?${q({ p })}`),
}

/** 재생 시각(시나리오 시작 후 h)을 사람 말로: 25 → '2일차 01:00' */
export function clock(h: number | null | undefined) {
  if (h === null || h === undefined || Number.isNaN(h)) return '—'
  const m = Math.round(h * 60)
  const d = Math.floor(m / 1440)
  const r = m % 1440
  return `${d + 1}일차 ${String(Math.floor(r / 60)).padStart(2, '0')}:${String(r % 60).padStart(2, '0')}`
}

const p2 = (n: number) => String(Math.floor(n)).padStart(2, '0')

/** 흐르는 시각을 초까지: 26.2 → '2일차 02:12:00' */
export function clockS(h: number) {
  let s = Math.floor(h * 3600 + 1e-6)
  const d = Math.floor(s / 86400)
  s -= d * 86400
  return `${d + 1}일차 ${p2(s / 3600)}:${p2((s % 3600) / 60)}:${p2(s % 60)}`
}

export function hms(h: number) {
  const s = Math.max(0, Math.round(h * 3600))
  return `${p2(s / 3600)}:${p2((s % 3600) / 60)}:${p2(s % 60)}`
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(digits)}%`

export const money = (x: number | null | undefined, unit = '만 원') =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${Math.round(x).toLocaleString('ko-KR')}${unit === '만 원' ? '만 원' : '점'}`
