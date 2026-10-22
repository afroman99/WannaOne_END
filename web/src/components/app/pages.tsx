import { Bell, CalendarClock, Download, FileCheck, History, ListTree, Monitor, RotateCcw, Save, SearchCheck, SlidersHorizontal, Truck } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'

import { Analysis, OntologyView } from '@/components/app/analysis'
import { FailCurve, PartsChart } from '@/components/app/charts'
import { Check, Crumb, Disclosure, Field, Metric, Notice } from '@/components/app/common'
import { useAction } from '@/lib/feedback'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { picked } from '@/lib/styles'
import { api, money, type ManagerRow, type Meta, type ModelInfo, type PartsCalc, type Policy, type PolicyOption } from '@/lib/api'

function Table({ head, rows, min = '40rem' }: { head: string[]; rows: ReactNode[][]; min?: string }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-border">
      <table className="w-full text-sm" style={{ minWidth: min }}>
        <thead className="bg-muted text-left"><tr>{head.map((h, i) => <th key={i} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i} className="border-t border-border align-top">{r.map((c, j) => <td key={j} className="px-3 py-2">{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}

const n0 = (x: number | null | undefined) => (x === null || x === undefined ? '—' : Math.round(x).toLocaleString('ko-KR'))
const f2 = (x: number | string | null | undefined) => (typeof x === 'number' ? x.toFixed(2) : x ?? '—')

/** 설비 상세 (하위 페이지): 설비 고르기 · 분석 근거 · 근거 따라가기 */
export function EquipPage({ meta, sid, rows, onSid, truth, version }: {
  meta: Meta; sid: string | undefined; rows: ManagerRow[]; onSid: (s: string) => void; truth: boolean; version: number
}) {
  const cur = sid && rows.some((r) => r.sid === sid) ? sid : rows[0]?.sid
  if (!cur) return null
  const r = rows.find((x) => x.sid === cur)!
  return (
    <section className="flex flex-col gap-6">
      <Field label="설비 (지금 위험한 순서)" htmlFor="ep-sid">
        <Select value={cur} onValueChange={onSid}>
          <SelectTrigger id="ep-sid" className="w-full sm:w-[32rem]"><SelectValue /></SelectTrigger>
          <SelectContent>{rows.map((x) => (
            <SelectItem key={x.sid} value={x.sid}>{x.name} ({x.sid}) · {x.stage}{x.risk !== null ? ` · 24시간 내 고장 위험 ${Math.round(x.risk * 100)}%` : ''}</SelectItem>
          ))}</SelectContent>
        </Select>
      </Field>
      <div className="flex flex-col gap-1">
        <h2 className="flex items-center gap-2 text-3xl font-semibold"><Monitor className="size-6" strokeWidth={2} aria-hidden />{r.name}</h2>
        <p className="text-sm text-muted-foreground">시나리오 {cur}</p>
      </div>
      <Analysis meta={meta} sid={cur} truth={truth} version={version} />
      <Disclosure title="근거 따라가기: 경보 → 예측 → 신호 → 정비 결정 (온톨로지)" icon={<ListTree className="size-4" strokeWidth={2} aria-hidden />}>
        <OntologyView sid={cur} version={version} />
      </Disclosure>
    </section>
  )
}

type Spec = [key: string, label: string, min: number, max: number, step: number, help?: string]
const SAFETY: Spec[] = [
  ['safety', '안전 기준: 정비 완료 전 고장 확률 상한', 0, 0.5, 0.01], ['prep', '부품 준비 시간 (조달 + 교체, h)', 4, 36, 1, '부품을 조달해야 하면 더 길어집니다.'],
  ['crew_wait', '정비팀이 시작할 수 있을 때까지 (h)', 0, 24, 1], ['n_crews', '동시에 교체할 수 있는 정비팀 수', 1, 10, 1],
  ['crew_day_h', '정비팀 1팀의 하루 작업 한도 (공수, 시간)', 4, 24, 1, '기본 8공수(상주 근무: 주 5일, 하루 8시간)'],
]
const COST: Spec[] = [
  ['f_cons', '소모품 (교체 1번: 볼트·너트·O링·윤활유)', 0, 10000, 10], ['f_part', '중요 부품 (개당: 밸브·진동장치)', 0, 10000, 10],
  ['f_labor_part', '공임 (파트당)', 0, 1000, 1], ['f_labor_ph', '정비 인력 (1명·1시간)', 0, 100, 0.5],
  ['f_crew', '정비팀 인원 (명)', 1, 10, 1, '현장: 보통 3~4인 1팀'], ['f_down', '생산 차질 (정지 1시간)', 0, 100000, 50],
  ['f_fail', '고장 1건 추가 손실 (불량·품질·안전 대응)', 0, 100000, 50],
]
const PARTS: Spec[] = [
  ['stock_v', '예비 밸브 (설비당)', 0, 10, 1, '정비 기록상 밸브 교체가 진동장치의 약 2배'], ['stock_o', '예비 진동장치 (설비당)', 0, 10, 1],
  ['lead', '정상 조달 (h)', 1, 720, 12], ['rush_h', '긴급 조달 (h)', 1, 168, 1, '웃돈을 내고 직접 수령'], ['rush_mk', '긴급 조달 웃돈 (부품값 대비)', 0, 3, 0.1],
]
const OTHER: Spec[] = [
  ['p_other', '함께 바꾸는 부품이 실제 원인일 확률', 0, 1, 0.05, '센서로 원인 부품을 구분할 수 없어 기본값은 0.5입니다.'], ['max_wait', '기다릴 수 있는 최대 시간 (h)', 0, 72, 1],
  ['delay_h', 'N시간 뒤 정비 (0 = 쓰지 않음, 예: 생산 영향이 적은 시간대)', 0, 48, 1], ['f_delay_rate', '그 시간대 생산 차질 (정지 1시간, 만 원)', 0, 100000, 50],
]

function AssumeForm({ p, onDone }: { p: Policy; onDone: () => void }) {
  const [a, setA] = useState(p.assumptions)
  const { busy, run } = useAction(onDone)
  const group = (title: string, specs: Spec[]) => (
    <div className="flex flex-col gap-3">
      <p className="font-semibold">{title}</p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {specs.map(([k, label, min, max, step, help]) => (
          <Field key={k} label={label} htmlFor={`as-${k}`} hint={help}>
            <Input id={`as-${k}`} type="number" min={min} max={max} step={step} value={a[k]}
              onChange={(e) => setA({ ...a, [k]: Number(e.target.value) })} />
          </Field>
        ))}
      </div>
    </div>
  )
  return (
    <Disclosure title="가정값 (바꾸면 추천·비용·배정할 경보가 다시 계산됩니다)" icon={<SlidersHorizontal className="size-4" strokeWidth={2} aria-hidden />}>
      <p className="text-sm text-muted-foreground">{p.note}</p>
      {group('안전·실행 가능성 (비용보다 먼저 지키는 제약)', SAFETY)}
      {group('비용 (만 원, 현장 가정)', COST)}
      {group('부품 조달 (데이터에 없어 미리 정한 가정값)', PARTS)}
      {group('기타', OTHER)}
      <div className="flex flex-wrap gap-2">
        <Button disabled={busy} onClick={() => run(() => api.setAssume(a), '가정값을 저장했습니다. 모든 화면이 이 값으로 다시 계산합니다')}>
          <Save strokeWidth={2} aria-hidden />가정값 저장
        </Button>
        <Button variant="outline" disabled={busy} onClick={() => run(() => api.resetAssume(), '기본값으로 되돌렸습니다').then(() => setA(p.defaults))}>
          <RotateCcw strokeWidth={2} aria-hidden />기본값으로 되돌리기
        </Button>
      </div>
    </Disclosure>
  )
}

function PolicyDetail({ p, version }: { p: Policy; version: number }) {
  const [sid, setSid] = useState(p.rows[0].sid)
  const [override, setOverride] = useState(false)
  const [po, setPo] = useState<Record<string, number>>({})
  const [o, setO] = useState<PolicyOption | null>(null)
  useEffect(() => { setOverride(false); setPo({}) }, [sid])
  useEffect(() => {
    let live = true
    api.policyOption(sid, override && Object.keys(po).length ? po : undefined).then((x) => live && setO(x)).catch(() => {})
    return () => { live = false }
  }, [sid, override, po, version])
  const key = (name: string) => (name === '지금 두 부품 교체' ? '지금 두 부품 교체' : name.startsWith('다음 정기') ? '정기 교체 때' : 'N시간 뒤 교체')
  return (
    <section className="flex flex-col gap-4">
      <h3 className="text-xl font-semibold">설비별 근거</h3>
      <Field label="설비를 골라 근거 보기" htmlFor="pd-sid">
        <Select value={sid} onValueChange={setSid}>
          <SelectTrigger id="pd-sid" className="w-full sm:w-96"><SelectValue /></SelectTrigger>
          <SelectContent>{p.rows.map((r) => <SelectItem key={r.sid} value={r.sid}>{r.name} ({r.sid})</SelectItem>)}</SelectContent>
        </Select>
      </Field>
      <Check id="pd-ov" label="고장 확률을 직접 입력 (현장 판단으로 덮어쓰기)" checked={override} onChange={setOverride} />
      {!o || o.sid !== sid ? <p className="text-muted-foreground">불러오는 중...</p> : (
        <>
          {override && (
            <div className="grid gap-3 sm:grid-cols-3">
              {o.opts.map((x) => (
                <Field key={x.name} label={`${x.name}: 완료 전 고장 확률`} htmlFor={`po-${key(x.name)}`}>
                  <Input id={`po-${key(x.name)}`} type="number" min={0} max={1} step={0.01}
                    value={po[key(x.name)] ?? Number(x.p.toFixed(2))} onChange={(e) => setPo({ ...po, [key(x.name)]: Number(e.target.value) })} />
                </Field>
              ))}
            </div>
          )}
          <Notice tone={o.best.startsWith('다음 정기') ? 'blue' : 'orange'} title={`${o.sid} · 추천: ${o.best}${o.urgent ? ` (${o.urgent})` : ''}`}>
            기대 총비용 {money(o.total, o.unit)} = (1 − 고장 확률 {o.p.toFixed(2)}) × (정비 {n0(o.maint)} + 생산 차질 {n0(o.down)}) + {o.p.toFixed(2)} × 고장 시 손실 {n0(o.loss)}
          </Notice>
          <p className="text-sm text-muted-foreground">이 설비 진행: {o.jobState} · 배정·보류·부품 발주는 설비 화면에서 합니다.</p>
          <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
            <div className="flex flex-col gap-3">
              <Table min="44rem" head={['추천', '선택지', '정비 완료까지(h)', '정비 비용', '생산 차질 비용', '완료 전 고장 확률', '고장 시 손실', '기대 총비용', '안전 기준 충족']}
                rows={o.opts.map((x) => [x.rec ? '✔' : '', x.name, n0(x.doneH), n0(x.maint), n0(x.down), x.p.toFixed(2), n0(x.loss), n0(x.total), x.safe ? '예' : '아니요'])} />
              <div>
                <p className="font-semibold">산정 근거</p>
                <ul className="flex list-disc flex-col gap-1 pl-5 text-sm">
                  {o.opts.map((x) => <li key={x.name}>{x.name}: {x.basis}</li>)}
                  <li>고장 확률: 학습 세트에서 경보가 처음 켜진 뒤 실제 고장까지 걸린 시간 분포(카플란-마이어). 이 설비는 경보 뒤 {o.elapsed.toFixed(0)}시간 동안 고장이 없었다는 조건으로 계산</li>
                  {override && <li>고장 확률은 현장 판단으로 덮어쓴 값</li>}
                </ul>
              </div>
              {o.boundary && <p className="text-sm text-muted-foreground">{o.boundary}</p>}
            </div>
            <div className="flex flex-col gap-2">
              <FailCurve curve={o.curve} points={o.points} safety={o.safety} />
              <p className="text-sm text-muted-foreground">파란 선: 지금부터 h시간 안에 고장 날 확률 · 초록 점: 선택지별 정비 완료 시점 · 빨간 점선: 안전 기준</p>
            </div>
          </div>
        </>
      )}
    </section>
  )
}

/** 정책·비용: 가정값, 전체 정비 결정과 근거 */
export function PolicyPage({ version, onChanged }: { version: number; onChanged: () => void }) {
  const [p, setP] = useState<Policy | null>(null)
  const { busy, run } = useAction(onChanged)
  useEffect(() => { api.policy().then(setP).catch(() => {}) }, [version])
  if (!p) return <p className="text-muted-foreground">불러오는 중...</p>
  const fb = p.feedback
  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h2 className="text-2xl font-semibold">정책·비용: 가정값, 전체 정비 결정과 근거</h2>
        <p className="text-sm text-muted-foreground">비용 단위: 만 원 (현장 가정값, 아래 "가정값"에서 바꿀 수 있음). 과제 공통 비용 규칙(점수)으로 낸 모델 평가는 결과보고서에 있습니다.</p>
      </div>
      <AssumeForm key={JSON.stringify(p.assumptions)} p={p} onDone={onChanged} />
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-xl font-semibold"><CalendarClock className="size-5" strokeWidth={2} aria-hidden />전체 정비 결정 · {p.rows.length}건</h3>
        {p.rows.length === 0 ? <Notice tone="green">지금 경보가 켜진 설비가 없습니다.</Notice> : (
          <>
            <Table min="76rem" head={['순위', '설비', '조치', '추천', '방법', '재고', '정비 완료까지 (h)', `기대 총비용 (${p.unit})`, '완료 전 고장 확률', '제약', '첫 경보 (h)', '다음 정기 교체']}
              rows={p.rows.map((r) => [r.rank ?? '', `${r.name} (${r.sid})`, r.state, r.best, r.method, r.stock, n0(r.doneH), n0(r.cost), f2(r.p), r.constraint, n0(r.firstAlarmH), r.nextPlan])} />
            <p className="text-sm text-muted-foreground">
              정비팀마다 교체({p.replaceH.toFixed(0)}시간)를 하루 {p.perDay}번까지 하고(출근 인원 × 8시간 ÷ 16인·시), 넘는 설비는 다음 날로 넘어갑니다. 차례를 기다리는 설비는 그만큼 늦게 끝나는 것으로 다시 계산합니다.
            </p>
            <PolicyDetail p={p} version={version} />
          </>
        )}
      </section>
      <Disclosure title="조치 기록과 점검 되먹임" icon={<History className="size-4" strokeWidth={2} aria-hidden />}>
        <div className="grid gap-3 sm:grid-cols-3">
          <Metric label="점검 결과 (원인 확인됨)" value={`${fb.n}건`} sub={`되먹임은 ${p.minConfirmed}건 이상부터 씁니다 (가정값)`} />
          <Metric label="점검으로 원인 부품을 맞힌 비율" value={fb.accuracy === null ? '—' : fb.accuracy.toFixed(2)} />
          <Metric label="교체 방법 추천" value={fb.ready ? '점검 후 한 부품' : '두 부품'}
            sub={fb.breakeven !== null ? `맞힌 비율이 손익분기 ${fb.breakeven.toFixed(2)} 이상이고 기록이 ${p.minConfirmed}건 이상이면 "점검 후 한 부품"` : undefined} />
        </div>
        <p className="text-sm text-muted-foreground">센서로는 원인 부품을 구분할 수 없어 지금은 두 부품을 함께 교체합니다. 현장이 점검 결과와 교체 뒤 확인된 원인을 입력하면, 점검으로 부품을 맞히는 비율이 쌓이고 이 값이 부품 손익의 손익분기를 넘으면 추천이 바뀝니다.</p>
        {p.log.length === 0 ? <p>이 데이터 세트에서 지금 시각까지 기록된 조치가 없습니다.</p> : (
          <Table min="64rem" head={Object.keys(p.log[0])} rows={p.log.map((r) => Object.values(r).map((v) => (typeof v === 'number' ? v.toFixed(1) : v)))} />
        )}
      </Disclosure>
      {p.caution.length > 0 && (
        <section className="flex flex-col gap-3">
          <h3 className="flex items-center gap-2 text-xl font-semibold"><SearchCheck className="size-5" strokeWidth={2} aria-hidden />주의 설비 준비</h3>
          {p.needOrder.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border p-4">
              <p><b>주의 설비 부품 발주 {p.needOrder.length}건</b> — 경보 전에 정상 조달({p.lead.toFixed(0)}시간)로 재고를 채웁니다</p>
              <Button disabled={busy} onClick={() => run(() => api.orderCaution())}><Truck strokeWidth={2} aria-hidden />한 번에 발주</Button>
            </div>
          )}
          <Table min="52rem" head={['설비', '다음 정기 교체', '할 일', '재고', '부품']} rows={p.caution.map((c) => [`${c.name} (${c.sid})`, c.nextPlan, c.todo, c.stock, c.parts])} />
          <p className="text-sm text-muted-foreground">발주 시점: 주의 단계에서 두 부품 교체에 모자라는 재고가 있을 때(정상 조달 {p.lead.toFixed(0)}시간). 경보 때 재고가 없으면 긴급 조달({p.rushH.toFixed(0)}시간, 웃돈 {Math.round(p.rushMk * 100)}%)로 교체가 늦어집니다.</p>
        </section>
      )}
      {p.sim.length > 0 && (
        <Disclosure title="정책 시뮬레이션 상세 (검증 세트, 기본 가정값)">
          <Table min="72rem" head={['정책', `기대 총비용 (${p.unit})`, '예상 절감액 (① 대비)', '고장', '고장 손실', '정비 횟수', '지금 교체', '정기 교체 때', '기다리다 고장', '고장 전 조치 완료율', '계획 밖 정지(h)', '미확정 경보']}
            rows={p.sim.map((s) => ['정책', '비용', '예상 절감액 (① 대비)', '고장', '고장 손실', '정비 횟수', '지금 교체', '정기 교체 때', '기다리다 고장', '고장 전 조치 완료율', '계획 밖 정지(h)', '미확정 경보']
              .map((k) => (typeof s[k] === 'number' ? (k === '고장 전 조치 완료율' ? (s[k] as number).toFixed(2) : n0(s[k] as number)) : s[k])))} />
          <p className="text-sm text-muted-foreground">같은 데이터·같은 모델 예측으로 세 정책을 돌려 실제 고장 시각으로 채점했습니다. 준비 시간·안전 기준을 바꾼 민감도는 analysis/10_dss_policy.py에 있습니다.</p>
        </Disclosure>
      )}
      <Disclosure title="이 화면 보는 법">
        <ul className="flex list-disc flex-col gap-1 pl-5">
          <li>경보 설비마다 정비 선택지(지금 / N시간 뒤 / 다음 정기 교체 때)의 기대 총비용 = 정비 비용 + 생산 차질 비용 + 고장 확률 × 고장 시 손실을 비교해 추천합니다.</li>
          <li>안전 기준과 실행 가능성(정비팀·부품 준비)은 비용보다 먼저 지킵니다.</li>
          <li>판단은 검증 가능한 예측값과 계산·규칙으로만 하며, 최종 결정은 정비 책임자가 합니다.</li>
        </ul>
      </Disclosure>
    </section>
  )
}

/** 모델 정보·한계 */
export function ModelPage({ version }: { version: number }) {
  const [m, setM] = useState<ModelInfo | null>(null)
  const [lim, setLim] = useState(['높음', '중간', '낮음'])
  useEffect(() => { api.model().then(setM).catch(() => {}) }, [version])
  if (!m) return <p className="text-muted-foreground">불러오는 중...</p>
  const v = m.point, e = m.event
  const nf = (x: unknown) => (typeof x === 'number' ? x : Number(x))
  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h2 className="text-2xl font-semibold">모델 정보와 검증 세트 성능</h2>
        <p className="text-sm text-muted-foreground">모델 저장: {m.createdAt}{m.cloud ? ' · 클라우드: 미리 계산한 결과만 사용 (원자료·모델 파일 없음)' : ''}</p>
      </div>
      <ul className="flex list-disc flex-col gap-1 pl-5">
        <li>모델: {m.model} · 특징 {m.nFeatures}개 ({m.features}) · {m.step}시간마다 예측</li>
        <li>학습: 학습 세트 216개 시나리오의 예측 시점 {m.nTrain.toLocaleString('ko-KR')}개</li>
        <li>경보 기준: 24시간 안 고장 확률 ≥ {m.th.toFixed(2)}. 학습 세트 교차검증에서 점진 고장을 모두 잡고 진짜 오경보가 0인 값 중 가장 낮은 값(오경보 없이 가장 일찍 경보)</li>
        <li>검증 세트: 72개 시나리오, 학습에 쓰지 않고 평가에만 사용(모델·기준 선택은 학습 세트 교차검증)</li>
      </ul>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="AP (순위 능력)" value={nf(v['AP']).toFixed(3)} sub="기준 예제 0.806" />
        <Metric label="정밀도 / 재현율" value={`${nf(v['정밀도']).toFixed(2)} / ${nf(v['재현율 (점진 고장만)']).toFixed(2)}`} sub={`재현율은 점진 고장 기준. 돌발 고장 포함 ${nf(v['재현율']).toFixed(2)}`} />
        <Metric label="고장 감지" value={`${e['감지']}/${e['고장']}`} sub="돌발 고장은 신호가 없어 못 잡음" />
        <Metric label="리드타임 중앙값" value={`${nf(e['리드타임 중앙값(h)']).toFixed(0)}시간`} />
      </div>
      <ul className="flex list-disc flex-col gap-1 pl-5">
        <li>오경보 {v['FP']}개 중 {v['FP 중 이른 경보']}개는 고장 24시간보다 일찍 울린 이른 경보이고, 고장 예정이 없는데 울린 진짜 오경보는 {v['FP 중 진짜 오경보']}개입니다.</li>
        <li>놓친 시점 {v['FN']}개 중 {v['FN 중 돌발 고장']}개는 신호 없이 갑자기 고장 난 돌발 고장입니다.</li>
        <li>위 숫자는 정답을 알 수 있는 시점만으로 셌습니다. 관측 끝부분의 미확정 경보는 "부품 손익" 페이지에 있습니다.</li>
      </ul>
      {m.caution && (
        <section className="flex flex-col gap-3">
          <h3 className="flex items-center gap-2 text-xl font-semibold"><Bell className="size-5" strokeWidth={2} aria-hidden />2단계 경보: 주의 단계</h3>
          <ul className="flex list-disc flex-col gap-1 pl-5">
            <li>주의 모델: {m.caution.model} · {m.caution.features} 특징 {m.caution.nFeatures}개. "지금 열화 중인가"를 맞힙니다(정답: 진단 정답의 열화 구간, 학습 때만 사용)</li>
            <li>주의 기준: 지금 열화 중일 확률 ≥ {m.caution.th.toFixed(2)}. 학습 세트 교차검증에서 주의 중 {Math.round(m.caution.minPrecision * 100)}% 이상이 실제 열화인 값 중 가장 낮은 값</li>
            <li>헛걸음을 조금 허용하는 이유: 현장에서는 주의를 받으면 장비 상태를 확인하고, 설비가 돌면 다음 정비 때 함께 정비합니다. 설비를 세우거나 바로 발주하지 않으므로 헛걸음 비용은 점검 한 번입니다.</li>
          </ul>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric label="주의 정밀도" value={nf(m.caution.v['주의 정밀도']).toFixed(2)} sub={`목표 ${m.caution.minPrecision}`} />
            <Metric label="주의가 켜진 점진 고장" value={`${m.caution.v['주의로 감지']}/${m.caution.v['점진 고장']}`} />
            <Metric label="경보보다 먼저 (평균)" value={`${nf(m.caution.v['추가 준비 시간 평균(h)']).toFixed(1)}시간`} />
            <Metric label="첫 주의 (중앙값)" value={`고장 ${nf(m.caution.v['주의 리드타임 중앙값(h)']).toFixed(0)}시간 전`} />
          </div>
          <p className="text-sm text-muted-foreground">검증 세트 주의 {m.caution.v['주의 묶음']}번 중 헛걸음 {m.caution.v['열화 아닌 설비의 주의']}번. 대부분은 열화가 시작되기 몇 시간 전이었고, 일부는 끝까지 정상인 설비였습니다. 자세한 근거는 결과보고서 7장</p>
        </section>
      )}
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-xl font-semibold"><FileCheck className="size-5" strokeWidth={2} aria-hidden />검증 세트 고장별 결과</h3>
        <Table head={['시나리오', '고장 종류', '감지', '리드타임(h)']} rows={m.events.map((r) => [r['시나리오'], r['고장 종류'], r['감지'] ? '예' : '아니요', typeof r['리드타임(h)'] === 'number' ? (r['리드타임(h)'] as number).toFixed(1) : '—'])} />
        <p className="text-sm text-muted-foreground">리드타임: 고장 4시간 전까지 난 경보 중 가장 이른 경보부터 고장까지의 시간. 자세한 근거는 결과보고서 5·6장</p>
      </section>
      <section className="flex flex-col gap-3">
        <h3 className="text-xl font-semibold">한계와 대응 방안</h3>
        <ToggleGroup type="multiple" variant="outline" value={lim} onValueChange={setLim} aria-label="중요도" className="w-fit">
          {['높음', '중간', '낮음'].map((k) => <ToggleGroupItem className={picked} key={k} value={k}>{k}</ToggleGroupItem>)}
        </ToggleGroup>
        <Table min="72rem" head={['중요도', '구분', '한계', '영향', '지금 대응', '추가 대응', '근거']}
          rows={m.limits.filter((l) => lim.includes(l['중요도'])).map((l) => ['중요도', '구분', '한계', '영향', '지금 대응', '추가 대응', '근거'].map((k) => l[k]))} />
        <p className="text-sm text-muted-foreground">돌발 고장·원인 부품·리드타임처럼 모델만으로 풀 수 없는 한계는 운영(정기 교체 유지, 점검 결과 되먹임, 예비 재고와 주의 단계 발주)으로 메웁니다. 자세한 내용은 결과보고서 8장.</p>
      </section>
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-xl font-semibold"><ListTree className="size-5" strokeWidth={2} aria-hidden />온톨로지: 지금 상태를 하나의 그래프로</h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="트리플 수" value={m.ontology.triples.toLocaleString('ko-KR')} />
          <Metric label="설비 / 예측" value={`${m.ontology.counts['설비'] ?? 0} / ${m.ontology.counts['예측'] ?? 0}`} />
          <Metric label="경보 / 주의" value={`${m.ontology.counts['경보 (WARNING)'] ?? 0} / ${m.ontology.counts['주의 (CAUTION)'] ?? 0}`} />
          <Metric label="규칙 점검" value={m.ontology.bad.length ? `${m.ontology.bad.length}건 어김` : '통과'}
            sub={m.ontology.bad.join('; ') || '신호→부품 링크 없음, 주의·경보 겹침 없음, 정비 결정마다 추천 있음'} />
        </div>
        <ul className="flex list-disc flex-col gap-1 pl-5">
          <li>스키마: ontology/casting_dss.ttl (OWL). 객체 = 설비·부품·신호·예측·경보 단계·정비 결정·선택지·정비 작업·정기 교체 계획·고장, 링크 = 가진다·관측한다·근거·요청한다·추천한다·교체한다 등</li>
          <li>인스턴스: 이 화면의 지금 시각·데이터 세트·가정값으로 만든다(예측·신호·정비 기록·DSS 결정)</li>
          <li>질의 예 (SPARQL): 경보 설비가 필요로 하는 부품 = {Object.keys(m.ontology.parts).length ? Object.entries(m.ontology.parts).map(([k, n]) => `${k} ${n}개`).join(' · ') : '없음'}</li>
        </ul>
        <Button variant="outline" className="w-fit" asChild><a href="/api/ontology.ttl" download><Download strokeWidth={2} aria-hidden />지금 그래프 내려받기 (Turtle)</a></Button>
      </section>
    </section>
  )
}

/** 부품 손익: 원인 부품을 얼마나 맞히면 한 부품만 바꿔도 되나 */
export function PartsPage({ version }: { version: number }) {
  const [p, setP] = useState(0.5)
  const [c, setC] = useState<PartsCalc | null>(null)
  useEffect(() => {
    let live = true
    const id = setTimeout(() => api.parts(p).then((x) => live && setC(x)).catch(() => {}), 150)
    return () => { live = false; clearTimeout(id) }
  }, [p, version])
  return (
    <section className="flex flex-col gap-6">
      <h2 className="text-2xl font-semibold">부품 손익: 원인 부품을 얼마나 맞히면 한 부품만 바꿔도 되나</h2>
      {!c ? <p className="text-muted-foreground">불러오는 중...</p> : (
        <>
          <div className="flex flex-col gap-1">
            <p>검증 세트 결과(고장 {c.event['고장']}건 중 {c.n.n_det}건 감지, 고장과 무관한 교체 {c.n.n_needed}번, 진짜 오경보 {c.n.n_false}번)에 현장 비용 가정(만 원, "정책·비용"의 가정값)을 적용합니다.</p>
            <ul className="flex list-disc flex-col gap-1 pl-5">
              <li>경보가 나면 바로 교체합니다(정책: 경보 즉시 교체). 한 부품만 바꾸는데 틀리면, 불필요 정비가 되고 고장도 못 막습니다.</li>
              <li>정기 교체 직전에 울린 맞는 경보도 교체 한 번으로 셉니다. 예정 교체와 묶어 처리하면 이 비용은 줄어듭니다.</li>
              <li>센서로는 원인 부품을 구분할 수 없어, 부품을 맞힐 확률은 현장 점검 능력에 달려 있습니다.</li>
            </ul>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex items-baseline justify-between"><span id="pp-p" className="font-medium">원인 부품을 맞힐 확률</span><span className="text-lg font-semibold text-primary tabular-nums">{p.toFixed(2)}</span></div>
            <Slider aria-labelledby="pp-p" min={0.5} max={1} step={0.01} value={[p]} onValueChange={([v]) => setP(v)} />
            <p className="text-sm text-muted-foreground">0.5 = 둘 중 하나를 찍음, 1.0 = 항상 맞힘</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric label="한 부품만 교체" value={money(c.one, c.unit)} sub={`아무것도 안 함 대비 ${money(c.one - c.none, c.unit)}`} />
            <Metric label="두 부품 모두 교체" value={money(c.both, c.unit)} sub={`아무것도 안 함 대비 ${money(c.both - c.none, c.unit)}`} />
            <Metric label="아무것도 안 함" value={money(c.none, c.unit)} />
          </div>
          {c.unlabeled && (
            <Notice tone="orange" title="위 비용은 정답을 알 수 있는 구간만 센 값입니다.">
              관측이 끝나 결과를 알 수 없는 시점 {c.unlabeled['시점']}개에서도 경보가 {c.unlabeled['경보 시점']}번(경보 묶음 {c.unlabeled['경보 묶음']}개) 울렸고, 실제 운영이라면 교체가 일어납니다.
              두 부품 교체 기준으로 경보 묶음마다 약 {money(c.two, c.unit)}, 모두 {money(Number(c.unlabeled['경보 묶음']) * c.two, c.unit)}까지 더 들 수 있습니다.
              이 경보가 관측 뒤의 고장을 막았는지, "아무것도 안 함"에서 관측 뒤에 고장이 났는지는 알 수 없어 운영 비용은 확정할 수 없습니다.
            </Notice>
          )}
          {c.even !== null && <Notice tone="blue">부품을 맞힐 확률이 {c.even.toFixed(2)} 이상이면 한 부품만 바꾸는 쪽이 두 부품 모두 바꾸는 것보다 쌉니다.</Notice>}
          <PartsChart curve={c.curve as unknown as Record<string, number>[]} p={p} />
        </>
      )}
    </section>
  )
}

/** 하위 페이지 머리 (← 정비 의사결정 / 페이지 이름) */
export function SubPage({ name, onBack, children }: { name: ReactNode; onBack: () => void; children: ReactNode }) {
  return <div className="flex flex-col gap-4"><Crumb back="정비 의사결정" onBack={onBack} here={name} />{children}</div>
}
