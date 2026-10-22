import { Activity, Network, Radio, TrendingUp } from 'lucide-react'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

import { Bars, RiskChart, SignalChart } from '@/components/app/charts'
import { Disclosure, Metric, Notice } from '@/components/app/common'
import { api, clock, type Detail, type Meta } from '@/lib/api'

/** 설비 하나의 분석 근거 (dashboard.py equipment_detail): 지표 · 입력 감시 · 위험 흐름 · 네 신호 · 지금 신호 · 기여도 */
export function Analysis({ meta, sid, truth, version }: { meta: Meta; sid: string; truth: boolean; version: number }) {
  const [d, setD] = useState<Detail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    api.detail(sid, truth).then((x) => live && setD(x)).catch((e: Error) => live && setErr(e.message))
    return () => { live = false }
  }, [sid, truth, version])
  if (err) return <Notice tone="red">{err}</Notice>
  if (!d || d.sid !== sid) return <p className="text-muted-foreground">불러오는 중...</p>
  const short = d.stage.split(' (')[0]
  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="상태" value={short} sub={d.stage !== short ? d.stage : undefined} />
        <Metric label="24시간 내 고장 위험" value={d.risk === null ? '—' : `${Math.round(d.risk * 100)}%`}
          sub={d.lastPredH !== null ? `마지막 예측: ${clock(d.lastPredH)}` : undefined} />
        {meta.caution ? (
          <>
            <Metric label="지금 열화 중일 확률" value={d.degraded === null ? '—' : d.degraded.toFixed(2)} />
            <Metric label="첫 주의 / 첫 경보" value={<span className="text-xl">{clock(d.firstCautionH)} / {clock(d.firstAlarmH)}</span>} />
          </>
        ) : <Metric label="첫 경보" value={clock(d.firstAlarmH)} />}
      </div>
      {d.latched && <p className="text-sm text-muted-foreground">지금 확률은 경보 기준보다 낮지만, 이번 운전 구간에서 이미 경보가 켜져 정비 전까지 경보로 유지합니다.</p>}
      {d.oor && <Notice tone="orange" title="입력 감시: 센서 확인">{d.oor}</Notice>}

      <section className="flex flex-col gap-2">
        <h4 className="flex items-center gap-2 text-lg font-semibold"><TrendingUp className="size-5" strokeWidth={2} aria-hidden />{meta.caution ? '위험 확률과 열화 확률' : '위험 확률'}</h4>
        <RiskChart d={d} th={meta.threshold} cth={meta.cautionThreshold} endH={meta.endH} />
        <p className="text-sm text-muted-foreground">
          파란 선과 점: 24시간 안 고장 확률(점 색 = 단계){meta.caution ? ' · 노란 점선: 지금 열화 중일 확률' : ''} · 빨간 가로 점선: 경보 기준 {meta.threshold.toFixed(2)}
          {meta.cautionThreshold !== null ? ` · 노란 가로 점선: 주의 기준 ${meta.cautionThreshold.toFixed(2)}` : ''} · 회색 띠: 정비 · 빨간 실선: 고장
          {truth ? ' · 빨간 세로 점선: 앞으로 일어날 실제 고장 (정답)' : ''}
        </p>
      </section>
      <section className="flex flex-col gap-2">
        <h4 className="flex items-center gap-2 text-lg font-semibold"><Activity className="size-5" strokeWidth={2} aria-hidden />네 신호의 흐름</h4>
        <SignalChart d={d} signals={Object.values(meta.signals)} endH={meta.endH} />
        <p className="text-sm text-muted-foreground">굵은 선: 네 신호 평균 점수 · 점선: 정상 범위 경계(1). 정비 후 재가동하면 처음 2시간은 기준을 잡는 시간이라 비어 있습니다.</p>
      </section>
      <div className="grid gap-6 lg:grid-cols-2">
        <section className="flex flex-col gap-2">
          <h4 className="flex items-center gap-2 text-lg font-semibold"><Radio className="size-5" strokeWidth={2} aria-hidden />지금 신호 상태</h4>
          {d.signalsNow ? (
            <>
              <Bars rows={d.signalsNow} danger={(v) => v > 1} refX={1} label="지금 네 신호의 표준화 점수" />
              <p className="text-sm text-muted-foreground">{d.signalsNote}</p>
            </>
          ) : <p>{d.signalsNote}</p>}
        </section>
        <section className="flex flex-col gap-2">
          <h4 className="text-lg font-semibold">모델이 본 이유 (기여도)</h4>
          {d.contrib ? (
            <>
              <Bars rows={d.contrib} danger={(v) => v > 0} refX={0} label={`특징 묶음별 위험 기여도 (${d.contribUnit ?? ''})`} />
              <p className="text-sm text-muted-foreground">{d.contribNote}</p>
            </>
          ) : <p>{d.contribNote}</p>}
        </section>
      </div>
    </div>
  )
}

/** 근거 사슬 (온톨로지): 그림 + 번호 목록 */
export function OntologyView({ sid, version }: { sid: string; version: number }) {
  const { resolvedTheme } = useTheme()
  const dark = resolvedTheme === 'dark'
  const [svg, setSvg] = useState<string | null>(null)
  const [trace, setTrace] = useState<[string, string][] | null | undefined>(undefined)
  useEffect(() => {
    let live = true
    setTrace(undefined)
    api.ontology(sid, dark).then(async (o) => {
      if (!live) return
      setTrace(o.trace)
      if (o.dot) {
        const { instance } = await import('@viz-js/viz')
        const viz = await instance()
        if (live) setSvg(viz.renderString(o.dot, { format: 'svg', engine: 'dot' }))
      }
    }).catch(() => live && setTrace(null))
    return () => { live = false }
  }, [sid, dark, version])
  if (trace === undefined) return <p className="text-muted-foreground">불러오는 중...</p>
  if (trace === null) return <p className="text-sm text-muted-foreground">근거를 불러오지 못했습니다. 서버를 다시 시작하면 대부분 해결됩니다. 비용과 배정은 그대로 쓸 수 있습니다.</p>
  return (
    <div className="flex flex-col gap-4">
      {svg && (
        <div role="img" aria-label="온톨로지 연결: 설비, 단계, 예측, 신호, 정비 결정, 선택지"
          className="overflow-x-auto [&_svg]:h-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
      )}
      <ol className="flex list-decimal flex-col gap-1 pl-6">
        {trace.map(([k, v], i) => <li key={i}><span className="font-semibold">{k}</span> · {v}</li>)}
      </ol>
      <p className="text-sm text-muted-foreground">
        설비·예측·신호·경보·정비 결정·정기 교체 계획을 하나의 그래프(온톨로지, OWL/RDF)로 잇고, 그 연결을 그림과 목록으로 보여 줍니다.
        판단은 경보 기준과 의사결정 규칙이 하고, 그래프는 그 근거를 이어 보여 줍니다.
      </p>
    </div>
  )
}

export function OntologyDisclosure({ sid, version }: { sid: string; version: number }) {
  return (
    <Disclosure title="근거: 경보 → 예측 → 신호 → 정비 결정 (온톨로지)" icon={<Network className="size-4" strokeWidth={2} aria-hidden />}>
      <OntologyView sid={sid} version={version} />
    </Disclosure>
  )
}
