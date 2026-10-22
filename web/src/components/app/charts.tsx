import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter,
  Tooltip, XAxis, YAxis,
} from 'recharts'

import { clock, pct, type Detail } from '@/lib/api'

// 차트 색: COMPASS 토큰 (Primary · 경보 = destructive · 주의 = warning · 정상 = success · 무채색)
const axis = { stroke: 'var(--muted-foreground)', fontSize: 13 }
const tip = { background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 12, color: 'var(--popover-foreground)' }
const STAGE_FILL: Record<string, string> = {
  '경보 (WARNING)': 'var(--destructive)', '주의 (CAUTION)': 'var(--warning)', '정상 (NORMAL)': 'var(--muted-foreground)',
}
const SIGNAL_COLORS = ['var(--primary)', 'var(--warning)', 'var(--success)', 'var(--chart-3)']

function Bands({ d, fails = true }: { d: Detail; fails?: boolean }) {
  return (
    <>
      {d.maintenance.map((m) => (
        <ReferenceArea key={`b${m.start}`} x1={m.start} x2={m.endShown} fill="var(--muted-foreground)" fillOpacity={0.18} ifOverflow="hidden" />
      ))}
      {fails && d.maintenance.filter((m) => m.fail !== null).map((m) => (
        <ReferenceLine key={`f${m.start}`} x={m.fail!} stroke="var(--destructive)" strokeWidth={2} />
      ))}
      {d.truthFails.map((h) => <ReferenceLine key={`t${h}`} x={h} stroke="var(--destructive)" strokeDasharray="4 4" />)}
    </>
  )
}

/** 위험 확률(점 색 = 단계) · 열화 확률 · 경보·주의 기준 · 정비 띠 · 고장 */
export function RiskChart({ d, th, cth, endH }: { d: Detail; th: number; cth: number | null; endH: number }) {
  return (
    <div className="h-72" role="img" aria-label={`${d.name}의 위험 확률과 열화 확률 흐름`}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={d.risk_series} margin={{ left: -12, right: 8, top: 8 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="h" type="number" domain={[0, endH]} tickFormatter={(h) => `${h}h`} {...axis} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
          <Tooltip contentStyle={tip} labelFormatter={(h) => clock(Number(h))} formatter={(v) => pct(Number(v))} />
          <Legend />
          <Bands d={d} />
          <ReferenceLine y={th} stroke="var(--destructive)" strokeDasharray="6 3" />
          {cth !== null && <ReferenceLine y={cth} stroke="var(--warning)" strokeDasharray="6 3" />}
          <Line name="24시간 안 고장 확률" dataKey="risk" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Scatter name="단계" dataKey="risk" isAnimationActive={false} legendType="none">
            {d.risk_series.map((r, i) => <Cell key={i} fill={STAGE_FILL[r.stage] ?? 'var(--muted-foreground)'} />)}
          </Scatter>
          {cth !== null && (
            <Line name="열화 확률" dataKey="degraded" stroke="var(--warning)" strokeDasharray="2 2" strokeWidth={2} dot={false} isAnimationActive={false} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/** 네 신호의 표준화 점수 · 평균 · 정상 범위 경계(1) */
export function SignalChart({ d, signals, endH }: { d: Detail; signals: string[]; endH: number }) {
  return (
    <div className="h-72" role="img" aria-label={`${d.name}의 네 신호 흐름`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={d.signal_series} margin={{ left: -12, right: 8, top: 8 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="h" type="number" domain={[0, endH]} tickFormatter={(h) => `${h}h`} {...axis} />
          <YAxis {...axis} />
          <Tooltip contentStyle={tip} labelFormatter={(h) => clock(Number(h))} formatter={(v) => Number(v).toFixed(2)} />
          <Legend />
          <Bands d={d} fails={false} />
          <ReferenceLine y={1} stroke="var(--foreground)" strokeDasharray="2 2" />
          {signals.map((s, i) => (
            <Line key={s} name={s} dataKey={s} stroke={SIGNAL_COLORS[i]} strokeOpacity={0.85} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          ))}
          <Line name="네 신호 평균" dataKey="평균" stroke="var(--foreground)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** 가로 막대 (지금 신호 상태 · 기여도). danger: 막대가 빨개지는 조건 */
export function Bars({ rows, danger, label, refX }: {
  rows: { name: string; value: number }[]; danger: (v: number) => boolean; label: string; refX?: number
}) {
  return (
    <div style={{ height: 40 + rows.length * 32 }} role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }}>
          <CartesianGrid stroke="var(--border)" horizontal={false} />
          <XAxis type="number" {...axis} />
          <YAxis type="category" dataKey="name" width={150} {...axis} />
          <Tooltip contentStyle={tip} formatter={(v) => Number(v).toFixed(3)} cursor={{ fill: 'var(--muted)' }} />
          {refX !== undefined && <ReferenceLine x={refX} stroke="var(--foreground)" strokeDasharray="2 2" />}
          <Bar dataKey="value" name="값" isAnimationActive={false} radius={4}>
            {rows.map((r, i) => <Cell key={i} fill={danger(r.value) ? 'var(--destructive)' : 'var(--muted-foreground)'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/** 지금부터 h시간 안 고장 확률 곡선 · 선택지별 정비 완료 시점 · 안전 기준 */
export function FailCurve({ curve, points, safety }: { curve: { h: number; p: number }[]; points: { h: number; p: number; name: string }[]; safety: number }) {
  return (
    <div className="h-64" role="img" aria-label="지금부터 시간에 따른 고장 확률">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={curve} margin={{ left: -12, right: 8, top: 8 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="h" type="number" domain={[0, 48]} tickFormatter={(h) => `${h}h`} {...axis} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
          <Tooltip contentStyle={tip} labelFormatter={(h) => `${h}시간 뒤`} formatter={(v) => pct(Number(v))} />
          <ReferenceLine y={safety} stroke="var(--destructive)" strokeDasharray="6 3" />
          <Line name="그 안에 고장 날 확률" dataKey="p" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Scatter name="선택지 정비 완료" data={points} dataKey="p" fill="var(--success)" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/** 부품 손익: 원인 부품을 맞힐 확률에 따른 세 방식의 비용 */
export function PartsChart({ curve, p }: { curve: Record<string, number>[]; p: number }) {
  return (
    <div className="h-80" role="img" aria-label="원인 부품을 맞힐 확률에 따른 비용">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={curve} margin={{ left: 16, right: 8, top: 8 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="p" type="number" domain={[0.5, 1]} tickFormatter={(v) => v.toFixed(2)} {...axis} />
          <YAxis tickFormatter={(v) => Math.round(v).toLocaleString('ko-KR')} {...axis} />
          <Tooltip contentStyle={tip} labelFormatter={(v) => `맞힐 확률 ${Number(v).toFixed(2)}`} formatter={(v) => `${Math.round(Number(v)).toLocaleString('ko-KR')}만 원`} />
          <Legend />
          <ReferenceLine x={p} stroke="var(--foreground)" strokeDasharray="4 4" />
          <Line name="한 부품만 교체" dataKey="한 부품만 교체" stroke="var(--primary)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          <Line name="두 부품 모두 교체" dataKey="두 부품 모두 교체" stroke="var(--success)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          <Line name="아무것도 안 함" dataKey="아무것도 안 함" stroke="var(--muted-foreground)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
