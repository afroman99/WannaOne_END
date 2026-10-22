import { Badge } from '@/components/ui/badge'
import type { ShortStage } from '@/lib/api'
import { cn } from '@/lib/utils'

// 화면에 보일 단계 이름: 정상(NORMAL) → 주의(CAUTION, 열화 감지) → 경보(WARNING, 24시간 고장 위험)
const STAGE_LABEL: Record<ShortStage, string> = {
  경보: '경보 (WARNING)',
  주의: '주의 (CAUTION)',
  정상: '정상 (NORMAL)',
  '정비 중': '정비 중',
  '정비 후 예측 대기': '정비 후 예측 대기',
}

// 상태색: 경보는 COMPASS --destructive, 주의·정상은 index.css의 --warning·--success (COMPASS에 없는 상태색)
const STAGE_DOT: Record<ShortStage, string> = {
  경보: 'bg-destructive',
  주의: 'bg-warning',
  정상: 'bg-success',
  '정비 중': 'bg-muted-foreground',
  '정비 후 예측 대기': 'bg-muted-foreground',
}

export function StageDot({ stage, className }: { stage: ShortStage; className?: string }) {
  return <span aria-hidden className={cn('inline-block size-2.5 shrink-0 rounded-full', STAGE_DOT[stage], className)} />
}

export function StageBadge({ stage }: { stage: ShortStage }) {
  return (
    <Badge variant={stage === '경보' ? 'destructive' : 'outline'}>
      <StageDot stage={stage} />
      {STAGE_LABEL[stage]}
    </Badge>
  )
}
