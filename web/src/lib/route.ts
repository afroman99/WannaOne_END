import { useCallback, useEffect, useState } from 'react'

// 화면 위치는 주소의 # 뒤에 둔다 → 브라우저·마우스의 뒤로 가기/앞으로 가기가 된다 (dashboard.py의 ?view=…와 같은 역할)
//   #/m                 설비관리자 홈 (설비 상태)     #/m/orders · #/m/staff   홈의 다른 범주
//   #/m/stage/경보      상태별 설비 목록              #/m/assign[/S0294]       배정할 시나리오 목록 (+ 고른 설비)
//   #/m/eq/S0294        설비 화면 (정비 판단)         #/m/page/설비 상세/S0294 · 정책·비용 · 부품 손익 · 모델 정보·한계
//   #/w                 작업자 홈                     #/w/list/지금 · #/w/task/5 · #/w/task/5/report · #/w/task/5/stop
export function useRoute(): [string[], (path: string) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  const [parts, setParts] = useState(read)
  useEffect(() => {
    const on = () => setParts(read())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const go = useCallback((path: string) => {
    const next = '#/' + path.split('/').filter(Boolean).map(encodeURIComponent).join('/')
    if (window.location.hash !== next) window.location.hash = next
    window.scrollTo({ top: 0 })
  }, [])
  return [parts, go]
}
