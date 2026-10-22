import { createContext, useContext, useState } from 'react'

// 알림(토스트)과 버튼 동작 훅. 알림 상자 모양은 components/app/common.tsx (ToastProvider)
export type Tone = 'red' | 'orange' | 'green' | 'blue'
export const ToastCtx = createContext<(text: string, tone?: Tone, long?: boolean) => void>(() => {})
export const useToast = () => useContext(ToastCtx)

/** 버튼 동작: 요청 → 알림 → 다시 읽기. 실패하면 오류 알림 */
export function useAction(onDone: () => void) {
  const toast = useToast()
  const [busy, setBusy] = useState(false)
  const run = async (f: () => Promise<{ message?: string } | unknown>, ok?: string) => {
    setBusy(true)
    try {
      const r = (await f()) as { message?: string } | undefined
      toast(ok ?? r?.message ?? '저장했습니다', 'green')
      onDone()
      return true
    } catch (e) {
      toast((e as Error).message, 'red')
      return false
    } finally {
      setBusy(false)
    }
  }
  return { busy, run }
}
