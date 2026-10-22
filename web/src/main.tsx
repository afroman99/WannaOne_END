import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from '@/App'
import { ThemeProvider } from '@/components/theme-provider'
import '@/index.css'

// COMPASS ThemeProvider를 앱 최상위에: 기본은 시스템 설정, 라이트·다크 모두
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
)
