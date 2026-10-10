import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import DocMeta from './DocMeta.vue'
import './custom.css'

declare global {
  interface Window {
    __aiInfraNotes?: { onNavigate?: () => void }
  }
}

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'doc-top': () => h(DocMeta),
    }),
  enhanceApp({ router }) {
    // VitePress 是 SPA：章节间跳转不刷新页面，
    // 需要手动通知 notes.js 重新锚定本页高亮笔记
    if (typeof window === 'undefined') return // SSR 构建期无 window
    router.onAfterRouteChanged = () => {
      window.__aiInfraNotes?.onNavigate?.()
    }
  },
}
