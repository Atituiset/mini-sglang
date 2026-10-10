import { defineConfig } from 'vitepress'

const base = '/mini-sglang/'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  lang: 'zh-CN',
  title: 'mini-sglang 源码学习指南',
  titleTemplate: '从请求生命周期到调度器、KV Cache 与张量并行',
  description:
    '面向 AI Infra 初学者的 mini-sglang 源码学习指南：以约 8000 行 Python 实现的现代 LLM 推理引擎为教材，逐章精读整体架构、调度器、KV Cache、Attention 后端、CUDA Graph、模型实现与张量并行。',

  base,
  lastUpdated: true,
  cleanUrls: true,
  srcExclude: ['README.md'],

  markdown: {
    // shiki 3.x 不再内置 cuda 语法, 用 cpp 高亮兜底
    languageAlias: { cuda: 'cpp' },
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#2e7d5b', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#18191c', media: '(prefers-color-scheme: dark)' }],
    ['meta', { property: 'og:title', content: 'mini-sglang 源码学习指南' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '面向 AI Infra 初学者的 mini-sglang 源码学习指南：整体架构、调度器、KV Cache、Attention 后端、CUDA Graph、模型实现与张量并行的行号级源码精读。',
      },
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
    // 学习笔记悬浮组件（localStorage），VitePress SPA 路由切换由 theme/index.ts 通知其重挂载
    ['script', { src: `${base}notes.js` }],
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/favicon.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '导读', link: '/introduction' },
      { text: '整体架构', link: '/ch01-architecture' },
      { text: '调度器', link: '/ch02-scheduler' },
    ],

    sidebar: [
      { text: '导读：如何阅读本书', link: '/introduction' },
      {
        text: '基础篇',
        collapsed: false,
        items: [
          { text: '整体架构与请求生命周期', link: '/ch01-architecture' },
          { text: '服务化接口与性能评测', link: '/ch08-server-bench' },
        ],
      },
      {
        text: '核心篇',
        collapsed: false,
        items: [
          { text: '调度器与连续批处理', link: '/ch02-scheduler' },
          { text: 'KV Cache 与显存管理', link: '/ch03-kvcache' },
          { text: 'Attention 后端与核心算子', link: '/ch04-attention' },
          { text: 'CUDA Graph 与采样', link: '/ch07-graph-sample' },
        ],
      },
      {
        text: '进阶篇',
        collapsed: false,
        items: [
          { text: '模型实现与权重加载', link: '/ch05-models' },
          { text: '分布式与张量并行', link: '/ch06-distributed' },
        ],
      },
      {
        text: '附录',
        collapsed: false,
        items: [{ text: '术语表', link: '/appendix-glossary' }],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本页大纲',
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    editLink: {
      pattern: 'https://github.com/Atituiset/mini-sglang/edit/main/docs-vitepress/:path',
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/Atituiset/mini-sglang' }],

    docFooter: { prev: '上一页', next: '下一页' },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' },
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    skipToContentLabel: '跳转到内容',

    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2026 mini-sglang 源码学习指南',
    },
  },
})
