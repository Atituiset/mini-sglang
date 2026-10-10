<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'

const chapter = ref<string | null>(null)
const minutes = ref<number | null>(null)
const fontStep = ref(1)

const SIZES = [15.5, 16.5, 18]
const KEY = 'ai-infra-font-step'
const route = useRoute()

function applyFont(step: number) {
  const doc = document.querySelector<HTMLElement>('.vp-doc')
  if (doc) doc.style.fontSize = `${SIZES[step]}px`
  fontStep.value = step
  try { localStorage.setItem(KEY, String(step)) } catch {}
}

function cycleFont() {
  applyFont((fontStep.value + 1) % SIZES.length)
}

function compute() {
  // reset first — pages without a chapter number must clear the old chip
  chapter.value = null
  minutes.value = null

  const h1 = document.querySelector('.vp-doc h1')
  const m = h1?.textContent?.match(/第\s*(\d+)\s*章/)
  if (m) chapter.value = m[1].padStart(2, '0')

  const text = document.querySelector('.vp-doc')?.textContent ?? ''
  const chars = text.replace(/\s+/g, '').length
  minutes.value = Math.max(1, Math.round(chars / 450))

  applyFont(fontStep.value) // .vp-doc node is replaced on each navigation
}

onMounted(() => {
  let saved = 1
  try { saved = Number(localStorage.getItem(KEY) ?? '1') } catch {}
  if (Number.isInteger(saved) && saved >= 0 && saved <= 2) fontStep.value = saved
  compute()
})

watch(
  () => route.path,
  async () => {
    await nextTick()
    compute()
  },
)
</script>

<template>
  <div class="doc-meta">
    <span v-if="chapter" class="doc-meta-chip">{{ chapter }}</span>
    <span v-if="minutes" class="doc-meta-item">约 {{ minutes }} 分钟</span>
    <button class="doc-meta-font" title="切换字号" @click="cycleFont">Aa</button>
  </div>
</template>

<style scoped>
.doc-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 0 8px;
  min-height: 24px;
}
.doc-meta-chip {
  background: var(--vp-c-brand-1);
  color: var(--vp-c-bg);
  font-family: var(--vp-font-family-mono);
  font-size: 12.5px;
  font-weight: 600;
  letter-spacing: 0.08em;
  padding: 2px 9px 2px 10px;
  border-radius: 999px;
  line-height: 1.6;
}
.doc-meta-item {
  color: var(--vp-c-text-3);
  font-size: 13px;
}
.doc-meta-font {
  margin-left: auto;
  border: 1px solid var(--vp-c-divider);
  background: transparent;
  color: var(--vp-c-text-3);
  font-size: 13px;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 999px;
  cursor: pointer;
  transition: color 0.2s, border-color 0.2s;
}
.doc-meta-font:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}
</style>
