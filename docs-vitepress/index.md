---
layout: home

hero:
  name: mini-sglang
  text: 源码学习指南
  tagline: 面向 AI Infra 初学者——以约 8000 行 Python 的教学精简版 SGLang 为教材，行号级精读一个功能完整的现代 LLM 推理引擎
  actions:
    - theme: brand
      text: 开始阅读
      link: /introduction
    - theme: alt
      text: 整体架构
      link: /ch01-architecture

features:
  - title: 基础篇
    details: 跟踪一条请求从 HTTP 进入到文本返回的完整链路；理解 OpenAI 兼容服务接口与 TTFT/TPOT/吞吐等性能指标。
    link: /ch01-architecture
  - title: 核心篇
    details: 推理引擎的三大支柱：调度器与连续批处理、分页 KV Cache 与 RadixCache、Attention 后端与核心算子，外加 CUDA Graph 与采样。
    link: /ch02-scheduler
  - title: 进阶篇
    details: 模型代码如何组织、权重如何加载、MoE 如何实现，以及基于 NCCL 的多卡张量并行。
    link: /ch05-models
  - title: 附录
    details: 术语表：用通俗语言解释 token、KV Cache、PagedAttention 等核心概念。
    link: /appendix-glossary
---
