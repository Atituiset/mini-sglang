# Mini-SGLang 架构深度解析：大模型推理框架入门与进阶指南

> 本文档面向大模型推理领域的初学者，从零开始讲解 LLM 推理框架的核心概念，并以 mini-sglang 项目为实例深入剖析其实现细节。

---

## 目录

1. [什么是 LLM 推理框架](#1-什么是-llm-推理框架)
2. [推理的核心流程：Prefill 与 Decode](#2-推理的核心流程prefill-与-decode)
3. [Mini-SGLang 整体架构](#3-mini-sglang-整体架构)
4. [进程架构与请求生命周期](#4-进程架构与请求生命周期)
5. [调度器：推理框架的大脑](#5-调度器推理框架的大脑)
6. [KV Cache：推理性能的关键](#6-kv-cache推理性能的关键)
7. [Radix Tree 前缀缓存：智能复用](#7-radix-tree-前缀缓存智能复用)
8. [分块预填充（Chunked Prefill）](#8-分块预填充chunked-prefill)
9. [CUDA Graph：消除 CPU 开销](#9-cuda-graph消除-cpu-开销)
10. [重叠调度（Overlap Scheduling）](#10-重叠调度overlap-scheduling)
11. [张量并行（Tensor Parallelism）](#11-张量并行tensor-parallelism)
12. [注意力后端：FlashInfer 与 FlashAttention](#12-注意力后端flashinfer-与-flashattention)
13. [模型实现与权重加载](#13-模型实现与权重加载)
14. [采样策略](#14-采样策略)
15. [关键技术对比总结](#15-关键技术对比总结)
16. [附录：目录结构速查](#附录目录结构速查)

---

## 1. 什么是 LLM 推理框架

当你向 ChatGPT、DeepSeek 等大模型发送一条消息时，背后发生的事情远不止"把文本扔给模型"那么简单。一个生产级的 LLM 推理系统需要解决以下核心问题：

| 问题 | 为什么重要 |
|------|-----------|
| **如何高效管理 GPU 显存？** | 模型参数 + KV Cache 可能占满显存，需要精细分配 |
| **如何同时服务多个用户？** | 多个请求需要并行处理，但又不能互相干扰 |
| **如何复用重复计算？** | 相同的系统提示（system prompt）不需要每次重新计算 |
| **如何降低延迟？** | CPU 与 GPU 之间的协调、内核启动开销都会增加延迟 |
| **如何支持多卡并行？** | 单卡放不下大模型，需要跨多张 GPU 协作 |

**Mini-SGLang** 是 [SGLang](https://github.com/sgl-project/sglang) 的精简重实现（约 5000 行 Python），保留了生产级推理框架的所有关键特性，同时代码量极小，非常适合学习。

---

## 2. 推理的核心流程：Prefill 与 Decode

理解 LLM 推理，最关键的概念是 **Prefill（预填充）** 和 **Decode（解码）** 两个阶段：

```
用户输入: "请给我讲一个关于"
模型输出: "勇敢的小猫的故事。"

┌─────────────────────────────────────────────────────┐
│  Prefill 阶段（并行处理输入）                         │
│  输入: [请, 给, 我, 讲, 一, 个, 关, 于]              │
│  处理: 所有 token 同时送入模型，一次性计算              │
│  输出: 第一个生成 token "勇"                          │
│  特点: 计算密集型（compute-bound），GPU 算力满载       │
│  速度: 快（并行处理）                                 │
├─────────────────────────────────────────────────────┤
│  Decode 阶段（逐个生成 token）                        │
│  步骤1: [请,...,于, 勇] → "敢"                       │
│  步骤2: [请,...,于, 勇, 敢] → "的"                   │
│  步骤3: [请,...,于, 勇, 敢, 的] → "小"               │
│  ...                                                 │
│  特点: 内存密集型（memory-bound），每次只处理 1 token  │
│  速度: 慢（串行处理），是推理瓶颈                      │
└─────────────────────────────────────────────────────┘
```

**核心洞察**：Prefill 阶段因为并行处理所有输入 token，GPU 计算单元被充分利用；而 Decode 阶段每次只生成一个 token，大量时间花在从显存读取权重和 KV Cache 上，GPU 计算单元利用率很低。

这也是为什么推理框架要把 Prefill 和 Decode 分开管理——它们有着截然不同的性能特征和优化策略。

---

## 3. Mini-SGLang 整体架构

```
python/minisgl/
├── __main__.py          # 入口：python -m minisgl serve/shell
├── core.py              # 核心数据结构：Req, Batch, Context
├── env.py               # 环境变量配置（MINISGL_*）
│
├── server/              # ── 服务层 ──
│   ├── launch.py        # 启动所有子进程，搭建 ZMQ 通信
│   ├── api_server.py    # FastAPI HTTP 服务（OpenAI 兼容 API）
│   └── args.py          # 命令行参数定义
│
├── engine/              # ── 推理引擎 ──
│   ├── engine.py        # 模型加载、初始化、前向推理
│   ├── config.py        # 引擎配置
│   ├── graph.py         # CUDA Graph 捕获与重放
│   └── sample.py        # Token 采样（temperature, top-k, top-p）
│
├── scheduler/           # ── 调度器（核心） ──
│   ├── scheduler.py     # 主循环、重叠调度
│   ├── prefill.py       # Prefill 管理、分块预填充
│   ├── decode.py        # Decode 管理
│   ├── cache.py         # 页面分配、缓存淘汰
│   ├── table.py         # 请求表管理
│   └── io.py            # ZMQ I/O、跨 rank 同步
│
├── attention/           # ── 注意力计算后端 ──
│   ├── base.py          # 抽象接口
│   ├── fi.py            # FlashInfer 后端（推荐）
│   ├── fa.py            # FlashAttention 后端
│   └── trtllm.py        # TensorRT-LLM 后端
│
├── kvcache/             # ── KV 缓存 ──
│   ├── radix_cache.py   # Radix Tree 前缀缓存（核心优化）
│   ├── naive_cache.py   # 朴素缓存（无复用）
│   └── mha_pool.py      # GPU 显存 KV Cache 池
│
├── models/              # ── 模型实现 ──
│   ├── llama.py         # LLaMA
│   ├── qwen2.py         # Qwen2
│   ├── qwen3.py         # Qwen3
│   ├── qwen3_moe.py     # Qwen3 MoE（混合专家）
│   └── mistral.py       # Mistral
│
├── layers/              # ── 神经网络层（支持 TP） ──
│   ├── linear.py        # 张量并行线性层
│   ├── attention.py     # 注意力层
│   ├── rotary.py        # 旋转位置编码（RoPE）
│   ├── embedding.py     # 词嵌入
│   ├── norm.py          # RMSNorm
│   └── moe.py           # MoE 层
│
├── tokenizer/           # ── 分词器 ──
│   ├── tokenize.py      # 文本 → token IDs
│   └── detokenize.py    # token IDs → 文本
│
├── distributed/         # ── 多 GPU 通信 ──
│   ├── info.py          # rank/size 信息
│   └── impl.py          # NCCL/PyNCCL 通信实现
│
├── kernel/              # ── 自定义 CUDA 内核 ──
│   ├── store.py         # KV Cache 存储内核
│   ├── radix.py         # Radix Tree 快速比较内核
│   └── triton/          # Triton 内核（MoE 等）
│
├── message/             # ── ZMQ 消息协议 ──
│   ├── backend.py       # 后端消息（用户请求、退出等）
│   ├── frontend.py      # 前端消息（回复）
│   └── tokenizer.py     # 分词器消息
│
└── llm/                 # ── 离线生成 API ──
    └── llm.py           # LLM 类（无需启动服务即可使用）
```

---

## 4. 进程架构与请求生命周期

Mini-SGLang 采用**多进程架构**，各进程通过 **ZMQ**（控制消息）和 **NCCL**（GPU 张量数据）通信：

```
                    ┌──────────────┐
                    │  API Server  │  ← FastAPI + uvicorn
                    │  (HTTP 入口)  │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │  Tokenizer   │ │Scheduler │ │  DeTokenizer  │
    │  Worker ×N   │ │  ×TP_size│ │  Worker ×1   │
    │  文本→token  │ │  推理核心 │ │  token→文本   │
    └──────────────┘ └────┬─────┘ └──────────────┘
                          │
                    ┌─────┴──────┐
                    │   Engine   │  ← 每张 GPU 一个
                    │  模型推理   │
                    └────────────┘
```

### 一个请求的完整生命周期

```
1. 用户发送 HTTP 请求 → API Server
2. API Server → Tokenizer Worker（文本 → token IDs）
3. Tokenizer → Scheduler Rank 0（发送 token IDs）
4. Rank 0 广播给所有 Scheduler Rank（多卡时）
5. 所有 Scheduler 执行调度 + Engine.forward()
6. Rank 0 收集输出 token → DeTokenizer
7. DeTokenizer（token ID → 文本）→ API Server
8. API Server 流式返回给用户
```

启动代码（`server/launch.py`）展示了这一过程：

```python
# 为每个 GPU rank 启动一个 Scheduler 进程
for i in range(world_size):
    mp.Process(target=_run_scheduler, args=(new_args, ack_queue)).start()

# 启动 Tokenizer 工作进程
for i in range(num_tokenizers):
    mp.Process(target=tokenize_worker, ...).start()

# 启动 DeTokenizer 工作进程（只有1个）
mp.Process(target=tokenize_worker, ..., name="minisgl-detokenizer-0").start()
```

---

## 5. 调度器：推理框架的大脑

调度器（`scheduler/scheduler.py`）是整个框架最核心的组件，它负责决定**何时处理哪些请求**。

### 5.1 核心数据结构

```python
# core.py — 一个推理请求
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor   # 输入 token（CPU 上）
    table_idx: int            # 在请求表中的索引
    cached_len: int           # 已缓存的 token 数（已算过的部分不需要重算）
    output_len: int           # 最大输出长度
    uid: int                  # 请求唯一 ID
    sampling_params: SamplingParams  # 采样参数
    cache_handle: BaseCacheHandle    # KV Cache 句柄

# core.py — 一个推理批次
@dataclass
class Batch:
    reqs: List[Req]                    # 批次中的请求列表
    phase: Literal["prefill", "decode"] # 当前阶段
    input_ids: torch.Tensor            # 合并后的输入 token
    positions: torch.Tensor            # 位置编码
    attn_metadata: BaseAttnMetadata    # 注意力计算元数据
```

### 5.2 调度流程

```python
# scheduler.py — 简化的调度逻辑
def _schedule_next_batch(self):
    # 优先调度 prefill（新请求），然后是 decode（继续生成）
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

调度器由四个子管理器协同工作：

| 管理器 | 职责 |
|--------|------|
| **PrefillManager** | 管理等待预填充的新请求，支持分块 |
| **DecodeManager** | 管理正在逐 token 生成的请求 |
| **CacheManager** | 管理页面分配、前缀缓存匹配与淘汰 |
| **TableManager** | 管理请求表（token 池、页面表） |

---

## 6. KV Cache：推理性能的关键

### 6.1 什么是 KV Cache？

Transformer 的自注意力机制在计算时需要所有历史 token 的 Key 和 Value：

```
生成第 5 个 token 时，注意力计算需要：
  当前 token 的 Q（Query）
  所有历史 token 的 K（Key）和 V（Value）
  K1, K2, K3, K4, K5
  V1, V2, V3, V4, V5
```

**KV Cache** 就是把这些 K 和 V 缓存在 GPU 显存中，避免每次生成新 token 时重新计算历史 token 的 K 和 V。没有 KV Cache，生成第 N 个 token 的计算量是 O(N²)，有了 KV Cache 则是 O(N)。

### 6.2 页式管理（Paged KV Cache）

Mini-SGLang 借鉴操作系统的**虚拟内存分页**思想管理 KV Cache：

```
┌──────────────────────────────────────────┐
│           GPU 显存（KV Cache 池）          │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐   │
│  │Page││Page││Page││Page││Page││Page│   │
│  │ 0  ││ 1  ││ 2  ││ 3  ││ 4  ││ 5  │   │
│  └────┘└────┘└────┘└────┘└────┘└────┘   │
│     ↑         ↑              ↑           │
│     │         │              │           │
│  ┌──┴─────────┴──┐    ┌─────┴──────┐    │
│  │  Request A    │    │ Request B  │    │
│  │  Page Table:  │    │Page Table: │    │
│  │  [0, 1]       │    │ [2, 4, 5]  │    │
│  └───────────────┘    └────────────┘    │
└──────────────────────────────────────────┘
```

**关键优势**：
- **非连续存储**：请求的 KV Cache 不需要占用连续显存，减少内存碎片
- **按需分配**：只在需要时分配新页，避免预分配浪费
- **灵活共享**：不同请求可以共享相同的页（前缀缓存）

### 6.3 页面表实现

```python
# engine.py — 页面表初始化
# page_table[req_idx, pos] = kv_cache 中的物理位置
self.page_table = torch.zeros(
    (max_running_req + 1, aligned_max_seq_len),  # +1 给 dummy req
    dtype=torch.int32,
    device=self.device,
)
```

### 6.4 GPU KV Cache 池

```python
# mha_pool.py — 实际的 GPU 显存池
# 形状: [2, num_layers, num_pages, page_size, num_kv_heads, head_dim]
#   2 = Key + Value
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device, dtype=dtype,
)
```

### 6.5 页面分配与回收

```python
# cache.py — CacheManager 核心逻辑

# 分配新页面
def _allocate(self, needed_pages):
    if needed_pages > len(self.free_slots):
        # 空闲页不够，需要从缓存中淘汰
        evicted = self.prefix_cache.evict(
            (needed_pages - free_pages) * self.page_size
        )
        self.free_slots = torch.cat([self.free_slots, evicted_pages])
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated

# 惰性释放（延迟 free 以减少同步开销）
@contextmanager
def lazy_free_region(self):
    lazy_free_list = []
    self._free = lambda indices: lazy_free_list.append(indices[::page_size])
    try:
        yield  # 在这个上下文中，_free 只是记录要释放的页
    finally:
        del self._free
        self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
```

---

## 7. Radix Tree 前缀缓存：智能复用

这是 Mini-SGLang（以及 SGLang/vLLM）最精彩的优化之一。

### 7.1 问题：重复计算

想象一个场景：1000 个请求都使用相同的系统提示（如 "You are a helpful assistant."）：

```
请求1: [系统提示] + "什么是Python？"       → 系统提示部分重复计算！
请求2: [系统提示] + "解释量子力学"          → 系统提示部分重复计算！
请求3: [系统提示] + "写一首诗"             → 系统提示部分重复计算！
```

没有前缀缓存时，每个请求都要独立计算系统提示的 KV Cache，既浪费算力又浪费显存。

### 7.2 Radix Tree 解决方案

**Radix Tree（基数树）** 是一种压缩前缀树，它能自动识别和共享公共前缀：

```
                    [root]
                       │
            ┌──────────┴──────────┐
            │                     │
     [You are a help]       [The weather]
     (共享4个页)             (共享3个页)
            │                     │
     ┌──────┴──────┐         ┌───┴───┐
  [ful assis]    [ful AI]  [today] [yesterday]
  (请求A,B)      (请求C)    (请求D)  (请求E)
```

**核心操作**：
1. **前缀匹配（match_prefix）**：新请求到来时，在树中查找最长公共前缀
2. **前缀插入（insert_prefix）**：将新生成的 KV Cache 插入树中
3. **节点分裂（split_at）**：当部分匹配时，将节点分裂以共享公共部分
4. **缓存淘汰（evict）**：当显存不足时，淘汰最近最少使用（LRU）的叶节点

### 7.3 核心实现

```python
# radix_cache.py — 树节点
class RadixTreeNode:
    children: Dict[Any, RadixTreeNode] = {}  # 子节点
    ref_count: int = 0                        # 引用计数（0=可淘汰）
    timestamp: int                            # 用于 LRU 淘汰

    def split_at(self, pos: int) -> RadixTreeNode:
        """将节点在 pos 位置分裂，公共前缀成为新父节点"""
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(self.parent)      # 接替自己的位置
        new_node.ref_count = self.ref_count
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)             # 自己成为新节点的子节点
        return new_node

# radix_cache.py — 前缀匹配（树遍历）
def _tree_walk(self, input_ids):
    node, prefix_len = self.root_node, 0
    while prefix_len < len(input_ids):
        child = node.children.get(key_fn(input_ids[prefix_len:]))
        if child is None:
            return node, prefix_len        # 没有匹配的子节点，返回
        node = child
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len
        if match_len != node.length:       # 部分匹配，需要分裂
            node = node.split_at(match_len)
    return node, prefix_len

# radix_cache.py — LRU 淘汰
def evict(self, size: int):
    leave_nodes = self._collect_leave_nodes_for_evict()
    heapq.heapify(leave_nodes)            # 按时间戳建堆（最老的优先淘汰）
    evicted_indices = []
    while evicted_size < size:
        node = heapq.heappop(leave_nodes)  # 淘汰最老的叶节点
        evicted_indices.append(node.value)
        # 如果父节点因此变成叶子且 ref_count=0，也加入候选
        if parent.is_leaf() and parent.ref_count == 0:
            heapq.heappush(leave_nodes, parent)
    return torch.cat(evicted_indices)
```

### 7.4 引用计数与保护机制

- **ref_count > 0**：该节点正在被某个请求使用（**受保护**，不可淘汰）
- **ref_count = 0**：该节点没有被任何请求使用（**可淘汰**）
- 当请求开始使用某个前缀时，路径上所有节点的 `ref_count += 1`
- 当请求完成时，路径上所有节点的 `ref_count -= 1`

这确保了**正在使用的 KV Cache 绝不会被淘汰**。

---

## 8. 分块预填充（Chunked Prefill）

### 8.1 问题：长提示阻塞

当一个请求的输入特别长（如 100K token 的文档）时，如果一次性做 prefill，会：
- 长时间占用 GPU，阻塞其他短请求
- 可能超出单次计算的最大 token 预算
- 导致延迟飙升

### 8.2 解决方案：分块处理

将长输入拆分成多个 chunk，逐步处理：

```
输入: [A B C D E F G H I J K L]  (12 tokens, budget=5)

Round 1 (Prefill): 处理 [A B C D E]，生成 ChunkedReq
Round 2 (Prefill): 处理 [F G H I J]，继续 ChunkedReq
Round 3 (Prefill): 处理 [K L]，完成 prefill，开始 decode
```

### 8.3 实现

```python
# prefill.py — ChunkedReq 是特殊的 Req 子类
class ChunkedReq(Req):
    def append_host(self, next_token):
        raise NotImplementedError("ChunkedReq 不应该被采样")
    
    @property
    def can_decode(self):
        return False  # 还没 prefill 完，不能进入 decode 阶段

# prefill.py — PrefillAdder 控制每次能处理多少 token
class PrefillAdder:
    token_budget: int      # 本次可用的 token 预算
    
    def _add_one_req(self, pending_req, cache_handle, table_idx, cached_len):
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len    # 是否需要分块
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size         # 扣减预算
        return CLS(...)
```

**效果**：短请求不必等长请求处理完，每轮 prefill 都在预算内处理多个请求的一部分，实现**公平调度**。

---

## 9. CUDA Graph：消除 CPU 开销

### 9.1 问题：Decode 阶段的 CPU 瓶颈

Decode 阶段每次只生成 1 个 token，但每次前向推理都需要：
1. CPU 发送内核启动命令给 GPU
2. CPU 准备参数、检查条件
3. GPU 执行计算（非常快）
4. CPU 准备下一次的命令

在这个流程中，**CPU 的开销可能比 GPU 计算时间还长**，导致 GPU 空闲等待。

### 9.2 CUDA Graph 原理

CUDA Graph 的工作方式类似"录制与回放"：

```
┌─ 录制阶段（只做一次）──────────────────────┐
│  1. 模拟一次 decode 前向推理               │
│  2. 记录所有 GPU 操作（内核启动、内存拷贝等）│
│  3. 保存为 CUDA Graph                      │
└────────────────────────────────────────────┘

┌─ 回放阶段（每次 decode 时）─────────────────┐
│  1. 更新输入缓冲区的数据（token IDs 等）     │
│  2. 调用 graph.replay() — 一次性提交所有操作  │
│  3. 读取输出缓冲区的结果                     │
└────────────────────────────────────────────┘
```

**效果**：将 N 次 CPU→GPU 内核启动合并为 1 次，消除 CPU 调度开销。

### 9.3 实现

```python
# graph.py — 录制 CUDA Graph
class GraphRunner:
    def _capture_graphs(self, max_seq_len, vocab_size, model):
        # 为不同的 batch size 分别录制 graph
        # 例如 bs = [1, 2, 4, 8, 16, 32, ...]
        for bs in self.graph_bs_list:
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            self.buffer.set_batch(batch)
            
            # 预热 + 录制
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                self.buffer.logits[:bs] = model.forward()
            
            self.graph_map[bs] = graph  # 保存录制的 graph

    def replay(self, batch):
        # 回放：只需复制输入数据 + 重放 graph
        self.buffer.copy_from(batch)         # 更新输入缓冲区
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()                           # 一键重放！
        return self.buffer.logits[:batch.size]
```

### 9.4 Batch Size 填充

CUDA Graph 要求回放时的 batch size 与录制时完全一致。因此需要将实际 batch **填充（pad）** 到录制的某个大小：

```python
# graph.py — 填充到最近的录制大小
def pad_batch(self, batch):
    padded_size = next(
        bs for bs in self.graph_bs_list if bs >= batch.size
    )
    # 用 dummy_req 填充多余位置
    batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
```

---

## 10. 重叠调度（Overlap Scheduling）

### 10.1 问题：CPU 与 GPU 的串行等待

在朴素实现中，CPU 处理（接收消息、准备数据）和 GPU 计算是串行的：

```
朴素调度:
CPU: [处理消息][准备batch]    [处理结果][处理消息][准备batch]    [处理结果]
GPU:                  [推理]                        [推理]
     ├─────────────── 时间 ────────────────────────────────►
```

GPU 在 CPU 处理期间是空闲的，造成浪费。

### 10.2 重叠调度的思路

使用**两个 CUDA Stream**，让 CPU 处理与 GPU 计算重叠：

```
重叠调度:
Stream1(CPU): [处理消息][准备batch_B]   [处理batch_A结果][准备batch_C]
Stream2(GPU):                    [推理A]                   [推理B]
              ├────────────────── 时间 ──────────────────────────►
```

- 在 GPU 执行 batch_A 的推理时，CPU 同时在准备 batch_B
- batch_A 推理完成后，batch_B 已准备好，可以立即提交 GPU

### 10.3 实现

```python
# scheduler.py — 重叠调度主循环
class Scheduler(SchedulerIOMixin):
    def __init__(self, config):
        self.stream = torch.cuda.Stream()         # 调度器 stream
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)

    def overlap_loop(self, last_data):
        # Step 1: 处理新消息（在调度器 stream 上）
        blocking = not (last_data is not None     # 有在跑的 batch 就不阻塞
                        or self.prefill_manager.runnable
                        or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Step 2: 调度下一个 batch（在调度器 stream 上）
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:           # 切到引擎 stream
                self.engine.stream.wait_stream(self.stream)  # 等待数据准备
                ongoing_data = (forward_input, self._forward(forward_input))

        # Step 3: 处理上一个 batch 的结果（在调度器 stream 上）
        self._process_last_data(last_data)
        return ongoing_data
```

---

## 11. 张量并行（Tensor Parallelism）

### 11.1 为什么需要张量并行

大模型（如 70B 参数）无法放入单张 GPU。张量并行将模型的**每个层**切分到多张 GPU 上：

```
                    单 GPU                    2-GPU 张量并行
                ┌──────────┐             ┌──────────┐┌──────────┐
MLP:            │  Gate:   │             │  Gate/2  ││  Gate/2  │
(FFN)           │  Up:     │      →      │  Up/2    ││  Up/2    │
                │  Down:   │             │  Down/2  ││  Down/2  │
                └──────────┘             └─────┬────┘└─────┬────┘
                                               │            │
                                               └──AllReduce─┘

Attention:      │  QKV:    │             │  QKV/2   ││  QKV/2   │
                │  O:      │             │  O       ││  O       │
                └──────────┘             └─────┬────┘└─────┬────┘
                                               │            │
                                               └──AllReduce─┘
```

### 11.2 线性层的并行策略

Mini-SGLang 在 `layers/linear.py` 中定义了四种并行线性层：

| 层类型 | 切分方式 | 通信 | 用途 |
|--------|---------|------|------|
| **LinearColParallelMerged** | 输出维度按列切分 | 无 | MLP 的 Gate/Up 投影 |
| **LinearQKVMerged** | Q/K/V 的 head 数切分 | 无 | Attention 的 QKV 投影 |
| **LinearRowParallel** | 输入维度按行切分 | AllReduce | MLP 的 Down 投影 |
| **LinearOProj** | 输入维度按行切分 | AllReduce | Attention 的 O 投影 |

```python
# linear.py — QKV 合并投影层
class LinearQKVMerged(_LinearTPImpl):
    def __init__(self, hidden_size, head_dim, num_qo_heads, num_kv_heads, has_bias):
        tp_info = get_tp_info()
        # Q heads 按 tp_size 均分，KV heads 允许复制（GQA 场景）
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 每张卡只存 1/tp_size 的 Q 和对应的 KV
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim

# linear.py — O 投影层（需要 AllReduce）
class LinearOProj(_LinearTPImpl):
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)  # 跨卡求和
        return y
```

### 11.3 通信实现

支持两种后端：
- **NCCL**：PyTorch 原生 NCCL 后端
- **PyNCCL**：自实现的 Python NCCL 包装器，延迟更低

---

## 12. 注意力后端：FlashInfer 与 FlashAttention

### 12.1 为什么需要专门的注意力内核

标准 PyTorch 实现的注意力计算：
- 显存占用 O(N²)（需要存储完整的注意力矩阵）
- 不支持 Paged KV Cache
- 无法高效处理变长序列

FlashAttention 系列内核解决了这些问题：
- **显存 O(N)**：不存储完整注意力矩阵，使用分块计算
- **IO 优化**：减少 HBM（显存）读写次数
- **支持 Paged KV Cache**：直接从非连续页面读取

### 12.2 可插拔后端架构

```python
# attention/__init__.py — 注册表模式
SUPPORTED_ATTENTION_BACKENDS = Registry("Attention Backend")

@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config):
    return FlashInferBackend(config)

@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config):
    return FlashAttentionBackend(config)

@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config):
    return TensorRTLLMBackend(config)

# 支持混合后端：prefill 用 FA，decode 用 FI
def create_attention_backend(backend, config):
    if "," in backend:  # 如 "fa,fi"
        p_backend, d_backend = backend.split(",")
        return HybridBackend(
            create_attention_backend(p_backend, config),  # prefill
            create_attention_backend(d_backend, config),  # decode
        )
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)
```

### 12.3 后端对比

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| **FlashInfer (fi)** | 支持 Paged KV Cache，decode 优化好 | 通用推荐 |
| **FlashAttention (fa)** | 原始 FA 内核，prefill 性能好 | 短序列 |
| **TensorRT-LLM (trtllm)** | NVIDIA 优化，SM100 原生支持 | H100/B200 |
| **Hybrid (fa,fi)** | prefill 用 FA，decode 用 FI | 兼顾两者优势 |

---

## 13. 模型实现与权重加载

### 13.1 模型架构

以 LLaMA 为例，展示推理框架如何实现一个模型：

```python
# models/llama.py
class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.model = LlamaModel(config)    # Transformer 主体
        self.lm_head = ParallelLMHead(...) # 语言模型头（输出 logits）

    def forward(self):
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits

class LlamaDecoderLayer(BaseOP):
    def __init__(self, config, layer_id):
        self.self_attn = LlamaAttn(config, layer_id)    # 注意力
        self.mlp = LlamaMLP(config)                      # MLP
        self.input_layernorm = RMSNormFused(...)          # 前层归一化
        self.post_attention_layernorm = RMSNormFused(...)  # 后层归一化

    def forward(self, x, residual):
        x, residual = self.input_layernorm.forward(x, residual)  # Pre-Norm
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual
```

### 13.2 模型注册表

```python
# models/register.py — 架构名 → 模型类
# 使用时：model = create_model(model_config)
# 根据模型的 architecture 字段自动选择对应的类
```

### 13.3 权重加载与 TP 切分

```python
# models/weight.py — 加载权重时自动按 TP 切分
# 1. 从 HuggingFace 格式加载完整权重
# 2. 根据 TP rank 和 size 截取对应的分片
# 3. 处理 GQA 的 KV head 复制
```

### 13.4 支持的模型

| 模型 | 特点 |
|------|------|
| **LLaMA** | 标准 Decoder-Only 架构 |
| **Qwen2** | GQA（Grouped Query Attention） |
| **Qwen3** | 支持 thinking 模式 |
| **Qwen3MoE** | 混合专家模型（MoE），路由部分 token 到不同专家 |
| **Mistral** | 滑动窗口注意力 |

---

## 14. 采样策略

推理框架的最后一步是从 logits 中选择下一个 token：

```python
# engine/sample.py — 采样参数
@dataclass
class SamplingParams:
    temperature: float = 0.0   # 温度（0=贪心，越高越随机）
    top_k: int = -1            # Top-K 采样（只考虑概率最高的 K 个）
    top_p: float = 1.0         # Top-P 核采样（累积概率达到 P 时截断）
    ignore_eos: bool = False   # 是否忽略结束符
    max_tokens: int = 1024     # 最大生成 token 数

    @property
    def is_greedy(self):
        # temperature ≤ 0 或 top_k == 1 时为贪心解码
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

### 采样流程

```
Logits [vocab_size]
    │
    ▼
┌─ temperature 缩放 ─┐   temperature=0 → 贪心
│  logits /= temp    │   temperature>0 → 缩放概率分布
└────────┬───────────┘
         ▼
┌─ Top-K 过滤 ───────┐   只保留概率最高的 K 个 token
│  保留 top K logits │   其余设为 -inf
└────────┬───────────┘
         ▼
┌─ Top-P 过滤 ───────┐   按概率降序排列
│  累积概率 ≤ P      │   保留累积概率 < P 的 token
└────────┬───────────┘
         ▼
┌─ 采样 ─────────────┐
│  从剩余 token 中   │
│  按概率随机选择    │
└────────────────────┘
```

---

## 15. 关键技术对比总结

### 推理框架的核心优化技术一览

| 技术 | 解决的问题 | 核心思路 | 对应模块 |
|------|-----------|---------|---------|
| **Paged KV Cache** | 显存碎片、预分配浪费 | 类似 OS 虚拟内存的分页管理 | `kvcache/mha_pool.py` |
| **Radix Tree 前缀缓存** | 重复计算公共前缀 | 树结构共享前缀 KV Cache | `kvcache/radix_cache.py` |
| **Chunked Prefill** | 长请求阻塞短请求 | 拆分长输入为多个 chunk | `scheduler/prefill.py` |
| **CUDA Graph** | Decode 阶段 CPU 开销大 | 录制+回放 GPU 操作序列 | `engine/graph.py` |
| **Overlap Scheduling** | CPU/GPU 串行等待 | 双 Stream 重叠 CPU 与 GPU | `scheduler/scheduler.py` |
| **张量并行 (TP)** | 单卡放不下大模型 | 每层按维度切分到多卡 | `layers/linear.py`, `distributed/` |
| **Flash Attention** | 注意力计算 O(N²) 显存 | 分块计算 + IO 优化 | `attention/fi.py`, `fa.py` |
| **连续批处理** | 请求间 GPU 空闲 | 动态组合 prefill+decode | `scheduler/` 全模块 |

### 从初学者到进阶的学习路径

```
入门 ──────────────────────────────────────────────────────────► 进阶

1. 理解 Prefill/Decode    →  6. 理解 Radix Tree 前缀缓存
2. 理解 KV Cache 原理     →  7. 理解 CUDA Graph 录制/回放
3. 理解连续批处理         →  8. 理解 Overlap Scheduling
4. 理解 Paged KV Cache    →  9. 理解张量并行切分策略
5. 理解采样策略           →  10. 理解自定义 CUDA 内核优化
```

**建议阅读顺序**：
1. `core.py` — 理解基本数据结构
2. `engine/engine.py` — 理解模型初始化流程
3. `scheduler/decode.py` — 最简单的管理器，理解 decode 阶段
4. `scheduler/prefill.py` — 理解分块预填充
5. `kvcache/radix_cache.py` — 最精彩的优化
6. `engine/graph.py` — CUDA Graph 实现
7. `scheduler/scheduler.py` — 把所有东西串起来
8. `layers/linear.py` — 张量并行的实现细节

---

## 附录：目录结构速查

```
mini-sglang/
├── python/minisgl/           # 主要源码
│   ├── core.py               # Req, Batch, Context 数据结构
│   ├── server/               # HTTP 服务 + 进程启动
│   ├── engine/               # 推理引擎 + CUDA Graph + 采样
│   ├── scheduler/            # 调度器（框架核心）
│   ├── attention/            # 注意力计算后端
│   ├── kvcache/              # KV Cache 管理
│   ├── models/               # 模型实现（LLaMA, Qwen2/3, Mistral）
│   ├── layers/               # 神经网络层（支持 TP）
│   ├── tokenizer/            # 分词/反分词
│   ├── distributed/          # 多 GPU 通信
│   ├── kernel/               # 自定义 CUDA/Triton 内核
│   ├── message/              # ZMQ 消息协议
│   ├── llm/                  # 离线生成 API
│   └── utils/                # 工具函数
├── tests/                    # 测试
├── benchmark/                # 性能基准测试
├── docs/                     # 文档
├── Dockerfile                # Docker 构建（CUDA 12.8）
└── pyproject.toml            # 项目配置
```

---

> 本文档基于 mini-sglang 代码库分析编写。如需深入某个主题，建议直接阅读源码——整个项目只有约 5000 行，是非常好的学习材料。
