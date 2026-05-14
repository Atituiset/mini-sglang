# Mini-SGLang 深度分析：面向 LLM 推理性能优化的源码解读

> 本文面向 LLM 推理新手，逐层拆解 mini-sglang（~5000 行 Python）的架构设计和性能优化技巧，为修改 SGLang 源码、极致压榨显卡推理性能提供知识基础。

---

## 一、整体架构：进程拓扑与数据流

`mini-sglang` 用一个单行命令启动整个分布式推理系统：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --tp 4
```

这行命令在内部 `spawn` 出以下进程：

```
                    ┌─────────────┐
   HTTP/SSE ───────►│  API Server │  (FastAPI + uvicorn, asyncio)
                    └──────┬──────┘
                           │ ZMQ
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌─────────────┐
        │Tokenizer │ │Tokenizer │ │ Detokenizer  │ (每个TP Rank一个)
        │ Worker 0 │ │ Worker N │ │   Worker     │
        └────┬─────┘ └────┬─────┘ └──────┬──────┘
             │             │              │
             └──────┬──────┘              │
                    │ ZMQ                 │ ZMQ
                    ▼                     │
        ┌───────────────────────┐         │
        │ Scheduler (TP Rank 0) │◄────────┘
        │   ┌───────────────┐   │
        │   │    Engine     │   │
        │   └───────────────┘   │
        └──────────┬────────────┘
                   │ NCCL / PyNCCL (all_reduce, all_gather)
        ┌──────────┼────────────┐
        ▼          ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Scheduler│ │Scheduler│ │Scheduler│  (TP Rank 1..N-1, 各自拥有 Engine)
   │ Engine  │ │ Engine  │ │ Engine  │
   └─────────┘ └─────────┘ └─────────┘
```

### 各进程职责

| 进程 | 职责 | 关键代码 |
|------|------|----------|
| **API Server** | 提供 OpenAI 兼容 `/v1/chat/completions` 端点，SSE 流式返回 | `server/api_server.py` |
| **Tokenizer Worker** | 文本 → token_ids，通过 ZMQ 发给 Scheduler | `tokenizer/` |
| **Detokenizer Worker** | token_ids → 文本，通过 ZMQ 返回 API Server | `tokenizer/` |
| **Scheduler** | 核心调度进程。每个 GPU 一个进程。Rank 0 负责接收/分发请求、收集结果 | `scheduler/scheduler.py` |
| **Engine** | 拥有模型权重、KV Cache、Attention 后端，执行前向计算 | `engine/engine.py` |

### 通信层

- **控制面**：ZMQ (IPC)，消息类型定义在 `message/`
- **数据面 (TP 通信)**：NCCL 或 PyNCCL（自定义 NCCL wrapper），具体在 `distributed/impl.py` 和 `kernel/pynccl.py`

---

## 二、请求生命周期：从 HTTP 到 token 再回来

```
1. User 发送 POST /v1/chat/completions
2. API Server → Tokenizer Worker (ZMQ)
3. Tokenizer: text → token_ids, 发送 UserMsg 给 Scheduler Rank 0
4. Scheduler Rank 0 通过 broadcast 将请求同步到所有 TP Rank
5. 每个 Scheduler 调度请求 (prefill / decode)，调用 Engine.forward_batch()
6. Engine 执行模型前向计算 → logits → sampler → next_token
7. Scheduler Rank 0 收集 next_token，发给 Detokenizer (ZMQ)
8. Detokenizer: token → text，发给 API Server
9. API Server 通过 SSE 流式返回给 User
10. 重复 5-9 直到所有请求完成或达到 max_tokens
```

---

## 三、核心数据结构

### 3.1 `Req` — 单个推理请求 (`core.py`)

```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor      # 完整 input token 序列（CPU tensor）
    table_idx: int               # 在 page_table 中的行索引（槽位）
    cached_len: int              # 已被 Radix Cache 命中的前缀长度
    output_len: int              # 用户请求的最大输出 token 数
    uid: int                     # 请求唯一 ID
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle  # KV Cache 前缀句柄

    # 自动推导：
    device_len: int              # 当前已在 GPU 上的 token 数 (初始化 = len(input_ids))
    max_device_len: int          # 最大 GPU token 数 = input_len + output_len

    @property
    def extend_len(self) -> int:    # 本次需要计算的 token 数
        return self.device_len - self.cached_len

    @property
    def remain_len(self) -> int:    # 还可以生成的 token 数
        return self.max_device_len - self.device_len

    def complete_one(self) -> None:
        """每生成一个 token 后调用：cached_len = old_device_len, device_len += 1"""
        self.cached_len = self.device_len
        self.device_len += 1
```

### 3.2 `Batch` — 推理批次 (`core.py`)

```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]

    # 由 Scheduler 在 _prepare_batch 中填充：
    input_ids: torch.Tensor   # GPU tensor, [total_extend_tokens]
    positions: torch.Tensor   # GPU tensor, 每个 token 的位置编码
    out_loc: torch.Tensor     # GPU tensor, page_table 查表结果 (KV cache 写入位置)
    padded_reqs: List[Req]    # padding 到 CUDA Graph 支持的 batch size

    # 由 Attention Backend 在 prepare_metadata 中填充：
    attn_metadata: BaseAttnMetadata
```

### 3.3 `Context` — 全局运行时上下文 (`core.py`)

```python
@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor       # [max_running_req, max_seq_len] → int32
    attn_backend: BaseAttnBackend
    moe_backend: BaseMoeBackend
    kv_cache: BaseKVCachePool
    _batch: Batch | None           # 当前正在 forward 的 batch
```

通过 `set_global_ctx()` / `get_global_ctx()` 全局访问，避免在各层之间显式传递。

---

## 四、模型层与 Tensor Parallelism (TP)

### 4.1 模型架构 (以 Llama 为例)

```
LlamaForCausalLM
├── model: LlamaModel
│   ├── embed_tokens: VocabParallelEmbedding          ← 词表并行 (切 vocab dim)
│   ├── layers: OPList[LlamaDecoderLayer] × N
│   │   ├── self_attn: RopeAttn
│   │   │   ├── qkv_proj: LinearQKVMerged (ColParallel)   ← Q/K/V 合并投影，按 dim=0 切
│   │   │   ├── attn: AttentionLayer                      ← FlashAttention/FlashInfer
│   │   │   └── o_proj: LinearOProj (RowParallel)         ← 输出投影，需要 all_reduce
│   │   ├── mlp: GatedMLP
│   │   │   ├── gate_up_proj: LinearColParallelMerged     ← gate/up 合并，按 dim=0 切
│   │   │   └── down_proj: LinearRowParallel              ← 下行投影，需要 all_reduce
│   │   ├── input_layernorm: RMSNormFused
│   │   └── post_attention_layernorm: RMSNormFused
│   └── norm: RMSNormFused
└── lm_head: ParallelLMHead                          ← 词表并行 (切 vocab dim)
```

### 4.2 TP 权重切分策略 (`layers/linear.py`)

| 层类型 | 切分方式 | 是否需要通信 |
|--------|----------|--------------|
| `LinearQKVMerged` (qkv_proj) | dim=0 按 TP 切输出维度 | 否 |
| `LinearColParallelMerged` (gate_up_proj) | dim=0 按 TP 切输出维度 | 否 |
| `LinearOProj` (o_proj) | dim=1 按 TP 切输入维度 | **是 — all_reduce** |
| `LinearRowParallel` (down_proj) | dim=1 按 TP 切输入维度 | **是 — all_reduce** |
| `VocabParallelEmbedding` | dim=0 按 TP 切词表 | 否 |
| `ParallelLMHead` | dim=0 按 TP 切词表 | 否（每个 rank 只算自己那部分 logits） |

**合并投影优化**：Q/K/V 三个投影的权重在加载时被合并为一个 `qkv_proj` 的大矩阵（`models/weight.py`），避免三次矩阵乘法，减少 kernel launch 开销。gate/up 同理。

### 4.3 权重加载 (`models/weight.py`)

流式读取 safetensors 文件 → 按 TP rank 实时切分 → 合并 Q/K/V 和 gate/up → yield (name, tensor)。对 MoE 模型，将多个 expert 的权重堆叠为 3D tensor。

---

## 五、KV Cache — 推理框架的核心

### 5.1 为什么需要 KV Cache？

自回归生成中，每个新 token 的 attention 需要看到所有历史 token。如果没有 KV cache，每步都要重新计算所有历史 token 的 K/V，计算量是 O(n²)。KV cache 将已计算的 K/V 存下来，每步只算新 token 的 K/V，attention 时拼接历史和新 K/V，计算量降为 O(n)。

### 5.2 PagedAttention (`kvcache/mha_pool.py`)

传统做法的痛点：每个请求的 KV cache 需要连续内存，导致碎片化和显存浪费。

**PagedAttention** 将 KV cache 按"页"（page）分块，每个请求通过 **page table** 做逻辑→物理映射：

```
物理 KV Cache: [page0] [page1] [page2] [page3] [page4] [page5] ...

请求 A 的 page_table: [page2, page0]           → 逻辑 token 0→page2, token 1→page0
请求 B 的 page_table: [page5, page3, page1]    → 逻辑 token 0→page5, ...
```

默认 `page_size = 1`（FlashInfer 限制），此时每个 token 独立一页，完全消除碎片化。

### 5.3 Radix Cache — 跨请求前缀共享 (`kvcache/radix_cache.py`)

这是 **SGLang 的招牌优化**。多个请求如果共享前缀（比如同一个 system prompt），只需计算一次。

**数据结构：Radix Tree（基数树）**

```
Root (ref_count=1, 永不被 evict)
├── key="你"  value=[page3]
│   ├── key="好"  value=[page7]        ← ref_count=2 (请求A和B都在用)
│   │   ├── key="啊"  value=[page2]    ← 请求A 独有 (ref_count=1)
│   │   └── key="吗"  value=[page5]    ← 请求B 独有 (ref_count=1)
│   └── key="坏"  value=[page1]        ← 请求C 独有
```

**核心操作**：

- **`match_prefix(input_ids)`**：沿树向下走，逐节点匹配。使用 `fast_compare_key`（C++ kernel, `kernel/radix.cpp`）做快速比较。返回匹配长度 + 对应 indices。
- **`insert_prefix(input_ids, indices)`**：将新前缀插入树。如果部分匹配，分裂节点（`split_at`）。
- **`evict(size)`**：LRU 淘汰。`ref_count=0` 的叶子节点可被 evict。通过最小堆（`heapq`）按 `timestamp` 找最旧节点。
- **`lock_handle()` / `unlock()`**：引用计数管理。`ref_count > 0` → protected 不可淘汰。

### 5.4 CacheManager (`scheduler/cache.py`)

在 Radix Cache（逻辑层面）和物理页面分配之间做协调：

```
CacheManager
├── prefix_cache: RadixPrefixCache   ← Radix Tree (逻辑)
├── free_slots: Tensor[int]          ← 空闲物理页
├── page_table: Tensor               ← [max_req, max_seq_len] int32
└── allocate_paged(reqs):
    1. 计算需要的页数
    2. 空闲不足 → prefix_cache.evict() 淘汰旧节点
    3. 更新 page_table[req.table_idx, positions] = allocated_indices
```

**延迟回收优化**：`lazy_free_region()` context manager，在 decode 阶段批量释放完成的页面，减少 GPU 同步次数。

---

## 六、调度器 (Scheduler) — 系统的"大脑"

### 6.1 主循环 (`scheduler/scheduler.py`)

```python
def run_forever(self):
    if DISABLE_OVERLAP_SCHEDULING:
        while True:
            normal_loop()    # 收消息 → 调度 → 执行 → 处理结果
    else:
        data = None
        while True:
            data = overlap_loop(data)   # 重叠模式
```

### 6.2 Normal Loop（非重叠模式）

```
┌─ Step 1: 接收消息 ───────────────────────────────────┐
│  receive_msg(blocking=...)                             │
│  ├─ UserMsg → prefill_manager.add_one_req()           │
│  ├─ AbortMsg → 释放请求资源                            │
│  └─ ExitMsg → 关闭系统                                 │
├─ Step 2: 调度批次 ────────────────────────────────────┤
│  _schedule_next_batch()                                │
│  ├─ 优先 prefill: prefill_manager.schedule_next_batch()│
│  └─ 无 prefill 时: decode_manager.schedule_next_batch()│
├─ Step 3: 准备批次 ────────────────────────────────────┤
│  _prepare_batch(batch)                                 │
│  ├─ graph_runner.pad_batch()        # padding 到 graph BS│
│  ├─ cache_manager.allocate_paged()  # 分配物理页        │
│  ├─ 构建 positions, input_ids, out_loc tensors          │
│  └─ attn_backend.prepare_metadata() # 准备 attention 元数据│
├─ Step 4: 执行前向 ────────────────────────────────────┤
│  engine.forward_batch(batch, sample_args)               │
│  ├─ CUDA Graph replay (decode) 或 正常 forward (prefill)│
│  └─ sampler.sample(logits) → next_tokens               │
└─ Step 5: 处理结果 ────────────────────────────────────┤
   _process_last_data(data)
   ├─ 将 next_token 追加到 req.input_ids
   ├─ 已完成 → 释放资源，发送 DetokenizeMsg
   └─ prefill 完成 → cache_manager.cache_req() 插入 Radix Cache
```

### 6.3 Overlap Scheduling — 核心性能优化

```
时间 ───────────────────────────────────────────────────►

  GPU Stream:
  │  forward(batch₁)  │  forward(batch₂)  │  forward(batch₃)  │
  
  CPU Stream:
  │ schedule batch₁ │ process batch₁ │ schedule batch₂ │ process batch₂ │
                    │ recv msgs      │                  │ recv msgs      │
                    
       ▲ GPU 不空闲！CPU 调度与 GPU 计算完美重叠 ▲
```

实现：两个 CUDA Stream (`engine.stream` 和 `scheduler.stream`)，通过 `torch.cuda.Event` 做同步。

```python
def overlap_loop(self, last_data):
    # 1. 接收新消息 (不阻塞)
    for msg in self.receive_msg(blocking=False):
        self._process_one_msg(msg)

    # 2. 调度下一个 batch
    forward_input = self._schedule_next_batch()

    # 3. 在 engine stream 上执行
    with torch.cuda.stream(self.engine.stream):
        self.engine.stream.wait_stream(self.stream)  # 等待 CPU stream 的 H2D 完成
        ongoing_data = (forward_input, self._forward(forward_input))

    # 4. 处理上一批的结果 (在 CPU stream 上，与 GPU 并行)
    self._process_last_data(last_data)
    return ongoing_data  # 下次循环的 last_data
```

### 6.4 Chunked Prefill — 长 prompt 内存优化 (`scheduler/prefill.py`)

```python
@dataclass
class PrefillAdder:
    token_budget: int       # 每次 prefill 最大 token 数 (max_extend_tokens)

    def try_add_one(self, pending_req):
        # 如果 input_len - cached_len > token_budget
        #   只取 token_budget 个 token → ChunkedReq (不可 decode，不可采样)
        # 如果 ≤ token_budget
        #   一次全部取完 → 普通 Req
```

长 prompt 被切成多个 chunk，分散到多个 step 执行：
```
Step 1: prefill tokens [0:8192)     → ChunkedReq (已计算 8192/16384)
Step 2: prefill tokens [8192:16384) → 普通 Req (全部计算完毕，可开始 decode)
Step 3: decode token 0
...
```

好处：将长 prompt 的显存峰值打散到多个 step，避免 OOM。代价：增加调度次数。

### 6.5 DecodeManager (`scheduler/decode.py`)

```python
@dataclass
class DecodeManager:
    running_reqs: Set[Req]       # 当前正在 decode 的请求
    page_size: int

    def schedule_next_batch(self) -> Batch | None:
        # 所有 running_reqs 在一个 batch 中一起 decode
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def inflight_tokens(self) -> int:
        # 估算当前 decode 请求在未来需要的总 token 数
        # = sum(remain_len) + (page_size - 1) * len(running_reqs)
```

---

## 七、Engine — 模型执行核心 (`engine/engine.py`)

### 7.1 初始化流程

```
Engine.__init__()
├─ Step 1: 设置 device, CUDA stream, 随机种子
├─ Step 2: 初始化分布式通信 (NCCL 或 PyNCCL)
├─ Step 3: 在 meta device 上创建模型结构
│          model = create_model(config)  # 只建图，不分配显存
├─ Step 4: 加载权重到 GPU
│          model.load_state_dict(load_weight(model_path, device))
├─ Step 5: 计算 KV Cache 大小
│          num_pages = (memory_ratio × free_memory - model_size) / cache_per_page
│          例: 0.9 × 80GB - 1.2GB / (2 × 128 × 8 × 2B × 28) ≈ 7.8M tokens
├─ Step 6: 初始化 Attention / MoE 后端
├─ Step 7: 捕获 CUDA Graph (decode 阶段加速)
└─ Step 8: 记录显存使用情况
```

### 7.2 forward_batch

```python
def forward_batch(self, batch, args) -> ForwardOutput:
    with self.ctx.forward_batch(batch):    # 设置全局 batch
        if self.graph_runner.can_use_cuda_graph(batch):
            # Decode 且 batch_size ≤ max_graph_bs → CUDA Graph 重放
            logits = self.graph_runner.replay(batch)
        else:
            # Prefill 或大 batch → 正常 forward
            logits = self.model.forward()

    # 更新每个 req 的状态
    for req in batch.reqs:
        req.complete_one()    # cached_len = old_device_len, device_len += 1

    # 采样
    next_tokens_gpu = self.sampler.sample(logits[:batch.size], args)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
    # 记录 copy 完成事件 (用于 overlap scheduling 同步)
    copy_done_event = torch.cuda.Event()
    copy_done_event.record(self.stream)
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

### 7.3 Sampler (`engine/sample.py`)

```python
def prepare(self, batch) -> BatchSamplingArgs:
    if all(p.is_greedy for p in params):
        return BatchSamplingArgs(temperatures=None)  # 快速路径：argmax

    # 否则：用 FlashInfer 的 sampling 模块
    # - top_p_sampling_from_probs
    # - top_k_sampling_from_probs
    # - top_k_top_p_sampling_from_probs

def sample(self, logits, args):
    if args.temperatures is None:
        return torch.argmax(logits, dim=-1)    # 贪婪采样
    return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
```

---

## 八、Attention 后端

### 8.1 后端选择 (`engine/config.py`)

```python
def _adjust_config(config):
    if attention_backend == "auto":
        if is_sm100():  # Blackwell (B200/B100)
            backend = "trtllm"
        elif is_sm90():  # Hopper (H100/H200)
            backend = "fa,fi"   # prefill=FlashAttention3, decode=FlashInfer
        else:            # Ampere 及以前
            backend = "fi"      # FlashInfer 全部
```

### 8.2 HybridBackend 模式 (`attention/base.py`)

```python
class HybridBackend(BaseAttnBackend):
    def __init__(self, prefill_backend, decode_backend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)
```

### 8.3 FlashInfer Backend (`attention/fi.py`)

```python
class FlashInferBackend:
    def prepare_metadata(self, batch):
        # 构建 FlashInfer 需要的元数据：
        # - cu_seqlens_q / cu_seqlens_k (累积序列长度)
        # - indices (page_table 查表结果)
        # - last_page_len (每个请求最后一页的有效长度)
        # → 调用 wrapper.plan(...) 异步做 kernel 规划

    def forward(self, q, k, v, layer_id, batch):
        # 1. 将新的 k, v 写入 kv_cache
        kv_cache.store_kv(k, v, batch.out_loc, layer_id)
        # 2. 将 kv_cache 展平为 [num_tokens, 1, num_kv_heads, head_dim]
        # 3. 执行 paged attention
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

**关键优化**：`plan()` 是异步的（`non_blocking=True`），利用 `torch.cuda.Event` 确保前一次 plan 完成后再做下一次，避免 GPU stall。

### 8.4 AttentionLayer (`layers/attention.py`)

```python
class AttentionLayer(StateLessOP):
    def forward(self, qkv: torch.Tensor):
        # 切分 Q/K/V
        q, k, v = qkv.split([qo_dim, kv_dim, kv_dim], dim=-1)
        # QK Norm (可选，Qwen3 有)
        if self.q_norm: self.q_norm.forward_inplace(q)
        if self.k_norm: self.k_norm.forward_inplace(k)
        # RoPE 旋转位置编码
        q, k = self.rotary.forward(positions, q, k)
        # Attention (FlashInfer / FlashAttention / TRT-LLM)
        o = ctx.attn_backend.forward(q, k, v, layer_id, batch)
        return o.reshape(-1, qo_dim)
```

---

## 九、CUDA Graph — Decode 阶段的关键加速 (`engine/graph.py`)

### 9.1 为什么需要 CUDA Graph？

每个 decode step 只计算 1 个新 token，计算量很小（~0.1ms），但 kernel launch overhead 可能高达 0.05ms（~50% overhead）。CUDA Graph 将整个 forward 图录制下来，重放时只需要一个 kernel launch，消除 overhead。

### 9.2 捕获流程

```python
class GraphRunner:
    graph_map: Dict[int, CUDAGraph]    # batch_size → graph
    graph_bs_list: [1, 2, 4, 8, 16, ...]  # 支持的 batch size

    def _capture_graphs(self):
        for bs in graph_bs_list:
            # 创建 dummy_req 的 batch (输入全是 0)
            batch = Batch(reqs=[dummy_req] * bs, phase="decode")

            # 先跑一次 warmup
            self.buffer.logits[:bs] = model.forward()

            # 录制 CUDA Graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.buffer.logits[:bs] = model.forward()

            self.graph_map[bs] = graph
```

所有 graph 共享同一个 `pool`（CUDA memory pool），减少显存占用。

### 9.3 重放

```python
def replay(self, batch):
    buffer.copy_from(batch)      # 将实际 input_ids, positions, out_loc 拷入 buffer
    graph = self.graph_map[batch.padded_size]
    graph.replay()               # 一个 kernel launch 执行整个 forward
    return buffer.logits[:batch.size]
```

### 9.4 Graph Batch Size 选择

```python
def _determine_cuda_graph_bs():
    if free_memory > 80GB:  # H200
        max_bs = 256
    else:
        max_bs = 160
    return [1, 2, 4] + list(range(8, max_bs + 1, 8))
    # → [1, 2, 4, 8, 16, 24, ..., 160]
```

实际运行时用 `pad_batch()` 将 batch 对齐到最近的 graph size：
```python
def pad_batch(self, batch):
    padded_size = next(bs for bs in graph_bs_list if bs >= batch.size)
    batch.padded_reqs = batch.reqs + [dummy_req] * (padded_size - batch.size)
```

---

## 十、Custom CUDA Kernels (`kernel/`)

通过 `apache-tvm-ffi` 做 Python ↔ CUDA 绑定，支持 **AOT** 和 **JIT** 两种编译模式。

### 10.1 关键 Kernel

| Kernel | 文件 | 用途 | 输入/输出 |
|--------|------|------|-----------|
| `store_cache` | `store.cu` | 将新计算的 K/V 写入 paged cache | `(k, v, indices) → kv_cache[indices] = k, v` |
| `indexing` | `index.cu` | 按 indices 从权重矩阵 gather | `lm_head_weight[token_ids] → logits` |
| `fast_compare_key` | `radix.cpp` | CPU 端快速找两个 token 序列的第一个不同位置 | `compare(x, y) → first_diff_pos` |
| `fused_moe_kernel` | `moe_impl.py` | Triton 实现的 fused MoE | `hidden_states + router_logits → expert_outputs` |
| `moe_sum_reduce` | `moe_impl.py` | MoE 中多个 expert 输出的加权求和 | `expert_outputs + weights → final_hidden_states` |

### 10.2 JIT 编译流程 (`kernel/utils.py`)

```python
def load_jit(*args, cuda_files, cuda_wrappers):
    # 1. 读取 CUDA 源文件 (kernel/jit/*.cu)
    # 2. 将 C++ template 参数拼入 kernel 名称
    # 3. 注入 ffi wrapper (__global__ kernel → C function)
    # 4. tvm_ffi.cpp.load_inline() → JIT 编译 → 返回 Python callable
    pass
```

模板参数示例：`StoreKernel<element_size=1024, num_threads=128, ...>`，不同参数编译出不同 kernel 实例。

---

## 十一、分布式通信 (`distributed/`)

### 11.1 插件链模式

```python
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x):
        return self.plugins[-1].all_reduce(x)   # 用最后一个插件

    def all_gather(self, x):
        return self.plugins[-1].all_gather(x)
```

### 11.2 PyNCCL (`kernel/pynccl.py`)

当 `TP size > 1` 且 `use_pynccl=True`：

```python
def enable_pynccl_distributed(tp_info, tp_cpu_group, max_bytes):
    # init_pynccl: gloo (barrier 同步) + NCCL (数据传输)
    comm = init_pynccl(tp_info.rank, tp_info.size, tp_cpu_group, max_bytes)
    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))
```

相比原生 NCCL，PyNCCL 的优势：
- 避免 NCCL group 创建开销
- gloo 做 barrier 同步更轻量
- 更好的错误处理和资源管理

---

## 十二、消息系统 (`message/`)

### 消息类型

```python
UserMsg(uid, input_ids, sampling_params)  # API Server → Scheduler: 新请求
DetokenizeMsg(uid, next_token, finished)  # Scheduler → Detokenizer: 生成的 token
BatchBackendMsg([msgs])                   # 批量包装
AbortBackendMsg(uid)                      # 中止请求
ExitMsg()                                 # 关闭系统
```

所有消息通过 ZMQ (IPC) 传输，支持自动序列化/反序列化（基于 `msgpack`）。

---

## 十三、可优化 / 可调整的关键参数

### 命令行参数

| 参数 | 默认值 | 说明 | 优化方向 |
|------|--------|------|----------|
| `--page-size` | 1 | KV cache 页大小 | 增大可减少 page table 查找，但可能浪费显存。**TRT-LLM 只支持 16/32/64** |
| `--max-prefill-length` | 8192 | Chunked Prefill 的最大 chunk | 太小→调度频繁；太大→显存峰值高 |
| `--memory-ratio` | 0.9 | KV cache 占总显存的比例 | 越大→更多并发但 OOM 风险 |
| `--cuda-graph-max-bs` | 自动 | CUDA Graph 最大 batch size | 越大→更多 graph 实例→更多显存 |
| `--cache` | `radix` | 缓存策略 | `naive` 禁用 Radix Cache (对比实验) |
| `--attn` | `auto` | Attention 后端 | `fa,fi` / `trtllm` / `fi` |
| `--model-source` | `huggingface` | 模型下载源 | `modelscope` 国内更快 |

### 环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MINISGL_DISABLE_OVERLAP_SCHEDULING` | `0` | 设为 `1` 禁用重叠调度 |
| `MINISGL_FLASHINFER_USE_TENSOR_CORES` | 自动 (GQA≥4) | 强制开关 FlashInfer Tensor Core |
| `MINISGL_PYNCCL_MAX_BUFFER_SIZE` | `1G` | PyNCCL 通信 buffer 大小 |

### 硬编码可改参数

| 位置 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `EngineConfig` | `max_running_req` | 256 | 最大并发请求数 |
| `EngineConfig` | `max_forward_len` | = `max_seq_len` | 每次 forward 最大 token 数 |
| `GraphRunner._determine_cuda_graph_bs` | H200→256, 其他→160 | CUDA graph BS 上限 |

---

## 十四、源码修改切入点（极致压榨性能）

### 入门级修改

1. **调整调度策略**
   - 文件：`scheduler/scheduler.py` → `_schedule_next_batch()`
   - 当前：优先 prefill
   - 可改为：decode-first（降低延迟）或 prefill/decode 按比例混合
   - 注意：decode batch 是 joined batch，所有 running reqs 一起跑

2. **调整 Radix Cache 淘汰策略**
   - 文件：`kvcache/radix_cache.py` → `evict()`
   - 当前：LRU（按 timestamp）
   - 可改为：LFU、基于 token 长度的加权淘汰等

3. **修改 CUDA Graph BS**
   - 文件：`engine/graph.py` → `_determine_cuda_graph_bs()`
   - 减小 BS 列表 → 更少 graph 实例 → 更多 KV cache 显存

### 中级修改

4. **PrefillAdder 的调度策略**
   - 文件：`scheduler/prefill.py` → `PrefillAdder`
   - 当前：简单的 token budget 分配
   - 可改进：基于请求优先级的调度、更好的显存估算（`estimated_len`）

5. **Attention 后端扩展**
   - 文件：`attention/` → 新建 backend 文件
   - 实现 `BaseAttnBackend` 接口
   - 在 `attention/__init__.py` 注册

### 高级修改

6. **Custom CUDA Kernel 优化**
   - 文件：`kernel/jit/store.cu`、`kernel/jit/index.cu`
   - 优化方向：warp tile、向量化加载/存储、shared memory 缓存
   - 这些是最热的 kernel，改动收益大

7. **FP8 / INT8 KV Cache 量化**
   - 当前：KV cache 存储 FP16/BF16
   - 需要修改：`kvcache/mha_pool.py` 分配量化存储、`kernel/jit/store.cu` 写入量化、`attention/fi.py` 解量化读取

8. **Speculative Decoding**
   - 新增模块：draft model 加载 + tree attention
   - 修改 Scheduler：支持 verify step

---

## 十五、学习路径建议

作为 LLM 推理新手，建议按以下顺序深入：

```
第一阶段：跑起来 + 宏观理解
├── python -m minisgl --model Qwen/Qwen3-0.6B --shell
├── 阅读 core.py：理解 Req、Batch、Context 三个核心数据结构
└── 阅读 scheduler/scheduler.py：理解 normal_loop 的 5 个步骤

第二阶段：模型执行
├── engine/engine.py：模型如何初始化、forward_batch 的流程
├── engine/graph.py：CUDA Graph 的捕获和重放
└── engine/sample.py：采样逻辑

第三阶段：显存管理（推理框架的核心）
├── kvcache/mha_pool.py：Paged KV Cache 的物理布局
├── kvcache/radix_cache.py：Radix Tree 的增删查改
├── scheduler/cache.py：CacheManager 的协调逻辑
└── scheduler/prefill.py：Chunked Prefill

第四阶段：Attention 和后端
├── attention/base.py：理解 HybridBackend 设计
├── attention/fi.py：FlashInfer 的元数据准备和执行
└── layers/attention.py：从 QKV 投影到 attention 输出的完整路径

第五阶段：分布式和底层
├── distributed/impl.py：NCCL vs PyNCCL
├── layers/linear.py：TP 下各种 Linear 的实现
├── models/weight.py：权重加载时的切分和合并
└── kernel/：自定义 CUDA kernel 和 tvm-ffi 绑定
```

---

## 附录：关键调用链

### Prefill 调用链
```
Scheduler._schedule_next_batch()
→ PrefillManager.schedule_next_batch()
  → PrefillAdder.try_add_one() × N
→ Scheduler._prepare_batch(batch)
  → GraphRunner.pad_batch()
  → CacheManager.allocate_paged()
  → AttentionBackend.prepare_metadata()
→ Scheduler._forward(batch)
  → Engine.forward_batch()
    → LlamaForCausalLM.forward()
      → LlamaModel.forward()
        → embed_tokens(input_ids)
        → for layer in layers:
            → RMSNormFused(x, residual)
            → RopeAttn.forward(x)
              → qkv_proj(x)           # [hidden] → [q,k,v]
              → AttentionLayer(qkv)
                → Rotary.forward(q, k)
                → FlashInferBackend.forward(q, k, v)
                  → store_kv(k, v, out_loc)   # ← custom CUDA kernel
                  → wrapper.run(q, paged_kv)   # ← FlashInfer paged attention
              → o_proj(o) + all_reduce
            → RMSNormFused(x, residual)
            → GatedMLP.forward(x)
              → gate_up_proj(x)
              → silu_and_mul(gate, up)
              → down_proj(y) + all_reduce
        → RMSNormFused(x, residual)
      → lm_head(output)              # → [vocab_size] logits
    → Sampler.sample(logits, args)   # → next_token
→ Scheduler._process_last_data()
```

### Decode 调用链（CUDA Graph 路径）
```
Scheduler._schedule_next_batch()
→ DecodeManager.schedule_next_batch()
→ Scheduler._prepare_batch()          # 同 prefill
→ Scheduler._forward()
  → Engine.forward_batch()
    → GraphRunner.replay(batch)
      → buffer.copy_from(batch)       # 拷贝实际数据
      → graph_map[bs].replay()        # 单 kernel launch: 整个 forward
    → Sampler.sample(logits, args)
→ Scheduler._process_last_data()
```

---

*分析基于 mini-sglang 源码，版本 0.1.0。所有路径相对于 `python/minisgl/`。*
