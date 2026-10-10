# 导读：如何阅读本书

## 这本书是什么

这是一本面向 **AI Infra 初学者** 的 [mini-sglang](https://github.com/sgl-project/mini-sglang) 源码学习指南。

mini-sglang 是 SGLang 的教学精简版：用约 8000 行 Python 实现了一个功能完整的现代 LLM 推理引擎，包含连续批处理（Continuous Batching）、分页 KV Cache（PagedAttention）、前缀缓存（RadixCache）、Chunked Prefill、CUDA Graph、张量并行（TP）、MoE 等工业级推理系统的核心技术。代码量只有 SGLang 的几十分之一，但骨架完整，是理解"大模型推理引擎到底在干什么"的最佳切入点之一。

本书的每一章都基于对源码的完整通读写成，所有引用都精确到 `文件:行号`，方便你随时跳回代码验证。

## 前置知识

你只需要：

- 会读 Python（不要求精通，能看懂类和数据结构即可）
- 知道 Transformer 的基本概念（token、注意力、KV cache 是什么——不知道也没关系，附录术语表有通俗解释）
- 有一块 GPU 并能跑通 README 里的 Quickstart（非必需，但强烈建议边跑边读）

不需要：CUDA 编程经验、分布式系统经验、读过 SGLang/vLLM 源码。

## 推荐阅读路径

本书按"由表及里"的顺序组织：

1. **基础篇**先建立全局观：第 1 章带你跟踪一条请求从 HTTP 进入到文本返回的完整链路，第 2 章（服务化与评测）让你理解引擎对外的接口长什么样、性能如何衡量（TTFT/TPOT/吞吐）。
2. **核心篇**深入推理引擎的三大支柱：调度器（怎么组批）、KV Cache 管理（显存怎么分配与复用）、Attention 后端（计算怎么落到 GPU kernel），外加让 decode 提速的 CUDA Graph 与采样。
3. **进阶篇**看模型侧：模型代码如何组织、权重如何加载、MoE 如何实现，以及多卡张量并行。

每章结尾都有"回顾与动手"小节，给出可以亲自验证/修改代码的小实验——读推理引擎代码，**改一改、跑一跑**比纯读有效得多。

## 边读边跑：最小上手路径

```bash
# 安装（详见仓库 README）
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && pip install -e .

# 启动一个 OpenAI 兼容服务（以 Qwen3-0.6B 为例）
python -m minisgl --model-path Qwen/Qwen3-0.6B

# 发一条请求
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "你好"}]}'
```

## 代码地图

```
python/minisgl/
├── core.py             # 进程编排：Tokenizer/Scheduler/Detokenizer 如何拉起
├── server/             # FastAPI 入口，OpenAI 兼容 API
├── tokenizer/          # tokenize / detokenize 进程
├── scheduler/          # 调度器：连续批处理的大脑
├── engine/             # 引擎：跑模型 forward、CUDA Graph、采样
├── kvcache/            # KV 池与 RadixCache
├── attention/          # FlashAttention / FlashInfer / TRTLLM 后端
├── models/             # Llama/Qwen2/Qwen3(Qwen3-MoE)/Mistral 模型定义
├── layers/             # 线性/Norm/RoPE/Embedding 等基础层
├── moe/, kernel/       # MoE 与底层 kernel 封装
└── distributed/        # NCCL 通信与张量并行
```

> 注：本书基于仓库当前 main 分支源码写成。mini-sglang 仍在快速演进，个别行号可能随版本漂移，但架构与设计思想是稳定的。
