# vLLM Hybrid KV-Cache "残留 NaN" Bug 与 PR #35219 修复机制

> **文档版本**: 1.0
> **分析对象**: [vllm-project/vllm#35219](https://github.com/vllm-project/vllm/pull/35219) — `[BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU`
> **关联 issue**: [#35138](https://github.com/vllm-project/vllm/issues/35138) — Qwen3.5-397B-A17B-FP8 在 Blackwell + FlashInfer 上多次评测精度递减
> **本地 vllm 源码状态**: 已经合入（commit `4ff8c3c8f`，本地 HEAD `02606b0b0`）
> **最后更新**: 2026-06-02

---

## 文档概述

这篇文档不是"PR 改了哪几行"的复述，而是想把这个 bug 当作一个**典型的"共享池 + 多 dtype + 内核掩码假设"组合事故**来拆开看：

> 为什么一个看起来 hybrid 模型独有的精度漂移问题，根因是 **fp32 比特模式在小 dtype 视图下的位重解释**，而压垮骆驼的最后一根稻草是 **attention kernel 的"乘以 0 抹除"假设在 NaN 面前失效**。

**目标读者**: 已经理解 vLLM v1 的 [`Scheduler`](../vllm_async_scheduler.md)、KV cache manager、block pool、`GPUModelRunner` 三段架构的工程师；想顺带摸清 v1 的 single-type KV cache manager / kernel block size / virtual block splitting 这些细节是怎么咬合在一起的。

**阅读指南**:

| 部分 | 内容 | 重点 |
|------|------|------|
| 第一部分 | 现象与触发条件 | 用户视角看到什么、复现的最小开关组合 |
| 第二部分 | Hybrid 模型的统一 block pool | Mamba/SSM 与 Attention 共享同一片 GPU buffer 的事实 |
| 第三部分 | 为什么 fp32 残留会变 NaN | 比特位重解释、subnormal、Inf 编码在 fp16/fp8 下的表现 |
| 第四部分 | 为什么 "0 × NaN = NaN" 是放大器 | FlashAttn / TRT-LLM 的掩码假设和 IEEE-754 行为 |
| 第五部分 | 修复设计 | scheduler 端追踪、worker 端 Triton 清零、CuMem 边界 |
| 第六部分 | 为什么只清 FullAttention 块 / 只对 hybrid 启用 | 工程取舍而不是物理必要 |
| 第七部分 | V2 model runner 是否仍有这个 bug | 本地源码审计结论 |
| 第八部分 | 怎么在 V2 上复现 | 一份最小重现脚本与可观察的信号 |
| 第九部分 | QA | 常见反直觉点 |

---

# 第一部分: 现象与触发条件

## 1.1 用户视角

Issue #35138 报告的现象很具体：

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 4 \
    --reasoning-parser qwen3 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --enforce-eager
# 然后连续跑两次 GSM8K 评测
python tests/evals/gsm8k/gsm8k_eval.py
python tests/evals/gsm8k/gsm8k_eval.py
```

| 第一次跑 | 第二次跑 | 之后 |
|----------|----------|------|
| 准确率 ~90% | 准确率显著下降 | 进一步退化 |

观察到的关键信号：

1. `Qwen3-Next-80B-A3B-Instruct` (也是 hybrid) **不出现**问题；
2. 切换到 `--attention-backend FLASH_ATTN` 立刻消失；
3. 关闭 prefix cache、async scheduling、chunked prefill、FP8 query 量化都**没有用**；
4. 在 FlashInfer native prefill 路径上能在 `trtllm_batch_decode_with_kv_cache` 输出里直接看到 NaN。

这四个观察合在一起，把怀疑面收得很窄：**问题不在 sampling/decode/prefill 路径本身，而在 KV cache buffer 的"被读出来时的数值"上**，并且 FlashInfer/TRT-LLM 后端是放大器。

## 1.2 触发条件归纳

把这些线索翻译成"组合开关"：

| 条件 | 必要性 | 解释 |
|------|--------|------|
| Hybrid 模型（带 Mamba/SSM 层）| **必要** | 决定了 block pool 内会出现 fp32 的状态写入 |
| `FullAttentionSpec` 层与 Mamba 层共享 block pool | **必要** | 决定了一个 block 可能被两种 dtype 先后占用 |
| 使用对 NaN 敏感的后端（FlashInfer-TRTLLM、FlashAttn3 某些路径）| **必要** | 决定了"乘以 0 抹除"的掩码模式 |
| 连续多轮请求 / KV cache 被多次复用 | **强信号** | 第一次很可能踩不到，必须等 Mamba 写过的 block 被 attention 拿走 |
| Async / Sync / prefix cache | **无关** | 已被排除 |
| FP8 / FP16 / BF16 query 量化 | **无关** | 它们都不会清掉旧数据 |

> 也就是说，**这不是"特定模型/特定后端/特定 dtype"的孤立 bug，而是"hybrid 模型的统一 block pool 没有跨 dtype 清零的不变量"**。一旦 attention kernel 的实现假设了"未触及位置 = 安全的 0/-inf"，bug 就触发。

---

# 第二部分: Hybrid 模型的统一 block pool

## 2.1 为什么 hybrid 模型要共享 block pool

vLLM v1 把 KV cache 抽象成统一的 page pool：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/kv_cache_interface.py
```

`KVCacheConfig.kv_cache_groups` 中每个 `KVCacheGroupSpec` 携带自己的 `kv_cache_spec`：

| 子类 | 用途 | 单 block 物理形状 | dtype |
|------|------|-------------------|--------|
| `FullAttentionSpec` | 标准 attention 层 | `[block, 2, block_size, num_kv_heads, head_size]`（不同 backend 顺序不同）| fp16 / bf16 / fp8（按 `cache_dtype`）|
| `MambaSpec` | Mamba/SSM 层 | conv_state + ssm_state | **fp32**（state 精度需求高）|

**关键事实**：两种 group 都从同一个 `BlockPool` 里 `get_new_blocks()`、`free_blocks()`。**block id 是同一组数字索引同一片 GPU buffer 的不同视图**。也就是说：

```text
block_id = 1234
attention layer 看它 → 一段 fp16 张量
mamba    layer 看它 → 一段 fp32 张量
两者底层指向 GPU 上的相同字节
```

> 这种共享是 v1 的核心设计目标之一：用一个 block pool 同时服务异构的 cache 类型，避免提前给两种 cache 各自规划独立池。它把内存利用率拉满，但**也建立了一个隐式不变量：任何 block 在被换给新 layer 之前都必须重置成对新视图无害的状态**。

## 2.2 谁负责"重置"？sync 路径下没有人

不在的不变量很容易丢。看看现状：

| 时机 | 当前行为 | 隐患 |
|------|----------|------|
| `BlockPool.get_new_blocks()` | 返回 id，从 free queue 弹出 | 不写值 |
| Mamba forward | **完整覆盖** 整个 conv/ssm state | 自身路径下没问题 |
| Attention forward | 只往 "本步要写的位置" 写 KV | 没写到的 byte 仍然是上次内容 |
| Attention kernel 读 | 按 `seq_len` 切片 + mask 掉 invalid 位置 | 假设没写到的位置已经是"无害的旧值" |

Mamba 路径自己没问题——它每步把状态整段写穿。问题在**先 Mamba 后 Attention**：Mamba 留下的 fp32 比特模式还在 block 里，Attention 拿到这个 block 时只覆盖了 KV cache 中"当前请求要写的部分"，剩下的 byte 仍然是 Mamba 视角下完美合法的 fp32 数，但 Attention 视角下…见下一节。

---

# 第三部分: 为什么 fp32 残留会变 NaN

## 3.1 IEEE-754 的视角错位

fp32 (single) 的位段是 `1 + 8 + 23`，fp16 是 `1 + 5 + 10`，fp8-e4m3 是 `1 + 4 + 3`。把一个 32-bit 字节序列重新解释成两个 fp16 或四个 fp8，那原本的指数段会被切成新的几片，且**绝大多数 fp32 数被重解释后落到 fp16/fp8 的 NaN / Inf 编码上**。

以最常见的 fp32 → fp16 视角错位为例：

```text
fp32 中 一个普通正数 1.0 = 0x3F800000
            S Exp(8)   Mantissa(23)
            0 01111111 00000000000000000000000

按 little-endian 读成两个 fp16:
  低 16 位 = 0x0000   → fp16 的 +0
  高 16 位 = 0x3F80   → fp16 的 0b0 01111 1110000000 = ~1.875

这个还算"看起来正常"，但只要 fp32 数的 exponent ≥ 159 (即数值大于 ~1e10)：
  高位 fp16 的 5-bit Exp 域全为 1
  fp16 解释为 ±Inf 或 NaN
```

对 Mamba 的 SSM/conv state 来说，参数初始化后的中间结果数量级常常落在 `1e5` 到 `1e20` 范围内（取决于 step、A/B/C 矩阵规模），**这正是导致 fp16/bf16 视角下高频出现 NaN/Inf 的区间**。fp8 更是几乎全部触发。

> 这一步的反直觉点是：**Mamba 写入的不是"非法数"。是 fp32 视角下完全合法的数值，在 fp16/fp8 视角下变成了 NaN/Inf 编码。**

## 3.2 attention 视角下的可见症状

Attention kernel 在读 KV cache 时执行 `Q @ K^T → softmax → @V`。一旦 K 或 V 里出现 NaN：

| 阶段 | NaN 行为 |
|------|----------|
| `Q @ K^T` | 任何乘加里含 NaN → 结果 NaN |
| `softmax` | 整行变 NaN（即使有效位置算出来正常）|
| `@ V` | NaN 继续向 hidden state 传播 |

更糟的是：

| 后端 | 掩码方式 | NaN 应对 |
|------|----------|----------|
| FlashAttn 经典实现 | 把 invalid 位置的 K/V 在加载时替换为 0 / -inf | 大概率"屏蔽掉"——但不绝对（看具体 kernel）|
| FlashAttn3 某些 fused 路径 | 计算后乘 0 把 invalid 位置抹掉 | **被 NaN 击穿** |
| FlashInfer TRT-LLM batch_decode | 同样依赖乘以 0 / mask 累加 | **被 NaN 击穿** |

issue #35138 报告 FlashInfer prefill 路径下能在 `trtllm_batch_decode_with_kv_cache` 输出里直接观察到 NaN，就是这个机制。把后端切到 `FLASH_ATTN` 之所以能 workaround，是因为它的 invalid-mask 在 load 阶段就先把 K/V 替换掉了，遇不到那个"乘 0"步骤。

---

# 第四部分: 为什么 "0 × NaN = NaN" 是这个事故的关键

## 4.1 IEEE-754 的明确规定

```text
0 × NaN = NaN
NaN + x = NaN
NaN × ∞ = NaN
NaN 不等于自身 (NaN != NaN)
```

任何 mask 函数如果用乘以 0 来"屏蔽"位置，都必须先保证那个位置的输入数本身不是 NaN/Inf。但很多高性能 attention kernel 出于以下原因偏好"乘 0 抹除"：

| 选择"乘 0"的理由 | 缘由 |
|------------------|------|
| 不破坏 SIMT lane 的 uniformity | 不需要分支 |
| 不需要在 `softmax` 内插值"行内 max 是否包含 invalid" | 让所有 lane 都进入同一通路 |
| 自动可被 tensor core 友好优化 | mma 操作不喜欢谓词 |

这意味着 attention kernel 默认**信任 KV cache 的 invalid 区域是数值"安全"的**——通常这只要求"不要把 ±Inf 或 NaN 留在那里"。普通 fp16 工作流里这不难，因为：

- 模型权重和激活值一般在 `[-1e4, 1e4]`；
- 之前的 attention 写入也都是有限数；
- 新分配的 block 内容是 PyTorch allocator 给的旧数据，但通常也是 fp16/fp8 合法数。

唯独 hybrid + 跨 dtype 重解释这一组合，违反了这个"暗约定"。

## 4.2 为什么 issue 报告"前两次评测之间"才发生衰减

复现条件是"跑两次"。这其实揭示了 bug 的传播模型：

1. 第一次评测：block pool 还是新的，Mamba 也是新启动；很多 block 还没被 Mamba 写过；attention 拿到的 block 大概率含旧 fp16 数据 → 不发病。
2. 第一次评测期间：随着 Mamba forward，越来越多的 block 上写满 fp32 state；当请求结束、block 回到 free pool。
3. 第二次评测期间：attention 大概率会拿到曾经被 Mamba 用过的 block；NaN 从 KV cache 传到 logits → 准确率塌方。
4. 之后越来越坏：被污染的 block 通过 attention 写入 fp16 NaN/Inf，进一步污染相邻 block；甚至 attention 自己的输出会"洗回"block 里。

> 这就解释了为什么"第一次跑没问题"——bug 是一个**统计学概率事件**，等到 block 的"被 Mamba 用过的比例"高到某个程度才稳定发病。

---

# 第五部分: PR #35219 的修复设计

修复思路一句话：**在每个 step 把"这一步新分配出去的 attention block"在 GPU 上清零，让 attention kernel 读到合法的 0 而不是 fp32 残留**。

复杂度都在"高效地做"和"只对必要的 block 做"。

## 5.1 三个改动锚点

| 层级 | 文件 | 改动 |
|------|------|------|
| Schema | `vllm/v1/core/sched/output.py` | 给 `SchedulerOutput` 加 `new_block_ids_to_zero: list[int] \| None` |
| Scheduler | `vllm/v1/core/single_type_kv_cache_manager.py` + `kv_cache_manager.py` + `scheduler.py` | 每次 allocate 累积 attention block id；schedule 完毕时 drain 进 `SchedulerOutput`；仅当 `kv_cache_config.needs_kv_cache_zeroing` 为真时才发送 |
| Worker | `vllm/v1/worker/gpu_model_runner.py` + `vllm/v1/worker/utils.py` | `_init_kv_zero_meta()` 一次性构建段地址表；每个 step 调一次 Triton `_zero_kv_blocks_kernel` |

## 5.2 Scheduler 侧 — 累积与漏取

```python
# vllm/v1/core/single_type_kv_cache_manager.py
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    ...
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)
    if type(self.kv_cache_spec) is FullAttentionSpec:
        self.new_block_ids.extend(b.block_id for b in new_blocks)
    return new_blocks

def take_new_block_ids(self) -> list[int]:
    ids = self.new_block_ids
    self.new_block_ids = []
    return ids
```

```python
# vllm/v1/core/sched/scheduler.py::schedule()
new_block_ids_to_zero = (
    (self.kv_cache_manager.take_new_block_ids() or None)
    if self.needs_kv_cache_zeroing
    else None
)
scheduler_output = SchedulerOutput(
    ...
    new_block_ids_to_zero=new_block_ids_to_zero,
)
```

设计要点：

| 决策 | 理由 |
|------|------|
| 在 `SingleTypeKVCacheManager` 里累积，而不是在 `BlockPool` | 只有 single-type manager 知道 spec 是不是 `FullAttentionSpec`，便于过滤 |
| 用 `type(...) is FullAttentionSpec` 而不是 `isinstance(...)` | 显式排除子类。如果将来出现 `EncoderOnlyAttentionSpec` 这类衍生类，不会被"顺带清零" |
| 每 step drain 一次（`take_new_block_ids`）| 把列表的生命周期对齐到 schedule，避免跨 step 累计漏发或重发 |
| 仅当 `needs_kv_cache_zeroing` 才发送 | 非 hybrid 模型连 `SchedulerOutput` 字段都是 None，零开销 |

`needs_kv_cache_zeroing` 来自 `KVCacheConfig.has_mamba_layers`，所以**普通纯 attention 模型完全不进这条路径**。

## 5.3 Worker 侧 — 一次 Triton launch 清所有段

worker 这边需要解决几个棘手的现实问题：

1. KV cache 不是一段连续显存，而是按 `attention_group` × `layer` × `K/V` 分散在多块 PyTorch tensor 上；
2. 不同 backend 的 block 维度位置不同（`block_dim=0` 还是 `block_dim=1`）；
3. 有 virtual block splitting（scheduler 看到的逻辑 block 不一定等于 kernel 内部 block size）；
4. CuMem 睡眠/唤醒不能让 metadata 失效。

修复用 `KVBlockZeroer`（`vllm/v1/worker/utils.py`）封装：

```python
class KVBlockZeroer:
    def init_meta(self, attn_groups_iter, kernel_block_sizes,
                  cache_dtype, runner_only_attn_layers,
                  static_forward_context):
        # 1) 遍历每个 attention group，只对 FullAttentionSpec
        # 2) 对每个 layer 的 kv_cache buffer 计算其 segment 数与起始 byte 地址
        #    - block_dim=0 layout: 一个 buffer 一个 segment
        #    - block_dim=1 layout: K 和 V 在 buffer 内被分成两段
        # 3) 把所有 segment 的绝对 GPU 地址塞到一个 int64 tensor 里
        #    => seg_addrs_ptr，给 Triton kernel 用
        # 4) 同时计算每 block 的字节大小 PAGE_SIZE_EL（int32 元素数）
        ...

    def zero_block_ids(self, block_ids: list[int]) -> None:
        # 1) 把 block_ids 拷到 pre-allocated pinned CPU tensor → 异步 H2D
        # 2) grid = (n_blocks * n_segs * chunks_per_block,)
        # 3) Triton kernel 直接按 (block_index, seg_index, chunk_index) 三维定位
```

Triton kernel 的核心循环：

```python
@triton.jit
def _zero_kv_blocks_kernel(seg_addrs_ptr, block_ids_ptr, n_blocks,
                           N_SEGS: tl.constexpr,
                           PAGE_SIZE_EL: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    chunks = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    block_index = pid // work_per_block
    if block_index >= n_blocks:
        return
    remainder = pid % work_per_block
    seg_index = remainder // chunks
    chunk_index = remainder % chunks
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    offset = (block_id.to(tl.int64) * PAGE_SIZE_EL
              + chunk_index.to(tl.int64) * BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))
```

把这段读懂的关键点：

| 关键点 | 解释 |
|--------|------|
| 用 `int32` 写 0 | 数据 dtype 不重要——只要把 32 bit 全清成 0，无论解释成 fp32 / 两个 fp16 / 四个 fp8，都仍然是 `+0` |
| 用绝对地址 (`int64`) | 让每个 segment 可以位于不同的 cuda allocation；不依赖某个 base tensor + offset |
| 用 `program_id(0)` 一维 grid | 让 block_index / seg_index / chunk_index 在 kernel 内部解出来，减少 host 端 grid 设置开销 |
| `BLOCK_SIZE` 用最大可被整除的 2 的幂 (`largest_power_of_2_divisor`) | 保证 chunks 整数划分 PAGE，不需要处理 tail |
| 一个 launch 处理所有 segment + 所有 block | 替代 15 次（Qwen3.5-397B 每层一次）的 `index_fill_` 调用，把 launch overhead 摊到几乎可忽略 |

## 5.4 CuMem 边界

```python
# vllm/v1/worker/gpu_worker.py
with self._maybe_get_memory_pool_context(tag="kv_cache"):
    self.model_runner.initialize_kv_cache(kv_cache_config)
...
# 注意：在 CuMem context 外部初始化 zeroer
if kv_cache_config.needs_kv_cache_zeroing and hasattr(
    self.model_runner, "_init_kv_zero_meta"
):
    self.model_runner._init_kv_zero_meta()
```

vLLM 用 CuMem pool 把 KV cache 显存放进一个特殊的池，sleep/wake 时这片显存的页表会被弃用并重新映射。**`seg_addrs` 表里的绝对地址在重新映射后通常会变化**——但 KV cache buffer 本身的 `data_ptr()` 也会变；问题是 `seg_addrs` tensor 自己**不能也住在 CuMem 池里**，否则它会被一起睡掉。

把 `_init_kv_zero_meta()` 放在 `with _maybe_get_memory_pool_context(...)` 之外，就让 zeroer 的 metadata（`seg_addrs`、`ids_gpu` 等）走标准 PyTorch allocator，sleep/wake 不影响。但请注意：**这只是 metadata tensor 安全；当 sleep/wake 改变了 KV cache 的实际地址时，仍然需要在 wake 后重新调 `_init_kv_zero_meta()` 让 seg_addrs 更新**（不在这个 PR 范畴）。

---

# 第六部分: 为什么只清 FullAttention block / 只对 hybrid 启用

## 6.1 Mamba block 不清

```python
if type(self.kv_cache_spec) is FullAttentionSpec:
    self.new_block_ids.extend(b.block_id for b in new_blocks)
```

只有 `FullAttentionSpec` 走累积路径。原因：

| 不清 Mamba block 的理由 | 解释 |
|--------------------------|------|
| Mamba 每步整段覆盖 conv/ssm state | 它不会"露出"旧数据 |
| Mamba 没有"mask 掉无效位置"的概念 | 没有"乘 0 抹除"路径，残留 fp16/fp8 数据对 Mamba 视图本身一般也是合法 fp32 数 |
| 清它会浪费时间 | hybrid 模型 Mamba layer 数远大于 attention layer 数；清 Mamba block 是无效功 |

> 这是一个**单向不变量**：Mamba 写入会污染 Attention，但 Attention 写入不污染 Mamba。

## 6.2 只对 hybrid 启用

```python
@property
def has_mamba_layers(self) -> bool:
    return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

@property
def needs_kv_cache_zeroing(self) -> bool:
    return self.has_mamba_layers
```

对纯 attention 模型，block 的所有占用者都是 fp16/bf16/fp8，跨 dtype 重解释这一条不成立；不会出现 NaN/Inf 残留，**清零反而是纯开销**。所以这个 gating 是必要的。

注意它写成两个独立的 property：未来如果有别的"需要清零"的场景（比如 Mamba 自己出 bug、或者新的 cache 类型），可以扩展 `needs_kv_cache_zeroing` 而不动 `has_mamba_layers` 的语义。

## 6.3 不清 prefix cache hit 的 block

PR 描述中明说：

> **Only freshly allocated blocks** — Prefix-cached blocks (cache hits) are not zeroed.

为什么？因为 prefix cache hit 走的是 `allocate_slots` 的"复用已有 block"路径，**根本不会进入 `new_block_ids` 累积**（`new_block_ids` 只在 `allocate_new_blocks` / `allocate_new_computed_blocks` 内追加）。这是数据流的自然边界，不是显式 if 判断。

prefix cache hit 的 block 此时持有的是合法的、刚被同一个 attention layer 写过的 KV——它本来就是有效数据，不应该被清。

---

# 第七部分: V2 model runner 是否仍有这个 bug

## 7.1 V2 model runner 是什么

本地源码里 `VLLM_USE_V2_MODEL_RUNNER` 这个环境变量打开后，worker 会用 `vllm/v1/worker/gpu/model_runner.py` 中的 `GPUModelRunner`（路径含 `gpu/`，是新版本），而不是 `vllm/v1/worker/gpu_model_runner.py` 这个旧版本：

```python
# vllm/v1/worker/gpu_worker.py
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import (
        GPUModelRunner as GPUModelRunnerV2,
    )
    self.model_runner = GPUModelRunnerV2(...)
else:
    from vllm.v1.worker.gpu_model_runner import (
        GPUModelRunner as GPUModelRunnerV1,
    )
    self.model_runner = GPUModelRunnerV1(...)
```

V2 是结构性重写，不是 V1 的子类。

## 7.2 审计结论：V2 没有这套清零代码

在本地 `02606b0b0` HEAD 上：

```bash
$ grep -n "_zero_block_ids\|_init_kv_zero_meta\|new_block_ids_to_zero\|KVBlockZeroer" \
       vllm/v1/worker/gpu/model_runner.py
# 无输出
```

而 V1 的 `gpu_model_runner.py` 这四个 token 全部命中。**V2 没有移植这个修复**。

更具体地：

| 检查点 | V1 (`gpu_model_runner.py`) | V2 (`gpu/model_runner.py`) |
|--------|----------------------------|-----------------------------|
| `_init_kv_zero_meta()` 方法 | 存在（`utils.py: KVBlockZeroer`）| **不存在** |
| `_zero_block_ids()` 方法 | 存在 | **不存在** |
| `_update_states()` 里调用 `_zero_block_ids` | 是 | V2 用 `add_requests/update_requests/finish_requests` 拆分，**没有任何位置消费 `scheduler_output.new_block_ids_to_zero`** |
| `gpu_worker.py` 中初始化 metadata 的代码 | `hasattr(self.model_runner, "_init_kv_zero_meta")` | `hasattr` 在 V2 上返回 **False**，**静默跳过**，无 warning |

**结论：当 `VLLM_USE_V2_MODEL_RUNNER=1` 时，PR #35219 完全失效**：

1. Scheduler 仍然如常追踪并填充 `SchedulerOutput.new_block_ids_to_zero`（scheduler 与 model runner 解耦）；
2. V2 model runner 拿到这个字段后**不读、不清**；
3. `gpu_worker.py` 的初始化代码用 `hasattr` 包裹，**不会报错也不会警告**——bug 是隐性的。

这是个非常典型的"重写时漏接 patch"的事故：原 PR 把 schema 改了、scheduler 改了，worker 端有两条实现路径——其中一条没人改。`hasattr` 的兜底等于把责任埋掉了。

## 7.3 风险评估

| 维度 | 评估 |
|------|------|
| 谁会踩 | 任何在 V2 model runner 下跑 hybrid 模型（Qwen3.5 系列、Hunyuan-Mamba、Falcon3-Mamba 等）且用 FlashInfer/FlashAttn3 fused 路径的人 |
| 触发概率 | 同 V1：第一次评测可能不出，连续多次后大概率出现精度衰减 |
| 用 V1 workaround | 设 `VLLM_USE_V2_MODEL_RUNNER=0` 即可彻底回避 |
| 后端 workaround | `--attention-backend FLASH_ATTN` 同样有效（issue #35138 已验证）|

V2 是 opt-in 的，所以这个 bug 在 V2 真正成为默认前，影响范围可控。但**一旦用户主动启用 V2 + hybrid 模型 + FlashInfer/TRTLLM 默认后端，就会复现**。

---

# 第八部分: 在 V2 上复现该 bug 的方法

## 8.1 最小复现矩阵

| 选项 | 必要值 | 备注 |
|------|--------|------|
| 模型 | `Qwen/Qwen3.5-397B-A17B-FP8` 或任何 Mamba-hybrid（Qwen3.5-7B-Hybrid、Falcon3-7B-Mamba 等亦可）| 必须有 `MambaSpec` group |
| Tensor parallel | 4（按 issue 原始组合；TP=1/2 也能复现，只是单机内存可能撑不住 397B）| |
| 后端 | `--attention-backend FLASHINFER`（默认）| 用 `FLASH_ATTN` 时即使有残留也会被屏蔽，看不到精度衰减 |
| Prefix cache | 任意（issue 报告里关了；关掉更纯净）| `--no-enable-prefix-caching` |
| Async scheduling | 任意（关掉减少干扰）| `--no-async-scheduling` |
| Enforce eager | 推荐 `--enforce-eager` | 排除 cudagraph 干扰 |
| **V2 开关** | `VLLM_USE_V2_MODEL_RUNNER=1` | **本次复现的核心** |

启动脚本：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --enforce-eager
```

如果没有 4×80G/96G 卡，把模型替换成参数量更小的 Mamba-hybrid（比如 `tiiuae/falcon-mamba-7b-instruct` 这种带 Mamba 层的模型，或 vLLM 仓库中自带的 hybrid test fixture）；只要确认 `KVCacheConfig.has_mamba_layers == True`，就可以触发同样的代码路径。

## 8.2 评测脚本

```bash
# Run 1 (cold)
python tests/evals/gsm8k/gsm8k_eval.py | tee run1.log

# Run 2 (warm) — KV cache 已被 Mamba 大量写过
python tests/evals/gsm8k/gsm8k_eval.py | tee run2.log

# 比较两次准确率
grep "Accuracy" run1.log run2.log
```

期望信号：

| 信号 | 含义 |
|------|------|
| Run 1 准确率 ~90% | 复现一致 |
| Run 2 / Run 3 / ... 准确率显著下降 | 命中 bug |
| 把 `VLLM_USE_V2_MODEL_RUNNER=0` 重跑 | 准确率稳定不退化 → 证明 V2 路径确实丢了 patch |
| 把 `--attention-backend FLASH_ATTN` 加上 | 准确率稳定 → 证明根因仍是 FlashInfer 的"乘 0 抹除"假设 |

## 8.3 更"实验室级"的探针

如果想直接观察 NaN 而不绕道评测准确率，可以在 V2 path 上加一行临时检测（提交时务必删）：

```python
# vllm/v1/worker/gpu/model_runner.py::execute_model 内，prepare_inputs 之后
import torch
for layer_name, ctx in self.compilation_config.static_forward_context.items():
    kv = ctx.kv_cache[0]
    if isinstance(kv, list):
        continue
    if torch.isnan(kv).any() or torch.isinf(kv).any():
        print(f"[NaN/Inf detected] layer={layer_name} shape={kv.shape}")
```

> 注意：直接调 `torch.isnan(kv).any()` 在 fp8 上未必有效（fp8 没有"统一的"isnan 语义，要按具体 e4m3/e5m2 单独判断）。BF16/FP16 视图下都能直接 isnan。最稳的方式是 view 成 `torch.int32`，再判断 32-bit 模式是否落在 fp16/fp8 NaN 编码区间。

如果不想改源码，也可以在 V2 路径上跑：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
NVIDIA_TF32_OVERRIDE=0 \
CUDA_LAUNCH_BLOCKING=1 \
vllm serve ... 2>&1 | grep -i "nan\|inf"
```

并把 FlashInfer 后端的 debug log 调高。如果之前观察到的 trtllm_batch_decode 出现 NaN，会被复现。

## 8.4 反向验证

要证明 V1 path 没问题（fix 起作用），把 V2 关掉重跑一遍：

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve ... # 准确率 run1 ≈ run2 ≈ runN
```

如果你想看到 `_zero_kv_blocks_kernel` 在 V1 下被实际调用，可以用 `nsys` 或者临时在 `KVBlockZeroer.zero_block_ids` 里加 `print(f"zeroing {len(block_ids)} blocks")`。每个 decode step 你应该看到几个 ~30 的数（参考 PR 测量数据）。

---

# 第九部分: QA

## Q1: 为什么不在 `BlockPool.get_new_blocks()` 里直接清？

可以做，但有两个问题：

1. `BlockPool` 不知道这个 block 给的是 Mamba 还是 Attention；如果一刀切全清，Mamba block 也会被清，**Mamba 那 fp32 大段写都白做了**——其实 Mamba 整段覆盖也不在乎旧值，但等于每分配一个 Mamba block 多浪费 ~MB 级清零时间。
2. `get_new_blocks` 是同步函数，在调度路径上，**清零是 GPU work，必须在 worker 上做**。在 scheduler 进程里发 GPU kernel 不可行（多进程架构下 scheduler 通常不持有 cuda device 句柄）。

把这个工作放在 worker 的 `_update_states` 里，是因为它**正好在所有 `prepare_inputs` 之前**——清完再 forward 就保证 attention 看不到旧字节。

## Q2: 为什么是 Triton kernel 而不是 PyTorch `tensor.zero_()` 或 `index_fill_`?

PR 描述里点了：Qwen3.5-397B 有约 15 个 attention 层，每个层是独立的 KV cache buffer（甚至 K/V 可能各占独立 segment）。**用 PyTorch 写就是 15+ 次 `index_fill_` 调用**，每次有自己的 launch overhead，host 端串行设置 grid，加起来在 decode 路径里就不可忽视（decode 整步才 13ms）。

Triton kernel 用一个 launch 处理所有 segment、所有 block、所有 chunk；segment 数 N_SEGS 是 `constexpr`，kernel 内部解三维坐标，host 几乎没什么 overhead。PR 实测在 decode 阶段 ~15μs，约整步的 0.1%。

## Q3: 用 `int32` 写 0 是不是有 endianness 问题？

不会。`0x00000000` 在任何字节序下都是 0。无论后续解释成 fp32 / fp16×2 / fp8×4，所有 bit 都是 0，对应的 IEEE-754 数都是 `+0`。这是这套修复能用"统一 int32 清零"覆盖所有 dtype 的根本原因。

## Q4: `largest_power_of_2_divisor(PAGE_SIZE_EL)` 这个工具函数为什么是这个 PR 引入的？

为了让 BLOCK_SIZE（每个 Triton program 处理的元素数）能整除 PAGE_SIZE_EL，避免在 kernel 里写 tail-handling 分支。`PAGE_SIZE_EL` 来自 `block_size * num_kv_heads * head_size * 2 / 4`（int32 视角下），经常不是 2 的整次幂，所以挑它的"最大 2 的幂因子"作为 BLOCK_SIZE 就既是合法 Triton block 又能整除。

## Q5: 为什么 V2 在审计时连一个"未实现"的 warning 都没有？

`gpu_worker.py` 那段是 `hasattr` 兜底：

```python
if kv_cache_config.needs_kv_cache_zeroing and hasattr(
    self.model_runner, "_init_kv_zero_meta"
):
    self.model_runner._init_kv_zero_meta()
```

设计意图很可能是"让 CPU model runner / 其他设备无痛跳过"——CPU runner 里 `_zero_block_ids` 是个 no-op（注释说 CPU attention 用 `-INF` 掩码，stale data 无影响）。但**V2 GPU runner 既不是 CPU 也不是无影响**，hasattr 把它当成"无影响设备"是错误的。

正确的设计应当至少有一处显式 `else: raise / warn`。这也是建议给社区提的小改动：把 hasattr 改成显式判别（"是 CPU runner 才放过，其他 GPU runner 必须实现"）。

## Q6: 为什么 Qwen3-Next 不出事？

issue 报告 `Qwen3-Next-80B-A3B-Instruct (tp2 and tp4) works fine`。两个可能：

1. 它的 KV cache layout 与 Qwen3.5-397B 不同，**Mamba state 和 Attention KV 没有共享 block pool**（v1 早期的某些 hybrid 模型走的是独立 cache 路径）；
2. 或者它的 Mamba state 数值范围比较小，落进 fp16 重解释后仍然是合法数，NaN/Inf 出现概率低。

第一种更可能。可以通过 grep `Qwen3Next` 在 `vllm/model_executor/models/` 里看它的 `get_mamba_state_shape` / `KVCacheConfig` 构造来核实——这超出本文档范围。

## Q7: 为什么前述 issue 评论里 vadiklyutiy 说"PR #35219 fix it"，但又是 hybrid 才修？

因为 Qwen3.5-397B 就是 hybrid。issue 报告时聚焦于 Qwen3.5，所以表面看像"FlashInfer 后端 + Qwen3.5 的 bug"。把视角放大到"任何 hybrid + 任何乘 0 掩码的后端"，这个修复才是普遍的。

## Q8: 这个 bug 会在哪些更细的场景下"假装好了"然后回来？

| 场景 | 行为 |
|------|------|
| 单次 evaluate，请求总量小 | KV cache 没被 Mamba 摸过几个 block，attention 拿到的几乎都是新池子 block → 不发病 |
| 模型规模小，free pool 远大于 active set | 同上，block 不容易被复用 → 不发病 |
| 用 `--attention-backend FLASH_ATTN` | mask 路径不依赖乘 0 → 表面正常但 KV 仍含 NaN 字节 |
| 用 V1 model runner | 已修 → 真的没问题 |
| 用 V2 model runner | **未修 → 上述复现路径** |
| sleep/wake CuMem 之后 | seg_addrs 表里的地址过期；即便是 V1 也可能有问题（需另一个 PR 验证）|

## Q9: 这个修复对 prefix cache 命中有副作用吗？

没有。前面说过，`new_block_ids` 只在 `allocate_new_blocks` / `allocate_new_computed_blocks` 内追加；**prefix cache hit 路径走的是 `req_blocks.append(cached_block)`，不在追加范围**。所以"被清零"的只是真正从 free pool 弹出来的新块，复用块原封不动。

## Q10: 这是性能修复还是正确性修复？

**正确性修复**，且带有非常小的性能代价（~0.1% 步时长）。两者都没办法 trade off——精度漂移不是"模型语义改变"，是 KV cache buffer 的字节级污染。性能开销之所以这么小，是因为只针对新分配 block（不是整个池），并且用一个 Triton launch 把所有清零工作合并。

---

# 总结

PR #35219 的修复，是给"共享 block pool + 异构 dtype"这种设计补一条**之前隐式存在但没人显式声明的不变量**：

> 任何一个 block，在它的下一个使用者是不同 dtype 的 attention 时，**必须先被清零**。

它由三处协作完成：

1. **scheduler 端**追踪每个 step 新弹出的 `FullAttentionSpec` block id，仅在 hybrid 模型上启用；
2. **worker 端**用 Triton kernel 在一次 launch 里把所有 segment 上对应 block 的字节清成 0；
3. **CuMem 边界**把 zeroer 的 metadata 安置在标准 PyTorch allocator，保证 sleep/wake 不抹掉。

整个修复的物理原理很简单——"被 Mamba 写过的 fp32 比特，在 fp16/fp8 视角下变成 NaN，攻击了 attention 的乘 0 掩码假设"——但工程实现要正确处理 layout / virtual block split / CuMem 这些细节才能做到不退性能。

**值得记住的反直觉点**：

| 反直觉 | 实际 |
|--------|------|
| "Mamba 写的肯定是合法数" | 是。fp32 视角下合法。但 fp16/fp8 重解释下大概率是 NaN/Inf |
| "attention 不读 invalid 位置" | 错。它读但用乘 0 抹除——而 0 × NaN = NaN |
| "block 池在 free 时应该清" | 没人在做，因为同 dtype 复用本来不需要；hybrid 跨 dtype 才需要 |
| "PR 合了应该全路径都修了" | **错**。V2 model runner 没有这套清零逻辑，hybrid + V2 仍在裸奔 |

**给本仓库的下一步建议**（如果你打算自己上 V2）：

1. 在 `vllm/v1/worker/gpu/model_runner.py` 增加 `_init_kv_zero_meta()` 与 `_zero_block_ids()`，直接复用 `KVBlockZeroer`；
2. 在 V2 的 `execute_model()` 进入 prepare_inputs 之前调用 `_zero_block_ids(scheduler_output.new_block_ids_to_zero)`；
3. 把 `gpu_worker.py` 那个 `hasattr` 检查换成"显式列举支持的 runner 类"——避免再出现 V3 又漏掉的事故；
4. 增加一个 unit test：构造 hybrid `KVCacheConfig` → 跑一个 step → 检查 `scheduler_output.new_block_ids_to_zero` 不为空 → 跑 forward → 检查 KV cache buffer 不含 NaN/Inf。这个 test 在两个 runner 上都跑，能挡住未来回归。
