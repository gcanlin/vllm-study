# vLLM Encoder Cache 与 Encode-PD 分离

> **文档版本**: 1.0
> **分析代码版本**: 当前 workspace 本地 `vllm` 源码（V1 引擎）
> **最后更新**: 2026-06-06

---

## 文档概述

多模态推理把"一个请求"切成两段截然不同的工作：

- **encode 段**：vision tower / audio tower 把像素 / 音频跑成一段 embedding。**计算密集、batch 不容易**。
- **prefill + decode 段**：拿到 embedding 后塞进 LLM，prefill 出 KV，再 decode 输出 token。**对 token throughput 敏感**。

两段的硬件偏好、scaling 曲线、batch 策略全部不一样。vLLM 在 V1 引擎里为此搭了**两层基础设施**：

1. **Encoder Cache**：单机内的多模态 embedding 缓存层 —— 同一张图被多个请求 / 多个 prefill chunk 共用时，vision tower 只跑一次。
2. **Encode-PD 分离 (EPD)**：把 encode 段拆到独立的 vLLM 实例上跑，通过 `ec_connector` 把 encoder 输出 push / pull 到 prefill 实例 —— 让两段硬件、模型并行度、副本数各自独立 scale。

这两层是同一套抽象的两个层级：**Encoder Cache 是单机 cache，EC connector 是跨实例 cache**。本文按"先单机、再跨机、最后端到端"展开。

| 部分 | 内容 |
|------|------|
| 第 1 部分 | 为什么要单独管理 encoder cache |
| 第 2 部分 | Scheduler 侧 `EncoderCacheManager`：簿记与预算 |
| 第 3 部分 | Worker 侧 `EncoderCache` + `EncoderRunner`：实际 tensor 与 lifecycle |
| 第 4 部分 | 一张图被 cache 的完整 lifecycle |
| 第 5 部分 | Encode-PD 分离：`ec_connector` 体系 |
| 第 6 部分 | EC vs KV connector：跨实例 cache 的同构设计 |
| 第 7 部分 | 1E1P1D 端到端 walk-through |
| 第 8 部分 | 源码索引 |

姊妹文档：[vllm_multimodal_pipeline_qwen3vl.md](vllm_multimodal_pipeline_qwen3vl.md)（Qwen3-VL pipeline 全景）、[vllm_qwen3_vl_vit_endtoend.md](vllm_qwen3_vl_vit_endtoend.md)（ViT 端到端）。本文与它们互补，只聚焦 encoder 输出的"存"与"传"。

---

# 第 1 部分：为什么要单独管理 encoder cache

## 1.1 一段 encoder 输出，会被复用三次以上

对一张 224×224 的图，Qwen3-VL 的 vision tower 输出约 49 个 embedding（spatial merge 后 14×14/4 = 49）。这段 embedding 的生命周期里：

```text
请求 A prefill chunk 1 → 需要前 30 个 embedding
请求 A prefill chunk 2 → 需要后 19 个 embedding
请求 B（同样图）prefill → 需要全部 49 个 embedding
```

如果不 cache：

- **chunked prefill**：同一张图的 vision tower 跑两次（每 chunk 跑一次）；
- **batch 内同图**：同一张图（同 mm_hash）vision tower 跑 N 次（N 个请求）；
- **多模态预处理流水线**：vision tower 占整条 prefill 时延 30%~60%（取决于模型）。

所以 vLLM 在 V1 里把 encoder 输出做成一个**独立的 cache 层**，单位是 `mm_hash`（一张图 / 一段视频的内容哈希）。

## 1.2 Encoder cache 的"量纲"是 embedding 数

容易混淆的一点：cache 不按 image 个数，也不按 placeholder token 数算，而是按 **encoder 输出的 embedding 数**算。

[`encoder_cache_manager.py:41-45`](../vllm/vllm/v1/core/encoder_cache_manager.py#L41-L45)：

> The EncoderCacheManager operates on the level of multimodal embeddings instead of encoder tokens. This means all break/text tokens in-between multimodal embeddings are not considered with respect to the cache size and the number of free slots.

理由：

- vision tower 输出"多少 embedding"是 cache 真正吃显存的指标；
- 同一段 multimodal placeholder 内可能夹杂 break token（Qwen-VL 的 `<|vision_start|> ... <|vision_end|>` 边界）—— 这些 break token 不占 cache 容量；
- 后面在 `PlaceholderRange.is_embed` 这个 mask 上能进一步细分到稀疏 embedding。

## 1.3 cache 容量 = `max_num_batched_tokens`（默认）

[`config/scheduler.py:96-107, 235-236`](../vllm/vllm/config/scheduler.py#L96-L107)：

```python
max_num_encoder_input_tokens: int = Field(init=False)   # 一步能编码多少 embedding
encoder_cache_size: int = Field(init=False)              # 总缓存容量

# __post_init__:
self.max_num_encoder_input_tokens = self.max_num_batched_tokens
self.encoder_cache_size = self.max_num_batched_tokens
```

两个预算默认都等于 `max_num_batched_tokens`：

- `encoder_compute_budget`：一步 schedule 内 vision tower 最多跑多少 embedding；
- `encoder_cache_size`：cache 总容量上限。

[`encoder_cache_manager.py:269-316`](../vllm/vllm/v1/core/encoder_cache_manager.py#L269-L316) 进一步保证 cache **至少能装下一张最大的图**，否则一张大图永远调度不进来：

```python
max_tokens_per_mm_item = max(mm_max_toks_per_item.values())
...
encoder_cache_size = max(scheduler_config.encoder_cache_size, max_tokens_per_mm_item)
```

`max_tokens_per_mm_item` 来自模型 config（`max_image_tokens` 等），通过 `MultiModalBudget` 获得。

---

# 第 2 部分：Scheduler 侧 `EncoderCacheManager`

[`encoder_cache_manager.py`](../vllm/vllm/v1/core/encoder_cache_manager.py)

这个类**只做簿记，不存 tensor**——tensor 全在 worker 进程的 GPU 上。Scheduler 用它回答"这张图是否在 cache 里"、"还能不能塞新的"、"哪些可以回收"。

## 2.1 四个核心数据结构

[`encoder_cache_manager.py:67-77`](../vllm/vllm/v1/core/encoder_cache_manager.py#L67-L77)：

```python
def __init__(self, cache_size: int):
    self.cache_size = cache_size
    self.num_free_slots = cache_size
    self.num_freeable_slots = cache_size

    # mm_hash → 引用这个 mm_hash 的 request_id 集合
    self.cached: dict[str, set[str]] = {}

    # mm_hash → 已经没人用、待回收的 embedding 数
    self.freeable: OrderedDict[str, int] = OrderedDict()
    self.freed: list[str] = []
```

| 字段 | 物理含义 |
|------|---------|
| `cache_size` | 全局上限（embedding 数） |
| `num_free_slots` | **当前**立刻可用容量 |
| `num_freeable_slots` | 当前容量 + 可立即回收容量。**等于** `cache_size - sum(还有引用的 entry 大小)` |
| `cached` | mm_hash → 引用它的 request_id 集合。**空集合 = 还在 cache 但没人引用** |
| `freeable` | 空引用集合的 mm_hash → 它占多少 embedding。OrderedDict 保证 FIFO 回收 |
| `freed` | 本步已经物理回收的 mm_hash，等 scheduler 打包给 worker |

**`num_free_slots` 与 `num_freeable_slots` 的关系**永远是：

```text
num_free_slots ≤ num_freeable_slots ≤ cache_size
```

- 当某 entry 的引用集合清空（`free_encoder_input`）时：`num_freeable_slots += size`，但 `num_free_slots` **不动**——entry 还留在 cache，只是变成"可回收"；
- 当 `can_allocate` 发现 `num_free_slots` 不够、需要真的回收（`popitem`）时：`num_free_slots += size`，此时两者重新相等于"无引用部分"。

**这是一种典型的 lazy eviction**：扔进 `freeable` 不会立即占用计算 / 通信资源，只在 cache 真的告急时才扫荷包。

## 2.2 `check_and_update_cache`：cache 命中 + 引用计数

[`encoder_cache_manager.py:91-117`](../vllm/vllm/v1/core/encoder_cache_manager.py#L91-L117)：

```python
def check_and_update_cache(self, request, input_id) -> bool:
    mm_hash = request.mm_features[input_id].identifier
    if mm_hash not in self.cached:
        return False

    # Cached but currently not referenced by any request
    if not self.cached[mm_hash]:
        num_encoder_embeds = self.freeable.pop(mm_hash)
        self.num_freeable_slots -= num_encoder_embeds

    self.cached[mm_hash].add(request.request_id)
    return True
```

三种情况：

- `mm_hash` 不在 `cached` → cache miss，返回 False，scheduler 接下来会判断 `can_allocate`；
- `mm_hash` 在 `cached` 但引用集合是空（之前进了 `freeable`）→ **"复活"**：从 `freeable` 拿回 size，把 `num_freeable_slots` 减掉（因为它又开始被引用，不能算可回收容量了），加上新 request_id；
- `mm_hash` 在 `cached` 且有引用 → 直接加上新 request_id。

**关键设计点**：cache hit 时不会增加 `num_free_slots`，因为 entry 始终占着位置。增减只发生在"引用从 0 变 1"或"从 1 变 0"这两个跨越点。

## 2.3 `can_allocate`：双预算检查 + lazy eviction

[`encoder_cache_manager.py:119-178`](../vllm/vllm/v1/core/encoder_cache_manager.py#L119-L178)：

```python
def can_allocate(self, request, input_id,
                 encoder_compute_budget, num_embeds_to_schedule) -> bool:
    num_embeds = request.get_num_encoder_embeds(input_id)

    # ① 计算预算检查
    if num_embeds > encoder_compute_budget:
        return False

    num_embeds += num_embeds_to_schedule

    # ② 空间预算检查
    if num_embeds <= self.num_free_slots:
        return True
    if num_embeds > self.num_freeable_slots:
        return False

    # ③ 触发 lazy eviction
    while num_embeds > self.num_free_slots:
        mm_hash, num_free_embeds = self.freeable.popitem(last=False)
        del self.cached[mm_hash]
        self.freed.append(mm_hash)
        self.num_free_slots += num_free_embeds
    return True
```

三层检查：

1. **`encoder_compute_budget`**：这一步还能让 vision tower 跑多少 embedding。如果一张图本身就比预算大，直接拒绝；
2. **`num_free_slots`**：立即可用容量；
3. **`num_freeable_slots`**：可立即 + 可回收容量。如果连这都不够，**真的塞不下**了。

只有在"②够但①不够"的中间情况下，才触发 `popitem(last=False)`（OrderedDict 的 FIFO 出队）→ 把最早进 `freeable` 的若干 entry 真物理回收。被回收的 mm_hash 追加进 `freed`，等下一次 `get_freed_mm_hashes()` 打包给 worker：

[`encoder_cache_manager.py:255-266`](../vllm/vllm/v1/core/encoder_cache_manager.py#L255-L266)：

```python
def get_freed_mm_hashes(self) -> list[str]:
    freed = self.freed
    self.freed = []
    return freed
```

Worker 在 [`gpu_model_runner.py:1153-1154`](../vllm/vllm/v1/worker/gpu_model_runner.py#L1153-L1154) 这边收到：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

——真正释放 GPU tensor。

## 2.4 `allocate` / `free_encoder_input` / `free`：簿记三剑客

[`encoder_cache_manager.py:180-205`](../vllm/vllm/v1/core/encoder_cache_manager.py#L180-L205)：

```python
def allocate(self, request, input_id) -> None:
    mm_hash = request.mm_features[input_id].identifier
    if mm_hash not in self.cached:
        self.cached[mm_hash] = set()
    num_encoder_embeds = request.get_num_encoder_embeds(input_id)
    # can_allocate 已经保证够
    assert self.num_free_slots >= num_encoder_embeds
    assert self.num_freeable_slots >= num_encoder_embeds
    self.cached[mm_hash].add(request_id)
    self.num_free_slots -= num_encoder_embeds
    self.num_freeable_slots -= num_encoder_embeds
```

"已经 can_allocate 过" 是个**强假设**：scheduler 调用顺序必须是 `can_allocate → allocate`，中间不能交错其他状态变更。

[`encoder_cache_manager.py:221-253`](../vllm/vllm/v1/core/encoder_cache_manager.py#L221-L253)：

```python
def free_encoder_input(self, request, input_id) -> None:
    req_id = request.request_id
    mm_hash = request.mm_features[input_id].identifier
    if not self.cached.get(mm_hash, None):
        return
    self.cached[mm_hash].discard(req_id)
    if not self.cached[mm_hash]:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.freeable[mm_hash] = num_encoder_embeds
        self.num_freeable_slots += num_encoder_embeds

def free(self, request) -> None:
    input_ids = self.get_cached_input_ids(request)
    for input_id in input_ids:
        self.free_encoder_input(request, input_id)
```

`free_encoder_input` **不动 `num_free_slots`**——entry 物理上还在，只是引用变 0，进 `freeable` 等待 lazy eviction。

`free(request)` 在两个地方被调用：
- 请求正常 finish（[`scheduler.py:958`](../vllm/vllm/v1/core/sched/scheduler.py#L958)、[`scheduler.py:1867`](../vllm/vllm/v1/core/sched/scheduler.py#L1867)）；
- 请求被 abort / cancel。

## 2.5 在 scheduler 主循环里的调用位置

[`scheduler.py:1192-1276`](../vllm/vllm/v1/core/sched/scheduler.py#L1192-L1276) 是 `_try_schedule_encoder_inputs` 的核心，按 mm_features 一个个走：

```python
for i, mm_feature in enumerate(request.mm_features):
    ...
    if self.encoder_cache_manager.check_and_update_cache(request, i):
        # 已经被本机 encode 过，跳过 vision tower
        continue
    ...
    if not self.encoder_cache_manager.can_allocate(
        request, i, encoder_compute_budget, num_embeds_to_schedule):
        # 容量满 → 把 num_new_tokens 砍到 mm item 之前，留到下一步再调度
        num_new_tokens = start_pos - num_computed_tokens
        break
    ...
    if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
        # EPD 模式下：连接器那边已有 → 本机不跑 vision tower，标记为外部加载
        external_load_encoder_input.append(i)
        num_embeds_to_schedule += num_encoder_embeds
        continue

    num_embeds_to_schedule += num_encoder_embeds
    encoder_compute_budget -= num_encoder_embeds
    encoder_inputs_to_schedule.append(i)
```

两个出口 + 一个"切尾巴"逻辑：

- `encoder_inputs_to_schedule`：本机要跑 vision tower 的 mm 项；
- `external_load_encoder_input`：本机**不跑**、由 EC connector 拉过来的；
- 容量满时 `num_new_tokens = start_pos - num_computed_tokens` —— scheduler **把 prefill 截在 mm 占位区之前**，避免一个 chunk 跨过未编码的 mm 边界。这是 chunked prefill 与 encoder cache 的关键耦合点。

调度结果最后写进 `SchedulerOutput`（[`scheduler.py:912-920`](../vllm/vllm/v1/core/sched/scheduler.py#L912-L920)）：

```python
scheduled_encoder_inputs=scheduled_encoder_inputs,         # {req_id → [mm_input_id...]}
free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
```

worker 拿这两个字段就够干活了。

---

# 第 3 部分：Worker 侧 `EncoderCache` + `EncoderRunner`

[`gpu/mm/encoder_cache.py`](../vllm/vllm/v1/worker/gpu/mm/encoder_cache.py) + [`gpu/mm/encoder_runner.py`](../vllm/vllm/v1/worker/gpu/mm/encoder_runner.py)

Worker 这边不需要复杂的容量簿记 —— scheduler 已经替它算好了。它只负责：**按 scheduler 的指令跑 encoder、把结果塞进 dict、按指令删 dict**。

## 3.1 `EncoderCache`：两个 dict 就完事

[`encoder_cache.py:8-41`](../vllm/vllm/v1/worker/gpu/mm/encoder_cache.py#L8-L41)：

```python
class EncoderCache:
    def __init__(self):
        # req_id → MM features (含 mm_position 等元数据)
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash → 实际 encoder 输出 tensor (GPU 上)
        self.encoder_outputs: dict[str, torch.Tensor] = {}

    def add_request(self, req_id, mm_features): ...
    def remove_request(self, req_id): ...
    def free_encoder_cache(self, mm_hash):
        self.encoder_outputs.pop(mm_hash, None)
```

两层 dict 的分工：

- **`mm_features`**：以 request 为主键，给 `gather_mm_embeddings` 用 —— 它要知道"这个 request 的第 k 个 mm item 对应哪个 mm_hash、放在 prompt 哪个 offset"；
- **`encoder_outputs`**：以 mm_hash 为主键，因为同一张图的 tensor 可以被多个 request 共享。

**注意**：`encoder_outputs` 的 value 是 vision tower 的输出 tensor，而不是 LLM 的 input_embeds。形状一般是 `(num_embeds, hidden_size)` 或 `(feature_size, hidden_size)`，dtype 跟 LLM hidden 一致（bf16 / fp16）。

## 3.2 `EncoderRunner`：三步把 encoder 跑完

[`encoder_runner.py`](../vllm/vllm/v1/worker/gpu/mm/encoder_runner.py)

### 3.2.1 `prepare_mm_inputs`：把 SchedulerOutput 翻译成 kwargs

[`encoder_runner.py:34-48`](../vllm/vllm/v1/worker/gpu/mm/encoder_runner.py#L34-L48)：

```python
def prepare_mm_inputs(self, scheduled_encoder_inputs) -> tuple[list[str], list[...]]:
    mm_hashes, mm_kwargs = [], []
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        mm_features = self.encoder_cache.mm_features[req_id]
        for mm_input_id in encoder_input_ids:
            mm_feature = mm_features[mm_input_id]
            if mm_feature.data is None:
                continue
            mm_hashes.append(mm_feature.identifier)
            mm_kwargs.append((mm_feature.modality, mm_feature.data))
    return mm_hashes, mm_kwargs
```

`mm_feature.data` 是 HF processor 的输出（pixel_values、grid_thw 等已经 packed 好），直接喂给 model 即可。

### 3.2.2 `execute_mm_encoder`：按 modality 分组 batch

[`encoder_runner.py:50-62`](../vllm/vllm/v1/worker/gpu/mm/encoder_runner.py#L50-L62)：

```python
@torch.inference_mode()
def execute_mm_encoder(self, mm_kwargs):
    encoder_outputs = []
    for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
        mm_kwargs, device=self.device, pin_memory=False,
    ):
        batch_outputs = self.model.embed_multimodal(**mm_kwargs_batch)
        sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
        encoder_outputs.extend(batch_outputs)
    return encoder_outputs
```

`group_and_batch_mm_kwargs` 把同 modality 的连续段拼起来一次 forward（vision 一组、audio 一组），既给 batch 留空间又保持 mm item 的顺序。

`gpu_model_runner.py` 里有更复杂的版本 [`_execute_mm_encoder`](../vllm/vllm/v1/worker/gpu_model_runner.py#L2869-L3078)：

- 处理 `prompt_embeds` 这个 passthrough modality（[`gpu_model_runner.py:2884-2904`](../vllm/vllm/v1/worker/gpu_model_runner.py#L2884-L2904)）—— 直接塞进 cache，不跑 encoder；
- 支持 `encoder_cudagraph_manager` 给 vision tower 开 CUDA Graph（[`gpu_model_runner.py:3053-3066`](../vllm/vllm/v1/worker/gpu_model_runner.py#L3053-L3066)）；
- 支持 multimodal LoRA mapping（tower-side adapter）；
- **最后两件事**（[`gpu_model_runner.py:3072-3076`](../vllm/vllm/v1/worker/gpu_model_runner.py#L3072-L3076)）：

```python
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

——同时存入本地 cache，并 **通知 EC connector**（如果启用）。这是 EPD producer 端的关键钩子，下一节展开。

### 3.2.3 `gather_mm_embeddings`：按 chunk 切片塞进 prefill

[`encoder_runner.py:64-134`](../vllm/vllm/v1/worker/gpu/mm/encoder_runner.py#L64-L134)：

```python
def gather_mm_embeddings(self, req_ids, total_num_scheduled_tokens, ...):
    ...
    for i, req_id in enumerate(req_ids):
        if not is_prefilling[i]:
            continue   # decode 不需要

        for mm_feature in self.encoder_cache.mm_features[req_id]:
            pos_info = mm_feature.mm_position
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            if start_pos >= query_end[i]:           break    # 还没到
            if start_pos + num_encoder_tokens <= query_start[i]:  continue   # 已经过去

            start_idx = max(query_start[i] - start_pos, 0)
            end_idx   = min(query_end[i] - start_pos, num_encoder_tokens)
            curr_embeds_start, curr_embeds_end = (
                pos_info.get_embeds_indices_in_range(start_idx, end_idx)
            )
            ...
            mm_hash = mm_feature.identifier
            encoder_output = self.encoder_cache.encoder_outputs[mm_hash]

            if (is_embed := pos_info.is_embed) is not None:
                is_embed = is_embed[start_idx:end_idx]
                mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
            else:
                mm_embeds_item = encoder_output[start_idx:end_idx]

            mm_embeds.append(mm_embeds_item)
            ...
    return mm_embeds, is_mm_embed
```

这是 **chunked prefill 的关键**：

- `[start_idx, end_idx)` 是当前 chunk 落在 mm item 内的相对区间；
- `pos_info.get_embeds_indices_in_range` 把"placeholder token 区间"翻译成"实际 embedding 区间"（因为 `is_embed` mask 可能让 placeholder 比 embedding 多）；
- 即使一张图被切成 3 chunk 处理，`encoder_output` 这个 tensor **整张都在 GPU 上不动**，每次 chunk 只切片读一段。

**这就是 encoder cache 存在的最直接收益**：vision tower 跑一次，prefill 切几片就切几片。

## 3.3 何时清 cache：scheduler 推给 worker

[`gpu_model_runner.py:1153-1154`](../vllm/vllm/v1/worker/gpu_model_runner.py#L1153-L1154)：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

[`gpu_model_runner.py:6233`](../vllm/vllm/v1/worker/gpu_model_runner.py#L6233)：

```python
self.encoder_cache.clear()    # 整体清空（reset / 权重换了）
```

worker 永远不主动决定回收，**只看 `scheduler_output.free_encoder_mm_hashes`**——这保证了 scheduler 的簿记和 worker 的实际状态总是一致。

---

# 第 4 部分：一张图被 cache 的完整 lifecycle

把上面拆开的 API 串起来，跟一张 224×224 单图（产生 49 个 embedding）在系统里的轨迹：

```text
T0：请求 A 到达，prompt 含 1 张图（mm_hash = H1，49 embeds）

T1：scheduler 主循环 _try_schedule_encoder_inputs：
    check_and_update_cache(A, 0):
      H1 not in cached → False
    can_allocate(A, 0, budget=512, scheduled=0):
      49 ≤ num_free_slots=8192 → True
    →  encoder_inputs_to_schedule.append(0)

T2：scheduler 主循环结束，调用 allocate(A, 0)：
    cached[H1] = {A}
    num_free_slots = 8192 - 49 = 8143
    num_freeable_slots = 8143
    SchedulerOutput.scheduled_encoder_inputs = {A: [0]}

T3：worker 执行 SchedulerOutput：
    _execute_mm_encoder：
      vision tower 跑出 encoder_outputs[H1] = Tensor(49, 4096) bf16
      maybe_save_ec_to_connector(H1)  # EPD producer 时落盘
    gather_mm_embeddings：
      请求 A 的 prefill chunk 1 取 H1 的前 30 个 embedding
    LLM forward。

T4：请求 A 继续 prefill chunk 2：
    （H1 已在 cache，scheduler check_and_update_cache(A, 0) → True，不再 schedule）
    gather_mm_embeddings 切 H1 的后 19 embedding。

T5：请求 B 到达，同样含 H1 这张图：
    check_and_update_cache(B, 0):
      H1 in cached, cached[H1]={A} 非空 → 直接 add(B) → True
    cached[H1] = {A, B}
    （没动 num_free_slots，因为容量没变）
    →  encoder_inputs_to_schedule（不含 B 的这张图）

T6：请求 A finish：
    scheduler.free(A) → free_encoder_input(A, 0)
    cached[H1] = {B}（还有 B）
    （没动 freeable，引用集合非空）

T7：请求 B finish：
    scheduler.free(B) → free_encoder_input(B, 0)
    cached[H1] = set() （空）
    freeable[H1] = 49
    num_freeable_slots += 49 → 8192
    （num_free_slots 还是 8143！）

T8：新请求 C 到达，需要新图 H2，95 embeds：
    can_allocate(C, 0, budget=512, scheduled=0):
      95 > num_free_slots=8143 ? 不会，绰绰有余 → True
    （H1 仍然留在 cache，"零成本"等待复用）

T9：...假设容量真的告急，新请求 D 需要 200 embeds 但 num_free_slots 只剩 100：
    can_allocate(D, 0, ...):
      ① 200 ≤ budget=512 ✓
      ② 200 > num_free_slots=100 → 跳到 ③
      ③ 200 ≤ num_freeable_slots=149 ?
         假设是 → while 循环：popitem H1 (49)
            del cached[H1]; freed.append(H1)
            num_free_slots = 100 + 49 = 149 ≥ 200 ?
            如果还不够，继续 popitem 下一个 freeable…
      最终 num_free_slots ≥ 200，return True
    →  allocate(D, 0)：num_free_slots -= 200, num_freeable_slots -= 200

T10：下一个 SchedulerOutput 携带 free_encoder_mm_hashes = [H1]：
     worker：encoder_cache.pop(H1, None) → 物理释放 49×4096×2 bytes
```

**几个反复出现的设计直觉**：

- **引用计数 + lazy eviction**：被释放的 entry 永远先进 `freeable`，只有真没空间才从 `freeable` 物理回收；
- **凡是回收，必然是 FIFO**：`OrderedDict.popitem(last=False)`；
- **`num_free_slots` 单调下降直到 eviction**：cache hit 既不消耗它也不增加它；
- **worker 是"奴隶"**：它的 cache 状态完全由 `SchedulerOutput.free_encoder_mm_hashes` 驱动。

---

# 第 5 部分：Encode-PD 分离 (`ec_connector` 体系)

[`distributed/ec_transfer/`](../vllm/vllm/distributed/ec_transfer/)

EPD 的动机是**异构 scaling**：

- vision tower 重算、轻内存（一张图几 GB pixel → 几十 KB embedding）；
- LLM prefill / decode 重 KV、轻算（每 token 几 MB KV，几 GFLOP）。

如果把它们绑在一个进程里，**算力 vs 显存比例固定**，无法分别 scale。EPD 让这俩 scaling 解耦：encode 实例可以堆"高算力 / 少显存"的卡，prefill / decode 实例可以堆"大显存 / 高带宽"的卡。

vLLM 的 `ec_connector` 抽象就是这件事情的"传送带"。

## 5.1 `ECTransferConfig`：角色与连接器

[`config/ec_transfer.py:10-108`](../vllm/vllm/config/ec_transfer.py#L10-L108)：

```python
ECProducer = Literal["ec_producer", "ec_both"]
ECConsumer = Literal["ec_consumer", "ec_both"]
ECRole = Literal[ECProducer, ECConsumer]

@config
class ECTransferConfig:
    ec_connector: str | None = None           # 连接器类名
    ec_role: ECRole | None = None             # 角色
    ec_rank: int | None = None                # 0 for encoder, 1 for pd instance
    ec_parallel_size: int = 1
    ec_buffer_device: str | None = "cuda"
    ec_connector_extra_config: dict = field(default_factory=dict)
    ec_connector_module_path: str | None = None
    ...
```

三种角色：

| 角色 | 行为 |
|------|------|
| `"ec_producer"` | 跑 vision tower，把结果发出去；本机不跑 LLM |
| `"ec_consumer"` | 不跑 vision tower，从远端拉 encoder 输出；本机跑 LLM |
| `"ec_both"` | 既跑又能接收（混合部署、debug 用） |

`ec_rank` 当前只支持 `1P1D`（实际上是 1E1PD）：rank 0 是 encoder 实例，rank 1 是 prefill/decode 实例。

## 5.2 `ECConnectorBase`：两侧抽象

[`ec_connector/base.py:59-278`](../vllm/vllm/distributed/ec_transfer/ec_connector/base.py#L59-L278)

抽象类被实例化成两份，一份在 scheduler 进程，一份在 worker 进程，两边角色不同：

```python
class ECConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1
```

### 5.2.1 Scheduler 侧方法

| 方法 | 何时被调 | 作用 |
|------|---------|------|
| `has_cache_item(mm_hash)` | `_try_schedule_encoder_inputs` 检查能否跳过本机 encoder | 判断 mm_hash 是否已在远端 ready |
| `ensure_cache_available(req, num_computed_tokens)` | 请求进 RUNNING 前 | 触发异步拉取；返回 False 时延后 schedule |
| `update_state_after_alloc(req, index)` | `allocate` 之后 | 记一笔"我要拉这个 mm_hash"，build_meta 时打包 |
| `build_connector_meta(scheduler_output)` | `SchedulerOutput` 打包前 | 把待拉/待存清单做成 `ECConnectorMetadata` |
| `request_finished(req)` | 请求结束时 | 决定是否同步落盘 |

### 5.2.2 Worker 侧方法

| 方法 | 何时被调 | 作用 |
|------|---------|------|
| `bind_connector_metadata(meta)` | `execute_model` 入口 | 把 scheduler 打包的 meta 注入 worker 实例 |
| `start_load_caches(encoder_cache, **kwargs)` | `gather_mm_embeddings` 前 | **consumer 端**：把远端 tensor 加载进本地 `encoder_cache` dict |
| `save_caches(encoder_cache, mm_hash, **kwargs)` | `_execute_mm_encoder` 完成一项 | **producer 端**：把 tensor 写到远端存储 |
| `get_finished(finished_req_ids)` | `execute_model` 出口 | 返回已完成异步传输的 req_ids |
| `clear_connector_metadata()` | `execute_model` 出口 | 清掉本步 meta |

[`ec_connector_model_runner_mixin.py:55-78`](../vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py#L55-L78) 用一个 contextmanager 把整套 lifecycle 围起来：

```python
@contextmanager
def _get_ec_connector_output(scheduler_output, encoder_cache, **kwargs):
    output = ECConnectorOutput()
    ec_connector = get_ec_transfer()
    ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

    if ec_connector.is_consumer:
        ec_connector.start_load_caches(encoder_cache, **kwargs)

    try:
        yield output
    finally:
        output.finished_sending, output.finished_recving = (
            ec_connector.get_finished(scheduler_output.finished_req_ids)
        )
        ec_connector.clear_connector_metadata()
```

`save_caches` 在另一条路径上调用（[`gpu_model_runner.py:3076`](../vllm/vllm/v1/worker/gpu_model_runner.py#L3076) 在 `_execute_mm_encoder` 内部），所以 producer 和 consumer 的 entry point 不同：producer 在跑完 vision tower 后立即 push，consumer 在 forward 前从 pull buffer 取。

## 5.3 参考实现 `ECExampleConnector`：共享文件系统版

[`ec_connector/example_connector.py`](../vllm/vllm/distributed/ec_transfer/ec_connector/example_connector.py)

最直白的实现，把 tensor 用 safetensors 写到共享 path 下：

```python
class ECExampleConnector(ECConnectorBase):
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        self._mm_datas_need_loads: dict[str, int] = {}
        self._storage_path = vllm_config.ec_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp",
        )

    def has_cache_item(self, identifier):
        return os.path.exists(self._generate_filename_debug(identifier))

    def update_state_after_alloc(self, request, index):
        mm_hash = request.mm_features[index].identifier
        if not self.is_consumer or not self.has_cache_item(mm_hash):
            return
        num_encoder_token = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(self, scheduler_output):
        meta = ECExampleConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    def start_load_caches(self, encoder_cache, **kwargs):
        metadata = self._get_connector_metadata()
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            filename = self._generate_filename_debug(mm_data.mm_hash)
            ec_cache = safetensors.torch.load_file(
                filename, device=current_platform.device_type,
            )["ec_cache"]
            encoder_cache[mm_data.mm_hash] = ec_cache

    def save_caches(self, encoder_cache, mm_hash, **kwargs):
        if not self.is_producer:
            return
        filename = self._generate_filename_debug(mm_hash)
        tensors = {"ec_cache": encoder_cache[mm_hash].detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
```

这版用于 debug，生产场景要换成 RDMA / shared GPU memory / object store 的实现。但接口形状不变 —— **传输细节被 connector 完全隐藏**。

## 5.4 在 scheduler 主循环里的两个钩子

### 钩子 1：跳过本机 encoder（[`scheduler.py:1258-1264`](../vllm/vllm/v1/core/sched/scheduler.py#L1258-L1264)）

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
    mm_hashes_to_schedule.add(item_identifier)
    external_load_encoder_input.append(i)
    num_embeds_to_schedule += num_encoder_embeds
    continue   # 不进入 encoder_inputs_to_schedule
```

只要 connector 报告"远端有"，本机就：
- 仍然走 `can_allocate / allocate`（**因为本机要占 cache 空间存这份 tensor**）；
- 但**不加进 `scheduled_encoder_inputs`** —— worker 不会跑 vision tower；
- 记进 `external_load_encoder_input`，下游 `update_state_after_alloc` 会把这条加到 connector 待拉清单。

### 钩子 2：延后 schedule（[`scheduler.py:633-642`](../vllm/vllm/v1/core/sched/scheduler.py#L633-L642)）

```python
if (self.ec_connector is not None
        and request.mm_features
        and not self.ec_connector.ensure_cache_available(request, num_computed_tokens)):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

如果 encoder 实例还没把数据传过来（远端还没 ready），就把请求**重新放回 WAITING 队列**，下一步再试。这是 "连接器 not ready → request defer" 的统一退路。

`ensure_cache_available` 在基类返回 True（[`base.py:214-229`](../vllm/vllm/distributed/ec_transfer/ec_connector/base.py#L214-L229)），具体连接器可以重载：

- ExampleConnector 用 `os.path.exists` 直接同步检查；
- 生产连接器可以发起异步预取，第一次返回 False，下一步发现 ready 再返回 True。

## 5.5 在 worker 主循环里的两个钩子

[`gpu_model_runner.py:3076`](../vllm/vllm/v1/worker/gpu_model_runner.py#L3076)（producer push）：

```python
# 在 _execute_mm_encoder 内部，每跑完一项立刻 save
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

[`ec_connector_model_runner_mixin.py:67-69`](../vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py#L67-L69)（consumer pull）：

```python
# 在 execute_model 的 contextmanager 入口
if ec_connector.is_consumer:
    ec_connector.start_load_caches(encoder_cache, **kwargs)
```

producer 是 **fire-and-forget**：跑完就 save，不等回执；consumer 是 **block 直到 ready**（更准确说，是"如果 `ensure_cache_available` 已经 True，那 `start_load_caches` 一定能加载成功"）。

---

# 第 6 部分：EC connector vs KV connector

vLLM 的 PD 分离用 `kv_connector` 走类似套路，两者放在一起看会更清楚：

| 维度 | EC connector | KV connector |
|------|-------------|-------------|
| 传输对象 | encoder 输出 tensor | KV cache block |
| 数据流向 | E → PD | P → D |
| 数据粒度 | 一个 mm_hash 一份大 tensor | 一个 block 一份小 tensor |
| 启动时机 | encoder 跑完立即 push | prefill 跑完 / 跑中 push |
| Consumer 退路 | `ensure_cache_available` False → defer | `WAITING_FOR_REMOTE_KVS` 状态 |
| Producer/Consumer 模型 | `is_producer / is_consumer` 标志 | 同 |
| Metadata 通道 | `ECConnectorMetadata` 挂在 `SchedulerOutput.ec_connector_metadata` | `KVConnectorMetadata` 挂在 `SchedulerOutput.kv_connector_metadata` |
| Cache 簿记 | `EncoderCacheManager`（embedding 数） | `KVCacheManager`（block 数） |
| 与 chunked prefill 配合 | `gather_mm_embeddings` 按 chunk 切片读 | 跨 chunk 时 KV 已经按 block 单位存 |

**核心同构**：

```text
scheduler  ──build_connector_meta──>  SchedulerOutput[.*_connector_metadata]
                                            │
                                            v
worker     ──bind_connector_metadata──>  start_load → forward → save / get_finished
```

两者都遵循 "scheduler 打包 → worker 一次性消费 → context manager 包住" 的 lifecycle，让连接器的具体实现可以是 NCCL / RDMA / Mooncake / 共享盘 / 共享内存…… 任意可选。

---

# 第 7 部分：1E1P1D 端到端 walk-through

场景：

- **E 实例**：`ec_role="ec_producer"`，只装 vision tower（实际上 LLM 也在但不跑 prefill）；
- **P 实例**：`ec_role="ec_consumer"` + KV connector producer，跑 LLM prefill；
- **D 实例**：KV connector consumer，跑 decode。

一个请求"`<text><image><text>`，图片 H1，49 embeddings"：

```text
T0：所有 3 个实例都收到这个请求（前端 router 复制）

E 实例
  scheduler:
    check_and_update_cache(req, 0) → False
    can_allocate(req, 0, ...) → True
    encoder_inputs_to_schedule = [0]
    allocate(req, 0)
    SchedulerOutput.scheduled_encoder_inputs = {req: [0]}
  worker:
    _execute_mm_encoder: vision tower 跑出 H1 tensor → encoder_cache[H1]
    maybe_save_ec_to_connector(H1) → 写文件 /shared/H1/encoder_cache.safetensors
  E 实例完成 encode，请求在 E 上结束。

P 实例
  第 1 个 scheduler step：
    ensure_cache_available(req, 0):
      ec_connector.has_cache_item(H1):
        os.path.exists(...) → 文件还没写出来 → False
      → return False
    request 退回 WAITING

  ...

  P 实例第 K 个 scheduler step（E 已经写完）：
    ensure_cache_available(req, 0) → True
    _try_schedule_encoder_inputs:
      check_and_update_cache → False（本机从没跑过）
      can_allocate → True，但下一步：
      ec_connector.has_cache_item(H1) → True
        → external_load_encoder_input.append(0)
        → 不进 scheduled_encoder_inputs
    update_state_after_alloc(req, 0):
      ec_connector._mm_datas_need_loads[H1] = 49
    build_connector_meta:
      ECExampleConnectorMetadata.mm_datas = [(H1, 49)]
    SchedulerOutput.ec_connector_metadata = meta
    SchedulerOutput.scheduled_encoder_inputs = {}      ← 不跑 vision tower

  P 实例 worker:
    execute_model contextmanager 入口：
      bind_connector_metadata(meta)
      is_consumer → start_load_caches(encoder_cache):
        for H1: safetensors.load → encoder_cache[H1] = ec_cache
    _execute_mm_encoder：mm_kwargs 为空，skip
    gather_mm_embeddings：从 encoder_cache[H1] 切片
    LLM prefill forward
    KV connector save：把 KV block 推到 D
    execute_model 出口：
      get_finished(finished_req_ids) → 返回已完成的传输
      clear_connector_metadata()

D 实例
  scheduler:
    KV connector: ensure_kv_cache_available → 等 P 推 KV block
    一旦 ready，正常 decode（mm_features 在 D 上没有任何特殊处理）
  worker:
    标准 decode loop，不碰 encoder_cache
```

**几个值得品的细节**：

1. **P 实例上仍然有 EncoderCacheManager**：因为 P 的本机 `encoder_cache` 也要簿记容量、要追踪引用 / 释放、`gather_mm_embeddings` 也要按 chunk 切。EC connector 只是替换了"数据从哪来"，没替换"数据怎么消费"。

2. **`can_allocate` 在 P 实例上仍然走**：尽管 P 不跑 encoder，它仍然需要拿出 49 个 slot 来存远端拉过来的 tensor —— cache 容量 = GPU 显存预算，跟数据来源无关。

3. **`encoder_compute_budget` 在 P 实例上"白给"**：因为 P 不跑 vision tower，理论上 budget 不消耗。当前实现里 EPD 路径 `continue` 在 `encoder_compute_budget -= ...` 之前，所以确实没扣。

4. **失败的恢复**：如果 E 实例在 push 之前挂了，P 的 `ensure_cache_available` 永远返回 False，请求会一直 defer。这是 fail-stop 设计，没有自愈，依赖外层 retry。

5. **同图复用跨实例**：如果 batch 内多个请求引用同一张图，E 只算一次（`mm_hash` 一样、scheduler `check_and_update_cache` 命中本机），P 也只 load 一次（`encoder_cache` 已有跳过）。**两层 cache 都吃这块红利**。

6. **P 上 `_execute_mm_encoder` 还在 `prompt_embeds` 路径上跑**：[`gpu_model_runner.py:2884-2904`](../vllm/vllm/v1/worker/gpu_model_runner.py#L2884-L2904) 处理 passthrough modality（已经是 embedding 的输入）—— 这种不通过 connector 也能直接走 `maybe_save_ec_to_connector`。

---

# 第 8 部分：源码索引

| 内容 | 路径 |
|------|------|
| **Scheduler 侧** | |
| `EncoderCacheManager` 与 `EncoderDecoderCacheManager` | [`vllm/v1/core/encoder_cache_manager.py`](../vllm/vllm/v1/core/encoder_cache_manager.py) |
| `compute_mm_encoder_budget` | [`encoder_cache_manager.py:269-316`](../vllm/vllm/v1/core/encoder_cache_manager.py#L269-L316) |
| Scheduler 主循环对 encoder cache 的所有调用 | [`vllm/v1/core/sched/scheduler.py`](../vllm/vllm/v1/core/sched/scheduler.py) lines 1149-1276（schedule），520-540 / 633-642（defer），955-960 / 1860-1870（finish） |
| `SchedulerOutput.scheduled_encoder_inputs` / `free_encoder_mm_hashes` / `ec_connector_metadata` | [`vllm/v1/core/sched/output.py`](../vllm/vllm/v1/core/sched/output.py) |
| 调度配置 (`encoder_cache_size`, `max_num_encoder_input_tokens`) | [`vllm/config/scheduler.py:96-107, 235-236`](../vllm/vllm/config/scheduler.py#L96-L107) |
| **Worker 侧** | |
| `EncoderCache` 本体 | [`vllm/v1/worker/gpu/mm/encoder_cache.py`](../vllm/vllm/v1/worker/gpu/mm/encoder_cache.py) |
| `EncoderRunner` 三步 | [`vllm/v1/worker/gpu/mm/encoder_runner.py`](../vllm/vllm/v1/worker/gpu/mm/encoder_runner.py) |
| `_execute_mm_encoder` / `_gather_mm_embeddings`（含 LoRA / cudagraph / connector hooks） | [`vllm/v1/worker/gpu_model_runner.py`](../vllm/vllm/v1/worker/gpu_model_runner.py) lines 2826-3078（execute），3080-3165（gather），1149-1160（free hashes）|
| **EC connector** | |
| 角色 / 配置 | [`vllm/config/ec_transfer.py`](../vllm/vllm/config/ec_transfer.py) |
| 抽象基类 | [`vllm/distributed/ec_transfer/ec_connector/base.py`](../vllm/vllm/distributed/ec_transfer/ec_connector/base.py) |
| 示例实现（共享盘） | [`vllm/distributed/ec_transfer/ec_connector/example_connector.py`](../vllm/vllm/distributed/ec_transfer/ec_connector/example_connector.py) |
| Worker 端 mixin | [`vllm/v1/worker/ec_connector_model_runner_mixin.py`](../vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py) |
| Connector factory | [`vllm/distributed/ec_transfer/ec_connector/factory.py`](../vllm/vllm/distributed/ec_transfer/ec_connector/factory.py) |
| 全局 helper `has_ec_transfer / get_ec_transfer` | [`vllm/distributed/ec_transfer/__init__.py`](../vllm/vllm/distributed/ec_transfer/__init__.py) |
| **多模态预算** | |
| 模型/processor 算 `mm_max_toks_per_item` | [`vllm/multimodal/encoder_budget.py`](../vllm/vllm/multimodal/encoder_budget.py) |
| **测试** | |
| 单元测试覆盖 EncoderCacheManager 全部边界 | [`tests/v1/core/test_encoder_cache_manager.py`](../vllm/tests/v1/core/test_encoder_cache_manager.py) |
