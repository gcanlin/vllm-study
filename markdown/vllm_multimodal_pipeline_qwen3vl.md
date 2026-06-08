# vLLM 多模态处理全流程：以 Qwen3-VL 为例

> 这是「数据从一张图像 + prompt 到一个采样 token」的完整调用链。
> ViT 内部结构见 [vllm_qwen3_vl_vit_endtoend.md](./vllm_qwen3_vl_vit_endtoend.md)，本文专注**外围基础设施**：
> registry / hash / cache / scheduler / model runner / encoder cudagraph / EVS。

---

## 0. 全景：一次 `LLM.generate` 都经过哪些模块

```
┌─ User ─────────────────────────────────────────────────────────────────┐
│   prompt = "Describe <|vision_start|><|image_pad|><|vision_end|>"      │
│   mm_data = {"image": [PIL.Image]}                                     │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌─ Frontend (API / LLM class) ─────────────────────────────────┐
│  InputPreprocessor.preprocess()                              │
│   → renderer._process_multimodal(prompt, mm_data)            │
│   → BaseMultiModalProcessor.apply(inputs)                    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌─ MultiModalProcessor (per-model, eg. Qwen3VLMultiModalProcessor) ──────┐
│  1. _call_hf_processor       — HF Qwen3VLProcessor 图像/视频预处理     │
│  2. _get_mm_fields_config    — 把 BatchFeature 拆成多 item             │
│  3. _get_prompt_updates      — 计算 token 占位符替换规则               │
│  4. apply_token_matches      — 把 <|image_pad|> 展开成 N 个 <|image|>  │
│  5. MultiModalHasher.hash_kwargs(mm_data, mm_kwargs)                   │
│     → mm_hashes                                                        │
│  6. ProcessorCache.put(hash, MultiModalKwargsItem)                     │
│     输出: prompt_token_ids + mm_kwargs + mm_hashes + mm_placeholders  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼ (per request, with mm_features list)
┌─ Engine Core ─ Scheduler.schedule() ─────────────────────────┐
│  EncoderCacheManager.check_and_update_cache()                │
│  encoder_compute_budget / encoder_cache_size 守门             │
│  → SchedulerOutput.scheduled_encoder_inputs                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌─ GPU Worker / GPUModelRunner ────────────────────────────────┐
│  _execute_mm_encoder(scheduler_output)                       │
│    group_and_batch_mm_kwargs() → 同 modality 合 batch        │
│    可能走 EncoderCudaGraphManager.execute(...)               │
│    否则 model.embed_multimodal(**mm_kwargs_batch)            │
│      → Qwen3VLForConditionalGeneration.embed_multimodal      │
│        → self.visual(pixel_values, grid_thw)                 │
│  encoder_outputs 存到 self.encoder_cache[mm_hash]            │
│                                                              │
│  _gather_mm_embeddings(scheduler_output)                     │
│    遍历 req.mm_features，从 encoder_cache 取出当前 chunk     │
│    建立 is_mm_embed mask                                     │
│                                                              │
│  model.embed_input_ids(input_ids, mm_embeds, is_mm_embed)    │
│    → _merge_multimodal_embeddings  (in-place index_put)      │
│    → 同时设置 deepstack_input_embeds buffer                  │
│                                                              │
│  language_model.model.forward(inputs_embeds, deepstack, ...) │
│  Sampler / logits / output                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. 注册：模型类如何接入 MM 体系

[qwen3_vl.py:1655](../vllm/vllm/model_executor/models/qwen3_vl.py#L1655)

```python
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGeneration(nn.Module, SupportsMultiModal, ...):
    ...
```

三件套：
- **`ProcessingInfo`**：纯只读，拥有 `get_hf_config / get_hf_processor / get_image_processor / get_data_parser`、以及计算「这个图会变多少 token」的工具方法 `_get_vision_info`、`get_num_frames_with_most_features`。
- **`DummyInputsBuilder`**：profile 阶段构造最大模型形状的假数据（决定 `max_num_batched_tokens` 内能塞多少图）。
- **`MultiModalProcessor`**：把用户输入 `(prompt, mm_data, mm_kwargs)` 变成 `(prompt_token_ids, MultiModalKwargsItems, mm_hashes, mm_placeholders)`。

`MULTIMODAL_REGISTRY.register_processor` 把三个工厂挂在 model class 上（`model_cls._processor_factory`）。
Engine 初始化时 `MULTIMODAL_REGISTRY.create_processor(model_config)` 拿出来用。

`SupportsMultiModal` 协议除上述外还要求 model 实现：
- `get_placeholder_str(modality, i)` —— 返回 `<|vision_start|><|image_pad|><|vision_end|>` 这种字符串占位符（chat template 里用）。
- `embed_multimodal(**mm_kwargs)` —— ViT/Audio encoder 入口。
- `embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)` —— 把 mm 特征合入 LLM 文本 embedding。

可选的协议：`SupportsEncoderCudaGraph`、`SupportsLoRA`、`SupportsPP`、`SupportsMRoPE`、`SupportsMultiModalPruning`。

---

## 2. 输入解析：图/视频/音频走进 HF Processor

[qwen3_vl.py:1264](../vllm/vllm/model_executor/models/qwen3_vl.py#L1264) `Qwen3VLMultiModalProcessor._call_hf_processor`

### 2.1 mm_data 的形状
Engine 接受的标准化形态（`MultiModalDataDict`）：

```python
mm_data = {
    "image": [PIL.Image | np.ndarray | dict(...)],   # 单图 list-of-one
    "video": [(np.ndarray, VideoMetadata)],          # video_array + metadata
    # "audio": [...]
}
```

`Qwen2VLMultiModalDataParser` 做归一化、`expected_hidden_size` 校验（防止用户直接传
image_embeds 时维度对不上）。

### 2.2 视频的「分而治之」
Qwen3-VL 视频处理不是一把 batch 完，是**每个视频独立** processor + 计算 timestamps：

```python
for video_array, metadata in videos:
    # 关键：do_sample_frames 决定 timestamps 怎么算
    timestamps = self.info._get_video_second_idx(metadata, ...)

    video_outputs = super()._call_hf_processor(
        prompt="<|vision_start|><|video_pad|><|vision_end|>",  # 单独跑一次
        mm_data={"videos": [[video_array]], "video_metadata": [[metadata]]},
        mm_kwargs=video_mm_kwargs,
        tok_kwargs=tok_kwargs,
    )

    # 算 tokens_per_frame；启用 EVS 时按剪枝率缩短
    if video_pruning_rate > 0:
        num_tokens = compute_retained_tokens_count(tokens_per_frame_base, num_frames, q)
        tokens_per_frame = [num_tokens, 0, 0, ..., 0]   # 把所有 token 放第一帧（占位用）
    else:
        tokens_per_frame = [tokens_per_frame_base] * num_frames

    video_repl = get_video_repl(tokens_per_frame, timestamps, ...)
    video_placeholder = tokenizer.decode(video_repl.full)
    prompt = prompt.replace("<|vision_start|><|video_pad|><|vision_end|>",
                            video_placeholder, 1)   # 把短占位符替换成完整 token 序列
```

注意 `prompt.replace(... 1)` —— **只替换一次**，保证多视频按出现顺序匹配。
图像的 prompt update 走第 4 步的 `_get_prompt_updates`（更通用的机制）。

### 2.3 HF Processor 输出
返回的 `BatchFeature` 是 dict：
- `input_ids`：含展开 placeholder 后的 token ids
- `pixel_values` / `pixel_values_videos`：packed patch tensor `(N_total_patches, 1176)`
- `image_grid_thw` / `video_grid_thw`：`(N_items, 3)` 整数 [t, h, w]
- `timestamps`：仅视频

---

## 3. `_get_mm_fields_config`：BatchFeature → 多 item

[qwen3_vl.py:1402](../vllm/vllm/model_executor/models/qwen3_vl.py#L1402)
→ [qwen2_vl.py:774](../vllm/vllm/model_executor/models/qwen2_vl.py#L774) `_create_qwen2vl_field_factory`

```python
return dict(
    pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid.prod(-1)),
    image_embeds=MultiModalFieldConfig.flat_from_sizes("image", grid.prod(-1) // m**2),
    image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
    pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_grid.prod(-1)),
    video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
    timestamps=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
)
```

它告诉 vllm「这个字段是怎么被多个 item 共享 / 切分的」：
- **`flat_from_sizes("image", sizes)`**：`pixel_values` 是把 N 张图的 patch 拼起来，每张图占 `sizes[i]` 个 patch；vllm 拆出每张图时按 cumsum 切。
- **`batched("image", keep_on_cpu=True)`**：`image_grid_thw` 第 0 维是 item 维，逐 item 取一个；`keep_on_cpu=True` 不上 device。

这些信息让 `MultiModalKwargsItems.from_hf_inputs` 能把 BatchFeature 解开成 per-item：

```python
MultiModalKwargsItems({
    "image": [
        MultiModalKwargsItem({"pixel_values": ..., "image_grid_thw": ...}),  # 第 1 张图
        MultiModalKwargsItem({"pixel_values": ..., "image_grid_thw": ...}),  # 第 2 张图
    ],
    "video": [...],
})
```

——为什么需要拆？因为 cache、hash、调度都以「一个 mm item」为单位，而不是「一个请求」。

---

## 4. `_get_prompt_updates`：让 token 序列长度匹配 ViT 输出

[qwen3_vl.py:1411](../vllm/vllm/model_executor/models/qwen3_vl.py#L1411)

输入 prompt 里只有一个 `<|image_pad|>` 占位（chat template 给的）；但 ViT 一张图会输出
`N = t*h*w / 4` 个 token。需要把这一个占位符**展开成 N 个**，对齐到 ViT 输出。

```python
def get_image_replacement_qwen3vl(item_idx: int):
    grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
    num_tokens = int(grid_thw.prod()) // merge_length     # = t*h*w / 4
    return [hf_processor.image_token_id] * num_tokens     # N 个 <|image|>

return [
    PromptReplacement(modality="image",
                      target=hf_processor.image_token,    # "<|image_pad|>"
                      replacement=get_image_replacement_qwen3vl),
    PromptReplacement(modality="video", target="<|vision_start|><|video_pad|><|vision_end|>",
                      replacement=get_video_replacement_qwen3vl),
]
```

`apply_token_matches` / `apply_text_matches` 用这些规则在 `prompt_token_ids` 上做替换，
同时产出 `PlaceholderRange(offset, length, is_embed)`——告诉调度器**第几个 token 到第几个 token**是
某张图的占位区段。

`is_embed` 是布尔 mask：对于视频带「帧时间戳文本」的情况，区段内并不是全部都是 image
token——`<|frame_0.5s|>` 等文本 token 散在中间，is_embed 标记哪些位置才放视觉特征。

---

## 5. MM Hash & Processor Cache

### 5.1 `MultiModalHasher.hash_kwargs(**mm_data, **mm_kwargs)`
[hasher.py:154](../vllm/vllm/multimodal/hasher.py#L154)

把图像 / 视频 / 张量字节流喂给 BLAKE3，得到稳定 hash：

```python
for k, v in sorted(kwargs.items()):
    for bytes_ in cls.iter_item_to_bytes(k, v):
        hasher.update(bytes_)
return hasher.hexdigest()
```

- PIL.Image：如果 EXIF 里有 `ImageID` UUID，直接用（避免重新序列化整张图）；否则按 mode + np.asarray 字节流 hash。
- bfloat16 tensor：先 view 成 uint8 字节流（NumPy 不支持 bf16）。
- numpy ndarray：直接拿底层 buffer。

**hash 包含什么**：mm_data 本身（图）+ mm_processor_kwargs（fps、min/max_pixels、num_frames、size 等会影响处理结果的参数）+ model_id（防止换模型时 cache 串味）。

**hash 不包含什么**：prompt text、tokenization_kwargs（这些不影响 ViT 输出）。

### 5.2 Processor Cache
[multimodal/cache.py](../vllm/vllm/multimodal/cache.py)
`BaseMultiModalProcessorCache`：用 mm_hash 作为 key，value 是 `MultiModalKwargsItem`（ViT 输入张量）。
两级：
- **本地 LRU**：进程内，每条 ~图像 size 的几 KB metadata + pixel_values。
- **远程 KV connector**（可选）：跨节点。

`_cached_apply_hf_processor` 的关键逻辑：
1. 拿 `mm_hashes`（按 item）
2. 查 cache，分离 `mm_is_cached` 和 `mm_missing_data_items`
3. **只对 missing items 调 HF processor**——避免重复跑 image_processor
4. 合并：把 cached items 和新算的拼回 `MultiModalKwargsItems`
5. 算 `_get_prompt_updates`，得到完整 prompt_token_ids

这一层 cache 的命中场景：同一个 system prompt + image 的多次请求、多轮对话。

---

## 6. mm_features：每个请求带的多模态特征清单

`Processor` 在请求构造时把上面 4 步的产物挂到 `Request.mm_features`：

```python
@dataclass
class MultiModalFeatureSpec:
    data: MultiModalKwargsItem | None          # 该 item 的张量（pixel_values / grid_thw / timestamps）
    modality: str                              # "image" / "video" / "audio" / "prompt_embeds"
    identifier: str                            # 该 item 的 mm_hash
    mm_position: PlaceholderRange              # 在 prompt token 序列里的 (offset, length, is_embed)
    lora_request: LoRARequest | None           # 若使用 tower LoRA
```

一个请求带 N 个图 = N 个 `MultiModalFeatureSpec`。后面调度器和模型 runner 都按 feature 维处理。

---

## 7. 调度器：encoder budget & encoder cache 守门

[v1/core/sched/scheduler.py:1141](../vllm/vllm/v1/core/sched/scheduler.py)
`_try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens,
encoder_compute_budget, ...)`

```
for each mm_feature in request.mm_features:
    start_pos    = feature.mm_position.offset
    encoder_len  = feature.mm_position.length
    embeds_num   = feature.mm_position.get_num_embeds()

    # 1. 落在已计算 token 之外？跳过
    if start_pos >= num_computed_tokens + num_new_tokens: break
    # 2. 已经被 LLM 消费了？KV cache 里已经有，不再编码
    if start_pos + encoder_len <= num_computed_tokens: continue
    # 3. encoder cache 命中（之前请求或同请求历史步算过）？跳过编码
    if self.encoder_cache_manager.check_and_update_cache(request, i): continue

    # 4. chunked prefill 模式下，决定要不要在 mm 之前停下来
    if disable_chunked_mm_input and 部分覆盖: roll back num_new_tokens, break
    # 5. budget 不够（compute budget 或 cache slot 都查）？停下来
    if not encoder_cache_manager.can_allocate(...): roll back, break

    # 6. 通过，记账
    encoder_compute_budget -= embeds_num
    encoder_inputs_to_schedule.append(i)
```

### 7.1 两类预算
- **`encoder_compute_budget`**：每步 forward 能跑多少 vision token（ViT 计算量上限）。
- **`encoder_cache_size`**：能同时缓存多少 vision token 的输出（embeddings 显存上限）。

`compute_mm_encoder_budget`（[encoder_cache_manager.py:269](../vllm/vllm/v1/core/encoder_cache_manager.py#L269)）从 `scheduler_config.encoder_cache_size` 和「最大 mm item 的 token 数」之间取 max——保证至少装得下最坏情况的一张图。

### 7.2 EncoderCacheManager
[encoder_cache_manager.py:17](../vllm/vllm/v1/core/encoder_cache_manager.py#L17)
- `cached: dict[mm_hash, set[request_id]]`：mm_hash → 引用它的 request set
- `freeable: OrderedDict[mm_hash, num_embeds]`：引用计数归零但还没物理释放，LRU 排序
- `allocate / free` 在请求开始/结束时调
- `get_freed_mm_hashes()` 让 worker 也把 GPU 上的 `encoder_cache[hash]` 张量丢掉

**淘汰策略**：分配时 if 不够：从 freeable 头部（最旧）evict 直到够；evict 的 hash 加进 `freed` 列表，下次 scheduler output 里同步给 worker。

### 7.3 scheduled_encoder_inputs
`SchedulerOutput.scheduled_encoder_inputs: dict[req_id, list[int]]`——每个请求这一步要算的 mm_feature index 列表。Worker 据此决定要不要跑 encoder。

---

## 8. Worker 侧执行：`_execute_mm_encoder`

[gpu_model_runner.py:2869](../vllm/vllm/v1/worker/gpu_model_runner.py#L2869)

```python
mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
# mm_kwargs: list[(modality, MultiModalKwargsItem)]

# 同 modality 合 batch，跨 modality 切开
for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(...):
    # cudagraph 可用？走 cudagraph
    if encoder_cudagraph_manager and encoder_cudagraph_manager.supports_modality(modality):
        batch_outputs = encoder_cudagraph_manager.execute(mm_kwargs_batch)
    else:
        batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
    encoder_outputs.extend(batch_outputs)

# 入 cache
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
```

### 8.1 `group_and_batch_mm_kwargs`
把 `MultiModalKwargsItem` 按 modality 分组，把同组的 `pixel_values` cat 起来，`image_grid_thw` stack 起来。GPU 上一次跑多张图。

### 8.2 跨 modality 不合 batch 的原因
注释明确写了「FIXME: hacky way」：当一个 batch 里既有 image 又有 video，按出现顺序拆开 follow-by-each-modality 处理，否则 mm_position 的 offset 与 encoder 输出顺序对不上。

### 8.3 EVS 视频的特殊降级
启用 `is_multimodal_pruning_enabled` 或 `requires_sequential_video_encoding` 时**视频每个 item 单独 forward**：因为 EVS 的 retention_mask 让输出 shape 变动态，batch 起来 shape 对不齐。

### 8.4 prompt_embeds 旁路
`prompt_embeds` modality 是「用户已经给好 LLM embedding」，不跑 encoder，直接 `pe_tensor.to(device)` 塞进 `encoder_cache[mm_hash]`，后续路径完全一样。

### 8.5 LoRA Tower
ViT 也可以挂 LoRA（`mm_lora_refs`）；通过 `LoRAMapping(type=LoRAMappingType.TOWER)` 单独激活，与 LLM 主模型的 LoRA 隔离。

---

## 9. `embed_multimodal`：模型类的 ViT 入口

[qwen3_vl.py:2734](../vllm/vllm/model_executor/models/qwen3_vl.py#L2734)

```python
def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings | None:
    mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    multimodal_embeddings: list[torch.Tensor] = []

    for modality in mm_input_by_modality:                  # 保留顺序：image 先于 video
        if modality == "image":
            image_embeddings = self._process_image_input(multimodal_input)
            image_embeddings = self._postprocess_image_embeds_evs(image_embeddings, ...)
            multimodal_embeddings.extend(image_embeddings)
        if modality == "video":
            video_embeddings = self._process_video_input(multimodal_input)
            if self.is_multimodal_pruning_enabled:
                video_embeddings = self._postprocess_video_embeds_evs(...)
            multimodal_embeddings.extend(video_embeddings)

    return tuple(multimodal_embeddings)
```

`_process_image_input`（[qwen3_vl.py:2150](../vllm/vllm/model_executor/models/qwen3_vl.py#L2150)）：
- DP ViT 模式：`run_dp_sharded_mrope_vision_model(self.visual, pixel_values, grid_thw_list, rope_type="rope_3d")`
- 否则：`self.visual(pixel_values, grid_thw=grid_thw)`
- 输出按每张图 token 数 `split`，返回 tuple

返回值是 `tuple[torch.Tensor, ...]`，长度 = item 数；每个 tensor shape `(t*h*w/4, hidden * (1+#deepstack))`。

---

## 10. Encoder CUDA Graph

[v1/worker/encoder_cudagraph.py:53](../vllm/vllm/v1/worker/encoder_cudagraph.py#L53) `EncoderCudaGraphManager`

### 10.1 配置
模型实现 `get_encoder_cudagraph_config`：

```python
EncoderCudaGraphConfig(
    modalities=["image", "video"],   # EVS 开就清空
    buffer_keys=["pixel_values", "pos_embeds", "rotary_pos_emb_cos",
                 "rotary_pos_emb_sin", "cu_seqlens", "max_seqlen",
                 "sequence_lengths"],
    out_hidden_size=visual.out_hidden_size,
    max_frames_per_video=...,
)
```

### 10.2 capture 阶段
- 用 `get_encoder_cudagraph_budget_range` 给出最小/最大 token budget。
- 跨多个 (item_count, max_input_size) 组合分别 capture（避免每次 replay 形状不同）。
- `prepare_encoder_cudagraph_capture_inputs` 给出最坏情况的 pixel_values、grid_thw、cu_seqlens
  —— 注意视频的 grid_config 用 T>1（每个 item 贡献 T 个 attention 序列）。
- `prepare_encoder_metadata(max_batch_size, max_frames_per_batch, max_seqlen_override)`
  让 cu_seqlens / max_seqlen padding 到最坏情况。

### 10.3 replay 阶段
- `execute(mm_kwargs_batch)` 根据 batch 的 item 数选择最贴近的 graph。
- 把实际 pixel_values 拷进 pre-allocated buffer，replay graph，从 output buffer 拷出来。
- 输出按 `item_specs.output_tokens` 切回 per-item tensor 列表。

### 10.4 与谁不兼容
- **EVS**：retention_mask 让输出 shape 动态变化；config 里 `modalities=[]` 关掉。
- **动态 FP8 scale**：amax buffer wrap 时会 `.item()` 同步，破坏 graph capture。

---

## 11. `_gather_mm_embeddings`：从 cache 取出当前 chunk 的特征

[gpu_model_runner.py:3080](../vllm/vllm/v1/worker/gpu_model_runner.py#L3080)

为什么不直接用 encoder_outputs？因为 **chunked prefill**：
- 第 1 步可能只算了 prompt 前 1024 个 token（图还没到）；
- 第 2 步算到图所在位置，要从 encoder_cache 里取该 mm_feature 的 embedding；
- 第 3 步图已经过了，文本继续。

`_gather_mm_embeddings` 做的事：
```python
for req in input_batch:
    for mm_feature in req.mm_features:
        # 计算本步 [num_computed, num_computed+num_scheduled) 与
        # mm_feature.mm_position [start_pos, start_pos+enc_len) 的交集
        if 不相交: continue
        encoder_output = self.encoder_cache[mm_feature.identifier]
        # 按 mm_position.is_embed 切片
        mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
        is_mm_embed[ req_start_pos+start_idx : req_start_pos+end_idx ] = True
        mm_embeds.append(mm_embeds_item)

return mm_embeds, is_mm_embed   # list of tensors + bool mask 长度 = total_scheduled_tokens
```

`is_mm_embed` 是 CPU 上的 bool 张量；交给 `_merge_multimodal_embeddings`
（[utils.py:456](../vllm/vllm/model_executor/models/utils.py#L456)）做 `inputs_embeds[is_multimodal] = mm_embeds_flat`——in-place index_put，没有额外 D2H 同步。

EVS 模式下额外触发 `model.recompute_mrope_positions`，重算 mRoPE 让剪枝后 token 的位置仍正确。

---

## 12. 文本 embedding + 视觉 embedding 的合并

[qwen3_vl.py:2805](../vllm/vllm/model_executor/models/qwen3_vl.py#L2805)
`Qwen3VLForConditionalGeneration.embed_input_ids`

```python
inputs_embeds = self._embed_text_input_ids(input_ids, self.language_model.embed_input_ids, ...)

if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
    return inputs_embeds                                      # 纯文本

if self.use_deepstack:
    deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(
        inputs_embeds, multimodal_embeddings, is_multimodal,
    )                                                         # split 通道 + scatter

inputs_embeds = _merge_multimodal_embeddings(
    inputs_embeds, multimodal_embeddings, is_multimodal,
)                                                             # 把视觉 embed 写到占位符位置

if deepstack_input_embeds is not None:
    self._set_deepstack_input_embeds(deepstack_input_embeds)  # 写到 buffer 等 LLM 用
```

`_compute_deepstack_embeds` 内部：
1. cat 所有 multimodal_embeddings（last dim = `(1+#deepstack) * out_hidden`）
2. `torch.split([visual_dim, multiscale_dim], dim=-1)` 拆成主特征 + 多 scale 特征
3. 多 scale 特征 reshape 成 `(seq_len, #deepstack, visual_dim)`，permute 为 `(#deepstack, seq, dim)`
4. 写进 `self.deepstack_input_embeds[i]`（pre-allocated）

---

## 13. LLM 端消费 deepstack

[qwen3_vl.py:1569](../vllm/vllm/model_executor/models/qwen3_vl.py#L1569) `Qwen3LLMModel.forward`

```python
for layer_idx, layer in islice(enumerate(self.layers), start_layer, end_layer):
    hidden_states, residual = layer(positions, hidden_states, residual)
    if deepstack_input_embeds is not None and layer_idx in range(len(deepstack_input_embeds)):
        hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
```

deepstack 在前 N 层 LLM 每层后加一次（残差注入），完全无需 LLM 权重变化。

Pipeline parallel：`start_layer >= len(deepstack_visual_indexes)` 时 PP 的非第一 stage 不参与
deepstack 注入（assert 保证 deepstack 全部发生在 PP rank 0）。

---

## 14. mRoPE：图/视频 token 怎么编码位置

[qwen3_vl.py:2562](../vllm/vllm/model_executor/models/qwen3_vl.py#L2562) `get_mrope_input_positions`

LLM 端 positions shape `(3, seq_len)`——三段对应 t/h/w。

```
text token:      (p, p, p)        # 三个值相同
image token:     (t_idx, h_idx, w_idx)
video token:     (frame_idx * second_per_grid_t * tokens_per_sec,
                  h_idx, w_idx)
```

`compute_mrope_for_media`（EVS 路径用，[evs.py:95](../vllm/vllm/multimodal/evs.py#L95)）把
`(T, H, W)` 网格展开成 `(N, 3)` 的位置矩阵；`recompute_mrope_positions` 在剪枝后重排。

`get_rope(rope_parameters={"mrope_section": [...]})` 在 LLM rotary 内部按 section 划分 cos/sin 通道。

---

## 15. EVS（Efficient Video Sampling）剪枝细节

[evs.py:38](../vllm/vllm/multimodal/evs.py#L38)

```python
def compute_retention_mask(video_embeds, (T, H, W), spatial_merge_size, q):
    # video_embeds: (T*H*W/m^2, hidden)，已经过 ViT
    video_embeds = video_embeds.reshape(T, H/m, W/m, hidden)
    # 帧间 cosine similarity
    sim = F.cosine_similarity(video_embeds[1:], video_embeds[:-1], dim=-1)  # (T-1, H/m, W/m)
    dissim = 1 - sim                                                         # 越不像 → 保留
    dissim = cat([255 * ones_like(first_frame_tokens), dissim])              # 第一帧全保留
    order = argsort(dissim.flatten(), descending=True)
    keep_K = compute_retained_tokens_count(tokens_per_frame, T, q)
    mask = zeros; mask[order[:keep_K]] = True
    return mask
```

`q ∈ [0, 1)` 是剪枝比例。EVS 的位置在 **ViT 输出之后、merge 到 input_embeds 之前**——
所以 cache 里存的还是完整 ViT 输出，剪枝是后处理。

EVS 链路上的额外工作：
- HF processor 阶段就要把 placeholder 数缩短（[qwen3_vl.py:1339](../vllm/vllm/model_executor/models/qwen3_vl.py#L1339)）让 prompt token 数对得上
- `_postprocess_video_embeds_evs`：剪枝后还要把 video 替换成 "frame_t1.0s ... frame_t2.5s ..." 的文字 token 序列（`get_video_repl`），并把视觉特征 scatter 到 `is_embed=True` 的位置
- model runner 的 `_gather_mm_embeddings` 触发 `recompute_mrope_positions`

---

## 16. DP ViT（`mm_encoder_tp_mode="data"`）

[model_executor/models/vision.py:383](../vllm/vllm/model_executor/models/vision.py#L383) `run_dp_sharded_mrope_vision_model`

```python
tp_size, tp_rank = ...
patches_per_image = [t*h*w for ...]                # 每图 patch 数
image_to_tp_rank, gpu_sample_counts, _ = \
    get_load_balance_assignment(patches_per_image, tp_size)   # 贪心负载均衡
# 本 rank 拿到的 image idx 列表
image_idxs_local = image_to_tp_rank[start:end]
pixel_values_local = cat(pixel_values[per-image slice for i in image_idxs_local])

# 本 rank 跑完整 ViT 副本
image_embeds_local = vision_model(pixel_values_local, grid_thw=local_grid_thw_list)

# 跨 rank all-gather（要 padding 到 max_len_per_rank）
output = tensor_model_parallel_all_gather(padded_local)
# 按原顺序重组
```

适用场景：
- 图像数量 >> tp_size（每 rank 拿到一两张图算得快）
- 各图分辨率差异大，TP allreduce 通信热点

代价：
- 每 rank 完整 ViT 权重副本
- all-gather 需要 padding

---

## 17. Pipeline 终点：Sampler 与 logits

deepstack 注入完成、ViT embeddings 合入 inputs_embeds 后，剩下就是普通的 LLM 推理：

```python
hidden_states = self.language_model.model(input_ids=None,
                                          positions=positions,
                                          inputs_embeds=inputs_embeds,
                                          deepstack_input_embeds=deepstack_input_embeds)
logits = self.language_model.lm_head(hidden_states[selected_positions])
sampled_token = sampler(logits, sampling_params)
```

`selected_positions` 是 `scheduler_output.num_scheduled_tokens` 累计到 prefill 末尾或 decode 位置；
mm_features 对应 token 位置永远不会被 sample（它们是 placeholder，prompt 一部分）。

---

# 一些跨模块的概念澄清

## C1. mm_kwargs / mm_data / mm_processor_kwargs 区别

| 名称 | 来源 | 内容 | 典型字段 |
|---|---|---|---|
| `mm_data` | 用户 | 原始媒体 | `{"image": [PIL.Image], "video": [(array, metadata)]}` |
| `mm_processor_kwargs` | 用户 | 影响 HF processor 的参数 | `min_pixels=256*28*28, max_pixels=1280*28*28, fps=2, num_frames=8` |
| `mm_kwargs`（HF 输出） | HF processor | 张量化的输入 | `pixel_values, image_grid_thw, video_grid_thw, timestamps` |
| `mm_kwargs_batch`（worker） | `group_and_batch_mm_kwargs` | 同 modality 合 batch 后的张量 | 同上但 N item 拼起来 |

`mm_data` + `mm_processor_kwargs` → HF → `mm_kwargs`；`mm_kwargs` 进 hash & cache & ViT。

## C2. encoder_cache 在三个地方各自指什么

| 位置 | 含义 |
|---|---|
| `EncoderCacheManager.cached`（**调度器**） | mm_hash → 引用 request set；记账用，不存张量 |
| `GPUModelRunner.encoder_cache`（**worker**） | mm_hash → 真实 GPU tensor（ViT 输出） |
| `BaseMultiModalProcessorCache`（**frontend**） | mm_hash → ViT 输入（pixel_values 等），不存输出 |

调度器和 worker 通过 `SchedulerOutput.scheduled_encoder_inputs` 和 `free_encoder_mm_hashes` 保持同步：
调度器决定调度哪些 mm_feature、释放哪些 hash；worker 按指令算并释放。

## C3. 一张图的完整数据流（packed）

```
PIL.Image(H, W, 3)
   │ HF image_processor: smart_resize + patchify
   ▼
pixel_values    (t*h*w, 1176)        ← packed 单图
image_grid_thw  (3,) = [t, h, w]
   │ MultiModalKwargsItem
   ▼
mm_hash = BLAKE3(pixel_values + grid_thw + mm_processor_kwargs + model_id)
   │ Scheduler 拍板：本步要算这个 hash
   ▼ (worker)
pixel_values: cat with other images → (Σ tᵢ hᵢ wᵢ, 1176)
   │ group_and_batch_mm_kwargs
   ▼
model.embed_multimodal:
   visual(pixel_values, grid_thw)
   │ Conv3d patch_embed + pos_embed interpolate
   │ 27 × Qwen3_VisionBlock (with rotary pos emb)
   │ deepstack 抽 3 层
   │ PatchMerger (spatial 2×2 合并)
   │ cat([main, deepstack_1, deepstack_2, deepstack_3])
   ▼
encoder_output  (t*h*w/4, (1+3)*out_hidden)
   │ split by image
   ▼
self.encoder_cache[mm_hash] = output      ← worker 缓存
   │ _gather_mm_embeddings: slice 当前 chunk
   ▼
mm_embeds, is_mm_embed
   │ _compute_deepstack_embeds: split [visual | scale1 | scale2 | scale3]
   │ _merge_multimodal_embeddings: inputs_embeds[is_mm] = visual
   │ _set_deepstack_input_embeds: buffer[i] = scale_i
   ▼
inputs_embeds (seq_len, LLM_hidden)
   │ Qwen3LLMModel.forward
   │ layer 0..2: hidden += deepstack_buffer[i]
   ▼
final hidden → lm_head → sample
```

## C4. Profile 阶段的 dummy 数据
[qwen3_vl.py:1115](../vllm/vllm/model_executor/models/qwen3_vl.py#L1115) `Qwen3VLDummyInputsBuilder`
- `get_dummy_text`：拼最长 prompt 字符串（按 `mm_counts` 决定多少图/视频占位符）。
- `get_dummy_mm_data`：构造最大分辨率的图、最长视频。
- Engine 启动时跑一遍 dummy forward 决定 `max_num_batched_tokens` 和 encoder_compute_budget。

## C5. 多模态相关的几个 env / config 关键字
- `VLLM_MM_ENCODER_MAX_PATCHES_PER_CHUNK`：env，ViT chunk forward 阈值
- `VLLM_MM_HASHER_ALGORITHM`：env，BLAKE3 / SHA256 等
- `mm_encoder_tp_mode`：config，`"data"` 走 DP ViT
- `mm_encoder_attn_backend`：config，强制 backend
- `mm_encoder_attn_dtype`：config，`"fp8"` 启用 FP8
- `mm_encoder_fp8_scale_path` / `mm_encoder_fp8_scale_save_path`：config
- `video_pruning_rate`：config，EVS 剪枝率
- `enable_mm_embeds`：config，允许用户跳过 ViT 直接传 embeddings
- `disable_chunked_mm_input`：scheduler config，禁止 mm item 跨 step 切分

---

# 把所有「关键文件」按本流程列一份

| 阶段 | 关键文件 | 关键类 / 函数 |
|---|---|---|
| 1. registry | [vllm/multimodal/registry.py](../vllm/vllm/multimodal/registry.py) | `MultiModalRegistry.register_processor` |
| 2. 模型 MM 三件套 | [qwen3_vl.py:922](../vllm/vllm/model_executor/models/qwen3_vl.py#L922) | `Qwen3VLProcessingInfo / DummyInputsBuilder / MultiModalProcessor` |
| 3. HF processor 调用 | [qwen3_vl.py:1264](../vllm/vllm/model_executor/models/qwen3_vl.py#L1264) | `_call_hf_processor` |
| 4. fields config | [qwen2_vl.py:774](../vllm/vllm/model_executor/models/qwen2_vl.py#L774) | `_create_qwen2vl_field_factory` |
| 5. prompt updates | [qwen3_vl.py:1411](../vllm/vllm/model_executor/models/qwen3_vl.py#L1411) | `_get_prompt_updates`, `PromptReplacement` |
| 6. hash | [vllm/multimodal/hasher.py](../vllm/vllm/multimodal/hasher.py) | `MultiModalHasher.hash_kwargs` |
| 7. processor cache | [vllm/multimodal/cache.py](../vllm/vllm/multimodal/cache.py) | `BaseMultiModalProcessorCache` |
| 8. apply 主流程 | [vllm/multimodal/processing/processor.py:1663](../vllm/vllm/multimodal/processing/processor.py#L1663) | `BaseMultiModalProcessor.apply` |
| 9. 调度器编码守门 | [vllm/v1/core/sched/scheduler.py:1141](../vllm/vllm/v1/core/sched/scheduler.py) | `_try_schedule_encoder_inputs` |
| 10. encoder cache 调度器侧 | [vllm/v1/core/encoder_cache_manager.py:17](../vllm/vllm/v1/core/encoder_cache_manager.py#L17) | `EncoderCacheManager` |
| 11. encoder 执行 | [vllm/v1/worker/gpu_model_runner.py:2869](../vllm/vllm/v1/worker/gpu_model_runner.py#L2869) | `_execute_mm_encoder` |
| 12. gather 当前 chunk | [vllm/v1/worker/gpu_model_runner.py:3080](../vllm/vllm/v1/worker/gpu_model_runner.py#L3080) | `_gather_mm_embeddings` |
| 13. encoder cudagraph | [vllm/v1/worker/encoder_cudagraph.py:53](../vllm/vllm/v1/worker/encoder_cudagraph.py#L53) | `EncoderCudaGraphManager` |
| 14. embed_multimodal | [qwen3_vl.py:2734](../vllm/vllm/model_executor/models/qwen3_vl.py#L2734) | `Qwen3VLForConditionalGeneration.embed_multimodal` |
| 15. ViT | [qwen3_vl.py:521](../vllm/vllm/model_executor/models/qwen3_vl.py#L521) | `Qwen3_VisionTransformer.forward` |
| 16. MMEncoderAttention | [vllm/model_executor/layers/attention/mm_encoder_attention.py](../vllm/vllm/model_executor/layers/attention/mm_encoder_attention.py) | `MMEncoderAttention` |
| 17. merge text + mm | [vllm/model_executor/models/utils.py:456](../vllm/vllm/model_executor/models/utils.py#L456) | `_merge_multimodal_embeddings` |
| 18. deepstack 注入 | [qwen3_vl.py:1569](../vllm/vllm/model_executor/models/qwen3_vl.py#L1569) | `Qwen3LLMModel.forward` |
| 19. EVS | [vllm/multimodal/evs.py](../vllm/vllm/multimodal/evs.py) | `compute_retention_mask`, `recompute_mrope_positions` |
| 20. DP ViT | [vllm/model_executor/models/vision.py:383](../vllm/vllm/model_executor/models/vision.py#L383) | `run_dp_sharded_mrope_vision_model` |

---

# 常见踩坑 / 性能 checklist

1. **prompt 里 placeholder 数对不上 ViT 输出 token 数** → `_get_prompt_updates` 没正确 expand。
2. **`is_embed` 出现 None 但视频带文本 token** → 调度器 / gather 切片错位，视觉特征写错位置。
3. **mm_hash 重复但 cache 没命中** → mm_processor_kwargs 或 model_id 没纳入 hash；或者用户改了 `min_pixels` 但没 invalidate cache。
4. **encoder_compute_budget 太小** → 每步只能跑一张图，prefill 拖长。提高 `encoder_cache_size` 或 `max_num_batched_tokens`。
5. **EVS + encoder cudagraph** → 默认相互关闭；强行打开会 hit shape mismatch。
6. **DP ViT 卡 padding** → 单图分辨率超大且其他 rank 闲置：换回 TP 或调整 load balance。
7. **chunked prefill 切到 mm 中间** → 默认允许；设 `disable_chunked_mm_input=True` 强制不切。
8. **重新生成同 prompt 同图但慢** → ProcessorCache 满了 evict，或换了 model_id；考虑增大本地 cache 或挂远端 KV connector。
9. **多视频精度漂移** → `do_sample_frames`/`fps`/`num_frames` 之间互斥；前端不一致会产生不同 timestamps，影响 mRoPE。
10. **FP8 ViT 与动态 scale 上 CUDA Graph** → 必报错；用静态 scale。

---

# 一句话总结

vLLM 的多模态在 Qwen3-VL 上分三个边界：**`MultiModalProcessor`** 把用户媒体转为带 hash 的张量 item；
**`Scheduler + EncoderCacheManager`** 用 compute budget 和 cache slot 守门，决定本步算谁、释放谁；
**`GPUModelRunner._execute_mm_encoder` → `model.embed_multimodal` → `_merge_multimodal_embeddings`** 把 ViT 输出按 placeholder mask 写入 LLM inputs_embeds，
中间额外有 **EncoderCudaGraph / EVS / Deepstack / DP ViT** 这几条优化分支。

整个体系最关键的设计是「**以 mm_hash 为 cache 主键**」：让相同的图在多请求、多步调度、跨节点之间复用同一份 ViT 输出，而上层只用引用计数和 placeholder mask 协调谁在哪一步看到它。
