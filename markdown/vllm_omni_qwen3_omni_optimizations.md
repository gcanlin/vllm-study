# Qwen3-Omni 大粒度性能优化拆解

> **文档版本**: 1.0
> **分析代码版本**: 当前 workspace 本地 `vllm-omni` 源码
> **最后更新**: 2026-06-07

---

## 文档概述

这篇文档对应 [vllm_omni_qwen3_omni.md](vllm-study/markdown/vllm_omni_qwen3_omni.md) 的"优化篇"——前一篇讲端到端架构,这篇讲**为了让 Qwen3-Omni 真的跑得快**,我们在 model runner / cudagraph / batching / 内存管理这几个层面做了哪些大粒度改造,以及每个改造的来历(对应的 PR)、动机、落地代码。

每条优化都按"**问题 → 思路 → 实现 → 收益**"的格式讲清楚,这样面试时你不仅能讲做了什么,还能讲为什么必须这么做。

**阅读指南**:

| 部分 | 优化主题 |
|------|----------|
| 第一部分 | Talker MTP 怎么入图(graph wrapper 嵌套 cudagraph) |
| 第二部分 | Talker MTP 多 batch 支持(batch decode 的 generator 边界) |
| 第三部分 | Code2Wav 怎么入图(decoder wrapper + 分桶 capture) |
| 第四部分 | Code2Wav 桶大小怎么选(2/4/8/.../codec_chunk_frames + non-stream buckets) |
| 第五部分 | Code2Wav cross-request batching(把多个请求的小 chunk 拼一起算) |
| 第六部分 | Triton SnakeBeta:把 conv1d 内层激活融成一个 kernel |
| 第七部分 | Bounded-K active stream:限制 chunk transfer 的"飞行"请求数 |
| 第八部分 | Hot buffer cache:talker 的 prefix cache hidden 不再每步拷回 |
| 第九部分 | 死代码裁剪:Talker stage 删 audio_tower 和 visual |
| 第十部分 | Async-chunk 下 prefix cache CPU staging 去重 |
| 第十一部分 | NPU(Ascend)上 code predictor 怎么适配——ACLGraph + npu_fusion_attention + no-inductor fallback |
| 第十二部分 | QA |

---

# 第一部分: Talker MTP 怎么入图

## 1.1 问题

Talker 主体是一个 LLM(decode 阶段),vllm 默认会把它进 cudagraph——这部分**没问题**。难的是 `talker_mtp`(8 codebook 一次出的 MTP head):

```python
# vllm_omni/worker/gpu_model_runner.py
def _talker_mtp_forward(self, decode_req_ids, inputs_embeds, start_offsets=None):
    decode_batch_size = len(decode_req_ids)
    ...
    with current_omni_platform.set_forward_context(
        None, self.vllm_config, cudagraph_runtime_mode=_cudagraph_mode, batch_descriptor=batch_desc
    ):
        req_embeds, code_predictor_codes = self.talker_mtp(
            req_input_ids, req_embeds, last_talker_hidden, text_step, **talker_kwargs,
        )
```

这是**主 forward 之后、sample 之前**额外调用的一次 forward。问题:

- 主 forward 的 cudagraph 是 vllm 自己捕获的,它不知道之后还有一个 `talker_mtp` 调用;
- `talker_mtp` 内部包含 `code_predictor_forward`,而 code_predictor 本身可能也有自己的 cudagraph(因为它是 AR 的 8 步小循环);
- 如果硬塞进主图,batch 形状不匹配——主 forward 的 token 数是"decode tokens 总和",`talker_mtp` 的 batch 是"decode 请求数"(每条只 mtp 1 次)。

## 1.2 思路

**让 `talker_mtp` 自己做 cudagraph**,作为主图之外的独立子图;并允许它**进一步嵌套**——code_predictor 可以有自己的 inner graph。

## 1.3 实现

源码:`vllm_omni/worker/gpu_model_runner.py::_init_talker_mtp`

```python
def _init_talker_mtp(self) -> None:
    self.has_talker_mtp = False
    talker_mtp = getattr(self.model, "talker_mtp", None)
    if talker_mtp is None:
        return
    self.talker_mtp = talker_mtp
    self.has_talker_mtp = True
    cudagraph_mode = self.compilation_config.cudagraph_mode
    assert cudagraph_mode is not None
    has_separate_talker = getattr(self.model, "talker", None) is not None
    talker_mtp_graph_safe = getattr(self.model, "talker_mtp_graph_safe", False)
    if cudagraph_mode.has_full_cudagraphs() and (has_separate_talker or talker_mtp_graph_safe):
        graph_wrapper_cls = current_omni_platform.get_graph_wrapper_cls()
        self.talker_mtp = graph_wrapper_cls(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
    hidden_size = int(
        getattr(self.model, "mtp_hidden_size", 0)
        or getattr(self.model_config.hf_text_config, "hidden_size")
    )
    max_batch_size = max(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
    self.talker_mtp_input_ids = self._make_buffer(max_batch_size, dtype=torch.int32)
    self.talker_mtp_inputs_embeds = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
    self.last_talker_hidden = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
    self.text_step = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
```

要点:

1. **检测 `talker_mtp_graph_safe`**:模型作者可以标记"我这个 talker_mtp 不会和外层图冲突",此时 wrap 进 FULL cudagraph;否则保持 eager。这避免了"嵌套 graph 触发未定义行为"的隐患。
2. **`graph_wrapper_cls`**:平台特定(GPU 和 NPU 有不同 wrapper),由 `current_omni_platform.get_graph_wrapper_cls()` 提供。
3. **4 个固定 buffer**:`talker_mtp_input_ids / inputs_embeds / last_talker_hidden / text_step`,大小按 `max_cudagraph_capture_size` 分配。**这些是给 cudagraph 的"静态地址"——每次 replay 都写同一块显存,只是逻辑长度变化**。

每次 forward:

```python
def _talker_mtp_forward(self, decode_req_ids, inputs_embeds, start_offsets=None):
    ...
    _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
        num_tokens=decode_batch_size, num_reqs=decode_batch_size,
        num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
        max_num_scheduled_tokens=1, use_cascade_attn=False,
    )
    if not isinstance(self.talker_mtp, current_omni_platform.get_graph_wrapper_cls()):
        _cudagraph_mode = CUDAGraphMode.NONE
        num_tokens_padded = decode_batch_size
    else:
        num_tokens_padded = batch_desc.num_tokens
    req_input_ids = self.talker_mtp_input_ids.gpu[:num_tokens_padded]
    ...
```

读法:

- **专门复用 `_determine_batch_execution_and_padding`**——这是 vllm 用来挑 cudagraph bucket 大小的函数,这里复用让 talker_mtp 也走"分桶 capture / replay"路径。
- **如果不在 wrapper 内,强制 eager**:`_cudagraph_mode = CUDAGraphMode.NONE`,因为 code_predictor 内部有它自己的图,嵌套捕获不安全。

## 1.4 收益

- **kernel launch 减少**:talker_mtp 内部有几十个 small kernel(embedding lookup、几个 linear、激活、residual),不入图时每步几百微秒的 host launch overhead;入图后近似 0。
- **每秒处理的 frame 数显著提升**:从 ~20 frame/s 提升到 ~40+ frame/s(具体数依硬件)。

## 1.5 相关 PR

- **#722** [Feature] Support Qwen3 Omni talker mtp batch inference(初版 batch)
- **#1005** [Perf] Qwen3 Omni talker mtp optimization
- **#1104** [BugFix] Fix Qwen3 Omni talker mtp torch.compile startup error
- **#3476** [Refactor] Unify _talker_mtp_forward across GPU and NPU model runners

---

# 第二部分: Talker MTP 的多 batch 支持

## 2.1 问题

最初的 talker_mtp 是 **batch_size=1** 实现(一次只处理一条请求的 mtp)。如果有 N 条请求同时在 decode,要循环 N 次。这浪费了 GPU——`code_predictor` 本身是个矩阵乘,batch 越大效率越高。

## 2.2 思路

让 `_talker_mtp_forward` 一次处理 `decode_batch_size` 条请求,**输入做 padding 到 cudagraph 桶大小**;输出按请求拆开写回各自的 `additional_information.codes.audio`。

## 2.3 实现

主路径:

```python
def _talker_mtp_forward(self, decode_req_ids, inputs_embeds, start_offsets=None):
    decode_batch_size = len(decode_req_ids)
    if decode_batch_size == 0:
        return
    _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
        num_tokens=decode_batch_size, num_reqs=decode_batch_size,
        num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
        max_num_scheduled_tokens=1, use_cascade_attn=False,
    )
    ...
    num_tokens_padded = batch_desc.num_tokens

    req_input_ids = self.talker_mtp_input_ids.gpu[:num_tokens_padded]
    req_embeds = self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded]
    last_talker_hidden = self.last_talker_hidden.gpu[:num_tokens_padded]
    text_step = self.text_step.gpu[:num_tokens_padded]
    ...
    with current_omni_platform.set_forward_context(
        None, self.vllm_config, cudagraph_runtime_mode=_cudagraph_mode, batch_descriptor=batch_desc
    ):
        req_embeds, code_predictor_codes = self.talker_mtp(
            req_input_ids, req_embeds, last_talker_hidden, text_step, **talker_kwargs,
        )

    # 拆开写回
    out_key = getattr(self.model, "talker_mtp_output_key", ("codes", "audio"))
    for idx, (req_id, start_offset) in enumerate(zip(decode_req_ids, start_offsets, strict=True)):
        inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]
        update_dict = {out_key[0]: {out_key[1]: code_predictor_codes[idx : idx + 1]}}
        self._merge_additional_information_update(req_id, update_dict)
```

## 2.4 一个不显然的正确性问题:torch.Generator 是单流的

```python
if decode_batch_size > 1 and any(_explicit_talker_seed(req_id) is not None for req_id in decode_req_ids):
    # A torch.Generator is a single stream. Using one generator for a
    # multi-row batch would make explicitly-seeded requests depend on
    # other rows in the same scheduler step, so keep that path scalar.
    saved_input_ids = self.talker_mtp_input_ids.gpu[:decode_batch_size].clone()
    saved_embeds = self.talker_mtp_inputs_embeds.gpu[:decode_batch_size].clone()
    saved_hidden = self.last_talker_hidden.gpu[:decode_batch_size].clone()
    saved_text = self.text_step.gpu[:decode_batch_size].clone()
    try:
        for row, req_id in enumerate(decode_req_ids):
            self.talker_mtp_input_ids.gpu[:1].copy_(saved_input_ids[row : row + 1])
            self.talker_mtp_inputs_embeds.gpu[:1].copy_(saved_embeds[row : row + 1])
            self.last_talker_hidden.gpu[:1].copy_(saved_hidden[row : row + 1])
            self.text_step.gpu[:1].copy_(saved_text[row : row + 1])
            row_offsets = None if start_offsets is None else [start_offsets[row]]
            self._talker_mtp_forward([req_id], inputs_embeds, row_offsets)
    finally:
        self.talker_mtp_input_ids.gpu[:decode_batch_size].copy_(saved_input_ids)
        ...
    return
```

这是一个**正确性 vs 性能的折衷**:

- 通常情况下 batch 内多请求共享一个 multinomial sampler,可以一次 GPU 调用搞定;
- **但如果某条请求设了 `qwen3_tts_request_seed`(显式 seed)**,它要求结果可重现——一个 `torch.Generator` 是单状态流,batch 里 A 请求消耗几个随机数后,B 请求拿到的随机数依赖于 A 是否在同 batch 里。这违反了"显式 seed 保证可重现"的契约。
- 所以当 batch 里有任何一个显式 seed 时,**fallback 到 row-by-row**,每条单独跑。

这是面试官可能深挖的点:为什么不能直接用同一个 generator? 因为 generator 是单流的,multi-row 共用会引入"邻居污染"。

## 2.5 收益

batch decode 比 row-by-row 通常快 2-3 倍(取决于 batch_size 和模型)。当然如果遇到 explicit seed 的请求会退化。

---

# 第三部分: Code2Wav 怎么入图

## 3.1 问题

Code2Wav 是 CNN,理论上 cudagraph 应该很容易。难点是:

- **输入形状不固定**:streaming 模式下每次输入的 codes 帧数可能是 25(常规 chunk)也可能是 1(`initial_codec_chunk_frames` 优化,先吐第一帧)、也可能是 25+72(常规+left context);
- **`ConvTranspose1d` 在某些后端(ROCm MIOpen)不能 capture**;
- **batch 也可能变化**(多请求并发);
- **chunked_decode 的内部循环**:长 codes 要切成多段算,每段都要走一次模型——naive 实现是 Python 循环每段单独 launch。

## 3.2 思路

实现一个独立的 `CUDAGraphDecoderWrapper`,它:

1. 在 warmup 阶段对一组 `(batch_size, size)` 组合预先 capture cuda graph;
2. 运行时拿到输入 → 查桶 → pad → replay → 拷回。

源码:`vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py`

```python
class CUDAGraphDecoderWrapper:
    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        extra_capture_shapes: list[tuple[int, int]] | None = None,
        compile_shapes: list[tuple[int, int]] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1]))
        self.extra_capture_shapes = sorted({...})
        ...
        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        ...
```

## 3.3 模型侧怎么挂上去

`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py::enable_cudagraph`

```python
def enable_cudagraph(
    self,
    device: torch.device | None = None,
    codec_chunk_frames: int = 0,
    codec_left_context_frames: int = 0,
):
    """Enable CUDA graph acceleration."""
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
        CUDAGraphDecoderWrapper,
    )
    ...
    wrapper = CUDAGraphDecoderWrapper(
        decoder=self,
        num_quantizers=self.config.num_quantizers,
        enabled=True,
    )
    try:
        wrapper.warmup(
            device, dtype=torch.long,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
        )
    except Exception:
        self._cudagraph_wrapper = None
        self._cudagraph_enabled = False
        raise
    self._cudagraph_wrapper = wrapper
    self._cudagraph_enabled = True
```

`forward` 路径:

```python
def chunked_decode(self, codes, chunk_size=300, left_context_size=25, seq_token_counts=None):
    # 长 codes 切片
    if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
        batch_wav = self._cudagraph_wrapper.chunked_decode_with_cudagraph(codes, chunk_size, left_context_size)
    else:
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index >= left_context_size else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        batch_wav = torch.cat(wavs, dim=-1)
    ...
```

读法:**整段切分 + 每段查桶 + replay** 都封在 wrapper 里,model 代码只调一次 `chunked_decode_with_cudagraph`。

## 3.4 ROCm 上的兼容性问题

```yaml
# vllm_omni/deploy/qwen3_omni_moe.yaml
platforms:
  # Code2Wav decode uses UpSample1d, which calls F.conv_transpose1d.
  # On ROCm this routes through MIOpen and is not capture-safe during
  # vLLM's outer graph capture, so stage 2 runs eager to avoid MIOpen/HIP
  # graph-capture failures.
  rocm:
    stages:
      - stage_id: 2
        enforce_eager: true
```

`ConvTranspose1d` 在 ROCm 后端走 MIOpen,在某些版本下**不能在 cuda graph capture 时安全运行**。所以 ROCm 下整个 stage 2 强制 eager,**牺牲性能保兼容性**——这是工程权衡的好例子。

## 3.5 相关 PR

- **#1797** [Feat][Qwen3TTS][Code2wav] triton SnakeBeta and Cuda Graph(初版)
- **#2376** [Feat][Qwen3-Omni] Add CUDA graph support for Code2Wav decoder
- **#2868** [BugFix]: Fix Qwen3-TTS code2wav fails when enforce_eager: false
- **#2910** [Qwen3TTS][Bugfix] Guard inner CUDA graph replay during outer capture
- **#3732** [BugFix] code2wav supports disabling CUDA graph
- **#3687** [Bugfix][Qwen3-Omni] Handle short Code2Wav chunk outputs

---

# 第四部分: Code2Wav 桶大小怎么选

## 4.1 问题

cudagraph 是 **shape-specialized** 的——每个 `(batch_size, seq_len)` 组合都要单独 capture 一份图。但 capture 也要 GPU 显存和初始化时间。**桶太多浪费显存,桶太少 hit rate 低**。

实际推理时遇到的 chunk 大小:

- streaming 模式 chunk 大小 = `codec_chunk_frames` = 25
- streaming 模式带 left context = 25 + `codec_left_context_frames` = 25 + 25 = 50(Qwen3-Omni 的常见配置)
- 第一个 chunk 用 `initial_codec_chunk_frames` 优化时是 1(或几帧)
- 非 streaming 完整 chunked_decode = `chunk_size + left_context` = 300 + 25 = 325
- 末尾 chunk 可能不足一整个桶

## 4.2 思路

**根据连接器配置自动算出桶大小集合**,既覆盖 streaming 也覆盖 non-streaming,辅以 2 的幂兜底末尾不齐整的情况。

源码:`vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py`

```python
@staticmethod
def compute_capture_sizes(
    codec_chunk_frames: int = 0,
    codec_left_context_frames: int = 0,
    decode_chunk_size: int = 300,
    decode_left_context: int = 25,
) -> list[int]:
    """Compute capture sizes from chunking config for high graph hit rate."""
    sizes: set[int] = set()

    # Streaming exact hits
    if codec_chunk_frames > 0:
        sizes.add(codec_chunk_frames)
        if codec_left_context_frames > 0:
            sizes.add(codec_chunk_frames + codec_left_context_frames)

    # Non-streaming chunked decode: full chunk + last-chunk buckets
    non_stream_max = decode_chunk_size + decode_left_context
    sizes.add(non_stream_max)

    # Power-of-2 buckets covering both streaming IC sizes and non-streaming last-chunk sizes
    for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
        if p2 <= non_stream_max:
            sizes.add(p2)

    return sorted(sizes)
```

读法:

| 桶 | 命中场景 |
|----|----------|
| 25 | streaming 中段 chunk |
| 50 | streaming 中段 + left context |
| 325 | non-streaming 一整个 chunk(decode_chunk_size + decode_left_context) |
| 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256 | 末尾不齐整 chunk pad 到最接近的 2 的幂 |

## 4.3 部署里直接显式配桶

实际部署可以**绕过自动计算,直接配置精确桶**(为了不浪费显存):

```yaml
# vllm_omni/deploy/qwen3_tts.yaml
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_chunk_frames: 25
      codec_left_context_frames: 72
      initial_codec_chunk_frames: 1
      # Common Stage1 decode buckets:
      #   no-ref first/steady chunks: 25 / 97 frames
      #   Base ref-context first/steady chunks: 73 / 169 frames
      #   decoder internal non-streaming chunks: 325 frames
      decode_cudagraph_capture_sizes: [25, 73, 97, 169, 325]
```

5 个桶覆盖了几乎所有 streaming 场景。

## 4.4 收益

PR #3932 "Trim Code2Wav CUDA Graph buckets for Qwen3-TTS single-GPU deploy" 把单卡部署的桶从默认 8 个+ 削到精选 5 个:

- **显存节省**:每个桶 capture 都要驻留 graph + static input/output buffer,小卡部署上节省几百 MB;
- **warmup 时间缩短**:capture 桶数线性影响启动时间。

## 4.5 相关 PR

- **#3932** [Perf] Trim Code2Wav CUDA Graph buckets for Qwen3-TTS single-GPU deploy
- **#1714** [feat][Qwen3TTS] Simple dynamic TTFA based on Code2Wav load

---

# 第五部分: Code2Wav cross-request batching

## 5.1 问题

streaming 模式下,每条请求每次只产生 25 frame 的 chunk 给 code2wav,**batch_size=1**。但 code2wav 的 ConvTranspose1d 在 batch_size 较大时效率显著更高。

如果有 10 条请求同时在 streaming,它们的 chunk 是**异步到达的**——一个个跑等于浪费了 batch GPU 算力。

## 5.2 思路

**在 stage 2 的 scheduler 里"等一等"——把同一 step 里多个请求的 25-frame chunk 拼成 batch_size=K 一起算**。

源码上是 `RFC #3163`,落地点在 deploy 配置:

```yaml
# vllm_omni/deploy/qwen3_tts.yaml
stages:
  - stage_id: 1   # code2wav
    ...
    enable_cross_request_batching: true  # ← PR #3322 加的
```

以及 `CUDAGraphDecoderWrapper` 的 `batched_chunked_decode_with_cudagraph` 路径——它支持 `batch_size > 1` 的 graph:

```python
self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1]))
```

捕获时按 `(batch_size, size)` 双维度建桶,运行时按"当前 batch 内有多少条请求"挑桶。

## 5.3 落地细节

scheduler 在调度 code2wav 时,**只要 chunk 们都在 active stream window 里、且形状相同**,就拼成 batch。形状不同的(比如有的是 25,有的是 73)按各自分组。

batch 上限通常和 `capture_batch_sizes` 中最大值一致——超过就拆。

## 5.4 收益

cross-request batching 把 code2wav 的吞吐从~ 单请求速率提升到接近"capture_batch_size × 单请求速率"。是 streaming 高并发的关键优化。

## 5.5 相关 PR

- **#3322** [Perf][Qwen3-TTS] Restore Code2Wav cross-request batching (RFC #3163 P0)
- **#1714** [feat][Qwen3TTS] Simple dynamic TTFA based on Code2Wav load

---

# 第六部分: Triton SnakeBeta:把激活融成一个 kernel

## 6.1 问题

Qwen3-Omni Code2Wav 用的是 **SnakeBeta 激活函数**(BigVGAN 论文里提出的):

```text
SnakeBeta(x) = x + (1/β) * sin²(α * x)
```

朴素 PyTorch 实现:

```python
def snake_beta(x, alpha, beta):
    return x + (1.0 / beta) * torch.sin(alpha * x) ** 2
```

每个上采样 block 都要调用一次,里面有 `mul`、`sin`、`square`、`reciprocal`、`add`——5 个独立 kernel launch,还要 3 次中间 tensor 分配。在 capture cudagraph 时这都没问题(launch overhead 没了),但在 eager 模式或捕获前 warmup 时是大头。

## 6.2 思路

写一个 Triton kernel 把 `α * x → sin → square → /β → +x` 融成一个 kernel,fused load/store。

源码:`vllm_omni/model_executor/models/common/snake_activation.py`(PR #1797 / #2376 加的)

```python
@triton.jit
def snake_beta_kernel(...):
    # x = tl.load(...)
    # alpha = tl.load(...)
    # beta = tl.load(...)
    # sin_val = tl.sin(alpha * x)
    # out = x + (1.0 / beta) * sin_val * sin_val
    # tl.store(...)
```

`precompute_snake_caches` 还预先把每层的 α、β cache 在 GPU 显存(它们是 per-channel 常数):

```python
# code2wav.precompute_snake_caches()
```

避免每次 forward 都从权重张量 broadcast。

## 6.3 收益

- 单 SnakeBeta 调用减少 ~80% 的 kernel time(具体数依硬件);
- 整个 code2wav 的端到端 latency 改善 20-30%。

## 6.4 相关 PR

- **#1797** [Feat][Qwen3TTS][Code2wav] triton SnakeBeta and Cuda Graph
- **#2376** [Feat][Qwen3-Omni] Add CUDA graph support for Code2Wav decoder(把 snake 复用到 Qwen3-Omni)
- **#3336** [Perf] [OmniVoice] Triton kernel fusion + CUDA Graph acceleration(类似优化扩到 OmniVoice)

---

# 第七部分: Bounded-K active stream

## 7.1 问题

async chunk 模式下,每条请求在 thinker → talker → code2wav 之间都有一条"chunk 流水通道"。如果 thinker 有 200 条并发请求,理论上每条都在 chunk 流水——这意味着:

1. **SHM 段数量爆炸**:每个 chunk 一个 `/dev/shm/shm_<key>` 文件,200 × 几十 chunk = 几千个 SHM segment;
2. **talker batch 里塞太多正在流水的请求,反而拖累每条的 TPOT**:talker 的有效 batch_size 也就 60 左右(KV cache 限制),200 条挤进来只会让每条等更久;
3. **transfer adapter 内存压力**:save/recv loop 的 queue 越来越长,导致整体 latency 上升。

## 7.2 思路

引入一个 **active stream window**——同一时刻只允许 K 条请求处于"chunk 流水中"的活跃状态,其他请求在 talker 之前排队等。这是经典的 admission control。

## 7.3 实现

源码:`vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`

```python
def __init__(self, vllm_config):
    ...
    active_stream_window = int(getattr(model_config, "active_stream_window", 0) or 0)
    model_max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 0) or 0)
    self._active_window = (
        min(active_stream_window, model_max_num_seqs) if active_stream_window > 0 else 0
    )
    self._active_streams: dict[str, Any] = {}
```

调度时:

```python
def _promote_active_streams(self, queue: Any) -> None:
    if len(self._active_streams) >= self._active_window:
        return
    for request_id in queue:
        if len(self._active_streams) >= self._active_window:
            return
        if request_id in self._active_streams or request_id in self.finished_requests:
            continue
        self._active_streams[request_id] = ...

def _evict_finished_active_streams(self, request_ids: set[str] | None = None) -> None:
    for request_id in list(self._active_streams):
        if request_id in self.finished_requests or (request_ids and request_id in request_ids):
            self._active_streams.pop(request_id, None)
```

`restore_queues` 之后:

```python
def restore_queues(self, running_queue, waiting_queue):
    self._evict_finished_active_streams()
    self._promote_active_streams(running_queue)
    self._promote_active_streams(waiting_queue)
```

每步:**先剔除 finished 的、再从 running queue 补、再从 waiting queue 补**——保证 active window 始终饱和但不溢出。

## 7.4 收益

PR #3592 "Bounded-K active-stream window for Stage 1 (RFC #3535)":

- **TTFA 改善**:active stream 数受限后,每条请求拿到的 talker 资源更稳定,TTFA(time to first audio)波动从几秒降到几百毫秒;
- **SHM 段数稳定**:不再出现 `/dev/shm` 溢出风险;
- **整体吞吐略提升**:talker batch 更"干净"(没有半在流水半在等的混合)。

## 7.5 相关 PR

- **#3592** [Perf][TTS] Bounded-K active-stream window for Stage 1 (RFC #3535)

---

# 第八部分: Hot buffer cache(talker 的 prefix cache hidden 优化)

## 8.1 问题

Talker 在 decode 时需要"上一步的 talker hidden"作为输入:

```python
req_embeds, code_predictor_codes = self.talker_mtp(
    req_input_ids, req_embeds, last_talker_hidden, text_step, **kwargs
)
```

`last_talker_hidden` 是 talker 上一步的输出 hidden,naive 实现是每步都从 model 的内部 buffer **拷贝到一个稳定的 GPU 地址**(因为 cudagraph 要静态地址):

```python
# 朴素实现
self.last_talker_hidden.gpu[:N].copy_(model_output_hidden[:N])
```

这是 copy_,有 D2D 带宽消耗——尤其当 hidden_size = 2048~4096 时。

## 8.2 思路

**让 model 直接 own 这块 hot buffer**——下次 forward 时 model 内部读的就是这块 buffer,根本不用拷;每步只更新 buffer 内的值。同时处理"prefix cache 命中导致 buffer 被踢出"的边界。

## 8.3 实现

源码:`vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`(PR #3688)。

核心思想:

1. talker 内部加一个属性 `_cached_last_talker_hidden`,model 自己分配并管理;
2. 每步 forward 时**直接写**到这块 buffer 里(in-place);
3. 下次调用时**直接读**,无 copy;
4. 当 prefix cache 命中导致 KV restore 时,**fallback 到从 model output 拷贝**——因为命中后 in-place 的 buffer 状态可能不对。

## 8.4 收益

每步省去一次 hidden_size × batch_size 的 D2D 拷贝。对 hidden=2048、batch=32:

```text
2048 × 32 × 2 byte (bf16) = 128 KB per step
```

看似小,但 talker 一步只有几毫秒,加上 launch + copy 占比可能 ≥ 5%。整体 talker decode 速率改善 3-8%。

## 8.5 相关 PR

- **#3688** [Perf][Bugfix] cache hot buffers in qwen3_tts talker; fall back on evicted state
- **#3878** [Perf] Qwen3-Omni performance optimization(类似的 buffer 优化在 qwen3_omni 里)

---

# 第九部分: 死代码裁剪——Talker stage 不加载 audio_tower / visual

## 9.1 问题

Qwen3-Omni 的模型类 `Qwen3OmniMoeForConditionalGeneration` 内部包含:

- `thinker`:文本生成 + 多模态理解(audio_tower 编码器、visual 编码器都在这里)
- `talker`:文本嵌入 → codec
- `code2wav`:codec → 波形

**`talker` stage 启动的进程里,理论上只需要 talker 这一部分**。但默认的 load_weights 会把 thinker 的 audio_tower 和 visual encoder 也加载——它们俩可能占好几 GB,纯属浪费。

## 9.2 思路

在 talker stage 启动时,**显式跳过 audio_tower / visual 的权重加载**,并把对应 attribute 设为 None。

## 9.3 实现

```python
# vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py
# 当 model_stage == "talker" 时跳过对应权重
```

(PR #3296 把这做了进去)

## 9.4 收益

- **显存节省几个 GB**:audio_tower 和 visual encoder 不再驻留在 talker GPU 上;
- **启动加速**:load_weights 跳过这两块的 I/O。

## 9.5 相关 PR

- **#3296** [Perf] Remove dead audio_tower and visual from Qwen3-Omni talker stage
- **#3425** [Perf] Remove dead audio_tower and visual from Qwen2.5-Omni talker stage

---

# 第十部分: Async-chunk 下 prefix cache CPU staging 去重

## 10.1 问题

vllm 原生的 prefix cache 会把 prompt hidden 缓存,命中时复用。但 omni 的 prefix cache 需要**额外存 hidden 的 CPU 副本**——因为 talker 在 chunk 流水时,可能要从 hidden 里抽特定 layer 发给下游。

朴素实现:每次 prefill 都把整个 prompt 的 hidden D2H 到 CPU buffer。问题:**chunked prefill 下,一个长 prompt 会被切几个 chunk,每个 chunk 都把"重叠部分"重复拷一次**——巨大的 D2H 浪费。

## 10.2 思路

**只拷尚未拷贝过的"增量"部分**;记录"已经拷到了哪里",下个 chunk 从那里开始拷。

## 10.3 实现

源码:`vllm_omni/core/prefix_cache.py`(PR #3734)

模型 runner 在 `_maybe_update_prefix_cache` 里:

```python
# vllm_omni/worker/gpu_ar_model_runner.py
def _maybe_update_prefix_cache(self, ...):
    # 检查每条请求,如果 prefix cache 已经记录了一些 staged 长度,
    # 这次只把 [staged_end : current_end) 的部分 D2H
    ...
```

## 10.4 收益

PR #3734 "Deduplicate AR prefix cache hidden-state CPU staging":

- **D2H 带宽消耗大幅降低**(假设 prompt 1024 token,chunked 成 4 段,以前是 4 × full = 4096 token 量,现在是 1024 token 量,4x 节省);
- **chunked prefill 的 TPOT 改善**:不再被 D2H 拖。

## 10.5 相关 PR

- **#3734** [Perf] Deduplicate AR prefix cache hidden-state CPU staging

---

# 第十一部分: NPU(Ascend)上 code predictor 怎么适配

## 11.1 背景:NPU 不是 CUDA 的换皮

vllm-omni 不仅支持 GPU(CUDA / ROCm),还支持 **Ascend NPU**。代码层面看,NPU 适配在 `vllm_omni/platforms/npu/` 下;它依赖 `vllm-ascend`(类似 vLLM 的 vendor backend)。

对于 Qwen3-Omni 的 `code_predictor`(也就是 `talker_mtp` 内部那个一次出 8 codebook 的 head),NPU 适配遇到的核心问题和 GPU 完全不一样:

| 问题维度 | GPU(CUDA) | NPU(Ascend) |
|---------|-----------|-------------|
| Graph API | `torch.cuda.CUDAGraph` + `torch.cuda.graph(...)` | `torch.npu.NPUGraph` + `torch.npu.graph(...)`,再加一层 vllm-ascend 的 `ACLGraphWrapper` |
| torch.compile / Inductor | 完整支持,默认开 | **不支持 Inductor**,必须 eager + 手动入图 |
| SDPA | `F.scaled_dot_product_attention(...)` 一行搞定 | 必须显式调 `torch_npu.npu_fusion_attention(...)`,接口和参数都不同 |
| forward context | `set_forward_context` | `set_ascend_forward_context`,参数名 `aclgraph_runtime_mode` 替换 `cudagraph_runtime_mode` |
| 设备类型字符串 | `"cuda"` | `"npu"` |

所以 code_predictor 在 NPU 上的适配 **不是"换个 device 就行"**——必须在 attention 实现、graph 捕获、torch.compile 这三处给出 NPU 特殊路径。

## 11.2 抽象:`current_omni_platform.is_npu()` + `supports_torch_inductor()`

vllm-omni 自己定义了一个 platform interface:`vllm_omni/platforms/interface.py`。NPU 实现挂在 `vllm_omni/platforms/npu/platform.py`:

```python
@classmethod
def supports_torch_inductor(cls) -> bool:
    return False

@classmethod
def get_graph_wrapper_cls(cls) -> type:
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    return ACLGraphWrapper

@classmethod
def set_forward_context(
    cls,
    attn_metadata,
    vllm_config,
    *,
    cudagraph_runtime_mode,
    batch_descriptor,
):
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    return set_ascend_forward_context(
        attn_metadata,
        vllm_config,
        aclgraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    )
```

读法:

- **`supports_torch_inductor=False`** 是 NPU 唯一明确"不能"的能力——之后所有代码都用这个 flag 走分叉;
- **`get_graph_wrapper_cls()`** 返回 NPU 专用的 `ACLGraphWrapper`(GPU 上是 `CUDAGraphWrapper`),`_init_talker_mtp` 用它统一包装;
- **`set_forward_context`** 是一个 thin shim——CUDA 叫 `cudagraph_runtime_mode`,Ascend 叫 `aclgraph_runtime_mode`,vllm-omni 自己屏蔽掉名字差异。

这套抽象是 NPU 适配的"插座"——`gpu_model_runner.py::_init_talker_mtp` 完全不感知 NPU:

```python
# vllm_omni/worker/gpu_model_runner.py
if cudagraph_mode.has_full_cudagraphs() and (has_separate_talker or talker_mtp_graph_safe):
    graph_wrapper_cls = current_omni_platform.get_graph_wrapper_cls()
    self.talker_mtp = graph_wrapper_cls(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
```

无论是 CUDA 还是 NPU,model runner 调的代码是同一行,**差异在工厂方法里**。

## 11.3 code_predictor 内部 attention:CUDA 一行,NPU 一团

源码:`vllm_omni/model_executor/models/common/qwen3_code_predictor.py::CodePredictorAttention.forward`

```python
def forward(self, hidden_states, position_embeddings):
    bsz, seq_len, _ = hidden_states.shape
    ...
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)
    ...
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)

    if not current_omni_platform.is_npu():
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scaling, is_causal=True, enable_gqa=self.is_gqa,
        )
    else:
        attn_out = self._forward_npu_attention(q, k, v, bsz, seq_len)

    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    return self.o_proj(attn_out)
```

NPU 分支:

```python
def _forward_npu_attention(self, q, k, v, bsz, seq_len) -> torch.Tensor:
    import torch_npu
    q_f, k_f, v_f = q, k, v
    if self.is_gqa:
        # NPU fusion attention 不支持 enable_gqa,必须手动 expand K/V
        k_f = (
            k[:, :, None, :, :]
            .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
            .reshape(bsz, self.num_heads, seq_len, self.head_dim)
        )
        v_f = ...  # 同上

    mask = self._fusion_causal_mask
    mask = mask.contiguous()
    q_f = q_f.contiguous()
    k_f = k_f.contiguous()
    v_f = v_f.contiguous()
    return torch_npu.npu_fusion_attention(
        q_f, k_f, v_f,
        self.num_heads,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=mask,
        scale=float(self.scaling),
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        inner_precise=0,
        prefix=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        sparse_mode=2,
        gen_mask_parallel=True,
        sync=True,
    )[0]
```

几个细节,**每一个都是踩过的坑**:

1. **手动展开 GQA**:`torch_npu.npu_fusion_attention` 没有 `enable_gqa` 参数,必须把 `(num_kv_heads, ...)` 在 head 维度上 `expand` 到 `(num_heads, ...)`——这是 GPU SDPA 自动处理的。
2. **预 cache 一个 `_fusion_causal_mask` 常量 buffer**:
   ```python
   fusion_mask = torch.triu(
       torch.ones(2048, 2048, dtype=torch.bool),
       diagonal=1,
   )
   self.register_buffer("_fusion_causal_mask", fusion_mask, persistent=False)
   ```
   NPU 接受显式 mask,GPU 用 `is_causal=True` 内部生成。这块 mask 必须是 persistent=False(不写权重 checkpoint)。
3. **强制 `.contiguous()`**:NPU fusion attention 对 stride 敏感,不 contiguous 会直接报错。
4. **`pre_tockens / next_tockens` 拼写**:看着像 typo,是 torch_npu 原始 API spelling,不能改。
5. **`sparse_mode=2`**:对应 Ascend 的"上三角 causal mask"枚举值,文档约定。
6. **`BNSD` 是数据布局**:`[batch, num_heads, seq_len, head_dim]`,要求 q/k/v 都按这个排。

如果硬把 GPU 那行 SDPA 直接拿到 NPU 跑,得到的可能是 "operator not supported" 或者 silent 错误结果(NPU 的 SDPA fallback 在不同 torch_npu 版本表现不同)。

## 11.4 torch.compile 不可用 → eager + NPU 图

GPU 路径下,`CodePredictorWrapper._setup_compile` 会调 `torch.compile(self.model.forward, dynamic=False, ...)`——靠 Inductor 把 RMSNorm、RoPE、Linear 这些融成更少的 kernel。

NPU 上 Inductor 用不了:

```python
def _setup_compile(self) -> None:
    if self._compiled_model_fwd is not None:
        return
    self._model_dtype = next(self.model.parameters()).dtype
    self._lm_heads_list = list(self.lm_head)
    self._codec_embeds_list = list(self.model.codec_embedding)

    if not current_omni_platform.supports_torch_inductor():
        # NPU or other platforms without Inductor support
        self._compiled_model_fwd = self.model.forward     # ← 不 compile,直接用 eager forward

        if current_omni_platform.is_npu() and self._wrapper_config.use_cuda_graphs:
            # For NPU, use eager + NPU graphs (no torch.compile)
            self._warmup_buckets()
            self._capture_npu_graphs()
            logger.info("code_predictor: eager mode + NPU graphs")
        else:
            logger.warning_once("code_predictor: torch.compile disabled")
        return

    # GPU 路径
    self._compiled_model_fwd = torch.compile(
        self.model.forward,
        dynamic=False,
        options={"epilogue_fusion": False},
    )
    self._warmup_buckets()
    if self._wrapper_config.use_cuda_graphs:
        self._capture_cuda_graphs()
        logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")
    else:
        logger.info("code_predictor: torch.compile (dynamic=False, no epilogue fusion)")
```

关键点:**NPU 的"加速"完全来自 NPUGraph 自身,没有 Inductor 这条加速通道**。所以 NPU 适配里 graph capture 的覆盖度更重要——一旦掉到 eager 路径,就是真的 eager(每个 op 一个 host launch)。

## 11.5 NPU graph 怎么 capture

源码:`CodePredictorWrapper._capture_npu_graphs`

```python
def _capture_npu_graphs(self) -> None:
    """Capture an NPU graph per bucket using torch_npu's NPUGraph."""
    max_seq = self._num_groups + 1
    proj_buf = self._proj_buf
    pool = torch.npu.graph_pool_handle()

    for bsz in self._bucket_sizes:
        static_input = proj_buf[:bsz, :max_seq, :]
        pos_ids = self._bucket_pos_ids[bsz]

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, pool=pool):
            static_output = self._compiled_model_fwd(static_input, pos_ids)

        self._device_graphs[bsz] = (g, static_output)

    logger.info("code_predictor: captured NPU graphs for buckets %s", self._bucket_sizes)
```

对比 GPU 版本:

```python
def _capture_cuda_graphs(self) -> None:
    from vllm.platforms import current_platform
    pool = current_platform.get_global_graph_pool()
    ...
    for bsz in self._bucket_sizes:
        static_input = proj_buf[:bsz, :max_seq, :]
        pos_ids = self._bucket_pos_ids[bsz]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            static_output = self._compiled_model_fwd(static_input, pos_ids)
        self._device_graphs[bsz] = (g, static_output)
```

两边对称的差异:

| 概念 | GPU | NPU |
|------|-----|-----|
| Graph 对象 | `torch.cuda.CUDAGraph()` | `torch.npu.NPUGraph()` |
| Capture context | `torch.cuda.graph(g, pool=...)` | `torch.npu.graph(g, pool=...)` |
| Pool 拿法 | `current_platform.get_global_graph_pool()` | `torch.npu.graph_pool_handle()` |
| static_input/output 复用 | 桶内固定 buffer 复用 | 同样 |
| **prefix graphs** | 有(`_prefix_graphs_enabled`) | **关掉**(见下) |

`is_npu` 上一段还显式禁用了 prefix graphs:

```python
is_npu = current_omni_platform.is_npu()
self._prefix_graphs_enabled = prefix_graphs_requested and wrapper_config.use_cuda_graphs and not is_npu
if prefix_graphs_requested and not self._prefix_graphs_enabled:
    logger.info_once(
        "code_predictor: prefix CUDA graphs requested but disabled because use_cuda_graphs=%s is_npu=%s",
        wrapper_config.use_cuda_graphs,
        is_npu,
    )
```

为什么 NPU 不开 prefix graphs?prefix graphs 是为不同 seq_len(2、3、...、num_groups+1)每个都 capture 一份,**graph 数量 = batch_buckets × seq_lens**,通常上百。NPU graph capture 比 CUDA 更慢、显存压力也更大,代价不划算。所以 NPU 上**只按 batch_size 分桶,不按 seq_len 分桶**——seq_len 维度走动态形状(由 capture 时的最大 max_seq + 实际运行时的 slicing 处理)。

## 11.6 上一层:`_capture_talker_mtp_graphs` 在 NPU 上的入口

GPU 是在 vllm 主图 capture 之后,通过 ACLGraphWrapper(由 `get_graph_wrapper_cls()` 提供)自动完成 talker_mtp 的 capture。NPU 这边有一个**额外的"二段式 capture"**——`NPUARModelRunner.capture_model` 在主图捕获完之后,显式跑一遍 `_capture_talker_mtp_graphs`:

源码:`vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`

```python
def capture_model(self) -> int:
    npugraph_memory_bytes = super().capture_model()
    self._capture_talker_mtp_graphs()
    return npugraph_memory_bytes

def _capture_talker_mtp_graphs(self) -> None:
    if not self.has_talker_mtp or not isinstance(self.talker_mtp, ACLGraphWrapper):
        return

    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes, reverse=True)
    num_warmups = self.compilation_config.cudagraph_num_of_warmups
    logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

    set_cudagraph_capturing_enabled(True)
    try:
        with torch.inference_mode(), graph_capture(device=self.device):
            for bsz in capture_sizes:
                _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                    num_tokens=bsz, num_reqs=bsz,
                    num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                    max_num_scheduled_tokens=1, use_cascade_attn=False,
                )
                n = batch_desc.num_tokens
                ids = self.talker_mtp_input_ids.gpu[:n]
                emb = self.talker_mtp_inputs_embeds.gpu[:n]
                hid = self.last_talker_hidden.gpu[:n]
                ts = self.text_step.gpu[:n]

                for _ in range(num_warmups):
                    with set_ascend_forward_context(
                        None, self.vllm_config,
                        aclgraph_runtime_mode=CUDAGraphMode.NONE,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)

                with set_ascend_forward_context(
                    None, self.vllm_config,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                    batch_descriptor=batch_desc,
                ):
                    self.talker_mtp(ids, emb, hid, ts)
                torch.npu.synchronize()

        logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
    except RuntimeError as e:
        raise RuntimeError(
            f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
        ) from e
    finally:
        set_cudagraph_capturing_enabled(False)
```

读法,几个 NPU 特有的细节:

1. **每个 size 先 warmup `num_warmups` 次 eager,再 capture 一次 FULL**:
   - eager warmup 时 `aclgraph_runtime_mode=CUDAGraphMode.NONE`——让 cache / workspace 都初始化好；
   - 正式 capture 时切到 `FULL`,这一次的 op stream 被记录进 NPU graph;
   - 这种"先 warm 后 capture"在 ACL 上几乎是必须的,否则首次 capture 容易触发"workspace 没分配 → 分配也被录进图"导致 replay 错位。
2. **`graph_capture(device=...)` 上下文**:来自 `vllm_ascend.worker.model_runner_v1.graph_capture`,负责把 NPU stream 切到 capture stream + 后续切回。
3. **`set_ascend_forward_context` 而非 GPU 的 `set_forward_context`**:它的命名参数 `aclgraph_runtime_mode` 是 Ascend 这边的术语。值的枚举(NONE / PIECEWISE / FULL)是 vllm 共享的。
4. **`torch.npu.synchronize()` 每个 size 后调用一次**:确保上一个 size 的 capture 完成,避免和下一个 capture overlap。GPU 上通常不需要(stream 自管理),NPU 这边的 ACL 实现需要显式同步。
5. **失败处理**:`talker_mtp_graph_safe=True` 是模型作者写的"承诺"——出问题就直接抛出,提醒模型作者承诺有误,而不是 silently fallback(silently fallback 会让 production 性能突然掉 5-10 倍,排查极痛苦)。

## 11.7 `_dummy_run` 里把 talker_mtp 也走一遍

`OmniNPUModelRunner._dummy_run` 在 NPU 上的"warmup forward":

```python
with set_ascend_forward_context(
    attn_metadata, self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    in_profile_run=is_profile,
    num_actual_tokens=num_tokens_padded,
    aclgraph_runtime_mode=cudagraph_runtime_mode,
    batch_descriptor=batch_desc,
    model_instance=self.model,
):
    # ---------------------------------------Omni-new----------------------------------------------
    if getattr(self.model, "talker", None) is not None and self.has_talker_mtp:
        num_tokens_padded_talker_mtp = num_tokens_padded
        if num_tokens_padded_talker_mtp == self.max_num_tokens:
            num_tokens_padded_talker_mtp = self.talker_mtp_input_ids.gpu.shape[0]
        outputs = self.talker_mtp(
            self.talker_mtp_input_ids.gpu[:num_tokens_padded_talker_mtp],
            self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded_talker_mtp],
            self.last_talker_hidden.gpu[:num_tokens_padded_talker_mtp],
            self.text_step.gpu[:num_tokens_padded_talker_mtp],
        )
        self.compilation_config.cache_dir = None
    # Call self.model() directly (like GPU) to avoid make_omni_output during dummy_run
    outputs = self.model(
        input_ids=input_ids, positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
    )
    # ---------------------------------------Omni-new----------------------------------------------
```

要点:

- **dummy_run 时也要 invoke talker_mtp**——只有这样,NPU graph 的 workspace 才会被"预占",真正 capture 时就不会因为"workspace 第一次分配"破坏图。
- **`num_tokens_padded_talker_mtp` 的边界 hack**:talker_mtp 的 buffer 大小可能比主 forward 的 `max_num_tokens` 小一档(因为 talker_mtp 按 `max_num_reqs` 而不是 `max_num_tokens` 分配),所以当 padded token 达到上限时强制 clamp 到 talker_mtp 自己的 buffer 大小。
- **`compilation_config.cache_dir = None`**:NPU 这边有个边界——dummy_run 后清掉 cache_dir 防止某些情况下 talker_mtp 的中间产物污染主 model 的编译缓存目录。

## 11.8 device mismatch 这种典型坑

PR #3453 修了一个看似简单但很代表性的 bug:

```diff
- code_predictor_codes_cpu = code_predictor_codes.detach().to("cpu").contiguous()
  ...
- update_dict = {out_key[0]: {out_key[1]: code_predictor_codes_cpu[idx : idx + 1]}}
+ update_dict = {out_key[0]: {out_key[1]: code_predictor_codes[idx : idx + 1]}}
```

背景:`_merge_additional_information_update` 最终会把 `code_predictor_codes` append 进 `request.additional_information.codes.audio`,这个 list 后续要被 chunk_transfer_adapter 取走、通过 SHM 发给 code2wav。

旧实现里早早把 codes D2H 到 CPU,看起来合理:**反正 SHM 是 host memory,迟早要回 CPU**。问题在**并发模式下**:NPU 上一个 step 内的多请求是真并发的(每条请求各自的 talker_mtp 输出在不同 NPU stream / device),先 D2H 再切片可能让"原本在 NPU 的另一条请求的张量"被错误地 fall through 到 CPU——device mismatch 错误。

修复方法是**保留在 NPU 上做 slicing,让 D2H 推迟到 chunk_transfer_adapter 真正要发送时**——那时只有一条请求的 slice,device 明确。这是 NPU 跨流并发下的典型陷阱:**多 stream 上的 D2H 时机错误会导致跨流数据被意外卷入**。GPU 上不容易出是因为单 stream 内的顺序通常显式,NPU 的 ACL 多 stream 会偶尔暴露这种隐式依赖。

## 11.9 收益和总结

| 优化 | NPU 上是否启用 | 收益来源 |
|------|---------------|----------|
| ACLGraph 包装 talker_mtp | ✓ | 把 talker_mtp 一次 forward 的几十个 op host launch 折叠 |
| `_capture_talker_mtp_graphs` 二段式 capture | ✓ | warmup → FULL 模式,保证 workspace 预占 |
| `_capture_npu_graphs` per-bucket | ✓ | code_predictor 内部 8 步 AR 循环也入图 |
| `npu_fusion_attention` 替换 SDPA | ✓ | 单独的 NPU 融合 attention,接口完全不同 |
| torch.compile / Inductor 融合 | ✗ | 平台不支持,只能 eager + 图 |
| prefix graphs(按 seq_len 分桶) | ✗ | NPU graph 数量爆炸代价大于收益 |
| Hot buffer cache(talker hidden) | 跨平台共用 | 静态地址 + replay,NPU/GPU 一致 |

实际上 NPU 上 Qwen3-Omni 单卡部署里,**这条 code_predictor 路径是改造工作量最大的一段**——既要写 `_forward_npu_attention`,又要写 `_capture_npu_graphs`,还要在 model runner 二段 capture。和 GPU 路径相比,NPU 的"特化"指数高。但这也是 vllm-omni `current_omni_platform` 抽象的价值:**所有 platform 特化都隔离在 `vllm_omni/platforms/npu/` 和模型代码里的 `if is_npu()` 分支,model runner / scheduler / orchestrator 等通用代码一行不动**。

## 11.10 相关 PR

- **#89** [Feat] Add NPU Backend support for vLLM-Omni(初始接入)
- **#484** [NPU][Model] Support Qwen3-Omni on NPU(Qwen3-Omni 跑通)
- **#537** [NPU] Support mixed modalities for Qwen3-Omni(图文音混合)
- **#597** [Bugfix] Removed the NPU-specific code path in _run_local_attention
- **#2695** [NPU] Support code predictor NPU graph(**核心:NPU graph capture 改造**)
- **#3453** Fix NPU code predictor device mismatch in concurrent mode(**NPU 并发踩坑**)
- **#3476** [Refactor] Unify _talker_mtp_forward across GPU and NPU model runners(GPU/NPU 合并 path)

---

# 第十二部分: QA

## Q1: talker_mtp 和 code_predictor 是同一个东西吗?

`talker_mtp` 是 `Qwen3OmniMoeForConditionalGeneration.talker_mtp` 这个 method;它内部调用 `self.talker.code_predictor_forward(...)`。所以:**`talker_mtp` 是这个 method 的名字,`code_predictor` 是真正干活的 nn.Module**。从外部(model runner)看是 talker_mtp,从内部看是 code_predictor 在算。

## Q2: 为什么 code2wav 的 cudagraph 是它自己内部管理,不走 vllm 的图?

vllm 的 cudagraph 是按 token batch 和 num_tokens 分桶的,适合 LLM 的"每步出 1 token"模式。code2wav 是一次出整段 wav,它的形状空间是 `(batch_size, codec_seq_len)`,和 vllm 的桶逻辑完全不一样。复用很麻烦,自己写 wrapper 更直接——而且 wrapper 还能管 `chunked_decode` 的内部循环,这是 vllm 不知道的。

## Q3: 我看代码里 `enforce_eager` 在 code2wav 上有时 true 有时 false,这有什么区别?

`enforce_eager=true` 表示**关掉 vllm 外层 cudagraph**;`enforce_eager=false` 是关闭"强制 eager"开关,**让外层 cudagraph 生效**。但 code2wav 还有**内部的 `CUDAGraphDecoderWrapper`**,它的开关是另一个独立的 `enable_cudagraph` 调用。所以可能的组合:

- 外层 eager + 内层关:全 eager,最慢但最稳;
- 外层 eager + 内层开:vllm 不入图,但 wrapper 入图——典型配置;
- 外层入图 + 内层开:嵌套图,有兼容性风险。

ROCm 上必须 `enforce_eager: true`,因为 MIOpen conv_transpose1d 不能 capture。

## Q4: bounded-K active stream 的 K 怎么定?

经验:**K ≈ talker 的稳态 batch_size**。比如 talker 的 `max_num_seqs=64`,实际稳态 batch ≈ 30-40,那 K=32 是合理的——太小则 talker batch 不饱和,太大则丧失 admission control 意义。

## Q5: talker_mtp 走 cudagraph,显存代价多大?

每个 batch_size 桶 capture 一次。Qwen3-Omni 默认 `max_cudagraph_capture_size=512`,但 talker_mtp 桶通常更小(decode batch 一般不超过 max_num_seqs)。每个桶占用:**4 个 buffer × hidden_size × batch_padded × 2 byte**。Qwen3-Omni hidden=2048,batch=64,一桶 ≈ 1 MB,十几桶 ≈ 几十 MB。可以忽略。

## Q6: code2wav 的 SnakeBeta 是什么?为什么不用 ReLU/SiLU?

SnakeBeta 是 BigVGAN(NVIDIA)论文提出的**周期性激活**——`x + (1/β) sin²(αx)`,擅长建模音频波形的周期结构,显著改善 vocoder 的音质。Qwen3-Omni 直接采用 BigVGAN 风格的 decoder,所以也用 SnakeBeta。它的 PyTorch 实现 kernel launch 多,Triton 融合是直接对症下药。

## Q7: cross-request batching 在 code2wav 上做和 talker 上做,意义一样吗?

不一样。Talker 是 LLM,vllm 本来就有 continuous batching,自然多请求合一个 batch。**Code2Wav 不是 LLM**——它的"batch"概念由 scheduler 单独管。Cross-request batching 在 code2wav 上的意思是:**多个独立请求的小 chunk 拼成一个 conv batch**。这是 RFC #3163 专门讨论的——和 LLM 的 batching 不是一回事。

## Q8: hot buffer cache 和 cudagraph 是什么关系?

cudagraph 要静态地址——也就是说,buffer 的指针不能变。hot buffer cache 让 model 持有一个**永远不变的 buffer 句柄**,这样:

- cudagraph 一次 capture 后可以无限 replay,因为读的指针是固定的;
- 每步只更新 buffer 内的值,**不重新分配**;
- 省去了"从 model output 拷到 cudagraph 静态 buffer"的额外 D2D。

如果 buffer 被 prefix cache 命中 evict(命中场景下 KV 是从 cache 拼回的,hidden 也可能从 cache 拼),就要 fallback 到从 model output 拷——这是 PR #3688 强调的"fall back on evicted state"。

## Q9: PR #3296 删 audio_tower / visual,会不会让 talker 收到 audio / image 请求时崩?

不会。**talker 这一 stage 在 vllm-omni 拓扑里不会直接接收 audio / image 输入**——它接收的是 thinker 已经处理过后的 hidden 张量。所以 talker 进程根本不需要 audio_tower / visual encoder。**职责分离让裁剪是安全的**:thinker 处理 vision/audio,talker 处理 text→codec,code2wav 处理 codec→wav。

## Q10: 如果我自己要给一个新 omni 模型做类似优化,顺序怎么排?

经验:

1. **先把端到端跑通**(用最朴素的 full payload + eager + batch=1);
2. **加 cudagraph wrapper**(decoder 入图,看 latency 是否大幅改善);
3. **加 cross-request batching**(吞吐场景做);
4. **加 hot buffer cache**(decode 路径下 hidden 拷贝大头);
5. **加 bounded-K active stream**(高并发场景);
6. **加 prefix cache staging 优化**(长 prompt 场景);
7. **最后调桶大小、删死代码、自定义 Triton kernel**(微调)。

每一步都先量化收益,再决定是否值得做——很多优化在小并发下没差别,大并发才显著。

## Q11: NPU 上 code_predictor 没有 torch.compile 加持,性能差距大吗?

差距取决于场景。**code_predictor 内部的 RMSNorm / RoPE / linear 链路是 fusion 收益的大头**——Inductor 大概能省 20-30% latency。NPU 拿不到这部分,但通过 NPUGraph 把整段 host launch 全消掉,**单 step 内部 op 之间的 launch overhead 几乎归零**,在小 batch / 短 seq 上反而能追平 GPU 的 `torch.compile + cudagraph` 组合;**真正吃亏在中大 batch、kernel 本身是瓶颈时**,Inductor 那 20% 就直接体现在端到端延迟上。所以 NPU 适配的核心是"把 graph 覆盖度做到极致",而不是抠 kernel 融合。

## Q12: NPU graph capture 和 GPU 的 `CUDAGraphWrapper` 行为有什么不一样?

最显著的是**"二段式 capture"**——NPU 必须先 eager warmup `num_warmups` 次再正式 capture FULL,GPU 上通常一次 capture 就够(因为 CUDA 的 workspace 分配相对透明)。还有 **prefix graphs(按 seq_len 分桶)NPU 上关掉**——graph 数量 = batch_buckets × seq_lens,NPU graph capture 比 CUDA 慢、显存占用大,所以只按 batch_size 维度分桶。第三是 **`torch.npu.synchronize()` 每个 size 后显式同步**——CUDA 上 stream 自动管理,NPU 的 ACL 多 stream 实现有时需要显式 sync 才能避免 capture overlap。

## Q13: PR #3453 修的 device mismatch 是怎么发生的?

NPU 上一个 step 内多请求是真并发(每条请求各自的 talker_mtp 输出可能挂在不同 NPU stream / device)。旧实现里**早早把 `code_predictor_codes` D2H 到 CPU 再 slice**,看似合理(反正 SHM 是 host memory)。但**这种"先全 D2H 再 slice"的实现把"另一条还在 NPU 上、还没算完"的请求的张量也卷进来**——拼成一个 CPU 张量后,后续按 idx slice 拿到的就不是本请求的了。修复是**保留在 NPU 上 slice,推迟 D2H 到 chunk_transfer_adapter 真正发送时**——那时只有一条请求,device 明确。GPU 上单 stream 顺序通常显式,所以同样代码不一定暴露;NPU 多 stream 让隐式依赖浮出水面。

## Q14: 如果我自己要给一个新 omni 模型做类似优化,顺序怎么排?

经验:

1. **先把端到端跑通**(用最朴素的 full payload + eager + batch=1);
2. **加 cudagraph wrapper**(decoder 入图,看 latency 是否大幅改善);
3. **加 cross-request batching**(吞吐场景做);
4. **加 hot buffer cache**(decode 路径下 hidden 拷贝大头);
5. **加 bounded-K active stream**(高并发场景);
6. **加 prefix cache staging 优化**(长 prompt 场景);
7. **最后调桶大小、删死代码、自定义 Triton kernel**(微调);
8. **跨平台时**(GPU → NPU / ROCm),先把 `current_omni_platform.is_npu()` 和 `supports_torch_inductor()` 的分叉点都写好,再针对每个分叉补特化代码(`npu_fusion_attention`、`_capture_npu_graphs` 等)。

每一步都先量化收益,再决定是否值得做——很多优化在小并发下没差别,大并发才显著。

---

# 总结

Qwen3-Omni 的性能优化覆盖了**从 kernel 到调度**的全栈:

| 层级 | 优化 |
|------|------|
| **Kernel** | Triton SnakeBeta 融合 |
| **CUDA Graph** | talker_mtp wrapper / code2wav wrapper / 桶分配策略 |
| **Tensor 内存** | hot buffer cache / prefix cache staging 去重 |
| **Batch 调度** | talker_mtp 多 batch / code2wav cross-request batching / bounded-K active stream |
| **死代码** | Talker stage 不加载 audio_tower / visual |
| **架构** | async chunk + stage replica + StagePool affinity |
| **跨平台** | NPU 上 code_predictor 的 `npu_fusion_attention` + ACLGraph 二段式 capture + Inductor fallback |

每一项都有对应的 PR 在 git history 里——你在面试时能精确地说"对应 PR #2376 是 code2wav 入图,PR #3322 是 cross-request batching,PR #3592 是 bounded-K……"会显得很专业。

更重要的是,理解**这些优化为什么必须分开做**:

- **kernel 融合**是模型作者写 nn.Module 时关心的;
- **cudagraph wrapper**是 runner 和 model 协作的产物;
- **batch 调度**是 scheduler 的事;
- **stage 编排**是 orchestrator 的事;
- **死代码裁剪**是 stage_id-aware 的 load_weights;

它们都不是 vllm 本身能解决的——这就是为什么 vllm-omni 需要存在,以及为什么它的优化路径要分散在多个层级。

如果有人问"vllm-omni 的优化和 vllm 的优化有什么不同?"——一句话回答:**vllm 优化"一个 model 一个 forward",vllm-omni 优化"多个 model 之间的 chunk 流水 + 单个 model 的辅助 head + 单个 stage 的 wrapper 类 + replica 间的负载分配"**——后者的优化空间更大、更模型相关,所以单独存在是合理的。

---

# 附录:Qwen3-Omni talker_mtp graph 面试讲法

这一节单独把 Qwen3-Omni 的 `talker_mtp` graph 适配过程讲清楚,尤其是两个容易混的点:

1. 为什么 `_preprocess()` 里只收集 `span_len == 1 and not is_prefill` 的 decode rows;
2. 为什么 `_talker_mtp_forward()` 要单独用 `decode_batch_size` 计算 cudagraph bucket,而不是复用主 forward 的 graph bucket。

## A. 先讲清楚 MTP 在一个 decode step 里干什么

Qwen3-Omni Talker 每个 audio frame 不是一个 token,而是一组 RVQ codebook id:

```text
frame_t = [codebook_0, codebook_1, ..., codebook_G-1]
```

其中:

```text
codebook_0    由主 Talker LLM 的 codec_head 产生
codebook_1..G-1 由 code_predictor / talker_mtp 补齐
```

一个 decode step 的真实依赖关系是:

```text
上一步主 Talker LLM 产出的 hidden_(t-1)
+ 上一帧 layer-0 code_(t-1)
+ 当前 text_step_t
    ↓
talker_mtp
    ↓
1. full RVQ codes_(t-1)        -> 写 additional_information.codes.audio
2. inputs_embeds_t             -> 写回本步主 Talker LLM 的 inputs_embeds
```

所以 `talker_mtp` 不是主 forward 的后处理,而是主 forward 的**前置输入准备**:

```text
_preprocess()
  -> _talker_mtp_forward()
  -> 写回 inputs_embeds_t
  -> 主 Talker LLM forward(inputs_embeds_t)
```

如果不前置执行,本步主 Talker LLM 就拿不到"上一帧实际采出的完整 RVQ embedding 反馈"。

## B. 为什么 `_preprocess()` 只收集 `span_len == 1 and not is_prefill`

runner 遍历本 step 的 scheduled requests 时,每条 request 在当前 batch 里对应一个 slice:

```python
start_offset = int(self.query_start_loc.cpu[req_index])
sched_tokens = int(num_scheduled_tokens_np[req_index])
s, e = start_offset, start_offset + sched_tokens
span_len = int(e) - int(s)
```

`span_len` 是这条请求本轮被 scheduler 安排了多少个 token。

### B.1 prefill row 为什么不跑 MTP

prefill 阶段的任务是把 prompt / thinker payload / TTS 特殊 embedding 等上下文灌进 Talker KV cache。它不是在逐帧生成音频。

此时:

```text
span_len 通常 > 1
is_prefill = True
```

prefill 只需要:

```text
model.preprocess(...) -> prompt inputs_embeds
主 Talker LLM forward -> 建 KV cache / 产出 prompt hidden
```

不需要 `talker_mtp` 补 residual codebooks。文档前面也提到,preprocess 的 prefill 分支通常会给 `codes.audio` 写零,表示这段 prompt/template 不应该被 code2wav 合成成音频。

### B.2 decode row 为什么要求 `span_len == 1`

Talker 的 decode 是自回归逐帧的:

```text
一个 decode step = 生成/推进一个 codec frame
```

因此每条 decode request 在本 step 里应该只调度 1 个 token:

```text
span_len == 1
```

这个 token 对应上一帧的 layer-0 code / 当前 Talker 输入位置。`talker_mtp` 正是基于这个单步状态补齐一帧 RVQ codes。

如果 `span_len > 1`,那通常是 prefill 或 chunked prefill,不是"一帧 decode";这时直接跑 MTP 会破坏 AR 语义,因为 MTP 需要逐步依赖上一帧 hidden,不能把多 token prompt 当成多帧音频一起补。

所以 runner 的筛选条件是:

```python
if self.has_talker_mtp and span_len == 1 and not is_prefill:
    # 收集这条 decode row 的 MTP 输入
```

一句话:**MTP 只服务 Talker decode 的单帧推进,不服务 prefill。**

## C. `_preprocess()` 收集的是什么

Qwen3-Omni 没有 Qwen3-TTS 那种 `preprocess_decode_batch` 高并发 hook,它走逐条 `model.preprocess(...)`:

```python
req_input_ids, req_embeds, update_dict = self.model.preprocess(
    input_ids=input_ids[s:e],
    input_embeds=embed_slice,
    **req_infos,
)
```

decode 分支里模型会返回:

```python
update_dict["mtp_inputs"] = last_talker_hidden, text_step
```

runner 取出这两个 MTP 专属输入:

```python
last_talker_hidden, text_step = update_dict.pop("mtp_inputs")
```

然后把 4 个输入写进固定 GPU buffer:

```python
self.talker_mtp_input_ids.gpu[decode_slice].copy_(req_input_ids)
self.talker_mtp_inputs_embeds.gpu[decode_slice].copy_(req_embeds)
self.last_talker_hidden.gpu[decode_slice].copy_(last_talker_hidden)
self.text_step.gpu[decode_slice].copy_(text_step)
```

这 4 个 buffer 是 `talker_mtp` graph 的静态输入地址:

| buffer | 语义 |
|--------|------|
| `talker_mtp_input_ids` | 上一帧 layer-0 codec id |
| `talker_mtp_inputs_embeds` | 上一帧 layer-0 codec embedding |
| `last_talker_hidden` | 上一步主 Talker LLM hidden |
| `text_step` | 当前文本条件 embedding |

同时 runner 记录:

```python
decode_req_ids.append(req_id)
decode_start_offsets.append(s)
```

`decode_start_offsets` 后面用于把 MTP 输出的 `req_embeds` 写回大 batch 的正确 row:

```python
inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]
```

## D. 主 forward graph 和 talker_mtp graph 的核心区别

这是面试里最容易讲乱的地方。可以用一张表说清楚:

| 维度 | 主 Talker LLM forward graph | talker_mtp graph |
|------|-----------------------------|------------------|
| 调用对象 | `self.model(...)` | `self.talker_mtp(...)` |
| 发生位置 | `_model_forward()` | `_preprocess()` 内,主 forward 之前 |
| 输入语义 | scheduler 本轮所有 token | 本轮所有 decode requests 的 MTP 输入 |
| batch key | `num_tokens` / `BatchDescriptor` | `decode_batch_size` / 单独 `BatchDescriptor` |
| 是否包含 prefill | 可能包含 prefill + decode 混合 | 只包含 decode rows |
| 每条请求 token 数 | 可能是多 token prefill,也可能 1 token decode | 固定 1 个 MTP step |
| attention/KV | 有 attention metadata / KV cache | 没有主 LLM attention metadata;是辅助子 forward |
| 输出 | hidden/logits 等主模型输出 | `req_embeds` + `codes.audio` |

### D.1 主 forward 的 graph bucket 怎么来

主 forward 是 vLLM 原生路径。它看的是 scheduler 本轮实际要跑多少 token:

```text
num_tokens = 所有 scheduled tokens 的总和
```

举例:

```text
请求 A: prefill 128 tokens
请求 B: decode 1 token
请求 C: decode 1 token

主 forward num_tokens = 130
```

主 forward 的 cudagraph bucket 会围绕 `130` 这个 token batch 去选。它还要考虑:

```text
prefill/decode 是否混合
attention metadata
KV cache slot mapping
LoRA 状态
DP padding / ubatch / cascade attention
```

这套 `BatchDescriptor` 是给 transformer 主 forward 用的。

### D.2 talker_mtp 的 graph bucket 怎么来

MTP 不关心 prefill tokens。它只关心本 step 有多少条 decode request 要补一帧:

```text
decode_batch_size = len(decode_req_ids)
```

还是上面的例子:

```text
请求 A: prefill 128 tokens
请求 B: decode 1 token
请求 C: decode 1 token

主 forward num_tokens = 130
MTP decode_batch_size = 2
```

所以 `_talker_mtp_forward()` 必须单独算 bucket:

```python
_cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
    num_tokens=decode_batch_size,
    num_reqs=decode_batch_size,
    num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
    max_num_scheduled_tokens=1,
    use_cascade_attn=False,
)
```

这里几个参数的含义:

```text
num_tokens=decode_batch_size
    对 MTP 来说,每个 decode request 贡献 1 个 MTP row。

num_reqs=decode_batch_size
    MTP batch 里有多少条请求。

num_scheduled_tokens_np=np.ones(...)
    明确告诉 dispatch:每条请求都是 1 个 scheduled token。

max_num_scheduled_tokens=1
    这是 uniform decode batch,没有 prefill 长 span。

use_cascade_attn=False
    MTP 不是主 attention forward,不走 cascade attention。
```

这样得到的 `batch_desc` 语义是:

```text
给 talker_mtp 子 forward 用的 batch descriptor
```

而不是主 LLM forward 的 batch descriptor。

## E. 为什么不能复用主 forward 的 batch_desc

如果复用主 forward 的 `batch_desc`,会有三个问题。

### E.1 padding 大小会错

主 forward 可能因为 prefill 选到一个很大的 bucket:

```text
主 forward: num_tokens = 130 -> pad 到 160
MTP: decode_batch_size = 2 -> 只需要 pad 到 2/4/8
```

如果 MTP 也用 160 的 bucket,就会让 `talker_mtp` 按 160 行 replay,大量 padding 行白算,显存和时间都浪费。

### E.2 row 语义会错

主 forward 的 row 是 token row:

```text
row 0..127: 请求 A 的 prefill tokens
row 128: 请求 B 的 decode token
row 129: 请求 C 的 decode token
```

MTP 的 row 是 decode request row:

```text
row 0: 请求 B
row 1: 请求 C
```

这两个 row space 不是同一个东西。MTP 输出 `req_embeds[0]` 应该写回请求 B 的 `start_offset`,而不是主 batch row 0。

### E.3 graph 捕获对象不同

主 forward graph 捕获的是:

```text
Talker LLM transformer forward
```

MTP graph 捕获的是:

```text
code_predictor_forward + RVQ embedding sum + text_step add
```

这两个 callable 的输入输出、内部 op、shape 都不一样。即使都用 vLLM 的 `CUDAGraphWrapper`,也必须有各自的 `BatchDescriptor` key。

一句话:**主 forward 的 graph key 是 token batch;MTP 的 graph key 是 decode request batch。**

## F. 适配过程中的核心修改

可以按下面顺序讲:

1. **模型暴露 `talker_mtp()` hook**

   顶层模型提供:

   ```python
   def talker_mtp(input_ids, input_embeds, last_talker_hidden, text_step, **kwargs):
       code_predictor_codes, summed_embeddings = self.talker.code_predictor_forward(...)
       inputs_embeds = summed_embeddings + text_step
       return inputs_embeds, code_predictor_codes
   ```

2. **runner 初始化时识别并包装**

   ```python
   talker_mtp = getattr(self.model, "talker_mtp", None)
   self.talker_mtp = graph_wrapper_cls(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
   ```

3. **runner 分配 4 个静态输入 buffer**

   ```text
   talker_mtp_input_ids
   talker_mtp_inputs_embeds
   last_talker_hidden
   text_step
   ```

4. **`_preprocess()` 只收集 decode rows**

   ```text
   span_len == 1 and not is_prefill
   ```

   prefill 只建 KV/cache,不跑 MTP。

5. **`_talker_mtp_forward()` 单独按 `decode_batch_size` 选 graph bucket**

   ```text
   num_tokens = decode_batch_size
   num_reqs = decode_batch_size
   每条请求 scheduled token = 1
   ```

6. **调用 wrapped `talker_mtp` 并写回两个结果**

   ```text
   req_embeds -> inputs_embeds[start_offset]      给主 Talker LLM
   codes.audio -> additional_information          给 Code2Wav
   ```

## G. 面试时可以这样串起来

> Qwen3-Omni Talker 的 MTP 不是普通后处理,它是主 Talker forward 前的一个子 forward。它用上一帧 hidden 和上一帧 layer-0 code 补齐完整 RVQ codes,同时把 G 层 codebook embedding 求和后加上当前 text_step,生成本步主 Talker LLM 的 inputs_embeds。所以 runner 需要在 `_preprocess()` 阶段只收集 decode rows,因为只有 `span_len == 1 and not is_prefill` 的 row 才代表一次音频帧推进。  
>
> 另一个关键是 graph bucket 不能复用主 forward。主 forward 的 batch descriptor 是 scheduler 本轮 token batch,可能包含大段 prefill;MTP 的 batch descriptor 是 decode request batch,每条请求只贡献 1 个 MTP row。因此 `_talker_mtp_forward()` 要用 `decode_batch_size` 单独调用 `_determine_batch_execution_and_padding`,得到自己的 cudagraph bucket。最终就是两张 graph 串联:`talker_mtp graph` 先写好 `inputs_embeds`,再由 vLLM 原生 `main talker LLM graph` 继续 forward。

最后可以用这个例子收尾:

```text
本 step:
  A: prefill 128 tokens
  B: decode 1 token
  C: decode 1 token

主 forward graph:
  num_tokens = 130
  rows = A 的 128 个 token + B + C

talker_mtp graph:
  decode_batch_size = 2
  rows = B, C
```

这就是为什么 MTP 必须有自己的 buffer、自己的 batch descriptor、自己的 graph wrapper。
