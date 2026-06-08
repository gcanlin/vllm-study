# MMEncoderAttention: NPU (vllm-ascend) vs GPU (vllm) 实现与优化历史

本文对比 vLLM 主仓 (`vllm/model_executor/layers/attention/mm_encoder_attention.py`)
与 vllm-ascend (`vllm_ascend/ops/mm_encoder_attention.py`、`vllm_ascend/_310p/ops/mm_encoder_attention.py`)
中 `MMEncoderAttention` 的实现差异，并梳理两边围绕该模块的演进与优化。

`MMEncoderAttention` 是多模态 (VL/OCR/Audio) 视觉编码器内部使用的**无 KV cache** 多头注意力，
形状一般为 `(batch, seq_len, hidden)`，由各 ViT 模型 (`qwen2_5_vl`, `qwen3_vl`, `glm4_1v`,
`siglip`, `intern_vit`, `clip`, `whisper` 等约 30+ 个) 共享调用。
它通过 `CustomOp.register_oot` 让平台（CUDA/ROCm/XPU/CPU/NPU/310P）替换实现。

---

## 1. 总体架构

`MMEncoderAttention` 继承自 `CustomOp`，并提供 `forward_native / forward_cuda /
forward_cpu / forward_xpu` 多套设备实现；NPU 实现以 OOT (out-of-tree) 方式注册：
[vllm_ascend/utils.py:826](../vllm-ascend/vllm_ascend/utils.py#L826) 把
`AscendMMEncoderAttention` 注册到 `"MMEncoderAttention"`，
[vllm_ascend/utils.py:876](../vllm-ascend/vllm_ascend/utils.py#L876) 在 310P 上覆写为
`AscendMMEncoderAttention310`。两者都通过 `forward_oot` 进入。

主要差异立刻可见的有 3 处：

| 维度 | vllm (GPU/XPU/CPU) | vllm-ascend (NPU) |
| --- | --- | --- |
| 入口 | `forward_cuda / forward_xpu / forward_cpu`（按 backend 分派） | `forward_oot`（设备适配器内部再分派 A2/A3/A5/310P） |
| 主算子 | FlashAttention(2/3/4), FlashInfer cuDNN, Triton ViT, Torch SDPA | `_npu_flash_attention_unpad` (A2/A3/310P) / `npu_fusion_attention` (A5) |
| 数据布局 | `(b, s, h, d)` 4D，varlen 用 `cu_seqlens`(GPU 上) | `(b*s, h, d)` 3D + **CPU 上的 `seq_lens`**（NPU 算子要求） |
| 量化 | FlashInfer + cuDNN ≥ 9.17.1 上支持 FP8（动态/静态 scale） | 当前无 FP8（FP8 通过 backend 模块走另一条路径） |
| GQA 处理 | 后端原生 `enable_gqa` 或 `num_queries_per_kv` | 显式 `torch.repeat_interleave` 把 KV 复制到 Q 头数 |
| head_dim padding | 仅 FlashInfer FP8 路径在 Triton kernel 中 pad | 始终把 `64 < head_dim < 128` 的 head padding 到 128（NPU 算子最优配置） |

---

## 2. GPU 端（vllm 主仓）实现要点

`MMEncoderAttention` 在 `__init__` 阶段调用 `get_vit_attn_backend()`，从配置和硬件能力中
选出 backend，可选项 `AttentionBackendEnum`:

- `FLASH_ATTN` / `ROCM_AITER_FA` → `_forward_fa` → `vit_flash_attn_wrapper`
- `TRITON_ATTN` → `_forward_triton`（PR #32183 引入，覆盖 head_dim 不被 FA 支持的情况）
- `FLASHINFER` → `_forward_flashinfer`（PR #34580 引入，使用 cuDNN prefill kernel）
- `TORCH_SDPA` → `_forward_sdpa`（fallback / CPU / 部分 XPU）

`vllm/v1/attention/ops/vit_attn_wrappers.py` 把每个 backend 用
`direct_register_custom_op` 注册成 `torch.ops.vllm.*`，这是为了让外层
ViT 能被 `torch.compile`（PR #30709 起，~5% 吞吐 / ~7% 时延收益）。

### 2.1 FlashAttention / Triton 路径

- 4D 输入 `(b, s, h, d)` 直接送进 `flash_attn_varlen_func / context_attention_fwd`。
- `cu_seqlens` 留在 device 上，`max_seqlen` 经过 `.item()`（被自定义 op 隔离避免破坏 `torch.compile`）。
- 没有 head_dim padding；GQA 由 `num_kv_heads != num_heads` 时 FA 内部完成。

### 2.2 FlashInfer cuDNN 路径（PR #34580）

为了利用 H100 / B200 上的 cuDNN ViT prefill kernel：

- 引入 batch-size / max-seqlen **bucketing**（`FLASHINFER_BATCH_BUCKETS`, `FLASHINFER_MAX_SEQLEN_BUCKETS`），
  避免 cuDNN graph cache 因每帧 shape 都不同而频繁重建。
- 引入静态 `_get_flashinfer_workspace_buffer()`（128 MiB），整张模型共用。
- 在 `maybe_recompute_cu_seqlens` 中把 `cu_seqlens` 重组为两段（Q/K/O 与 V），因为
  cuDNN 期望批 offsets，且 BF16 路径下 Q/K/V 来自同一 packed buffer，V 的 stride 是 3 倍。
- 在 `maybe_compute_seq_lens` 中额外提供 `sequence_lengths` 给 cuDNN（差值并 padding 到 bucket）。

### 2.3 FP8 ViT Attention（PR #38065，2026-04）

vllm 主仓最重的一次 ViT attention 改造，只在 FlashInfer + cuDNN ≥ 9.17.1 + 原生 FP8 GPU 上生效：

- 配置：`mm_encoder_attn_dtype="fp8"`，可选 `mm_encoder_fp8_scale_path`（静态）或自动收集动态 scale 并 `mm_encoder_fp8_scale_save_path` 落盘。
- 通过 `vllm/kernels/triton/qkv_padded_fp8_quant.py::quantize_fp8_maybe_pad_head_dim`
  在一次 Triton kernel 里完成「pad head_dim 到 2 的幂」+「per-tensor FP8 量化」+「scale 写回」。
- **动态 scale**：`_record_amax_and_update_scales` 维护长度 16 的循环 amax history（GPU 上），
  buffer wrap 时一次性落盘所有层 scale，避免 forward 路径上的 D2H sync。
  注释明确指出：使用 CUDA Graph 时必须切回静态 scale 文件。
- `_forward_flashinfer` 把 `q_scale/k_scale/v_scale/o_data_type` 一并传给 cuDNN，
  让 cuDNN 直接走 FP8 路径并输出 BF16；head_dim 若被 padding，最后切片回 `head_size`。

### 2.4 零拷贝 GQA（PR #33732）

之前 `_forward_sdpa` 需要先 `repeat_interleave` KV 头到 Q 头数；这个 PR 让 wrappers 直接传
`enable_gqa=True` 给 `F.scaled_dot_product_attention`，省掉一次 K/V 复制，对 multimodal & CPU 都生效。
**vllm-ascend 仍然保留显式 `repeat_interleave`**——因为底层 `_npu_flash_attention_unpad` /
`npu_fusion_attention` 当前不支持 GQA。

---

## 3. NPU 端（vllm-ascend）实现要点

NPU 路径远比 GPU 路径精简，因为 NPU 上只跑一个底层算子，没有多 backend 选择，但围绕「**让那个唯一的算子跑得快**」做了很多 host 侧优化。

### 3.1 与 GPU 不同的核心约束

NPU 上视觉 attention 用的是 Ascend 自家两条 API：

- `torch_npu._npu_flash_attention_unpad`（A2/A3/310P）—— **要求 `seq_len` 在 CPU 上**，需要 3D 输入 `(total_tokens, num_heads, head_dim)`，**不支持 GQA**，head_dim ∈ {64, 80, 96, 128} 等离散配置最优。
- `torch_npu.npu_fusion_attention`（A5/Ascend 950）—— TND 布局，接受 device 上的 `actual_seq_qlen` 累计形式。

这两点直接决定了 NPU 实现的所有「奇怪」之处：

1. **4D → 3D 重排**：`_reshape_qkv_to_3d` 先 `view` 成 `(b*s, h, d)`；返回时通过 `einops.rearrange` 回到原 shape，保留 `is_reshaped` 标志保证「**输出 shape 与输入一致**」（PR #5443 修复）。
2. **GQA 用 `repeat_interleave` 显式实现**：`num_queries_per_kv > 1` 时把 K/V 复制到 Q 头数。GPU 端则交给 backend。
3. **head_dim padding**：`self.enable_pad = MIN_PAD_SIZE < head_size < MAX_PAD_SIZE` (64 < d < 128) 时用 `F.pad` 把 q/k/v 末维补到 128，输出再切回原长度——NPU 上 128 是最优 head_dim。GPU 端只在 FP8 路径才 pad，且 pad 与量化融合到一个 Triton kernel；NPU 上仍是 3 次独立 `F.pad`（PR #6204 曾尝试 stack-then-pad 合并为 1 次启动，PR #6448 又 revert 回 3 次，因为 SP/TP 场景下 stack 反而引入 D2H 同步）。
4. **`seq_lens` 强制落 CPU**：`forward_oot` 里若上层没有预提供 `sequence_lengths`，就 `torch.diff(cu_seqlens).to("cpu")`，**每层都同步**。

### 3.2 A2/A3 vs A5 的算子分歧（PR #8671, 2026-04）

A5 (Ascend 950) 默认走 `npu_fusion_attention`（TND 布局，性能在 A5 上更好），但 A2/A3 上同一个算子比 `_npu_flash_attention_unpad` 慢。PR #8671 把这两条路径下沉到设备适配器：

```python
# vllm_ascend/device/device_op.py:301  (BaseDeviceAdaptor, A2/A3 default)
torch_npu._npu_flash_attention_unpad(query, key, value, seq_len=seq_lens_cpu, ...)

# vllm_ascend/device/device_op.py:1106 (A5DeviceAdaptor)
seq_lens_cpu = list(seq_lens_cpu.cumsum(0))
torch_npu.npu_fusion_attention(query, key, value, actual_seq_qlen=seq_lens_cpu,
                               actual_seq_kvlen=seq_lens_cpu, input_layout="TND", ...)
```

`AscendMMEncoderAttention.forward_oot` 统一调用 `DeviceOperator.npu_flash_attention(...)`，由设备适配器内部决定具体算子，效果上 A2/A3 上 1920×1080P 22 并发场景 290 → 300 tps。

### 3.3 310P 单独实现

`AscendMMEncoderAttention310`（PR #6117 refactor 拆出独立类）与基类几乎一致，唯一不同是直接调用 `torch_npu._npu_flash_attention_unpad`，不经过设备适配器——因为 310P 不属于 A2/A3/A5 体系（PR #7518 是为了修 A5 算子被合入主类后 310P 崩溃的 regression）。

### 3.4 OOT hook：`maybe_compute_seq_lens` 重写

`MMEncoderAttention` 在主仓里给 OOT 留了 3 个 classmethod 钩子：
`compute_max_seqlen / maybe_compute_seq_lens / maybe_recompute_cu_seqlens`
（PR #36605 增加 `maybe_get_oot_by_class` 让 OOT 类可以覆写这些静态行为）。
vllm-ascend 重写 `maybe_compute_seq_lens` 让模型层（在 vision blocks 之前一次性）把
`cu_seqlens` 转成 CPU 上的 `seq_lens` 并通过 forward 参数 `sequence_lengths` 一路传进 attention：

```python
# vllm_ascend/ops/mm_encoder_attention.py:72
seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
seq_lens = torch.from_numpy(seq_lens).to("cpu", non_blocking=True)
```

`forward_oot` 里若收到 `sequence_lengths`，就跳过本层的 `torch.diff(...).to("cpu")` 同步——这是 PR #7104 的核心优化，把 D2H sync 从每层一次降到整个 ViT 一次。

---

## 4. 优化历史时间线

### vllm 主仓

| 时间 | PR | 关键变化 |
| --- | --- | --- |
| 2026-01-09 | #31916 | `[1/N] Restructure attention`：拆出 `layers/attention/mm_encoder_attention.py` 独立目录 |
| 2026-01 | #30709 | LLaMa4 vision encoder 用 `torch.compile`；引入 `vit_attn_wrappers.py` 的自定义 op 边界 |
| 2026-02-05 | #33732 | Zero-copy GQA：SDPA / CPU 路径不再 `repeat_interleave` |
| 2026-02-15 | #32183 | 增加 Triton ViT attention backend |
| 2026-02-27 | #34580 | FlashInfer cuDNN backend + batch / max-seqlen bucketing + workspace buffer |
| 2026-03-12 | #36605 | OOT 钩子：让 NPU 等 OOT 平台覆写 `maybe_compute_seq_lens / maybe_recompute_cu_seqlens` |
| 2026-04-04 | #35010 | XPU 也支持 TORCH_SDPA / TRITON_ATTN backend |
| 2026-04-09 | #32974 | FA4 集成（FlashAttention 4 后端） |
| 2026-04-27 | #38065 | **FP8 ViT attention**：FlashInfer + cuDNN ≥ 9.17.1，动态/静态 scale，QKV padded fp8 quant Triton kernel |

### vllm-ascend

| 时间 | PR | 关键变化 |
| --- | --- | --- |
| 2025-07 | #1929 | Qwen2.5-VL ViT 启用 SP + mrope 融合算子（早期 patch 形式） |
| 2025-12 | #4750 | 把 `AscendMMEncoderAttention` 改造成 `CustomOp.register_oot`，删除原先的 monkey-patch |
| 2025-12 | #5443 | Bugfix：`AscendMMEncoderAttention` 输出 shape 与输入保持一致（4D in / 4D out） |
| 2026-01-24 | #6117 | 310P 拆独立类 `AscendMMEncoderAttention310` |
| 2026-01-26 | #6204 | 把 Q/K/V padding 并行：stack 后一次 `aclnnConstantPadNd`（TTFT -3.15%, 峰值吞吐 +4.20%） |
| 2026-02-26 | #6448 | `seq_lens` CPU cache + scale value 提前计算 + `enable_pad` 移到 `__init__`；**revert #6204**（stack 引入新同步） — TTFT -7.43%, 吞吐 +1.23% |
| 2026-03-06 | #6730 | `split_qkv_rmsnorm_mrope` 融合算子（Qwen3-VL 上游） |
| 2026-03-20 | #7046 | A5 (Ascend 950) 上把 `_npu_flash_attention_unpad` 替换为 `npu_fusion_attention` |
| 2026-03-23 | #7518 | Bugfix：A5 算子合入主类后 310P 崩溃，拆分 310P 路径 |
| 2026-03-23 | #7104 | **Pre-compute `seq_lens`**：利用主仓 #36605 的 OOT 钩子，把每层 D2H 同步降到整个 ViT 一次 |
| 2026-04-02 | #7762 | `qkv_rmsnorm_mrope` 融合（Qwen3-VL full attention） |
| 2026-04-03 | #7893 | Flash Comm V1（SP）支持 Qwen3-VL |
| 2026-04-29 | #8671 | A2/A3 上把 `npu_fusion_attention` 换回 `_npu_flash_attention_unpad`，通过 `BaseDeviceAdaptor` / `A5DeviceAdaptor` 二分派 |

---

## 5. 设计权衡总结

### NPU 没有走 GPU 的多 backend / FP8 路径，原因：

1. **底层算子收口在 `torch_npu` 两个 API**：FlashAttention/Triton 在 NPU 上要么不存在要么不是最优；唯一选择就是包好 NPU 的 unpad/fusion 算子。
2. **NPU 算子的硬约束驱动 host 改写**：CPU `seq_lens`、3D 输入、`head_dim==128`、不支持 GQA——这些迫使 vllm-ascend 在每个 forward 之前做 reshape / pad / repeat_interleave，而对应的优化全部花在「把这些 host 工作量降下来」上（CPU cache、pre-compute、A2/A3 与 A5 分派）。
3. **FP8 路径走在 attention 模块之外**：Ascend 当前 ViT FP8 通过 backend 模块（`vllm_ascend/quantization/...`）做权重量化和 GEMM，不像 GPU 这样把 Q/K/V 的 FP8 量化和 cuDNN attention 融合。

### GPU 端最大的工程难点是「为多个 backend / 多个 head_dim / `torch.compile` / FP8 / cuDNN graph cache」做泛化：

1. backend 选择放进 `get_vit_attn_backend(head_size, dtype)`，因为不同 head_dim 在 FA2/FA3/FA4 上支持度不同（FA4 整合见 PR #32974）。
2. `vit_attn_wrappers.py` 用 `direct_register_custom_op` 把 `.item()` / 不规则 reshape 隔离在 `torch.ops.vllm` 边界外，保证 `torch.compile` 不被打断（注释明确 +5% 吞吐 / +7% 时延）。
3. FlashInfer cuDNN bucketing 是为了把「每张图 shape 都不同」压成有限 graph，避免 cuDNN graph cache thrash。
4. FP8 引入了**动态 scale**这条不能与 CUDA Graph 共存的路径，因此提供静态 scale 文件方案；代码注释把这一点写得很清楚（`_record_amax_and_update_scales` doc）。

### 共同点（两边一致的处理）

- **VarLen 拼接**：都用 `cu_seqlens` 表达打包后的多张图/帧序列，避免对最长序列 padding。
- **`CustomOp.register_oot`**：vllm 主仓在 PR #30125 后规范了 OOT 注册方式；vllm-ascend 在 PR #4750 跟上，并通过 PR #36605 的 OOT 钩子（`maybe_get_oot_by_class`）扩展行为而非整体替换。
- **vision blocks 上游预处理**：两边都倾向把 `seq_lens / cu_seqlens` 在进入 ViT blocks 之前算好一次，逐层只读。
