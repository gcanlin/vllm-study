# vLLM RoPE 全景：从普通 RoPE 到 MRoPE / YaRN / DualChunk / FoPE

> **文档版本**: 1.0
> **分析代码版本**: 当前 workspace 本地 `vllm` 源码
> **最后更新**: 2026-06-06

---

## 文档概述

本文档讲 vLLM 推理热路径里 **旋转位置编码 (RoPE)** 的实现。RoPE 是当前主流 decoder-only LLM 唯一的位置编码方式：每个 token 在进入 attention 之前，先把 Q / K 旋转一个由 token 位置决定的角度，attention 计算出来的内积就只依赖相对位置。

但"普通 RoPE"在 vLLM 里其实是 **十几种变体**：

- **长上下文外推**：LinearScaling、NTK、DynamicNTK、Llama3、YaRN、Deepseek YaRN、Phi3 LongRoPE、TeleChat3 YaRN、DynamicNTKAlpha
- **多模态**：MRoPE (Qwen2/2.5/3-VL)、MRoPE-Interleaved (Pangu)、XDRoPE (4D)、Ernie4.5-VL、Llama4 Vision、Gemma4 Proportional
- **长序列结构**：DualChunkRoPE (Qwen-long)
- **可学习位置**：FoPE (Fourier RoPE)

它们的差别 **不是计算逻辑** —— 几乎都是 `q' = q·cos + rotate(q)·sin` —— 而是：

> **`cos_sin_cache` 怎么算、`positions` 怎么索引这个 cache、最后旋转哪几个维度**

理解这一点，整张地图就清晰了。本文按这个角度展开：

| 部分 | 内容 |
|------|------|
| 第一部分 | RoPE 的数学回顾和 vLLM 的统一抽象 |
| 第二部分 | 基础 RoPE：`RotaryEmbedding` / `RotaryEmbeddingBase` |
| 第三部分 | 长上下文家族：linear / NTK / YaRN / Llama3 / LongRoPE |
| 第四部分 | DeepSeek YaRN 和 V4 变体（MLA 路径） |
| 第五部分 | MRoPE：多模态 3D 位置 |
| 第六部分 | MRoPE 衍生：interleaved / XD / Ernie4.5-VL |
| 第七部分 | DualChunkRoPE：分块外推 |
| 第八部分 | FoPE / Gemma4 / Llama4-Vision 等特殊形态 |
| 第九部分 | `get_rope` 工厂与 `_ROPE_DICT` 缓存 |
| 第十部分 | 源码索引 |

**目标读者**：已经知道 RoPE 的基本思路、想在 vLLM 里区分各种 RoPE 变体的工程师。

---

# 第一部分：RoPE 的数学回顾与 vLLM 的统一抽象

## 1.1 普通 RoPE 做了什么

对一个 head 的 query 向量 q ∈ ℝ^d（`d = head_size` 或 `rotary_dim`），把它两两分成 pair `(q_{2i}, q_{2i+1})`，每对乘上一个 2×2 旋转矩阵：

```text
[ q'_{2i}   ] = [ cos(m·θ_i)  -sin(m·θ_i) ] [ q_{2i}   ]
[ q'_{2i+1} ]   [ sin(m·θ_i)   cos(m·θ_i) ] [ q_{2i+1} ]
```

其中：

- `m` 是 token 在序列中的位置；
- `θ_i = base^(-2i/d)` 是第 `i` 对的频率（`base` 一般是 `rope_theta = 10000`）。

对 K 做同样旋转后，`q'·k'` 只跟两个 token 的位置差 `m_q - m_k` 有关 —— 这就是"相对位置编码"。

> 直觉：把每个 channel pair 想成复平面上的一个点，按位置 `m` 转一个角度。后面所有变体本质上都是在调 `θ_i` 或 `m`。

## 1.2 NEOX-style vs GPT-J-style

同样的旋转，pair 在 head_dim 上有两种排布：

| style | pair 取法 | 半边重排函数 |
|-------|-----------|--------------|
| **NEOX (`is_neox_style=True`)** | `x1 = x[:d/2]`, `x2 = x[d/2:]` | `rotate_neox`：`[x, y] → [-y, x]` |
| **GPT-J (`is_neox_style=False`)** | `x1 = x[::2]`, `x2 = x[1::2]` | `rotate_gptj`：把奇偶交错 stack 再 flatten |

[common.py:18-28](../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L18-L28)：

```python
def rotate_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_gptj(x):
    x1 = x[..., ::2]; x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
```

两种 style 数学上等价，只是模型权重的内存排布约定不同。HF 上 Llama / Qwen / DeepSeek 都用 NEOX；早期 GPT-J / Falcon 用 GPT-J style。**vLLM 同一份 `cos_sin_cache` 配合 `is_neox_style` 这个 flag 就能切换**。

## 1.3 vLLM 的统一抽象

所有 RoPE 变体最终都被 [`get_rope()`](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py#L33-L384) 这个工厂函数创建：

```python
self.rotary_emb = get_rope(
    head_size,                       # 单 head 维度
    max_position=max_position,       # 训练时最大位置
    is_neox_style=True,
    rope_parameters={                # 模型 config 里 rope_scaling 的内容
        "rope_type": "yarn",
        "rope_theta": 1000000,
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        ...
    },
)
```

用法（[llama.py:230](../vllm/vllm/model_executor/models/llama.py#L230)）：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)   # ← in-place 旋转 q/k
attn_output = self.attn(q, k, v)
```

`get_rope` 内部根据 `rope_parameters["rope_type"]` 走 if-elif，命中以下分支之一：

```text
default       → RotaryEmbedding  (或 MRotaryEmbedding / FourierRotaryEmbedding)
proportional  → Gemma4RotaryEmbedding
llama3        → Llama3RotaryEmbedding
mllama4       → Llama4VisionRotaryEmbedding
linear        → LinearScalingRotaryEmbedding
ntk           → NTKScalingRotaryEmbedding
dynamic       → DynamicNTKScalingRotaryEmbedding / DynamicNTKAlphaRotaryEmbedding
xdrope        → XDRotaryEmbedding
yarn          → YaRNScalingRotaryEmbedding (或带 mrope 时 → MRotaryEmbedding)
deepseek_yarn / deepseek_llama_scaling → DeepseekScalingRotaryEmbedding / V4
longrope      → Phi3LongRoPEScaledRotaryEmbedding
openpangu     → MRotaryEmbeddingInterleaved
telechat3-yarn → TeleChat3RoPEScaledRotaryEmbedding
```

结果会被缓存在 `_ROPE_DICT`，**key 是参数 hash**，所以同一台机器同模型只构建一次。

---

# 第二部分：基础 RoPE — `RotaryEmbeddingBase`

[base.py:15-137](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L15-L137)

## 2.1 类结构

```text
CustomOp                          (vLLM 自定义算子基类)
  └── RotaryEmbeddingBase         # 算 inv_freq / cos_sin_cache，不实现 forward
        └── RotaryEmbedding       # 普通 RoPE，实现 forward_native/cuda/hip/xpu/cpu
              ├── LinearScalingRotaryEmbedding
              ├── NTKScalingRotaryEmbedding
              ├── DynamicNTKScalingRotaryEmbedding
              ├── DynamicNTKAlphaRotaryEmbedding ── XDRotaryEmbedding
              ├── Llama3RotaryEmbedding
              ├── YaRNScalingRotaryEmbedding
              ├── Gemma4RotaryEmbedding
              └── FourierRotaryEmbedding
        └── DeepseekScalingRotaryEmbedding
              └── DeepseekV4ScalingRotaryEmbedding
        └── MRotaryEmbedding
              ├── MRotaryEmbeddingInterleaved
              └── Ernie4_5_VLRotaryEmbedding
        └── Llama4VisionRotaryEmbedding
```

不属于这棵树的：

- `DualChunkRotaryEmbedding`：直接继承 `CustomOp`，维护 5 个 cache。
- `Phi3LongRoPEScaledRotaryEmbedding`：直接继承 `nn.Module`，维护 `long_short_cos_sin_cache`。

## 2.2 cos/sin cache 的算法

基类做两件事：

```python
# 1. 算 inv_freq：每个 channel pair 的频率
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

# 2. 算 cos/sin cache：max_position × rotary_dim/2
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = einsum("i,j -> ij", t, inv_freq)   # [max_pos, rotary_dim/2]
cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)   # [max_pos, rotary_dim]
```

**所有 RoPE 变体都通过重写下面这两个方法来"换皮"：**

```python
def _compute_inv_freq(self, base): ...
def _compute_cos_sin_cache(self): ...
```

cache 在 `__init__` 里算好一次，注册为 non-persistent buffer：

[base.py:58-74](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L58-L74)

```python
if init_cache:
    cache = self._compute_cos_sin_cache()
    if not self.use_flashinfer:
        cache = cache.to(dtype)
    self.register_buffer("cos_sin_cache", cache, persistent=False)
```

dtype 默认沿用模型 dtype（bf16/fp16）。**DeepSeek V4** 是少数把 cache 留在 fp32 的（[deepseek_scaling_rope.py:247-248](../vllm/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py#L247-L248)），因为 MLA 路径对 RoPE 精度敏感。

## 2.3 forward 路径：多后端分发

`RotaryEmbedding.forward_*` 是 `CustomOp` 标准接口，按设备分发：

| 后端 | 实现 | 入口 |
|------|------|------|
| `forward_native` | 纯 PyTorch，逐步 chunk / cos·x + sin·rotate(x) | [base.py:203-219](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L203-L219) |
| `forward_cuda` | C++ kernel `ops.rotary_embedding`，**in-place** 写回 q/k；或 FlashInfer 路径 | [base.py:221-252](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L221-L252) |
| `forward_hip` | ROCm AITER Triton kernel（开了开关时），否则 fallback CUDA 路径 | [base.py:254-271](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L254-L271) |
| `forward_xpu` | Intel XPU 路径 | [base.py:273-296](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L273-L296) |
| `forward_cpu` | 复用同一份 `ops.rotary_embedding` (CPU build) | [base.py:298-318](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L298-L318) |

**为什么必须 in-place？** RoPE 在 attention 之前，输入是刚 split 出来的 q/k tensor，下一步立刻喂给 attention kernel。在原地改写一次显存远比"分配新 tensor + 写一次 + 读一次"便宜，对 decode 阶段尤其重要。

CUDA kernel 真正干活的是 [pos_encoding_kernels.cu:9-33](../vllm/csrc/libtorch_stable/pos_encoding_kernels.cu#L9-L33)：

```cuda
inline __device__ void apply_token_rotary_embedding(
    scalar_t* arr, const cache_t* cos_ptr, const cache_t* sin_ptr,
    int rot_offset, int embed_dim, const bool inverse) {
  // NEOX: pair = (arr[i], arr[embed_dim + i])
  // GPT-J: pair = (arr[2i], arr[2i+1])
  const float x_f = arr[x_index];
  const float y_f = arr[y_index];
  arr[x_index] = x_f * cos_f - y_f * sin_f;
  arr[y_index] = y_f * cos_f + x_f * sin_f;
}
```

一个 block 处理一个 token，block 内线程并行处理 `num_heads × embed_dim` 个 pair。`inverse=True` 用来"反向旋转"（DeepSeek V4 indexer 会用到）。

## 2.4 FlashInfer 路径

[base.py:227-236](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L227-L236) 在 `use_flashinfer=True` 时调 `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace`。当前在 base 里默认关闭，只有 DeepSeek 路径在 dtype/head_size 满足条件时打开：

[deepseek_scaling_rope.py:61-67](../vllm/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py#L61-L67)：

```python
self.use_flashinfer = (
    self.enabled()
    and dtype in (torch.float16, torch.bfloat16)
    and current_platform.is_cuda()
    and has_flashinfer()
    and head_size in [64, 128, 256, 512]
)
```

FlashInfer 的 RoPE kernel 把 `lookup cos_sin → 旋转 → 写回`融成一个 kernel，并对 MLA 的 head 数 / head dim 做了专门 tuning。

## 2.5 `ApplyRotaryEmb`：被 MRoPE 系列复用的"通用旋转"

[common.py:124-289](../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L124-L289) 里的 `ApplyRotaryEmb` 是把"已经准备好的 cos/sin"应用到 x 上的 CustomOp，被 MRoPE 系列复用：

- `forward_native`：torch 实现，`q_rot = x1·cos - x2·sin; q_pass = x2·cos + x1·sin`；
- `forward_cuda`：调 `vllm_flash_attn.layers.rotary.apply_rotary_emb`（FlashAttn 的 Triton kernel）；
- `forward_hip`：调 `flash_attn.ops.triton.rotary.apply_rotary`。

它和 `ops.rotary_embedding` 的区别：

| 算子 | 是否 lookup cos/sin cache | 适用场景 |
|------|---------------------------|----------|
| `ops.rotary_embedding` | 是，传 `positions` 自己 lookup | 普通 LLM 路径 |
| `ApplyRotaryEmb` | 否，cos/sin 已外部准备好 | MRoPE 这种"cos/sin 要按多模态 section 拼装"的场景 |

---

# 第三部分：长上下文家族

下面这一组都是只改 `_compute_inv_freq` 或 `_compute_cos_sin_cache`，forward 直接复用基类 → C++ kernel。

## 3.1 LinearScalingRoPE：位置除以 factor

[linear_scaling_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/linear_scaling_rope.py)

把位置 `t` 整体压缩 `factor` 倍（"位置插值"，Position Interpolation, PI）：

```python
max_len = max_position * scaling_factor
t = torch.arange(max_len) / scaling_factor       # ← 关键：t 除以 factor
freqs = einsum("i,j -> ij", t, inv_freq)
```

支持 **多个 scaling_factor 共存**：把多个 cache 沿位置维拼在一起，再用 `scaling_factor_to_offset` 记录每段起点 —— 给 **LoRA** 用，不同 adapter 可以训不同 RoPE scaling，推理时用 offset 索引到对应那段 cache，**一个 kernel 就能 batch 处理多个 LoRA**。

## 3.2 NTKScalingRoPE：放大 base

[ntk_scaling_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/ntk_scaling_rope.py)

NTK 思路：高频频率（小角度变化大）做外推，低频做插值。最简单的 fixed-NTK 把 `base` 放大：

```python
base_new = base * scaling_factor              # mixed_b 模式见 kexue.fm/archives/9706
inv_freq = 1.0 / (base_new ** (arange / rotary_dim))
inv_freq = inv_freq / scaling_factor ** (2 / rotary_dim)
```

`mixed_b` 是混合 NTK，给每个频率单独算 lambda。

## 3.3 DynamicNTKRoPE：按当前序列长度动态算 base

[dynamic_ntk_scaling_rope.py:53-72](../vllm/vllm/model_executor/layers/rotary_embedding/dynamic_ntk_scaling_rope.py#L53-L72)

```python
base = base * ((scaling_factor * max_pos / max_trained_pos) - (scaling_factor - 1)) \
            ** (rotary_dim / (rotary_dim - 2))
```

"dynamic"指的是 base 跟实际长度挂钩，长度越长 base 越大。注意 vLLM 里这个 cache **是预先按 `max_position` 算好一次的，并不真的每步重算** —— 因为 KV cache 里旧 token 的 RoPE 已经写死，运行时再改 base 会破坏一致性。

## 3.4 DynamicNTKAlphaRoPE：Hunyuan / XD 的入口

[dynamic_ntk_alpha_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/dynamic_ntk_alpha_rope.py)

更简洁的形式，给一个 `scaling_alpha`：

```python
base = base * scaling_alpha ** (rotary_dim / (rotary_dim - 2))
```

Hunyuan 用它做长上下文；后面 **XDRoPE** 继承它扩展到 4D 多模态。

## 3.5 Llama3 RoPE：分段缩放

[llama3_rope.py:33-54](../vllm/vllm/model_executor/layers/rotary_embedding/llama3_rope.py#L33-L54)

Llama3 把频率分三段：

- **高频（wave_len < high_freq_wavelen）**：完全外推（freq 不变）
- **低频（wave_len > low_freq_wavelen）**：完全插值（freq / scaling_factor）
- **中频**：线性混合 `(1-smooth) * freq/factor + smooth * freq`

```python
smooth = (orig_max_pos / wave_len - low_freq_factor) / (high_freq_factor - low_freq_factor)
new_freqs = torch.where(
    wave_len < high_freq_wavelen, inv_freqs,
    torch.where(
        wave_len > low_freq_wavelen, inv_freqs / scaling_factor,
        (1 - smooth) * inv_freqs / scaling_factor + smooth * inv_freqs,
    ),
)
```

## 3.6 YaRN：扩展 + magnitude 修正

[yarn_scaling_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py)

YaRN 在 Llama3 思路上更精细：

1. **频率插值/外推混合**：用 `yarn_linear_ramp_mask(low, high, ...)` 在 `beta_fast`/`beta_slow` 之间做 ramp，避免"在哪个频段切换"的边界突变；
2. **magnitude 修正 (`mscale`)**：长上下文 RoPE 后 q·k 内积分布会变，用 `mscale = 0.1 * log(scale) + 1.0` 给 cos/sin 整体乘一个系数补偿。

cache 计算：

```python
inv_freq = (1 - mask) * inv_freq_interpolation + mask * inv_freq_extrapolation
freqs = einsum("i,j -> ij", arange(max_pos * scale), inv_freq)
cos = freqs.cos() * mscale
sin = freqs.sin() * mscale
```

## 3.7 Phi3 LongRoPE：双 cache 切换

[phi3_long_rope_scaled_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py)

Phi3 family 维护 **两份 cache**：

- **短 cache** `short_factor`：原始 `original_max_position_embeddings` 范围
- **长 cache** `long_factor`：扩展 `max_position` 范围

`__init__` 时拼成一个大 cache：

```python
short_cache = self._compute_cos_sin_cache(orig_max, short_factor, short_mscale)
long_cache  = self._compute_cos_sin_cache(max_pos, long_factor, long_mscale)
long_short_cache = torch.cat([short_cache, long_cache], dim=0)
```

运行时根据 `max_model_len > orig_max` 决定 `idx` 是否加上 `orig_max` 偏移，从 long 那段取：

```python
if self.use_long_rope:
    idx = positions + orig_max_position_embeddings   # ← 落到 long cache
else:
    idx = positions                                  # ← 落到 short cache
```

**注意这里没有走 C++ kernel**，是纯 torch forward —— Phi3 直接重写 `nn.Module.forward`，不走 `CustomOp` 那套分发。`mscale` 是 attention 的 softmax 温度补偿。

## 3.8 TeleChat3 YaRN

[__init__.py:351-380](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py#L351-L380) 里这个分支会把 `scaling_factor` 重新算成 `max_position / original_max_position`，然后走 `TeleChat3RoPEScaledRotaryEmbedding`（细节和 YaRN 几乎一致，只是 scaling_factor 的来源不同）。

---

# 第四部分：DeepSeek YaRN 和 V4

[deepseek_scaling_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py)

DeepSeek 的 MLA 把 head 拆成 `q_nope`（不带 RoPE）和 `q_rope`（带 RoPE），RoPE 部分有几个特别之处：

## 4.1 DeepseekScalingRotaryEmbedding

数学上和 YaRN 接近，但：

- **`mscale` 公式不同**：
  ```python
  mscale = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim) * attn_factor
  ```
- **直接继承 `RotaryEmbeddingBase`，不继承 `RotaryEmbedding`**：自己写 `forward_native` / `forward_cuda` / `forward_xpu` / `forward_hip`；
- **可能开 `use_flashinfer`**：[deepseek_scaling_rope.py:61-67](../vllm/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py#L61-L67)；
- **支持 `offsets` 参数**：`positions + offsets` 后再 lookup，留给推测解码（speculative decoding）调 RoPE 位置。

## 4.2 DeepseekV4ScalingRotaryEmbedding

[deepseek_scaling_rope.py:230-344](../vllm/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py#L230-L344) 在 V3 基础上又改了几点：

| 差别 | 说明 |
|------|------|
| **RoPE 应用在 head 的最后 `rotary_dim` 维** | V3 是前 `rotary_dim`，V4 是 `query[..., -rotary_dim:]` |
| **cos_sin_cache 保留 fp32** | 提高 RoPE 精度（attention indexer 对位置敏感） |
| **`forward` 多一个 `inverse: bool`** | 允许"反向旋转"，indexer 通过 `sin = -sin` 把 RoPE 抵消回去 |
| **K 可以为 None** | DeepSeek indexer 只对 Q 应 RoPE |
| **CUDA 路径用 `ops.rotary_embedding(..., rope_dim_offset, inverse=...)`** | C++ kernel 加了 `rope_dim_offset` 参数（[pos_encoding_kernels.cu:103-118](../vllm/csrc/libtorch_stable/pos_encoding_kernels.cu#L103-L118)） |

这些细节配合 DeepSeek MLA 的 "K rope 共享、Q rope per-head"的结构，是 V4 indexer 需要"先正向、再反向"才能保留位置不变量。

---

# 第五部分：MRoPE — 多模态 3D 位置

[mrope.py](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py)

MRoPE 是 Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Keye-VL / Qwen2.5-Omni 系列的统一位置编码。**核心点：把 head_dim 切成 T / H / W 三段，每段用不同的"位置"驱动旋转**。

## 5.1 为什么要 3D 位置

纯文本 token，位置就是 token 在 prompt 里的索引；但图像 token 在 prompt 里是连续展开的，**真实空间结构是 (t, h, w)**：

```text
prompt:  ... <|vision_start|> p0 p1 p2 p3 ... pN <|vision_end|> 接下来文本 ...
                              └─────── 一帧图像被铺平 ───────┘
```

如果硬给 p0..pN 一个 1D 递增的位置编码，模型会把图像也当成"一段长文本"处理，看不到二维结构。MRoPE 的做法是给图像 token 分配 `(t, h, w)` 三元组：

```text
text:    positions = [3, 3, 3]      # T=H=W 都一样，等价于普通 1D RoPE
image:   positions[0] = grid_t indices  (帧 / 时间)
         positions[1] = grid_h indices  (高)
         positions[2] = grid_w indices  (宽)
```

构造细节在每个 mm 模型里实现，例如 [qwen2_vl.py:1240-1277](../vllm/vllm/model_executor/models/qwen2_vl.py#L1240-L1277)。最终 `positions` 形状是 `[3, num_tokens]`。

## 5.2 mrope_section：head_dim 的切片

`MRotaryEmbedding.__init__` 接受 `mrope_section: list[int]`，比如 `[16, 24, 24]`，含义是：

- 前 16 个频率 pair 用 T 维位置驱动；
- 接下来 24 个用 H 维位置驱动；
- 最后 24 个用 W 维位置驱动；
- `sum(mrope_section) == rotary_dim // 2`。

[mrope.py:248-251](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L248-L251)：

```python
self.mrope_section = mrope_section
if self.mrope_section:
    assert sum(self.mrope_section) == rotary_dim // 2
```

文本 token 因为 t=h=w 一样，三段对外仍是同一份 cos/sin → 退化成普通 RoPE，**所以 MRoPE 是普通 RoPE 的严格扩展，对纯文本 prompt 完全兼容**。

## 5.3 cache 比正常 RoPE 大 4×

[mrope.py:236-246](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L236-L246)：

```python
# In Qwen2.5-VL, the maximum index value is related to the duration of
# the input video. We enlarge max_position_embeddings to 4 times to get
# a larger the cos and sin cache.
self.cache_max_position_num = max_position_embeddings * 4
super().__init__(head_size, rotary_dim, self.cache_max_position_num, ...)
```

视频 token 的 t 索引会被 `t_factor`（`second_per_grid_ts * tokens_per_second`）放大，4× 是经验值。

## 5.4 forward_native：纯 torch 实现

[mrope.py:263-322](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L263-L322)

关键几行：

```python
cos_sin = cos_sin_cache[positions]       # [3, num_tokens, head_dim]  ← 3D lookup
cos, sin = cos_sin.chunk(2, dim=-1)
if positions.ndim == 2:                  # multimodal
    cos = torch.cat(
        [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
        dim=-1,
    )
    # 等价于: cos[T-section] from cos[0], cos[H-section] from cos[1], cos[W-section] from cos[2]
    sin = torch.cat(...)
# 然后调 self.apply_rotary_emb(query_rot, cos, sin)
```

`cos.split(mrope_section, dim=-1)` 沿 head_dim 把 cos 切成 `[T_section, H_section, W_section]` 三块；每一块上保留三个轴（来自 3D positions）的对应轴，最后拼回去。直觉上：

```text
cos_T = cos[0][:, :T_section]      # 用 T 位置算的 cos，取 T 段
cos_H = cos[1][:, T_section:T+H]   # 用 H 位置算的 cos，取 H 段
cos_W = cos[2][:, T+H:]            # 用 W 位置算的 cos，取 W 段
cos   = [cos_T | cos_H | cos_W]     # 沿 head_dim 拼回
```

## 5.5 forward_cuda：Triton mrope kernel

[mrope.py:14-187](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L14-L187)

vLLM 里写了一个 Triton kernel `_triton_mrope_forward`（adapt 自 Liger Kernel），一个 program 处理一个 token：

```python
# cos / sin 的 stride 是按 (3, num_tokens, head_dim/2)
t_cos = cos + pid * half_rd
h_cos = t_cos + num_tokens * half_rd
w_cos = h_cos + num_tokens * half_rd
# 用 mask 决定 head_dim 上每个 pair 用 T / H / W 哪一段
t_mask = cos_offsets < mrope_section_t
h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

# 三段相加 = 选择
cos_row = t_cos_row + h_cos_row + w_cos_row     # mask 互斥，其实就是 select
sin_row = t_sin_row + h_sin_row + w_sin_row

# 旋转 q / k 的左右半
new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
# 同理 k
```

把"按 section 取 cos/sin + 旋转 q/k"融成一个 kernel，避免 `cos.split`/`cat` 那一连串临时 tensor。

## 5.6 `mrope_position_delta` 与 decode 阶段

MRoPE prefill 后会得到一个 `mrope_position_delta`：

```python
llm_positions, mrope_position_delta = self.get_mrope_input_positions(input_tokens, mm_features)
mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens))
```

例如：一张图占 100 个 token，但因为 (t, h, w) 索引并不是 0..99 顺序递增，`llm_positions.max()` 可能是 200，那 delta = 200 + 1 - prompt_len。

decode 时新 token 是普通文本，位置直接接着 `mrope_position_delta + context_len` 递增：

[mrope.py:386-414](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L386-L414)：

```python
@staticmethod
def get_next_input_positions_tensor(out, out_offset, mrope_position_delta,
                                     context_len, num_new_tokens):
    values = np.arange(
        mrope_position_delta + context_len,
        mrope_position_delta + context_len + num_new_tokens,
    )
    out[:, out_offset : out_offset + num_new_tokens] = values   # 同样赋给 3 个轴
```

decode 的新文本 token 又退回 t=h=w 的"对角"模式。

## 5.7 MRoPE + YaRN

`MRotaryEmbedding.__init__` 还接收 YaRN 参数（`scaling_factor`、`beta_fast` 等），当 `scaling_factor != None` 时：

[mrope.py:253-261](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L253-L261)：

```python
def _compute_inv_freq(self, base):
    if self.scaling_factor is None:
        return super()._compute_inv_freq(base)
    return YaRNScalingRotaryEmbedding._compute_inv_freq(self, base)

def _compute_cos_sin_cache(self):
    if self.scaling_factor is None:
        return super()._compute_cos_sin_cache()
    return YaRNScalingRotaryEmbedding._compute_cos_sin_cache(self)
```

这是给 Qwen2.5-VL 在长上下文场景下用的 —— `__init__.py:259-272` 的 yarn 分支会带 `mrope_section` 一起进 `MRotaryEmbedding`。

---

# 第六部分：MRoPE 衍生

## 6.1 MRotaryEmbeddingInterleaved（Pangu）

[mrope_interleaved.py](../vllm/vllm/model_executor/layers/rotary_embedding/mrope_interleaved.py)

OpenPangu MM 的 MRoPE 把 T / H / W 在 head_dim 上**交错排布**，而不是 chunked `[TTT...HHH...WWW]`。kernel 端：

[mrope.py:60-69](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L60-L69) 里 Triton kernel 已经支持 `is_interleaved` 模式：

```python
if is_interleaved:
    h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
    w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
    t_mask = ~(h_mask | w_mask)
```

`MRotaryEmbeddingInterleaved` 还多一个 `get_mrope_interleaved_id_list` 算法 —— 给定 (a, b, c) 数量，按"剩余最少且不和上一个相同"的贪心生成一个不连续重复的交错序列，保证频率连续性。

## 6.2 XDRoPE：4D 位置

[xdrope.py](../vllm/vllm/model_executor/layers/rotary_embedding/xdrope.py)

继承 `DynamicNTKAlphaRotaryEmbedding`，把位置从 3D 扩到 4D：

```python
# positions: [4, num_tokens]  (P/W/H/T)
cos = torch.cat(
    [m[i] for i, m in enumerate(cos.split(self.xdrope_section, dim=-1))],
    dim=-1,
)
```

`xdrope_section` 是 4 段。`P` 一般是 page / frame，用于视频 / 长多模态。基础变换沿用 NTK-alpha 做长上下文。

## 6.3 Ernie4_5_VLRotaryEmbedding

[ernie45_vl_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py)

也是 3D MRoPE，但 head_dim 切片方式更花：H 和 W 不是连续两段，而是 `[H W H W H W ...]` 交错，最后接一段 T：

```python
section_cos_h = cos[..., : section_h + section_w : 2]    # 偶数位
section_cos_w = cos[..., 1 : section_h + section_w : 2]  # 奇数位
section_cos_t = cos[..., -section_t:]
```

然后 `cos_hw = stack([cos_h, cos_w]).reshape(2x)` 再 cat `cos_t`。这是 Ernie4.5-VL 专门设计的频率分配 —— 在 H 和 W 上交替一个高一个低，保证空间各方向频率覆盖均匀。

---

# 第七部分：DualChunkRoPE — 分块外推

[dual_chunk_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py)

Qwen 长上下文专用，配合 **Dual Chunk Attention (DCA)**。核心思路：

> 把超长序列切成多个 `chunk_size` 的块，每个块内用相对位置 RoPE，块间用专门的"intra"和"succession"位置。

[dual_chunk_rope.py:72-120](../vllm/vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py#L72-L120)：

```python
chunk_len = chunk_size - local_size
q_t  = arange(chunk_len)                                  # q 的块内位置 [0..chunk_len)
qc_t = (arange(chunk_len) + chunk_len).clamp(max=chunk_size)
                                                          # q 的"下一块"位置
k_t  = arange(max_position) % chunk_len                   # k 的位置：按 chunk 取模
qc_no_clamp_t = arange(chunk_len) + chunk_len             # 不 clamp 的版本
q_inter_t     = arange(chunk_len) + chunk_size            # q 跨块 inter 位置
```

总共 **5 份 cos_sin cache**：`q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache`。

forward：

[dual_chunk_rope.py:122-184](../vllm/vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py#L122-L184)：

```python
key   = apply(k_cache[positions], ...)                    # K 只有一份
query = apply(q_cache[positions % chunk_len], ...)
query_succ = apply(qc_cache[positions % chunk_len], ...)
query_inter = apply(qc_cache[chunk_len - 1].repeat(...), ...)
query_succ_critical = apply(qc_no_clamp_cache[positions % chunk_len], ...)
query_inter_critical = apply(q_inter_cache[positions % chunk_len], ...)
# 把 5 份 query 沿 head_dim 拼起来，DCA attention kernel 内部用不同分片做不同 attention 子计算
query = torch.cat((query, query_succ, query_inter,
                   query_succ_critical, query_inter_critical), dim=-1)
```

attention kernel 看到的"q tensor"实际上是 5 份不同 RoPE 编码的 q 拼在一起 —— DCA attention 内部根据 token 之间是同块、邻块还是远块，分别和 K 做不同 q 的内积。这是少数 **q 维度被 RoPE 放大** 的情况，下游 attention backend 必须配合解读。

---

# 第八部分：其他特殊形态

## 8.1 FoPE：可学习的傅里叶 RoPE

[fope.py](../vllm/vllm/model_executor/layers/rotary_embedding/fope.py)

普通 RoPE 的 `cos(θ_i t)` 是固定频率；FoPE 把它当成"基函数"，再乘一组**可学习**的系数矩阵：

```python
# cos_coef, sin_coef: nn.Parameter, shape [num_kv_heads, input_dim, output_dim]
# inv_freq 截断到 num_inv_freq 个
freqs = einsum("j,i -> ji", t, self.inv_freq)
pos_cos = freqs.cos()
pos_sin = freqs.sin()
# 学习系数变换：output cos = pos_cos @ cos_coef
cos = einsum("htD, hDd -> thd", pos_cos, self.cos_coef.float())
sin = einsum("htD, hDd -> thd", pos_sin, self.sin_coef.float())
```

特点：

- **per-head 不同**：cos/sin 多了一个 head 维（`fope_sep_head=True` 时）；
- **TP 切分系数**：`weight_loader` 按 `tensor_model_parallel_rank` 切 head 维；
- **首次 forward 才填 cache**：因为 `cos_coef` / `sin_coef` 加载顺序在 `__init__` 之后 → `forward_native` 里检查 `self.update_cache` 标志。

## 8.2 Gemma4 Proportional RoPE：偏要按 head_dim 算频率

[gemma4_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/gemma4_rope.py)

Gemma4 全局 attention 用 "proportional" 缩放，关键差别：

- **频率指数的分母是 `head_dim`，不是 `rotary_dim`**；
- **不旋转的 pair 用 cos=1, sin=0 占位**（identity rotation）；
- 这样 `rotary_dim` 可以传 `head_size`，C++ kernel 始终对整个 head_dim 做"旋转"，但 nope 段没有变化：

```python
self.rope_angles = rotary_dim // 2
self.nope_angles = (head_size // 2) - self.rope_angles
super().__init__(head_size, head_size, ...)        # ← rotary_dim 传 head_size

inv_freq = 1.0 / (base ** (arange(0, 2*rope_angles, 2) / head_size))
inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles)])  # ← zero-pad
```

这一招让 partial RoPE 直接复用通用 CUDA kernel，没有额外分支。

## 8.3 Llama4 Vision RoPE：2D 图像位置 + 复数运算

[llama4_vision_rope.py](../vllm/vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py)

给 vision encoder 用，每个 patch 给一对 (x, y) 位置：

```python
num_patches_single_dim = int(sqrt(num_patches))
frequencies_x = img_idx % num_patches_single_dim
frequencies_y = img_idx // num_patches_single_dim
freqs_x = (frequencies_x + 1)[..., None] * inv_freq * repeat_interleave(2)
freqs_y = (frequencies_y + 1)[..., None] * inv_freq * repeat_interleave(2)
freqs = cat([freqs_x, freqs_y], dim=-1)[..., ::2]
cache = view_as_complex(stack([cos(freqs), sin(freqs)], dim=-1))   # ← 复数 cache
```

forward 用 `torch.view_as_complex` 把 q / k 也变成复数，直接乘：

```python
query_ = view_as_complex(query.float().reshape(..., -1, 2))
query_out = view_as_real(query_ * freqs_ci).flatten(3)
```

跟纯文本路径完全不同，不调 `ops.rotary_embedding`，专走 native。

---

# 第九部分：`get_rope` 工厂与 `_ROPE_DICT`

[__init__.py:33-384](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py#L33-L384)

## 9.1 接口

```python
def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict | None = None,            # 来自 config.rope_parameters
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict | None = None,
) -> RotaryEmbedding:
```

`rope_parameters` 几乎就是 HF config 里 `rope_scaling` 的内容外加 `rope_theta`、`partial_rotary_factor`、`mrope_section` 这些。

## 9.2 partial_rotary_factor

```python
partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
rotary_dim = int(head_size * partial_rotary_factor)
```

部分 RoPE：只对 head 前 `rotary_dim` 维做旋转，后面留作 nope dims。Qwen3-VL Vision 用 0.5（[qwen3_vl.py:575-580](../vllm/vllm/model_executor/models/qwen3_vl.py#L575-L580)）：

```python
self.rotary_pos_emb = get_rope(
    head_size=head_dim,
    max_position=8192,
    is_neox_style=True,
    rope_parameters={"partial_rotary_factor": 0.5},
)
```

如果 config 里给了显式 `rope_dim`，会优先用 `rope_dim`。

## 9.3 cache key

```python
key = (head_size, rotary_dim, max_position, is_neox_style,
       rope_parameters_args, dual_chunk_attention_args, dtype)
if key in _ROPE_DICT:
    return _ROPE_DICT[key]
```

`_ROPE_DICT` 是模块级全局 dict，**整个进程内同样参数的 RoPE 实例只创建一份**：

- 多层 attention 共用同一个 `cos_sin_cache`，节省显存；
- 多个相同 RoPE 设置的模型实例（如 PP 多 stage）共享 buffer；
- 但 LoRA 多 `scaling_factor` 时用 `LinearScalingRoPE` 内置 multi-scale cache 来 batch，不会塞进 `_ROPE_DICT` 多份。

## 9.4 特殊分支：YaRN + MRoPE

[__init__.py:259-272](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py#L259-L272)：

```python
elif scaling_type == "yarn":
    ...
    if "mrope_section" in rope_parameters:
        extra_kwargs.pop("apply_yarn_scaling", None)
        rotary_emb = MRotaryEmbedding(
            head_size, rotary_dim, original_max_position, base,
            is_neox_style, dtype,
            mrope_section=rope_parameters["mrope_section"],
            mrope_interleaved=rope_parameters.get("mrope_interleaved", False),
            scaling_factor=scaling_factor,
            **extra_kwargs,
        )
    else:
        rotary_emb = YaRNScalingRotaryEmbedding(...)
```

这是给 Qwen2.5-VL 长上下文用 —— `MRotaryEmbedding` 在 `_compute_*` 里把 YaRN 公式接进来。

---

# 第十部分：怎么读这张地图

最后给一个 cheat sheet，看模型 config 怎么决定走哪条路：

| `rope_parameters` 内容 | 走到 |
|------------------------|------|
| 无 | `RotaryEmbedding` |
| `partial_rotary_factor < 1.0` | 同上，但 `rotary_dim < head_size` |
| `rope_type: "linear"` | `LinearScalingRoPE`（支持多 LoRA） |
| `rope_type: "ntk"` | `NTKScalingRoPE` |
| `rope_type: "dynamic"` + `factor` | `DynamicNTKScalingRoPE` |
| `rope_type: "dynamic"` + `alpha` | `DynamicNTKAlphaRoPE`（Hunyuan） |
| `rope_type: "llama3"` | `Llama3RotaryEmbedding` |
| `rope_type: "yarn"` | `YaRNScalingRoPE` |
| `rope_type: "yarn"` + `mrope_section` | `MRotaryEmbedding` (Qwen2.5-VL 长上下文) |
| `rope_type: "deepseek_yarn"` 或 `"deepseek_llama_scaling"` | `DeepseekScalingRotaryEmbedding` |
| 同上 + `is_deepseek_v4: True` | `DeepseekV4ScalingRotaryEmbedding` |
| `rope_type: "longrope"` | `Phi3LongRoPEScaledRotaryEmbedding` |
| `rope_type: "telechat3-yarn"` | `TeleChat3RoPEScaledRotaryEmbedding` |
| `rope_type: "proportional"` | `Gemma4RotaryEmbedding` |
| `rope_type: "mllama4"` | `Llama4VisionRotaryEmbedding` |
| `rope_type: "xdrope"` | `XDRotaryEmbedding`（4D） |
| `rope_type: "openpangu"` + `mrope_interleaved` | `MRotaryEmbeddingInterleaved` |
| `mrope_section` 单独存在 | `MRotaryEmbedding` (Qwen2/2.5/3-VL 标准模式) |
| `use_fope: True` | `FourierRotaryEmbedding`（可学习系数） |
| 传 `dual_chunk_attention_config` | `DualChunkRotaryEmbedding` |

---

# 第十一部分：源码索引

| 类别 | 入口文件 | 关键符号 |
|------|----------|----------|
| 工厂 / 类型分发 | `vllm/model_executor/layers/rotary_embedding/__init__.py` | `get_rope`、`_ROPE_DICT` |
| 抽象基类、CUDA 分发 | `.../rotary_embedding/base.py` | `RotaryEmbeddingBase`、`RotaryEmbedding`、`forward_cuda/native/hip/xpu/cpu` |
| 公共工具 | `.../rotary_embedding/common.py` | `rotate_neox/gptj`、`yarn_*`、`ApplyRotaryEmb`、`_flashinfer_rotary_embedding` |
| Linear PI | `.../linear_scaling_rope.py` | `LinearScalingRotaryEmbedding`、`scaling_factor_to_offset` |
| NTK | `.../ntk_scaling_rope.py` | `NTKScalingRotaryEmbedding` |
| Dynamic NTK | `.../dynamic_ntk_scaling_rope.py`、`.../dynamic_ntk_alpha_rope.py` | `DynamicNTKScalingRotaryEmbedding`、`DynamicNTKAlphaRotaryEmbedding` |
| Llama3 三段 | `.../llama3_rope.py` | `Llama3RotaryEmbedding` |
| YaRN | `.../yarn_scaling_rope.py` | `YaRNScalingRotaryEmbedding` |
| DeepSeek YaRN / V4 | `.../deepseek_scaling_rope.py` | `DeepseekScalingRotaryEmbedding`、`DeepseekV4ScalingRotaryEmbedding`、`use_flashinfer` |
| Phi3 LongRoPE | `.../phi3_long_rope_scaled_rope.py` | `Phi3LongRoPEScaledRotaryEmbedding`、`use_long_rope` |
| TeleChat3 | `.../telechat3_scaling_rope.py` | `TeleChat3RoPEScaledRotaryEmbedding` |
| Gemma4 | `.../gemma4_rope.py` | `Gemma4RotaryEmbedding`（zero-pad nope） |
| MRoPE 主体 + Triton | `.../mrope.py` | `MRotaryEmbedding`、`_triton_mrope_forward`、`apply_interleaved_rope`、`get_next_input_positions_tensor` |
| MRoPE 交错（Pangu） | `.../mrope_interleaved.py` | `MRotaryEmbeddingInterleaved`、`get_mrope_interleaved_id_list` |
| Ernie4.5-VL | `.../ernie45_vl_rope.py` | `Ernie4_5_VLRotaryEmbedding` |
| XDRoPE（4D） | `.../xdrope.py` | `XDRotaryEmbedding`、`get_next_input_positions_tensor` |
| Llama4 Vision | `.../llama4_vision_rope.py` | `Llama4VisionRotaryEmbedding` |
| FoPE | `.../fope.py` | `FourierRotaryEmbedding`、`cos_coef/sin_coef`、`weight_loader` |
| DualChunk | `.../dual_chunk_rope.py` | `DualChunkRotaryEmbedding`、5 个 cache buffer |
| CUDA kernel | `csrc/libtorch_stable/pos_encoding_kernels.cu` | `rotary_embedding_kernel`、`apply_token_rotary_embedding`、`rope_dim_offset`、`inverse` |
| MRoPE 位置构造（示例） | `vllm/model_executor/models/qwen2_vl.py`、`qwen2_5_vl.py`、`qwen3_vl.py`、`qwen2_5_omni_thinker.py`、`keye.py` | `get_mrope_input_positions`、`mrope_position_delta` |

---

> 一句话总结：vLLM 的 RoPE 实现是"**一个 cos_sin_cache + 一个 in-place 旋转 kernel + 一个统一工厂**"的设计。各种长上下文 / 多模态变体只是在不同位置改 cache 计算（`_compute_inv_freq` / `_compute_cos_sin_cache`）和位置 lookup（1D / 3D / 4D / dual-chunk），旋转算子本身只有 NEOX/GPT-J 两套，C++ kernel 一个就够了。
