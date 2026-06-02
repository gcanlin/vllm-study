# vLLM 现有融合算子走读

> **文档版本**: 1.0  
> **分析代码版本**: 当前 workspace 本地 `vllm` 源码  
> **最后更新**: 2026-06-01

---

## 文档概述

本文档讲 vLLM 推理热路径里已经在用的 **融合算子**。这里的“融合”不只指 torch.compile 里的 fusion pass，也包括 vLLM 手写 CUDA/Triton kernel、FlashAttention/FlashInfer/AITER/CUTLASS/Marlin 等后端 kernel，以及模型层把多个线性层打包成一个大 GEMM 的工程融合。

重点不是 fusion pass 框架怎么写，而是：

> **vLLM 到底融合了哪些算子，拆开看原本应该做什么，融合后为什么会快，以及这些算子在 Transformer 推理中处在哪个位置。**

**目标读者**: 不熟悉 GPU 算子和 kernel 的工程师，希望先从 vLLM 现有融合点理解“算子为什么能融合、融合收益从哪里来”。

**阅读指南**:

| 部分 | 内容 | 重点 |
|------|------|------|
| 第一部分 | 先建立算子直觉 | kernel launch、显存读写、中间张量为什么贵 |
| 第二部分 | Transformer block 的融合地图 | 一层模型里哪些位置最值得融合 |
| 第三部分 | Dense 模型常见融合 | QKV / gate-up 打包、激活、RMSNorm、RoPE、KV cache、Attention、采样 |
| 第四部分 | 量化相关融合 | dequant + GEMM、RMSNorm + quant、activation + quant、attention output quant |
| 第五部分 | MoE 相关融合 | routing、token 重排、expert GEMM、激活、top-k reduce |
| 第六部分 | 通信和 LoRA 融合 | all-reduce + RMSNorm、async TP、Punica / fused MoE LoRA |
| 第七部分 | 怎么判断一个融合有没有收益 | memory-bound / launch-bound / compute-bound 的判断 |
| 第八部分 | 源码索引 | 按融合类别列入口文件 |

---

# 第一部分: 先建立算子直觉

## 1.1 什么是“算子”

在 PyTorch 里你写：

```python
y = torch.nn.functional.silu(x1) * x2
```

从数学上看很简单，但在 GPU 上通常会变成几个动作：

```text
1. 从显存读 x1
2. 算 sigmoid(x1) * x1，写出临时张量 silu_x1
3. 从显存读 silu_x1 和 x2
4. 做乘法，写出 y
```

每个动作可能对应一次 kernel launch。kernel launch 有固定开销，中间张量还会多读写显存。推理 decode 阶段每步 token 数不大，很多算子本身很小，这时 launch overhead 和显存往返会明显影响 TPOT。

融合算子的核心就是把上面的多步变成一个 kernel：

```text
1. 从显存读 x1、x2
2. 在寄存器里算 silu(x1) * x2
3. 只写一次 y
```

收益主要来自四件事：

| 收益来源 | 解释 |
|----------|------|
| 少启动 kernel | 多个小 kernel 合成一个，CPU 调度和 GPU launch 开销下降 |
| 少读写显存 | 中间结果不落显存，直接在 register / shared memory 里消费 |
| 更好利用缓存 | 一个 token / 一行 hidden 读进来后连续做完多步 |
| 更少同步边界 | 原本两个 kernel 之间隐含有依赖，融合后在一个 kernel 内部完成 |

## 1.2 不是所有融合都一定快

如果一个算子本来就是大 GEMM，GPU 已经跑满 Tensor Core，强行把复杂逻辑塞进去可能会降低矩阵乘吞吐。vLLM 里的融合通常集中在这些位置：

1. **GEMM 前后的小算子**：量化、反量化、bias、scale、activation。
2. **attention 里的固定流水**：QK、scale、mask、softmax、PV。
3. **KV cache 更新**：reshape、layout 转换、量化、scatter 写 cache。
4. **MoE 的稀疏调度**：top-k、token 按 expert 重排、专家 GEMM、top-k 加权求和。
5. **残差和 Norm**：residual add + RMSNorm 经常连续出现，且 memory-bound。

---

# 第二部分: Transformer block 的融合地图

先用 Llama 类 decoder block 的推理路径建立地图。简化后大概是：

```text
hidden_states
  │
  ├─ input_layernorm
  │
  ├─ qkv_proj                         # Q/K/V 三个线性层常被打包
  │    └─ split q, k, v
  │
  ├─ RoPE(q, k)                       # Q/K rotary position embedding
  │
  ├─ KV cache update                  # reshape + 写入 paged cache，可带量化
  │
  ├─ attention                        # QK^T + mask + softmax + V
  │
  ├─ o_proj
  │
  ├─ residual add + post_attention_rmsnorm
  │
  ├─ gate_up_proj                     # gate_proj/up_proj 两个线性层常被打包
  │
  ├─ activation                       # silu(gate) * up
  │
  ├─ down_proj
  │
  └─ residual add + final_rmsnorm
```

如果是 MoE block，`gate_up_proj -> activation -> down_proj` 会换成：

```text
router logits
  ├─ top-k experts
  ├─ token 按 expert 重排 / padding
  ├─ expert w1/w3 GEMM
  ├─ gated activation
  ├─ expert w2 GEMM
  └─ top-k 权重加权 + reduce 回原 token 顺序
```

vLLM 的融合基本就围绕这些区域展开。

---

# 第三部分: Dense 模型常见融合

## 3.1 QKV 打包: 三个 Linear 合成一个大 Linear

源码入口：

```text
vllm/model_executor/layers/linear.py::QKVParallelLinear
vllm/model_executor/layers/linear.py::MergedColumnParallelLinear
```

### 拆开是什么

标准 attention 里会有三个投影：

```python
q = x @ Wq
k = x @ Wk
v = x @ Wv
```

如果直接拆开跑，就是三次 GEMM，三次读 `x`，三次 kernel 调度，最后还要把结果组织成 Q/K/V。

### vLLM 怎么融合

`QKVParallelLinear` 把 Q、K、V 的 weight 沿输出维拼在一起：

```text
Wqkv = concat([Wq, Wk, Wv], dim=out_features)
qkv = x @ Wqkv
q, k, v = split(qkv)
```

源码里 `QKVParallelLinear` 明确说明 weight matrix 沿 output dimension 拼接，并处理 GQA/MQA 下 K/V head 数比 Q 少的情况。`MergedColumnParallelLinear` 则是更通用的“多个 column parallel linear 拼成一个”的层，MLP 里的 gate/up 也会用到它。

### 为什么有收益

| 拆开 | 打包后 |
------|--------|
| 读 3 次输入 `x` | 读 1 次输入 `x` |
| 3 次 GEMM launch | 1 次 GEMM launch |
| 3 份输出调度 | 1 份连续输出，后面 split view 即可 |
| TP shard 逻辑重复 3 次 | 一次按拼接后输出维切 shard |

这里的融合不是把矩阵乘内部改了，而是把多个同输入的矩阵乘变成一个更大的矩阵乘。大 GEMM 更容易喂满 Tensor Core，launch overhead 也少。

## 3.2 gate/up 打包 + SwiGLU

源码入口：

```text
vllm/model_executor/layers/linear.py::MergedColumnParallelLinear
vllm/model_executor/layers/activation.py::SiluAndMul
vllm/model_executor/layers/activation.py::SiluAndMulWithClamp
vllm/model_executor/layers/activation.py::MulAndSilu
vllm/model_executor/layers/activation.py::FatreluAndMul
```

### 拆开是什么

Llama / Qwen 这类 MLP 常见形式是 SwiGLU：

```python
gate = x @ W_gate
up = x @ W_up
hidden = silu(gate) * up
out = hidden @ W_down
```

注意 `gate` 和 `up` 输入一样，所以第一步也适合打包：

```text
gate_up = x @ concat([W_gate, W_up])
gate, up = split(gate_up)
hidden = silu(gate) * up
```

### vLLM 怎么融合

1. `MergedColumnParallelLinear` 把 `gate_proj` 和 `up_proj` 拼成一个 GEMM。
2. `SiluAndMul` 用一个 custom op 完成 `silu(x[..., :d]) * x[..., d:]`。
3. 有些模型变体使用 `MulAndSilu`、`SiluAndMulWithClamp`、`FatreluAndMul`，本质都是“把 gate 激活和逐元素乘法合在一个 kernel 里”。

`SiluAndMul` 的 native 逻辑非常直观：

```python
d = x.shape[-1] // 2
return F.silu(x[..., :d]) * x[..., d:]
```

CUDA/XPU 路径则调用 `torch.ops._C.silu_and_mul(out, x)`，输出宽度是输入的一半。

### 为什么有收益

`silu(gate) * up` 是典型 memory-bound elementwise 链：

```text
拆开:
  gate -> sigmoid -> silu_gate 临时张量
  silu_gate + up -> hidden

融合:
  读 gate/up
  register 内算 sigmoid、乘法
  写 hidden
```

这类算子没有 Tensor Core 大矩阵乘那么重，瓶颈常是显存带宽和 launch 开销，所以融合收益很直接。

## 3.3 RMSNorm + residual add

源码入口：

```text
vllm/model_executor/layers/layernorm.py::RMSNorm
vllm/kernels/vllm_c.py::rms_norm
vllm/kernels/vllm_c.py::fused_add_rms_norm
vllm/_custom_ops.py::rms_norm
vllm/_custom_ops.py::fused_add_rms_norm
```

### RMSNorm 拆开看

RMSNorm 公式是：

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

也就是：

```text
1. 对 hidden 维度求平方和
2. 除以 hidden size，开方
3. x 乘以倒数 rms
4. 再乘 learned weight
```

如果 block 有残差，一般还会先做：

```python
x = x + residual
residual = x
y = rms_norm(x)
```

### vLLM 怎么融合

`RMSNorm.forward_native()` 在有 residual 时走：

```python
ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, eps)
```

CUDA-like 平台会落到 `torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)`。这个 op 是 inplace：更新 `x` 为 norm 后输出，同时更新 `residual` 为 add 后的残差。

### 为什么有收益

拆开时至少会有两次显存写：

```text
residual_add_out = x + residual
norm_out = rms_norm(residual_add_out)
```

融合后可以在同一个 kernel 里：

```text
读 x/residual
算 sum = x + residual
用 sum 做 RMSNorm
写 norm_out，同时保留 residual=sum
```

对 hidden size 比较大的模型，RMSNorm 本身需要读完整行 hidden 做 reduction。把 residual add 合进去，可以避免 add 结果单独落显存。

## 3.4 RoPE: Q/K 一起旋转

源码入口：

```text
vllm/model_executor/layers/rotary_embedding/base.py::RotaryEmbedding
vllm/_custom_ops.py::rotary_embedding
vllm/_custom_ops.py::fused_qk_norm_rope
vllm/compilation/passes/fusion/qk_norm_rope_fusion.py
```

### RoPE 拆开看

RoPE 会对 Q/K 的前 `rotary_dim` 个维度做二维旋转。每两个维度一组：

```text
x0' = x0 * cos - x1 * sin
x1' = x1 * cos + x0 * sin
```

对每个 token、每个 head 都要按 position 取对应的 cos/sin。

### vLLM 怎么融合

普通路径里，`RotaryEmbedding.forward_cuda()` 调 `ops.rotary_embedding(positions, query, key, ...)`，这个 custom op 同时处理 query 和 key，并且是 inplace 修改。

对带 Q/K Norm 的 MLA / DeepSeek 类模型，vLLM 还有 `fused_qk_norm_rope`：

```text
qkv
  ├─ 对 q 做 RMSNorm
  ├─ 对 k 做 RMSNorm
  └─ 对 q/k 做 RoPE
```

这个融合点把 Q/K norm 和 RoPE 放在同一个 custom op 中，避免 norm 后的 Q/K 中间结果反复读写。

### 为什么有收益

RoPE 是逐元素变换，但它访问模式比较固定：

```text
读 q/k
读 position 对应 cos/sin
算旋转
写回 q/k
```

Q 和 K 都要做同样的 position lookup。合在一个 kernel 里可以减少 launch，并且让 Q/K 的 shape/layout 处理只做一次。对 Q/K Norm + RoPE，收益更明显，因为 norm 需要 reduction，RoPE 需要 elementwise，拆开会产生 norm 后中间张量。

## 3.5 RoPE + KV cache update

源码入口：

```text
vllm/compilation/passes/fusion/rope_kvcache_fusion.py
vllm/v1/attention/backend.py::fused_rope_kvcache_supported
vllm/v1/attention/backend.py::do_rope_and_kv_cache_update
vllm/model_executor/layers/attention/attention.py::unified_kv_cache_update
```

### 拆开是什么

decode / prefill 每层 attention 都会做两件连续的事：

```text
1. 对 q/k 做 RoPE
2. 把 k/v 写入 KV cache
```

KV cache 写入不是简单 copy。vLLM 的 paged KV cache 用 slot mapping 表示“这个 token 应该写到哪个 block 的哪个位置”，所以写 cache 通常包含：

```text
reshape key/value
根据 slot_mapping 找位置
按 cache layout 写入 key_cache/value_cache
如果 cache 是 FP8/NVFP4，还要量化和写 scale
```

### vLLM 怎么融合

attention backend 暴露了两个 hook：

```python
fused_rope_kvcache_supported()
do_rope_and_kv_cache_update(...)
```

如果 backend 支持，`torch.ops.vllm.fused_rope_and_unified_kv_cache_update` 会把 RoPE 和 KV cache update 合到一起。这样 K 做完 RoPE 后不必先写回一个普通 K tensor，再被 reshape/cache writer 重新读走。

### 为什么有收益

这个融合非常符合“生产者和消费者紧挨着”的模式：

```text
拆开:
  k -> RoPE -> k_rot 写回
  k_rot -> reshape/cache writer -> key_cache

融合:
  读 k
  做 RoPE
  直接写 key_cache
```

少一次 K 的显存往返，少一个 kernel launch。decode 阶段每步 token 少，KV cache update 本来就是小 kernel，launch 减少也有价值。

## 3.6 reshape_and_cache: reshape + scatter 写 KV cache

源码入口：

```text
vllm/v1/attention/ops/paged_attn.py::PagedAttention.write_to_paged_cache
vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.do_kv_cache_update
vllm/_custom_ops.py::reshape_and_cache
vllm/_custom_ops.py::reshape_and_cache_flash
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
```

### 拆开是什么

模型输出的 `key` / `value` 通常是：

```text
[num_tokens, num_kv_heads, head_dim]
```

KV cache 希望按 block 存：

```text
[num_blocks, block_size, num_kv_heads, head_dim]
```

所以 cache update 要做：

```text
1. 从 key/value 读一个 token 的一个 head
2. 根据 slot_mapping 算 block_id 和 block_offset
3. 按 backend 需要的 layout 写进 key_cache/value_cache
4. 如果 KV cache dtype 是 fp8，做 scale/quant
```

### vLLM 怎么融合

`PagedAttention.write_to_paged_cache()` 调 `ops.reshape_and_cache(...)`。FlashAttention backend 的 cache layout 不同，走 `reshape_and_cache_flash(...)`。

代码注释里也强调：`reshape_and_cache_flash` 使用 `slot_mapping` 的 shape 判断 actual tokens，所以 key/value 即便有 padding，也不需要 Python 侧先 slice 到 actual token 数。

### 为什么有收益

如果用普通 PyTorch 写，会涉及 `view`、`index`、`copy_`、可能还有 quant。vLLM 用一个 cache writer kernel 直接完成 layout 转换和 scatter 写入，避免多个小 op，也减少 Python 调度。

## 3.7 Attention: FlashAttention / PagedAttention 本身就是大融合

源码入口：

```text
vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward
vllm/v1/attention/ops/paged_attn.py::PagedAttention
vllm/_custom_ops.py::paged_attention_v1
vllm/_custom_ops.py::paged_attention_v2
vllm/vllm_flash_attn/flash_attn_interface.py
```

### 拆开是什么

标准 attention 数学是：

```python
scores = q @ k.transpose(-1, -2)
scores = scores * scale
scores = scores + mask
prob = softmax(scores)
out = prob @ v
```

如果真的把 `scores` 和 `prob` 全量存下来，显存量是：

```text
[batch, heads, query_len, kv_len]
```

长上下文下这非常大。推理 decode 虽然 query_len 小，但 kv_len 可能很长，而且 KV cache 是 paged/block layout。

### vLLM 怎么融合

vLLM 通过不同 backend 选择不同 attention kernel：

| 场景 | 常见 backend | 融合内容 |
------|--------------|----------|
| prefill / chunked prefill | FlashAttention | QK、scale/mask、online softmax、PV |
| decode + paged KV | PagedAttention / FlashAttention paged path / FlashInfer / AITER | 按 block table 读 KV、QK、softmax、PV |
| cascade attention | `merge_attn_states` | prefix/suffix attention 结果按 LSE 合并 |

FlashAttentionImpl.forward 最终调用 `flash_attn_varlen_func(...)`，传入 Q、KV cache、block table、seq_lens、sliding window、softcap、scale 等。PagedAttention 则调用 `_C.paged_attention_v1/v2` 这类 custom op。

### 为什么有收益

Attention 的融合收益不是只少 launch，而是算法级别少存中间矩阵：

```text
普通 attention:
  写 scores 矩阵
  读 scores 做 softmax，写 prob
  读 prob 和 V 算 out

Flash/Paged attention:
  分块读 K/V
  在 SRAM/register 内维护 max、sum、局部 out
  最后只写 out
```

这就是为什么 attention kernel 往往是推理框架最核心的融合点。它把 `QK^T + softmax + V` 合成一个流式 kernel，避免 `[query_len, kv_len]` 大中间张量落显存。

## 3.8 logits processor / sampler 融合

源码入口：

```text
vllm/v1/sample/sampler.py::Sampler
vllm/v1/sample/ops/topk_topp_sampler.py::TopKTopPSampler
vllm/_custom_ops.py::apply_repetition_penalties
```

### 拆开是什么

采样阶段一般是：

```text
logits
  ├─ repetition / frequency / presence penalty
  ├─ temperature
  ├─ top-k mask
  ├─ top-p mask
  ├─ softmax
  └─ random sample
```

这些操作作用在 vocab 维度。vocab 很大，但 batch 可能不大，很多操作都是小 kernel 或需要排序。

### vLLM 怎么融合

1. repetition penalty 有 CUDA inplace custom op：`apply_repetition_penalties_`。
2. top-k/top-p 在 CUDA 上可用 FlashInfer sampler，或者批量足够大时走 Triton `apply_top_k_top_p_triton`。
3. XPU 有 `xpu_topk_topp_sampler`，把 top-k/top-p 和 sampling 放到一个 backend kernel 里。
4. PyTorch fallback 里，vLLM 也尽量 inplace 修改 logits，避免额外张量。

### 为什么有收益

采样不是 Transformer block 里的大头，但 decode 每步都要做。top-p 如果用普通 sort，会对整个 vocab 排序；FlashInfer 采样用 rejection sampling 等方式减少排序成本。vLLM 自己的 `random_sample` 也避免了 `torch.multinomial` 导致的 CPU-GPU 同步。

---

# 第四部分: 量化相关融合

量化模型里，融合更重要。因为量化引入了额外步骤：

```text
float activation -> quant activation
quant weight -> dequant 或 scaled GEMM
输出可能还要 quant 给下一层
```

如果这些步骤拆开，会把“省显存带宽”的收益又吃掉一部分。

## 4.1 dequant + GEMM: AWQ/GPTQ/Marlin/CUTLASS/Machete

源码入口：

```text
vllm/_custom_ops.py::awq_gemm
vllm/_custom_ops.py::gptq_gemm
vllm/model_executor/kernels/linear/mixed_precision/marlin.py
vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
vllm/model_executor/kernels/linear/scaled_mm/*
vllm/model_executor/kernels/linear/nvfp4/*
vllm/model_executor/kernels/linear/mxfp4/*
```

### 拆开是什么

以 4-bit weight-only 量化为例，数学上相当于：

```text
W_fp16 = dequant(W_int4, scale, zero_point)
Y = X_fp16 @ W_fp16
```

如果真的先 dequant 整个 weight，再 GEMM，会产生很大的临时 `W_fp16`，显存和带宽都会爆。

### vLLM 怎么融合

量化 linear kernel 直接读 packed weight 和 scale，在 GEMM 内部反量化：

```text
读 packed int4/int8/fp8 weight
读 scale / zero point
在寄存器中还原成计算 dtype
做 dot
写 output
```

Marlin 这类 kernel 还要求权重加载后预处理布局，把 weight/scale 变成 kernel 更容易连续读取的格式。`MarlinLinearKernel.process_weights_after_loading()` 里会 repack weight、permute scales、准备 workspace。

CUTLASS scaled MM 则走 `ops.cutlass_scaled_mm(...)` / `ops.cutlass_scaled_mm_azp(...)`，把量化输入、量化权重、scale、bias 都交给一个 GEMM kernel。

### 为什么有收益

这类融合的收益非常大：

| 拆开 | 融合后 |
------|--------|
| 先展开大 weight 到 fp16/bf16 | packed weight 常驻，边读边反量化 |
| dequant 写完整临时矩阵 | dequant 值只在寄存器 / fragment 中存在 |
| dequant kernel + GEMM kernel | 一个专用 GEMM kernel |
| 显存带宽高 | 用更少 bit 读 weight，scale 小得多 |

这也是量化推理能快的关键：不是只把权重存小，还要避免运行时把它完整展开回来。

## 4.2 input quant + scaled GEMM

源码入口：

```text
vllm/model_executor/kernels/linear/scaled_mm/cutlass.py::CutlassInt8ScaledMMLinearKernel
vllm/model_executor/layers/quantization/input_quant_fp8.py
vllm/_custom_ops.py::scaled_int8_quant
vllm/_custom_ops.py::cutlass_scaled_mm
```

### 拆开是什么

W8A8 / FP8 动态量化会先把 activation 量化：

```text
x_scale = max(abs(x)) / range
x_q = round(x / x_scale)
y = scaled_mm(x_q, w_q, x_scale, w_scale)
```

### vLLM 怎么融合

对 int8 CUTLASS 路径，`apply_weights()` 先调用 `ops.scaled_int8_quant(x, ...)`，再调用 `ops.cutlass_scaled_mm(...)`。严格说这是两个 kernel，不是完全融合成一个，但它们是为量化 GEMM 成对设计的：前者输出 quantized activation 和 scale，后者在 GEMM epilogue 里应用 scale/bias。

在 torch.compile pass 场景，vLLM 还会把前面的 RMSNorm 或 activation 与 quant 合并，进一步减少中间写出。

## 4.3 RMSNorm + quant

源码入口：

```text
vllm/_custom_ops.py::rms_norm_dynamic_per_token_quant
vllm/_custom_ops.py::rms_norm_per_block_quant
vllm/compilation/passes/fusion/rms_quant_fusion.py
vllm/compilation/passes/fusion/rocm_aiter_fusion.py
```

### 拆开是什么

量化模型里常见链路：

```text
norm_out = rms_norm(x)
norm_q, scale = quant(norm_out)
next = scaled_mm(norm_q, w_q, scale, w_scale)
```

### vLLM 怎么融合

vLLM 提供：

```text
rms_norm_dynamic_per_token_quant
rms_norm_per_block_quant
```

它们在一个 kernel 里完成：

```text
RMSNorm
计算 per-token 或 per-block scale
把 norm 后结果量化成 fp8/int8
输出 quant tensor + scale
```

还支持带 residual 的变体，通过参数把 residual 一起传进去。

### 为什么有收益

RMSNorm 本来要完整读写 hidden。quant 也要完整扫描 hidden 取 max/scale。融合后可以在一次遍历中把 norm 后的值直接量化，避免 `norm_out` 这个 fp16/bf16 中间张量落显存。

## 4.4 activation + quant

源码入口：

```text
vllm/_custom_ops.py::silu_and_mul_per_block_quant
vllm/compilation/passes/fusion/act_quant_fusion.py
vllm/compilation/passes/fusion/rocm_aiter_fusion.py
vllm/kernels/helion/ops/silu_mul_fp8.py
```

### 拆开是什么

MLP 里常见：

```text
hidden = silu(gate) * up
hidden_q, scale = quant(hidden)
down = scaled_mm(hidden_q, w_down_q, scale, w_scale)
```

### vLLM 怎么融合

`silu_and_mul_per_block_quant` 把两个动作合起来：

```text
读 gate/up
算 silu(gate) * up
按 block/group 求 scale
写 quantized hidden 和 scale
```

ROCm AITER 也有 `act_mul_and_fp8_group_quant`；Helion 路径有 `silu_mul_fp8`。

### 为什么有收益

`silu_and_mul` 输出宽度是输入一半，但仍然是一个大 hidden tensor。如果下一步马上量化它，单独写 fp16 hidden 再读回来量化很浪费。融合后只写最终 quantized tensor 和 scale。

## 4.5 attention output + quant

源码入口：

```text
vllm/v1/attention/backend.py::fused_output_quant_supported
vllm/compilation/passes/fusion/attn_quant_fusion.py
vllm/compilation/passes/fusion/mla_attn_quant_fusion.py
```

### 拆开是什么

attention 输出后如果下一层 linear 需要量化输入，会有：

```text
attn_out = attention(q, k, v)
attn_q, scale = quant(attn_out)
o = scaled_mm(attn_q, o_proj_weight)
```

### vLLM 怎么融合

attention backend 可以声明自己是否支持 fused output quant。支持时，fusion pass 会把 attention 输出量化并入 attention backend 的输出路径。

当前并不是所有 backend 都支持。例如 FlashAttentionImpl 里明确对 `output_scale` / `output_block_scale` 抛 `NotImplementedError`，说明这条融合依赖具体 backend 能力。

### 为什么有收益

attention output 是 `[tokens, heads, head_dim]`，紧接着通常进入 `o_proj`。如果能直接产出量化后的 output，就能少一次 fp16/bf16 output 的显存往返。

---

# 第五部分: MoE 相关融合

MoE 比 dense MLP 更依赖融合，因为它除了矩阵乘，还有 routing 和 token 重排。

## 5.1 FusedMoE 层把专家 MLP 当成一个整体

源码入口：

```text
vllm/model_executor/layers/fused_moe/layer.py::FusedMoE
vllm/model_executor/layers/fused_moe/fused_moe.py
vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py
vllm/model_executor/layers/fused_moe/modular_kernel.py
vllm/model_executor/layers/fused_moe/experts/*
```

### 拆开是什么

MoE 层数学上是：

```text
router_logits = gate(x)
topk_weights, topk_ids = topk(router_logits)

for token in tokens:
  for expert in topk_ids[token]:
    y[token] += topk_weight * expert_mlp[expert](x[token])
```

expert MLP 通常又是：

```text
expert_mlp(x) = w2( activation(w1(x), w3(x)) )
```

如果按 token 循环，会非常慢；如果按 expert 批处理，就要把 token 重排到对应 expert。

### vLLM 怎么融合

`FusedMoE` 层包含：

| 阶段 | vLLM 中的角色 |
------|---------------|
| router | `create_fused_moe_router(...)` |
| prepare/finalize | `prepare_finalize/*` |
| experts GEMM | `experts/*`，如 Triton/CUTLASS/DeepGEMM/FlashInfer/AITER/TRTLLM |
| top-k 权重和 reduce | `topk_weight_and_reduce.py` |
| shared experts | `runner/shared_experts.py` |

`FusedMoEModularMethod.apply()` 把 hidden states、w1/w2、topk_weights、topk_ids、activation、expert_map、shared_experts 一起交给 `FusedMoEKernel.apply()`，让 kernel 组合选择如何准备 token、跑专家、合并输出。

### 为什么有收益

MoE 的融合收益来自三个层面：

1. **把 token 按 expert 聚合**：同一个 expert 的 token 批量 GEMM，避免 token-by-token 小矩阵乘。
2. **把 gate/up/down 与 activation 串起来**：专家内部仍然是 MLP，可以复用 dense MLP 的融合思想。
3. **把 top-k 加权和 reduce 合并到 finalize**：专家输出形状常是 `[num_tokens, top_k, hidden]`，最终要乘 router weight 再沿 top-k 求和。

## 5.2 token 重排 + expert GEMM

源码入口：

```text
vllm/model_executor/layers/fused_moe/moe_align_block_size.py
vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py
vllm/model_executor/layers/fused_moe/experts/triton_moe.py
vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
```

### 拆开是什么

top-k 后，每个 token 会被送到多个 expert。为了让 GEMM 高效，需要把 token 从原顺序：

```text
token0 -> expert 3
token1 -> expert 1
token2 -> expert 3
token3 -> expert 7
```

变成按 expert 聚集：

```text
expert 1: token1
expert 3: token0, token2
expert 7: token3
```

还要 padding 到 block size，方便 Triton/CUTLASS kernel 做规整 tile。

### vLLM 怎么融合

`fused_moe_kernel_gptq_awq` 这类 Triton kernel 通过 `sorted_token_ids` 和 `expert_ids` 决定当前 program block 应该处理哪个 expert 的哪些 token。kernel 内部直接根据 expert id 读对应 expert weight，根据 token id 读 hidden。

这避免了为每个 expert 单独切 tensor、单独 launch GEMM 的 Python 级循环。

## 5.3 top-k weight + reduce

源码入口：

```text
vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py
vllm/_custom_ops.py::moe_sum
```

### 拆开是什么

专家输出如果是：

```text
expert_out: [num_tokens, top_k, hidden]
topk_weights: [num_tokens, top_k]
```

最终输出是：

```python
out[token] = sum_j expert_out[token, j, :] * topk_weights[token, j]
```

### vLLM 怎么融合

`TopKWeightAndReduceContiguous.apply()` 先 inplace 乘 topk weight，然后调用 `ops.moe_sum(fused_expert_output, output)` 做 top-k 维度求和。

有些 prepare/finalize backend 会把 weight application 和 reduction 已经做掉，这时 `TopKWeightAndReduceNoOP` 直接返回。

### 为什么有收益

这个阶段纯 memory-bound。如果拆成：

```text
mul kernel
sum kernel
copy/scatter kernel
```

会多次读写 `[num_tokens, top_k, hidden]`。能在 finalize 或专门 reduce kernel 里合并，就能少写中间结果。

## 5.4 ROCm AITER fused MoE / shared experts fusion

源码入口：

```text
vllm/_aiter_ops.py::rocm_aiter_fused_moe
vllm/_aiter_ops.py::rocm_aiter_fused_topk
vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py
```

ROCm AITER 路径额外提供 fused MoE、fused top-k、shared experts fusion 等能力。`FusedMoE` 初始化时会根据：

```text
rocm_aiter_ops.is_fused_moe_enabled()
rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
```

决定是否启用 AITER 的 MoE 融合。

shared experts fusion 的意义是：有些模型既有 routed experts，也有 shared experts。如果分开算，需要 routed MoE 一套路径、shared expert 一套 MLP，再相加。AITER 支持时可以把 shared expert 的部分合进 MoE 路径，减少额外 GEMM/activation/加法开销。

---

# 第六部分: 通信和 LoRA 融合

## 6.1 all-reduce + RMSNorm

源码入口：

```text
vllm/compilation/passes/fusion/allreduce_rms_fusion.py
vllm/_aiter_ops.py::rocm_aiter_fused_allreduce_rmsnorm
vllm/compilation/passes/pass_manager.py
```

### 拆开是什么

Tensor Parallel 下，某些 linear 输出需要 all-reduce：

```text
partial = local_linear(x)
full = all_reduce(partial)
out = rms_norm(full + residual)
```

### vLLM 怎么融合

编译 pass 可以把 all-reduce 和后续 RMSNorm 组合。ROCm AITER 还有 `fused_ar_rms` 对应的 fused allreduce rmsnorm op。

### 为什么有收益

all-reduce 是通信，RMSNorm 是本地显存读写。融合的目标不是让通信消失，而是减少通信结果落地后的额外搬运，并在可能时把后续 norm 的读写与通信输出衔接起来。

## 6.2 GEMM + communication overlap / async TP

源码入口：

```text
vllm/compilation/passes/fusion/collective_fusion.py::AsyncTPPass
vllm/compilation/passes/pass_manager.py
```

这类更接近调度融合：把 tensor parallel 的 GEMM 与 collective 通信重排或异步化，让通信和计算重叠。它不是“两个数学算子变成一个 CUDA kernel”，但推理性能收益来源类似：减少等待边界。

## 6.3 LoRA: shrink / expand / fused MoE LoRA

源码入口：

```text
vllm/lora/punica_wrapper/punica_gpu.py
vllm/lora/ops/triton_ops/lora_shrink_op.py
vllm/lora/ops/triton_ops/lora_expand_op.py
vllm/lora/ops/triton_ops/fused_moe_lora_op.py
vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py
```

LoRA 的数学是：

```text
y = base_linear(x) + B(A(x)) * scale
```

多 LoRA serving 难点是：一个 batch 内不同请求可能挂不同 LoRA adapter。Punica 路径用 `sampler_indices` / `prompt_mapping` 把请求映射到 adapter，使用 batched gather-matvec kernel 做 shrink/expand，避免对每个请求或每个 adapter 单独 launch。

MoE + LoRA 场景还有 `fused_moe_lora`，把专家路径和 LoRA 路径组合，减少 routed expert 与 adapter 更新之间的额外搬运。

---

# 第七部分: 怎么判断一个融合有没有收益

## 7.1 看它是不是 memory-bound

这类最适合融合：

```text
elementwise: silu, mul, clamp, bias
reduction + elementwise: RMSNorm
layout/scatter: reshape_and_cache
small-vocab 操作: penalties / sampling mask
```

因为它们算术量不大，主要花在读写显存和 launch 上。

## 7.2 看中间张量是不是马上被消费

典型可融合链：

```text
RMSNorm -> quant -> scaled GEMM
silu_and_mul -> quant -> down_proj
RoPE(k) -> reshape_and_cache(k)
attention -> output quant
expert_out -> topk weight -> reduce
```

如果中间张量只被下一个算子消费，且没有复杂分支，融合通常值得做。

## 7.3 看 batch/token 规模

decode 阶段每步 token 少，小 kernel launch overhead 占比高，所以小算子融合更值钱。prefill 阶段 token 多，大 GEMM/attention 更容易跑满，这时收益更多来自算法级融合，比如 FlashAttention 避免存 scores/probs。

## 7.4 看 backend 支持

vLLM 很多融合不是所有设备都有：

| 融合 | 依赖 |
------|------|
| FlashAttention | CUDA capability、FA2/FA3/FA4 支持、head size、dtype |
| FlashInfer sampler / attention | FlashInfer 版本和 GPU capability |
| AITER fused MoE / Norm / RoPE | ROCm + AITER 环境变量和能力探测 |
| Marlin/Machete | NVIDIA CUDA、特定 compute capability、weight layout |
| CUTLASS FP8/FP4 | CUDA capability 和 CUTLASS kernel 支持 |

所以看“vLLM 有这个融合”时，要同时看当前平台、dtype、量化格式、head size、block size 是否满足条件。

---

# 第八部分: 源码索引

## 8.1 按 Transformer 位置索引

| 位置 | 融合点 | 主要源码 |
|------|--------|----------|
| QKV projection | Q/K/V 三个 Linear 打包 | `vllm/model_executor/layers/linear.py::QKVParallelLinear` |
| MLP gate/up | gate/up 两个 Linear 打包 | `vllm/model_executor/layers/linear.py::MergedColumnParallelLinear` |
| SwiGLU | `silu(gate) * up` | `vllm/model_executor/layers/activation.py::SiluAndMul` |
| RMSNorm | RMSNorm custom kernel | `vllm/model_executor/layers/layernorm.py::RMSNorm`、`vllm/kernels/vllm_c.py` |
| Residual + RMSNorm | `x + residual` 和 RMSNorm | `torch.ops._C.fused_add_rms_norm` |
| RoPE | Q/K 同时 rotary | `vllm/model_executor/layers/rotary_embedding/base.py`、`vllm/_custom_ops.py::rotary_embedding` |
| QK Norm + RoPE | Q/K RMSNorm + RoPE | `vllm/_custom_ops.py::fused_qk_norm_rope`、`qk_norm_rope_fusion.py` |
| KV cache | reshape + scatter cache write | `reshape_and_cache`、`reshape_and_cache_flash` |
| Attention | QK + mask + softmax + V | `flash_attn.py`、`paged_attn.py`、`_custom_ops.py::paged_attention_v1/v2` |
| Sampler | top-k/top-p + sampling | `vllm/v1/sample/ops/topk_topp_sampler.py` |

## 8.2 按量化索引

| 融合点 | 主要源码 |
|--------|----------|
| AWQ/GPTQ GEMM | `vllm/_custom_ops.py::awq_gemm`、`gptq_gemm` |
| Marlin weight-only GEMM | `vllm/model_executor/kernels/linear/mixed_precision/marlin.py` |
| CUTLASS scaled MM | `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py` |
| FP8/NVFP4/MXFP4 linear | `vllm/model_executor/kernels/linear/fp8|nvfp4|mxfp4` |
| RMSNorm + quant | `rms_norm_dynamic_per_token_quant`、`rms_norm_per_block_quant` |
| activation + quant | `silu_and_mul_per_block_quant`、`act_quant_fusion.py` |
| attention output + quant | `attn_quant_fusion.py`、`mla_attn_quant_fusion.py` |

## 8.3 按 MoE 索引

| 融合点 | 主要源码 |
|--------|----------|
| FusedMoE 层 | `vllm/model_executor/layers/fused_moe/layer.py` |
| modular fused MoE | `fused_moe_modular_method.py`、`modular_kernel.py` |
| token align / permute | `moe_align_block_size.py`、`moe_permute_unpermute.py` |
| expert kernels | `fused_moe/experts/*` |
| top-k weight reduce | `topk_weight_and_reduce.py`、`ops.moe_sum` |
| ROCm AITER MoE | `vllm/_aiter_ops.py` |

## 8.4 按编译 pass 名称索引

这里只列名字，本文不展开框架：

```text
RMSNormQuantFusionPass
ActivationQuantFusionPass
AttnQuantFusionPass
MLAAttnQuantFusionPass
QKNormRoPEFusionPass
RopeKVCacheFusionPass
MLARoPEKVCacheCatFusionPass
AllReduceFusionPass
AsyncTPPass
RocmAiterRMSNormQuantFusionPass
RocmAiterSiluMulFp8GroupQuantFusionPass
RocmAiterAllReduceFusionPass
MLADualRMSNormFusionPass
```

它们的注册顺序在 `vllm/compilation/passes/pass_manager.py`，大概说明 vLLM 当前重点优化的链路：Norm/Activation/Attention output quant、RoPE+KV cache、QK Norm+RoPE、AllReduce+RMSNorm、MoE/ROCm AITER 相关融合。

---

# 总结

vLLM 现有融合可以按收益来源理解：

| 类型 | 代表融合 | 核心收益 |
|------|----------|----------|
| 打包 GEMM | QKV、gate/up | 一个大 GEMM 替代多个小 GEMM |
| elementwise 融合 | SiLU+Mul、clamp+mul、penalty | 少 launch、少中间张量 |
| reduction 融合 | residual+RMSNorm、RMSNorm+quant | 一次读 hidden，顺手完成后续操作 |
| layout/cache 融合 | reshape_and_cache、RoPE+KV cache | 减少 reshape/scatter/copy 的显存往返 |
| attention 算法融合 | FlashAttention、PagedAttention | 不落地 scores/probs 大矩阵 |
| quant GEMM 融合 | Marlin、CUTLASS、AWQ/GPTQ | packed weight 边读边反量化，不展开 weight |
| MoE 融合 | FusedMoE、expert GEMM、top-k reduce | 稀疏 token 聚合成批量 GEMM，减少重排和 reduce 开销 |
| 通信融合 | all-reduce+RMSNorm、async TP | 减少通信后的等待和搬运 |

如果只记一个判断标准：**融合最值钱的地方，是“一个中间张量刚写完马上被下一个算子读走”的地方**。vLLM 的这些融合基本都在消灭这类中间张量，或者把多个小 kernel 变成一个更适合 GPU 执行的大 kernel。
