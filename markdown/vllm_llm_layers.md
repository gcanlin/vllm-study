# LLM 常用 Layer 精读（面试向）

> 一个累积型文档：每一节走一个 layer，写出**能在面试白板上手写的版本**。
> 数学先讲清楚，代码尽量短，每行一句注释，特殊点放在代码后面补充。

| 节 | Layer | 状态 |
|---|---|---|
| §1 | RMSNorm | ✅ |
| §2 | ParallelLMHead（TP Linear / Vocab Parallel） | ✅ |
| §3 | MergedColumnParallelLinear & SwiGLU MLP（gate_up_proj / down_proj / SiluAndMul） | ✅ |
| §4 | QKVParallelLinear（fused QKV + GQA/MQA 切分） | ✅ |

---

## §1 RMSNorm

### 1.1 数学

对每个 token 的 hidden 向量 `x ∈ ℝ^d`，做：

$$
y_i \;=\; w_i \cdot \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 \;+\; \varepsilon}}
$$

- `d = hidden_size`，归一化在**最后一维**（per-token，token 之间互不影响）。
- `w ∈ ℝ^d` 是可学习的 channel-wise 缩放，**所有 token 共享同一份 w**。
- `ε` 防 `sqrt(0)`，典型 `1e-6`。

直觉：分母是 `x` 这个向量的 **RMS（均方根）**。`x / RMS(x)` 把向量长度归一到 `√d`（当各分量方差为 1 时），方向不变；再乘 `w` 让每个 channel 有独立缩放。**本质是"向量长度归一化"**。

#### 和 LayerNorm 的区别

| 项 | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | `(x - μ)` | **不减** |
| 除标准差 | `σ = √Var(x)` | `σ = √mean(x²)` |
| 可学习参数 | `weight + bias` | **只有 weight** |
| reduction 次数 | 2（`μ` 一次，`var` 一次） | **1（`mean(x²)`）** |

**为什么 LLM 都用 RMSNorm？** Zhang & Sennrich 2019 的实验发现去掉减均值对 Transformer 训练稳定性几乎没影响。Pre-LN residual 流自带零均值倾向，re-centering 多余。代价小一半（少一次 reduction、少一个 bias 参数），结果一样好，于是 LLaMA / Qwen / Mistral 全跟进。

### 1.2 主体实现

#### 普通 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))   # (1)
        self.eps = eps                                        # (2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype                                  # (3)
        x = x.to(torch.float32)                               # (4)
        variance = x.pow(2).mean(dim=-1, keepdim=True)        # (5)
        x = x * torch.rsqrt(variance + self.eps)              # (6)
        x = x.to(orig_dtype) * self.weight                    # (7)
        return x
```

逐行：

- **(1)** `weight` 默认全 1，等价于 identity 缩放。模型 load 时会被 checkpoint 覆盖。
- **(2)** `eps` 是数值安全项，要加在 `sqrt` **里面**（`sqrt(var + eps)`），不能加外面——`var = 0` 的极端情形外面写法会先得到 `0/0`。
- **(3)(4)** **FP32 upcast**：模型一般是 BF16 / FP16，但 `pow(2).mean(...)` 在 `hidden=4096` 这种长度上累积误差会进有效位。统计量在 FP32 算，最后 cast 回去。这是数值精度的关键。
- **(5)** `pow(2).mean(dim=-1, keepdim=True)`：每个 token 单独算 `mean(x²)`。`keepdim=True` 让结果 shape 是 `(..., 1)`，后面广播自然对齐。
- **(6)** **`rsqrt` 而不是 `1/sqrt`**：硬件有专门的 `rsqrt` 指令（CUDA 上是 `__frsqrt_rn`），一条指令算完 reciprocal square root，比"先 sqrt 再除"快。
- **(7)** 先 cast 回 `orig_dtype` 再乘 weight，避免乘出 FP32 中间张量浪费显存带宽。

### 1.3 vLLM 真正用的版本：fused add + RMSNorm

普通 RMSNorm 在 Transformer 里只完成"归一化"。但实际 Pre-LN 的标准写法是：

```python
h = x + attn(rmsnorm(x))     # 一次 add + 一次 norm
```

里头 `x` 被读两次（一次给 norm，一次给残差加法），还要写出 `attn` 输出再加回去。vLLM 把这步改成 **add 和 norm 融合**：上一层算完 `attn(...)` 直接喂给下一层 norm，norm 先加再归一化，省一次 elementwise pass。

接口因此多一个 `residual` 参数：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is None:                                   # (1) 纯 norm 路径
            orig_dtype = x.dtype
            x = x.to(torch.float32)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (x.to(orig_dtype) * self.weight)

        # fused: out = norm(x + residual), new_residual = x + residual
        orig_dtype = x.dtype
        x = x.to(torch.float32) + residual.to(torch.float32)   # (2) 先加
        new_residual = x.to(orig_dtype)                        # (3) 未归一化的 sum
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        out = x.to(orig_dtype) * self.weight
        return out, new_residual                               # (4) 两个返回值
```

逐行：

- **(1)** 没传 residual：第一层入口（残差流还没建立）或独立使用的场景，走纯 RMSNorm。
- **(2)** 在 FP32 上加，避免 BF16 加法的精度损失。
- **(3)** **关键设计**：`x + residual` 的结果（**未归一化**）作为新 residual 返回。下一次 add+norm 会用它再加上下一次 sub-layer 输出。
- **(4)** 返回 `(归一化后的输出, 未归一化的 sum)`。两者语义完全不同：第一个会过 attn / mlp，第二个直接穿过去等下一次 add。

#### 在 Pre-LN Transformer block 里的用法

```python
def forward(self, x, residual):
    # ===== Attention =====
    if residual is None:
        residual = x
        x = self.input_layernorm(x)             # 纯 RMSNorm
    else:
        x, residual = self.input_layernorm(x, residual)   # fused
    x = self.self_attn(x)

    # ===== MLP =====
    x, residual = self.post_attention_layernorm(x, residual)  # 又一次 fused
    x = self.mlp(x)

    return x, residual                          # residual 流穿过这一层
```

注意 `self_attn` / `mlp` **不自己加 residual**——加法被推迟到下一次 norm 入口跟那次 norm 合并。

#### 收益

| 项 | 传统写法 | fused 写法 |
|---|---|---|
| 每层 kernel 数 | 4（2 add + 2 norm） | 2（fused add+norm × 2） |
| hidden 张量的 HBM pass | 多 2 次 | 0 冗余 |

LLM 36 层 × 2 norm 点 = 72 次冗余 elementwise pass 被消掉。decode 是 memory-bound 的，每少一次访存直接转成 token/s。

### 1.4 常见疑问

**Q1：为什么 reduction 在 FP32 做，乘 weight 又 cast 回 BF16？**
- reduction 累加误差敏感，FP32 必须；
- 乘 weight 是 channel-wise elementwise，BF16 精度足够，cast 回去省显存带宽（输出张量本来就是 BF16）。

**Q2：`weight` 默认初始化全 1 有什么用？**
- 推理框架 weight 一定从 checkpoint 加载，全 1 只是 placeholder；
- 训练时也常这么初始化——一开始 RMSNorm 等价于 identity，后续训练学出 channel-wise 缩放。

**Q3：能不能不要 weight？**
- 数学上可以，叫 "RMS scaling"，但表达力下降。weight 让每个 channel 有独立"重要性"，让后续 GEMM 学到的方向不依赖 channel 之间的尺度。

**Q4：`rsqrt(var + eps)` 和 `1 / sqrt(var + eps)` 数值上一样吗？**
- 数学一样，硬件不一样。`rsqrt` 是一条 fused 指令，`1 / sqrt` 是两条。差距在大 batch 下肉眼可见。

**Q5：QK Norm（每个 head 上单独 RMSNorm）是同一个类吗？**
- 是。只是 `hidden_size = head_dim`（如 128），作用维度从 hidden 变成 head_dim，所有 head 共享一份 `(128,)` 的 weight。

**Q6：为什么 norm 写在 sub-layer 前面（Pre-LN）而不是后面（Post-LN）？**
- Post-LN 的 `LN(x + sub(x))` 在深层 Transformer 训练时梯度爆炸 / 消失，需要 warmup。Pre-LN 的 `x + sub(LN(x))` 残差流是恒等映射，梯度直接穿，训练稳定。代价是末尾要补一次 final norm。LLaMA / Qwen / GPT-NeoX 全是 Pre-LN。

---

## §2 ParallelLMHead（TP Linear / Vocab Parallel）

`ParallelLMHead` 是 vLLM 里输出 logits 的并行词表头。它本身很轻：沿 vocab 维切权重，每个 TP rank 只算一段局部 logits，真正的 gather、padding 截断、scale / soft cap 等逻辑交给 `LogitsProcessor` 做。

为了讲清楚它，需要顺带理解 vLLM 里同一套 tensor-parallel layer 体系：

| layer | 切分维度 | 通信 | 用在哪 |
|---|---|---|---|
| `ColumnParallelLinear` | weight 的输出维 / column | 可选 all-gather | `qkv_proj`、`gate_up_proj` |
| `RowParallelLinear` | weight 的输入维 / row | all-reduce | `o_proj`、`down_proj` |
| `QKVParallelLinear` | Q/K/V 合并后按 head 切 | 无 gather | attention 输入投影 |
| `VocabParallelEmbedding` | vocab 维 | all-reduce | token embedding |
| `ParallelLMHead` | vocab 维 | logits gather 在 `LogitsProcessor` 做 | 输出 logits |

这些 layer 的共同思路是：**把一块大 Linear / Embedding 按 TP rank 切开，让每张卡只存一片权重、只算一片结果，再用一次 collective 把数学语义补回来。** `ParallelLMHead` 是其中沿 vocab 维切分的输出头。

### 2.1 为什么 LLM 需要 lm_head

decoder-only LLM 的主体 Transformer block 输出的是 hidden state：

$$
h_t \in \mathbb{R}^{H}
$$

它是第 `t` 个位置的语义表示，不是 token 概率。生成下一个 token 时，模型必须回答一个分类问题：

> 在 vocab 里的 `V` 个 token 中，下一个 token 最像哪一个？

所以需要一个从 hidden space 到 vocab space 的投影层：

$$
\text{logits}_t = h_t W_{lm}^{T} + b,\quad W_{lm} \in \mathbb{R}^{V \times H}
$$

这就是 `lm_head`。它的输出 shape 是：

$$
\text{logits}_t \in \mathbb{R}^{V}
$$

每个维度对应一个 token id 的未归一化分数。训练时，`logits` 进 cross entropy：

$$
\mathcal{L} = -\log \operatorname{softmax}(\text{logits}_t)[y_t]
$$

推理时，vLLM 不一定显式算完整 softmax，而是直接在 logits 上做 temperature、top-k、top-p、greedy / sampling 等操作。重点是：**Transformer 主体负责把上下文压成 hidden 表示，lm_head 负责把 hidden 表示翻译成“词表上的分数”。**

#### 为什么不直接让 Transformer 输出 vocab 维？

如果每层都维持 `[T, V]`，代价会非常大。`V` 通常是 32K、100K 甚至更高，而 `H` 一般是 4K、8K 量级。Transformer 内部需要反复做 attention、MLP、residual、norm，这些都应该在较紧凑的 hidden space 里完成。只有最后需要预测 token 时，才投影到 vocab space。

所以结构是：

```text
token_id -> embedding -> transformer blocks -> hidden_state -> lm_head -> logits
```

#### lm_head 和 embedding 的关系

输入侧 embedding 是：

$$
h_0 = E[\mathrm{token\_id}],\quad E \in \mathbb{R}^{V \times H}
$$

输出侧 lm_head 是：

$$
\text{logits} = h W_{lm}^{T}
$$

很多模型会做 **weight tying**：

$$
W_{lm} = E
$$

直觉是：输入 token 和输出 token 用同一套 token 向量空间。这样既省一份 `[V, H]` 的大权重，也通常不损失效果。vLLM 模型代码里常见：

```python
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens
else:
    self.lm_head = ParallelLMHead(...)
```

这也是为什么 `ParallelLMHead` 继承自 `VocabParallelEmbedding`：两者权重形状相同，都是 `[vocab, hidden]`，只是一个用于查表输入，一个用于输出投影。

#### 为什么 lm_head 在 vLLM 里尤其重要？

`lm_head` 往往是最后一个很大的矩阵。以 `V=151936, H=4096, BF16` 为例，单这一个权重大约：

$$
151936 \times 4096 \times 2 \approx 1.16\text{GB}
$$

如果每张卡都复制一份，显存浪费很明显；如果每步都 gather 完整 logits，通信也很贵。因此 vLLM 把它做成 `ParallelLMHead`：

- 权重沿 vocab 维切，每张卡只存 `V / TP`；
- 每张卡只算局部 logits `[T, V / TP]`；
- 真正需要采样时，再由 `LogitsProcessor` gather 或走局部 argmax 优化；
- PP 场景下只有最后一个 pipeline rank 需要 `lm_head`，非最后 rank 用 `PPMissingLayer` 占位。

### 2.2 数学：一块 Linear 怎么切

普通 Linear：

$$
Y = XW + b,\quad X \in \mathbb{R}^{T \times H},\ W \in \mathbb{R}^{H \times O}
$$

其中 `T = num_tokens`，`H = hidden_size`，`O = output_size`。TP 并行有两种基本切法。

#### Column parallel：切输出维

把 `W` 按输出维切成 `p` 片：

$$
W = [W_0, W_1, \dots, W_{p-1}]
$$

每个 rank 算：

$$
Y_i = XW_i
$$

`Y_i` 的 shape 是 `[T, O/p]`。如果下一层也能吃这个分片结果，就不 gather；如果后面需要完整 `Y`，再做：

$$
Y = \operatorname{concat}(Y_0, Y_1, \dots, Y_{p-1})
$$

直觉：**每张卡负责一部分输出 channel**。这适合 `qkv_proj`、`gate_up_proj`，因为它们后面通常还能继续沿最后一维分片计算。

#### Row parallel：切输入维

把 `X` 和 `W` 按输入维配套切开：

$$
X = [X_0, X_1, \dots, X_{p-1}],\quad
W =
\begin{bmatrix}
W_0 \\
W_1 \\
\dots \\
W_{p-1}
\end{bmatrix}
$$

每个 rank 算局部贡献：

$$
Z_i = X_iW_i
$$

完整输出是所有 rank 的和：

$$
Y = \sum_i Z_i
$$

所以 row parallel 的核心通信是 **all-reduce(sum)**。直觉：**每张卡只算输入的一部分对完整输出的贡献，最后把贡献加起来**。这适合 `o_proj`、`down_proj`，因为它们通常负责把前面分片的中间维度收回到完整 hidden。

### 2.3 最短可手写版本

下面是忽略量化、bias loader、PP 占位后的最小版本。

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_rank, tp_size):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.out_per_rank = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))

    def forward(self, x, gather_output=False):
        y_local = F.linear(x, self.weight)          # [T, O / TP]
        if gather_output:
            return all_gather_last_dim(y_local)     # [T, O]
        return y_local
```

逐行：

- `weight` 只保存当前 rank 的输出分片，显存直接除以 `tp_size`。
- `F.linear` 里 PyTorch weight 是 `[out, in]`，数学里的 `XW` 对应代码里的 `x @ weight.T`。
- `gather_output=False` 是 vLLM 的常态：能把分片张量继续往后传，就不要急着拼回完整张量。

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_rank, tp_size):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.in_per_rank = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))

    def forward(self, x, input_is_parallel=True):
        if input_is_parallel:
            x_local = x                              # [T, H / TP]
        else:
            x_local = split_last_dim(x, self.tp_size)[self.tp_rank]

        y_partial = F.linear(x_local, self.weight)   # [T, O]
        return all_reduce_sum(y_partial)             # [T, O]
```

逐行：

- `input_is_parallel=True` 表示上一层已经产出了 `[T, H/TP]`，不用再 split。
- 每个 rank 的 `y_partial` shape 都是完整 `[T, O]`，但数值只是局部贡献。
- `all_reduce_sum` 后，每个 rank 都拿到完整 `Y`，可以继续进入 replicated 的 residual / norm 路径。

### 2.4 vLLM 的典型组合：Column → Row

以 Qwen / LLaMA 的 MLP 为例，源码结构基本是：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
)

def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)   # [T, 2I / TP]
    x = silu(gate) * up                 # [T, I / TP]
    x, _ = self.down_proj(x)            # all-reduce -> [T, H]
    return x
```

这里有三个关键点：

1. `gate_proj` 和 `up_proj` 被 `MergedColumnParallelLinear` 合成一次大 GEMM，输出 `[gate, up]` 拼在一起，少一次 kernel launch。
2. 激活函数 `SiluAndMul` 在分片的 intermediate 维度上做，不需要通信。
3. `down_proj` 是 `RowParallelLinear`，负责把分片 intermediate 的贡献 reduce 回完整 hidden。

attention 也是同一套思路：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
```

`qkv_proj` 按 head 切，当前 rank 只生成自己负责的 Q/K/V heads；attention 在本 rank 的 heads 上算；`o_proj` 再通过 row parallel all-reduce 回完整 hidden。

### 2.5 `QKVParallelLinear`：为什么它比普通 Column 稍复杂

QKV 本质还是 column parallel，只是有两个额外细节：

```python
output_size =
    local_q_heads * head_dim * tp_size
  + local_kv_heads * head_dim * tp_size
  + local_kv_heads * v_head_dim * tp_size
```

vLLM 需要处理 GQA / MQA：

- Q heads 通常很多，可以平均分给 TP ranks；
- KV heads 可能比 Q heads 少；
- 当 `tp_size >= total_num_kv_heads` 时，KV heads 不能继续切，只能在多个 rank 上复制；
- 当 `tp_size < total_num_kv_heads` 时，KV heads 才按 TP 切。

所以 `QKVParallelLinear` 不是简单把 `3 * hidden_size` 一刀切，它要知道 `total_num_heads`、`total_num_kv_heads`、`head_size`，按 head 语义切权重。这样后面的 attention backend 才能看到正确的本地 head 数。

### 2.6 Vocab parallel：embedding 和 LM head 怎么切

词表维度也很大，尤其输出 `lm_head` 的权重 shape 是：

$$
W_{lm} \in \mathbb{R}^{V \times H}
$$

`VocabParallelEmbedding` 和 `ParallelLMHead` 都沿 vocab 维切：

```python
class VocabParallelEmbedding(nn.Module):
    def forward(self, input_ids):
        masked_input, input_mask = build_vocab_mask(input_ids, local_vocab_range)
        out_local = embedding(masked_input, self.weight)   # [T, H]
        out_local.masked_fill_(input_mask[..., None], 0)
        return all_reduce_sum(out_local)                   # [T, H]
```

为什么 embedding 用 all-reduce？

- 每个 token id 只属于一个 rank 的 vocab shard；
- 不属于本 rank 的 token 被 mask 掉，输出置 0；
- 所有 rank 相加后，刚好只留下真正那一份 embedding。

`ParallelLMHead` 继承 `VocabParallelEmbedding`，但它的 `forward()` 故意报错：

```python
def forward(self, input_):
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

原因是 vLLM 不把 `lm_head` 当普通 module 调 forward，而是在 `LogitsProcessor` 里直接用它的 weight 算 logits：

```python
logits_local = lm_head.quant_method.apply(lm_head, hidden_states)
logits = tensor_model_parallel_gather(logits_local)
logits = logits[..., :org_vocab_size]
```

也就是：

1. 每个 rank 算 `[T, V/TP]` 的局部 logits；
2. sampling 前 gather 成 `[T, V]`；
3. 去掉 padding vocab；
4. 后续再做 temperature、top-k、top-p、采样。

vLLM 还提供了 `get_top_tokens()` 这种局部 argmax 优化：每个 rank 只返回本地最大值和本地 index，再 gather `(value, index)` 对，而不是 gather 整个 vocab logits。通信量从 `O(T * V)` 变成 `O(T * TP)`，适合 greedy / argmax 路径。

### 2.7 权重加载：切分必须和 checkpoint 名字对上

TP layer 的 forward 很短，真正容易错的是权重加载。

`ColumnParallelLinear.weight_loader()` 的关键逻辑是：

```python
shard_size = param_data.shape[output_dim]
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

即：**沿输出维拿当前 rank 的那一段**。

`RowParallelLinear.weight_loader()` 则沿输入维切：

```python
shard_size = param_data.shape[input_dim]
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
```

`MergedColumnParallelLinear` 还要处理 checkpoint 里分开的 `gate_proj` / `up_proj`，把它们加载到一个 fused 参数里。模型文件里常见的映射长这样：

```python
packed_modules_mapping = {
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

所以面试里问"为什么 vLLM 模型代码里有一堆 `packed_modules_mapping`"，答案不是框架炫技，而是：**运行时想用 fused GEMM，checkpoint 仍然可能按原始模块名存权重，加载器必须把名字和切片关系对齐。**

### 2.8 和 PP / DP 的边界

`ParallelLMHead` 以及前面涉及的 parallel linear / embedding 说的是 **TP 层内并行**，不要和 PP / DP 混在一起：

| 并行 | 切什么 | 在这节 layer 里的体现 |
|---|---|---|
| TP | hidden / head / vocab 维 | `ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding` |
| PP | transformer 层深度 | 不属于本 PP rank 的 layer 用 `PPMissingLayer` 占位 |
| DP | 请求批次 / replica | dense 模型里每个 DP rank 各自跑一套 TP/PP world |

一句话：**TP 改的是一个 layer 的数学实现；PP 改的是哪些层在本 rank 上存在；DP 改的是哪些请求进哪个 engine。**

### 2.9 常见疑问

**Q1：为什么 column 后面不立刻 all-gather？**
- 因为很多后续操作能直接吃分片张量，例如 MLP 的激活和 attention 的每-head 计算；
- 早 gather 会让通信提前发生，还会把本来可分片的中间张量变大。

**Q2：为什么 row parallel 是 all-reduce，不是 all-gather？**
- row parallel 每个 rank 算的是同一个输出维度上的"局部和"；
- 完整结果是求和，不是拼接，所以必须 all-reduce(sum)。

**Q3：bias 在 row parallel 里为什么只在 rank 0 加？**
- 因为所有 rank 的 `y_partial` 会相加；
- 如果每个 rank 都加 bias，all-reduce 后 bias 会被加 `tp_size` 次；
- vLLM 里 `RowParallelLinear.forward()` 只让 rank 0 把 bias 融进 GEMM。

**Q4：`ParallelLMHead` 为什么没有正常 forward？**
- sampler 需要控制 logits gather、padding 截断、soft cap、scale、logits processor；
- 这些逻辑集中在 `LogitsProcessor` 更清楚，也避免普通 module forward 隐式触发大 vocab gather。

**Q5：这类 layer 为什么说"轻量"？**
- 它们本身没有复杂状态机，只是在 `Linear/Embedding` 周围加三件事：权重切片、局部 GEMM、collective；
- 真正复杂的是分布式 group 初始化、权重加载映射、量化参数切片、PP/DP 调度组合。

**Q6：TP size 越大越好吗？**
- 不是。TP 降低单卡权重和 GEMM 压力，但会增加 all-reduce / all-gather 通信；
- decode 阶段 batch 小、GEMM 变瘦，通信更容易成为瓶颈；
- 一般是"模型放得下就少切，放不下再切"，再根据吞吐和延迟实测调。

---

## §3 MergedColumnParallelLinear & SwiGLU MLP

§2 里 `ColumnParallelLinear` 是单个 Linear 沿输出维切。但 LLaMA / Qwen 的 MLP 里 `gate_proj` 和 `up_proj` 是两个**输入相同、shape 相同、切法相同**的 Linear——vLLM 把它们合并成一次 GEMM，这就是 `MergedColumnParallelLinear`。配上 `SiluAndMul` 和 `down_proj`，整个 SwiGLU MLP 在 TP 下只剩两次 GEMM + 一次 all-reduce。

### 3.1 SwiGLU 数学

$$
\text{MLP}(x) \;=\; \big(\,\operatorname{SiLU}(x W_{\text{gate}}) \;\odot\; x W_{\text{up}}\,\big)\, W_{\text{down}}
$$

- `W_gate, W_up ∈ ℝ^(H × I)`，`W_down ∈ ℝ^(I × H)`。
- `I = intermediate_size`，通常是 `~2.67 H`（LLaMA-7B 是 `H=4096, I=11008`）。
- `SiLU(z) = z · sigmoid(z)`，又叫 Swish。
- `⊙` 是 element-wise 乘，**起 gate 门控作用**：`up` 路径携带主信号，`SiLU(gate)` 路径决定每个 channel 通过多少。

直觉：相比传统 `Linear → GeLU → Linear`，SwiGLU 多出一条 gate 路径，让 MLP 有一个学到的门控。同 FLOPs 下效果略好，是 LLaMA 系列的默认。

#### 为什么 intermediate_size 是 `~2.67 H` 而不是 `4 H`？

传统 GeLU MLP 是 `H → 4H → H`，两块权重共 `8 H²`。SwiGLU 有三块权重（gate / up / down），如果还用 `4H`，参数量是 `12 H²`，多了 50%。为了对齐总参数预算，中间维降到 `~2.67 H`，`gate + up + down ≈ 8 H²`。LLaMA-7B `4096 → 11008 → 4096` 就是这么算出来的。

### 3.2 为什么要 merge gate 和 up

朴素写法：

```python
gate = x @ W_gate.T   # GEMM #1: [T, H] × [H, I] -> [T, I]
up   = x @ W_up.T     # GEMM #2: [T, H] × [H, I] -> [T, I]
```

但 `W_gate` 和 `W_up` 的 K 维（=`H`）相同、N 维（=`I`）相同，输入也都是同一个 `x`。把它们在输出维拼起来：

$$
W_{\text{merged}} = \begin{bmatrix} W_{\text{gate}} \\ W_{\text{up}} \end{bmatrix} \in \mathbb{R}^{2I \times H}
$$

只需要一次 GEMM：

```python
gate_up = x @ W_merged.T   # [T, H] × [H, 2I] -> [T, 2I]
gate, up = gate_up.split(I, dim=-1)
```

收益：

1. **省一次 kernel launch**。两个独立 GEMM 各启动一次 CUDA kernel，合并后只启动一次。decode 阶段 GEMM 本身很瘦，launch overhead 占比不小。
2. **省一次 `x` 的 HBM 读**。朴素写法 `x` 要被 GEMM 读两次。
3. **更大 GEMM 更接近 roofline**。cuBLAS / cutlass 在 N 翻倍后吞吐通常 >2x，特别是 N 比较小的时候。

### 3.3 TP 切法：gate 和 up 必须切同一段

`MergedColumnParallelLinear` 沿输出维切，但**有约束**：gate 和 up 的切片必须对齐。TP=2 的正确切法：

```
rank 0: W_gate[:I/2, :],  W_up[:I/2, :]
rank 1: W_gate[I/2:, :],  W_up[I/2:, :]
```

每个 rank 拿到的 `param` shape 是 `[2 * I/TP, H]`，前一半是 gate 的本地分片，后一半是 up 的本地分片。

为什么不能 rank 0 拿整个 gate、rank 1 拿整个 up？后面要算 `SiLU(gate) ⊙ up`——这是 element-wise 乘，两个张量必须在**同一个 channel**上对应。不同 rank 拿不同 channel 等于 broken。

forward 出来的 `gate_up_local` shape 是 `[T, 2I/TP]`：

```
gate_up_local = [ gate[rank * I/TP : (rank+1) * I/TP],
                  up  [rank * I/TP : (rank+1) * I/TP] ]
```

split 后 `gate_local`、`up_local` 都是 `[T, I/TP]`，element-wise 完美对齐。

### 3.4 weight_loader：用 `loaded_shard_id` 分别加载 gate / up

checkpoint 里 `gate_proj.weight` 和 `up_proj.weight` 是分开存的。`MergedColumnParallelLinear.weight_loader` 会被调用两次，靠 `loaded_shard_id ∈ {0, 1}` 区分这次加载的是 gate 还是 up，写到 `param` 的对应一半。

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    # loaded_shard_id: 0 -> gate, 1 -> up
    tp_rank, tp_size = get_tp_rank_and_size()

    shard_size = self.output_sizes[loaded_shard_id] // tp_size   # I/TP
    shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size   # gate=0, up=I/TP

    # 在 param 里定位本 rank 该写的那一段
    param_slice = param.data.narrow(0, shard_offset, shard_size)

    # 从 checkpoint 整片里切出本 rank 那一段
    weight_slice = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)

    param_slice.copy_(weight_slice)
```

模型代码里的映射表负责告诉 loader "这两个 checkpoint 名字塞进同一个 fused 参数"：

```python
packed_modules_mapping = {
    "gate_up_proj": [
        ("gate_proj", 0),   # loaded_shard_id = 0
        ("up_proj",   1),   # loaded_shard_id = 1
    ],
}
```

这也是 §2.7 里那句"框架不是炫技、是为了把 checkpoint 名字和 fused 切片关系对齐"的具体落地。

### 3.5 SiluAndMul：又一次 elementwise fusion

split + silu + mul 的朴素写法：

```python
gate, up = gate_up.chunk(2, dim=-1)   # view，免费
y = F.silu(gate) * up                  # 两次 elementwise pass
```

`gate_up` 要被读两次（算 silu(gate) 一次，乘 up 一次），中间 `silu(gate)` 还要写出一份再读回来。vLLM 提供 `SiluAndMul`（背后是 CUDA kernel `vllm._C.ops.silu_and_mul`）：

```python
# 一个 kernel 完成：
# for i in 0..I/TP:
#     out[t, i] = silu(gate_up[t, i]) * gate_up[t, i + I/TP]
```

- 读 `gate_up` 一次（`[T, 2I/TP]`）；
- silu 和 mul 在寄存器里完成；
- 写出 `[T, I/TP]` 一次。

HBM 带宽消耗从 `~4 * T * I/TP * 2B` 降到 `~3 * T * I/TP * 2B`，少 25%。decode 阶段是 memory-bound，这一刀直接转成 token/s。

### 3.6 down_proj：Row Parallel 把 intermediate 收回 hidden

MLP 出口必须回到完整 `hidden` 维 `H`（让 residual / 下一层 norm 用），但此时 intermediate 在分片状态 `[T, I/TP]`。所以 `down_proj` 是 `RowParallelLinear`：

- `W_down ∈ ℝ^(H × I)` 沿**输入维** `I` 切给 TP，每个 rank 拿 `W_down_local ∈ ℝ^(H × I/TP)`。
- 每个 rank 算 `y_partial = x_local @ W_down_local.T`，shape `[T, H]`，但只是局部贡献。
- all-reduce(sum) 拿回完整 `[T, H]`。

> 为什么不让 down_proj 也走 column？那样输出就是 `[T, H/TP]`，得再 all-gather 才能给 residual。通信量同阶，但 all-reduce 后每张卡都拿到完整 hidden，可以直接进 replicated 的 residual / norm，逻辑更顺。

### 3.7 完整的 SwiGLU MLP

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],  # [gate, up]
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)   # [T, 2I/TP], 1 次 GEMM (fused)
        x = self.act_fn(gate_up)            # [T,  I/TP], fused silu+mul kernel
        x, _ = self.down_proj(x)            # all-reduce -> [T, H]
        return x
```

数一下整层的 collective：**只有 1 次 all-reduce**。所有 fuse 都是为了把"应该一刀切的事情"塞进一个 kernel：gate+up 合成一次 GEMM，silu+mul 合成一次 elementwise，最后才让 down_proj 做唯一的一次跨卡通信。

#### 通信节点表

| 子模块 | 输入 shape | 输出 shape | 是否通信 |
|---|---|---|---|
| `gate_up_proj` (Column) | `[T, H]` | `[T, 2I/TP]` | 无 |
| `SiluAndMul` | `[T, 2I/TP]` | `[T, I/TP]` | 无 |
| `down_proj` (Row) | `[T, I/TP]` | `[T, H]` | **all-reduce(sum)** |

这就是为什么 SwiGLU MLP 在 TP 下被叫做 "Column → Activation → Row" 模式，attention 那边 (`QKVParallelLinear` → `Attention` → `o_proj`) 是同一个套路。

### 3.8 常见疑问

**Q1：fused GEMM 和 cuBLAS batched GEMM 是同一个东西吗？**
- 不是。fused 是把两份权重拼成更大的单个矩阵，跑一次大 GEMM；batched GEMM 是同时跑 N 个独立 GEMM。`MergedColumnParallelLinear` 走前者。

**Q2：为什么 `gate_up_proj` 没 bias，传统 MLP 有？**
- 经验结论：Pre-LN + RMSNorm 之后 bias 对效果几乎没影响，去掉省参数省计算。LLaMA 全系列、Qwen 都把 `bias=False` 作为默认。

**Q3：`down_proj` 的 bias 怎么处理？**
- 通常也是 `bias=False`。如果有 bias，按 §2.9 Q3 的规则，row parallel 里只在 rank 0 加一次（all-reduce 后只被加一次）。

**Q4：MLP 里有没有可能用上 PP？**
- PP 切的是层深度，不会切单个 MLP 的内部结构。`LlamaMLP` 整体属于某一个 transformer block，要么完整在本 PP rank 上，要么整层是 `PPMissingLayer`。

**Q5：gate/up 合并对量化（INT8、FP8）有影响吗？**
- 有，但合并方向是有利的。每个 fused 权重整体走同一个 scale/zero-point 时，scale 数量减半；如果是 per-channel 量化，gate 和 up 各保留自己的一组 channel scale，loader 也得跟着 narrow 出对应那段 scale。vLLM `weight_loader_v2` 里的 `qkv` / `gate_up` 都按这个套路兼容多种量化。

**Q6：为什么 `forward` 返回的是 `(output, bias)` 元组？**
- vLLM 的 `LinearBase.forward` 把 bias 单独返回，让上层决定是 fuse 到下一个算子里还是马上加。MLP 没 bias 时直接 `_` 丢掉。这是 vLLM 自带的 fuse hook，不是 PyTorch 的约定。

---

## §4 QKVParallelLinear（fused QKV + GQA/MQA 切分）

§2.5 里给了 `QKVParallelLinear` 的形状直觉，这一节把它拆透：它是带 GQA / MQA 语义约束的 `MergedColumnParallelLinear`。

### 4.1 数学：三个 Linear 合一

attention 入口需要三个投影：

$$
Q = x W_Q^\top,\quad K = x W_K^\top,\quad V = x W_V^\top
$$

- `W_Q ∈ ℝ^(N_q · d_h × H)`，`N_q = num_heads`，`d_h = head_dim`；
- `W_K, W_V ∈ ℝ^(N_kv · d_h × H)`，`N_kv = num_kv_heads`（GQA 时 `N_kv < N_q`，MQA 时 `N_kv = 1`）。

三个 Linear 共享输入 `x`、共享 K 维 `H`，自然合并成一次 GEMM：

$$
W_{\text{QKV}} = \begin{bmatrix} W_Q \\ W_K \\ W_V \end{bmatrix} \in \mathbb{R}^{(N_q + 2 N_{kv}) d_h \times H}
$$

```python
qkv = x @ W_QKV.T                 # [T, (N_q + 2 N_kv) * d_h]
q, k, v = qkv.split([N_q * d_h, N_kv * d_h, N_kv * d_h], dim=-1)
```

跟 `gate_up_proj` 是同一个 merge 思路，只是 sizes 不再相等。

### 4.2 切分约束：按 head 切，不按 channel 切

`gate_up` 可以直接按输出维 `I` 等分给 TP（因为 silu+mul 是 element-wise）。QKV **不行**——后面的 attention 算 `softmax(QK^T / √d) V` 是**按 head 独立**的，head 内的 `d_h` 个 channel 必须留在同一个 rank，否则 `QK^T` 算不出来。

所以 `QKVParallelLinear` 切的单位是 head，不是 channel：

```
rank r 拿到:
  Q heads: [r * N_q/TP : (r+1) * N_q/TP]    每 head d_h 维
  K heads: [r * N_kv/TP : (r+1) * N_kv/TP]
  V heads: [r * N_kv/TP : (r+1) * N_kv/TP]
```

local 输出维：

$$
O_{\text{local}} \;=\; \frac{N_q}{\text{TP}}\, d_h \;+\; 2\,\frac{N_{kv}}{\text{TP}}\, d_h
$$

### 4.3 GQA / MQA：当 `N_kv < TP` 时复制 KV head

GQA 的设计哲学是少存 KV cache：`N_kv` 通常远小于 `N_q`（LLaMA-2-70B：`N_q=64, N_kv=8`；Mistral-7B：`32 / 8`）。但 `N_kv` 可能 **小于 TP**：

- TP=8、`N_kv=8`：每张卡刚好分到 1 个 KV head，正常切；
- TP=8、`N_kv=4`：4 个 KV head 没法平均给 8 张卡——vLLM 选择**在多个 rank 上复制 KV head**，每 2 个 rank 共用同一个 KV head；
- TP=8、`N_kv=1`（MQA）：1 个 KV head 在 8 张卡上**全复制**。

复制 KV head 的代价是显存稍微浪费一点（每张卡多存一份 K/V），但避免了在 head 内部切 channel 这种 broken 切法，attention 后端逻辑保持简单。vLLM 在初始化时计算：

```python
num_kv_heads_per_rank = max(1, total_num_kv_heads // tp_size)
num_kv_head_replicas  = max(1, tp_size // total_num_kv_heads)
```

### 4.4 weight_loader：三种 `loaded_shard_id`

和 `MergedColumnParallelLinear` 同构，只是 shard id 多一个：

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    # loaded_shard_id ∈ {"q", "k", "v"}
    if loaded_shard_id == "q":
        shard_offset = 0
        shard_size = self.num_heads * self.head_size               # 本地 Q
    elif loaded_shard_id == "k":
        shard_offset = self.num_heads * self.head_size
        shard_size = self.num_kv_heads * self.head_size            # 本地 K
    elif loaded_shard_id == "v":
        shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        shard_size = self.num_kv_heads * self.head_size            # 本地 V

    # 在 fused 参数里定位本 rank 的 q/k/v 段
    param_slice = param.data.narrow(0, shard_offset, shard_size)

    # 从 checkpoint 切出本 rank 的 head 段（注意 KV 复制时不再乘 tp_rank）
    if loaded_shard_id == "q":
        start = self.tp_rank * shard_size
    else:
        start = (self.tp_rank // self.num_kv_head_replicas) * shard_size

    param_slice.copy_(loaded_weight.narrow(0, start, shard_size))
```

关键差异：

- Q 永远按 `tp_rank` 切；
- KV 在 `num_kv_head_replicas > 1` 时，**多个相邻 rank 加载同一段权重**（这就是"复制 KV head"的实现）。

模型层的映射表：

```python
packed_modules_mapping = {
    "qkv_proj": [
        ("q_proj", "q"),
        ("k_proj", "k"),
        ("v_proj", "v"),
    ],
}
```

### 4.5 attention block 里的位置

```python
class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            bias=False,
        )
        self.rotary_emb = ...           # RoPE
        self.attn = Attention(...)      # PagedAttention backend

    def forward(self, x, positions, kv_cache):
        qkv, _ = self.qkv_proj(x)                   # [T, (N_q_local + 2 N_kv_local) * d_h]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v, kv_cache)          # [T, N_q_local * d_h]
        out, _ = self.o_proj(out)                   # all-reduce -> [T, H]
        return out
```

通信节点：

| 子模块 | 输入 shape | 输出 shape | 是否通信 |
|---|---|---|---|
| `qkv_proj` (QKV column) | `[T, H]` | `[T, (N_q_local + 2 N_kv_local) d_h]` | 无 |
| `rotary_emb` | q, k | q, k | 无 |
| `attn` (PagedAttention) | q, k, v | `[T, N_q_local · d_h]` | 无 |
| `o_proj` (Row) | `[T, N_q_local · d_h]` | `[T, H]` | **all-reduce(sum)** |

整层 attention **只在 `o_proj` 通信一次**，跟 MLP 同节奏。一个 transformer block 的 collective 数 = 2 次 all-reduce（attention 一次、MLP 一次），36 层就是 72 次——vLLM 后续的 `compile` / `sequence parallel` / `async TP` 等优化基本都围绕这些点继续榨。

### 4.6 常见疑问

**Q1：为什么不像 `gate_up` 一样直接按等比例切？**
- gate 和 up 的 channel 之间相互独立（element-wise 乘）；
- attention 的 head 内 channel 之间相互耦合（`Q K^T` 是 head 内 dot product）；
- 所以 QKV 必须按 head 切，head 内不分。

**Q2：`num_kv_head_replicas` 让 KV head 在多个 rank 上重复，attention 后端怎么用？**
- 每个 rank 拿到的 K/V shape 都是"本地 Q head 数对应的 KV"——也就是说 backend 看到的 KV head 数 = `N_q_local / heads_per_kv`；
- 物理上 KV 权重是复制的，但语义上每张卡都自洽，attention 实现不用区分"原生切" vs "复制切"。

**Q3：为什么不把 `o_proj` 也 fuse 进 attention kernel？**
- 部分 backend（FlashAttn 系）做过尝试，叫 "flash attention with output projection fused"。vLLM 主线还是分开，原因是 `o_proj` 本身是个大 GEMM，让 cuBLAS / cutlass 选最优 algo 更好，并且 row parallel 要 all-reduce，融合后 collective 边界更混乱。

**Q4：MQA（`N_kv = 1`）下 `QKVParallelLinear` 还有意义吗？**
- 有。`N_q · d_h` 还是要按 head 切给 TP，KV 那一份全复制。fused GEMM 的形状变成 `[T, (N_q_local + 2) · d_h]`，merge 收益还在。

**Q5：DeepSeek / MLA 这种结构走的还是 `QKVParallelLinear` 吗？**
- 不是。MLA 把 KV 压成 `kv_lora` + 解压矩阵，QKV 的入口投影变成另一组 `q_a_proj` / `kv_a_proj_with_mqa` / `q_b_proj` 等，vLLM 里有专门的 `MLAAttention` 路径。`QKVParallelLinear` 主要服务 GQA / MHA / MQA 这一系。

**Q6：`q_size`、`kv_size` 在 forward 里怎么得到？**
- 初始化时算好并缓存：`self.q_size = num_heads_per_rank * head_dim`，`self.kv_size = num_kv_heads_per_rank * head_dim`。这两个值是 backend 切 Q/K/V 的依据，也是 vLLM 模型代码里频繁出现的"形状常量"。
