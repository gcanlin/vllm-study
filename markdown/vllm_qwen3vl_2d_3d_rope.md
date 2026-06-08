# Qwen3-VL 的 2D RoPE 与 3D MRoPE：算什么、怎么算、怎么省

> **文档版本**: 1.0
> **分析代码版本**: 当前 workspace 本地 `vllm` 源码
> **最后更新**: 2026-06-06

---

## 文档概述

Qwen3-VL 在一条 prompt 里同时跑两套 RoPE：

- **ViT 内部的 2D RoPE**：把每个 patch 按 `(h_id, w_id)` 两个坐标旋转，让 vision tower 拿到二维空间结构。
- **LLM 端的 3D MRoPE**：把 head_dim 切成 T/H/W 三段，分别用 `(t, h, w)` 三个轴的位置驱动，让 LLM 能区分"同一帧不同位置"与"不同帧同一位置"的 image token。

这两套 RoPE 的数学骨架完全相同——都是 `q' = q·cos + rotate(q)·sin`——区别只在于 **cos/sin 的来源** 和 **每个 channel pair 用哪个轴的位置驱动**。本文沿这条主线展开，重点放在 vLLM 为它们做的工程优化（cos/sin cache、lru_cache、Triton 融合 kernel、in-place 写回、cache 4× 扩容、`mrope_position_delta` 等）。

| 部分 | 内容 |
|------|------|
| 第 1 部分 | 为什么 ViT 和 LLM 要用不同的 RoPE |
| 第 2 部分 | ViT 端 2D RoPE 的实现与优化 |
| 第 3 部分 | LLM 端 3D MRoPE 的实现与优化 |
| 第 4 部分 | 两套 RoPE 的并列对照 |
| 第 5 部分 | 一张 224×224 单图的端到端 walk-through |

姊妹文档：[vllm_rope.md](vllm_rope.md)（vLLM RoPE 全景）、[vllm_qwen3_vl_vit_endtoend.md](vllm_qwen3_vl_vit_endtoend.md)（ViT 端到端）。本文与它们互补，只聚焦"两套 RoPE 的差异和优化"。

---

# 第 1 部分：为什么 ViT 和 LLM 要用不同的 RoPE

## 1.1 ViT 看见的"位置"是二维的

ViT 的输入是 packed patch 序列。一张 `224×224` 单图被切成 `14×14 = 196` 个 patch，**逻辑上是 2D**，但喂给 attention 的时候铺成 1D。如果直接用 1D RoPE，模型会把 `(0,0)→(0,1)→...→(0,13)→(1,0)` 这条"扫描线"当成一段长文本，**完全看不到 `(h, w)` 两个方向的对称性**。

所以 vLLM 在 ViT 里给每个 patch 一对 `(h_id, w_id)`，head_dim 的前半看 h，后半看 w：

```text
patch (h, w):  q ∈ ℝ^{head_dim}
                    │
                    ├── 前 head_dim/2 个 channel pair → 用 h 旋转
                    └── 后 head_dim/2 个 channel pair → 用 w 旋转
```

两段共享同一份 cos/sin 表（[`base.py:94-103`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L94-L103) 的 `_compute_cos_sin_cache`），靠"前半 vs 后半"在 head_dim 上自然分轴。

## 1.2 LLM 看见的"位置"是三维的

进了 LLM 端之后，image / video token 在 prompt 里变成一段连续展开。LLM 不知道这段里哪里换行、哪里换帧。所以 Qwen-VL 系列引入 MRoPE：每个 token 用 `(t, h, w)` 三个独立的位置：

```text
text:    positions = (n, n, n)         # 三轴相同，退化成普通 1D RoPE
image:   positions = (t_frame, h_id, w_id)
```

为了让 head_dim 同时编码三个轴，MRoPE 把 head_dim 切成 T/H/W 三段 ([`mrope.py:248-251`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L248-L251))：

```python
self.mrope_section = mrope_section            # e.g. [24, 20, 20]
assert sum(self.mrope_section) == rotary_dim // 2
```

这是 ViT 里 2D RoPE 的自然扩展：从"切成两半"变成"切成三段"。

## 1.3 两套 RoPE 的物理分工

| 层级 | 旋转单位 | 轴 | 谁负责构造位置 |
|------|---------|-----|----------------|
| **ViT 内** | 单个 patch | 2D `(h_id, w_id)` | [`rot_pos_ids`](../vllm/vllm/model_executor/models/qwen3_vl.py#L634-L659) 在 ViT 内部算 |
| **LLM 内** | 一段 image / video / text | 3D `(t, h, w)` | [`get_mrope_input_positions`](../vllm/vllm/model_executor/models/qwen3_vl.py#L2562-L2647) 在 prefill scheduler 算 |

**ViT 不在时间维做 RoPE**——视频帧靠 `repeat(t, 1)` 复用同一组 2D 位置，时间关系完全交给 LLM 端的 MRoPE。这是一个有意的"职责切分"，让 vision tower 保持纯空间编码。

---

# 第 2 部分：ViT 端 2D RoPE 的实现与优化

## 2.1 复用通用 `get_rope` + `partial_rotary_factor=0.5`

[`qwen3_vl.py:575-580`](../vllm/vllm/model_executor/models/qwen3_vl.py#L575-L580)：

```python
self.rotary_pos_emb = get_rope(
    head_size=head_dim,                   # 72
    max_position=8192,                    # 远大于实际 max_grid_size
    is_neox_style=True,
    rope_parameters={"partial_rotary_factor": 0.5},
)
```

只传 `partial_rotary_factor=0.5`，没传 `rope_type`：

- `get_rope` 算出 `rotary_dim = head_size // 2 = 36`（[`__init__.py:66-72`](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py#L66-L72)）；
- 没有 `rope_type` → 走 default 分支 → 普通 [`RotaryEmbedding`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L139)；
- 基类构造时直接算 cos/sin cache（[`base.py:58-63`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L58-L63)），shape `(8192, rotary_dim) = (8192, 36)`，前 18 列是 cos，后 18 列是 sin。

**"rotary_dim = head_dim // 2" 是 2D RoPE 的关键**：剩下的 head_dim/2 留给"另一半"，正好和"前半看 h、后半看 w"的语义对齐。

### 优化点 ①：cos_sin_cache 在 `__init__` 算一次，永驻 GPU buffer

[`base.py:94-103`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L94-L103)：

```python
def _compute_cos_sin_cache(self):
    inv_freq = self._compute_inv_freq(self.base)
    t = torch.arange(self.max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)   # (max_pos, rotary_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)             # (max_pos, rotary_dim)
```

[`base.py:58-63`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L58-L63)：

```python
if init_cache:
    cache = self._compute_cos_sin_cache()
    if not self.use_flashinfer:
        cache = cache.to(dtype)            # 通常 bf16
    self.register_buffer("cos_sin_cache", cache, persistent=False)
```

收益：

- `cos` / `sin` 不再每 forward 算一次三角函数；
- non-persistent buffer 不进 checkpoint，但跟着 `model.to(device)` 一起搬；
- dtype 跟模型走（bf16 / fp16），避免运行时 cast。

这一份 cache **同时被 ViT 的 forward 和 `prepare_encoder_metadata` 用**，全模型一份，TP 不切。

### 优化点 ②：`get_cos_sin` 直接切片，不重排不复制

[`base.py:133-136`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py#L133-L136)：

```python
def get_cos_sin(self, seqlen):
    cos_sin = self.cos_sin_cache[:seqlen]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin
```

`max_grid_size` 一般只有十几（14、28、48），而 cache 有 8192 行——一次切片几乎免费。`chunk` 是 view，不拷贝。

## 2.2 `rot_pos_ids`：构造 2D 位置 + lru_cache

[`qwen3_vl.py:634-659`](../vllm/vllm/model_executor/models/qwen3_vl.py#L634-L659)：

```python
@staticmethod
@lru_cache(maxsize=1024)
def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
    M = spatial_merge_size
    hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
    hpos_ids = hpos_ids.reshape(h//M, M, w//M, M).transpose(0, 2, 1, 3).flatten()

    wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
    wpos_ids = wpos_ids.reshape(h//M, M, w//M, M).transpose(0, 2, 1, 3).flatten()

    return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))  # (h*w, 2)
```

`reshape + transpose(0, 2, 1, 3)` 是 **spatial-merge 重排**：让 patch 序列里连续 `M×M` 个 token 是同一个 merge block。要和 `PatchMerger.view` 的 reshape 共用同一个 order，否则下游 view 会读串位。

### 优化点 ③：`@lru_cache(maxsize=1024)` 缓存位置表

- 同一 batch 里多张同样分辨率的图共用 `(h, w, M)` —— 只算一次；
- 不同请求之间同样可以复用；
- 1024 容量足够覆盖常见分辨率组合；
- 缓存的是 CPU `torch.Tensor`，每次 `to(device, non_blocking=True)` 搬过去 —— 因为图片分辨率谱有限，复用率非常高。

`rot_pos_emb` 在外面对结果做 `torch.cat([...]).to(self.device, non_blocking=True)`（[`qwen3_vl.py:669`](../vllm/vllm/model_executor/models/qwen3_vl.py#L669)），保证 lru_cache 的 tensor 永远停在 CPU，多个 GPU 都能复用同一份缓存。

## 2.3 `rot_pos_emb`：2 次 lookup + flatten 拼出 2D cos/sin

[`qwen3_vl.py:661-677`](../vllm/vllm/model_executor/models/qwen3_vl.py#L661-L677)：

```python
def rot_pos_emb(self, grid_thw):
    max_grid_size = max(max(h, w) for _, h, w in grid_thw)
    pos_ids = [
        self.rot_pos_ids(h, w, self.spatial_merge_size)
        if t == 1
        else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
        for t, h, w in grid_thw
    ]
    pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

    cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)   # 复用 cache
    cos_combined = cos[pos_ids].flatten(1)
    sin_combined = sin[pos_ids].flatten(1)
    return cos_combined, sin_combined
```

关键操作 `cos[pos_ids].flatten(1)` 的形变：

```text
pos_ids                 (seq_len, 2)          # 每行 (h_id, w_id)
cos[pos_ids]            (seq_len, 2, head_dim/4)
  ↑ 第一段对 h_id 查表、第二段对 w_id 查表
.flatten(1)             (seq_len, head_dim/2) = (seq_len, rotary_dim)
  ↑ 沿 head_dim 拼成 [cos(h·θ_low..θ_high) | cos(w·θ_low..θ_high)]
```

### 优化点 ④：复用 `cos_sin_cache`，不再算第二份 2D cache

最朴素的做法是把 `(h_id, w_id)` 两个轴展平成线性 ID（比如 `h_id * grid_w + w_id`），然后单独建一份 2D cache。但 vLLM 选择 **共用 1D cache + 二次查表**：

- 1D cache 已经在 `__init__` 算好；
- 二次查表两次 `cos[pos_ids]` 是张量 fancy-indexing，融合在一次 kernel；
- `flatten(1)` 是 view，0 拷贝；
- 不需要为每张图重新构造 cache。

实际效果是一张 196 patch 的图，构造 cos/sin 只需要一次 `lru_cache hit`（CPU→GPU 拷贝 392 个 int64）+ 一次 GPU fancy-indexing。

### 优化点 ⑤：视频帧直接 `repeat(t, 1)`，不在 ViT 算时间 RoPE

```python
... if t == 1 else self.rot_pos_ids(h, w, M).repeat(t, 1)
```

视频 t 帧共用同一组 2D 位置 —— ViT 把每帧当作独立 2D 图，时间关系交给 LLM 端 MRoPE。这把 ViT 的 RoPE 复杂度从"$O(t·h·w)$ 三轴"压到 "$O(h·w)$ 单一缓存表"，同时让 [`cu_seqlens`](../vllm/vllm/model_executor/models/qwen3_vl.py#L736) 把每帧切成独立 attention 段。

## 2.4 `ApplyRotaryEmb`：half-head 旋转 + FlashAttn Triton kernel

[`common.py:124-183`](../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L124-L183) 是 vLLM 通用的"接受外部 cos/sin 的旋转算子"。NEOX 路径：

```python
cos = cos.unsqueeze(-2)                # (seq, 1, rotary_dim)，广播 head 维
sin = sin.unsqueeze(-2)
x1, x2 = torch.chunk(x, 2, dim=-1)     # 各 (..., head_dim/2)
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
output = torch.cat((o1, o2), dim=-1)   # 拼回 head_dim
```

### 优化点 ⑥：head_dim 沿中线切两半，省掉 HF 那次 cat

HF Transformers 的 ViT RoPE 实现里，cos/sin 长度是 `head_dim`（先 cat 一次 `cat(emb, emb)`），然后用 `rotate_half`。vLLM 这里 **直接让 cos/sin 长度 = head_dim/2**：

```text
x1 = x[..., :head_dim/2]    → 对应 cos/sin 的 "h 半"   (左前 rotary_dim/2 个 channel pair)
x2 = x[..., head_dim/2:]    → 同一份 cos/sin 的 "w 半"   (右后 rotary_dim/2 个 channel pair)
```

两半各自做 `cos·x ± sin·rotate(x)`，数学等价于 HF 的 `rotate_half(cat(emb, emb))`，但 **省掉一次 cat、少一份显存**。

### 优化点 ⑦：`forward_cuda` → FlashAttn 的 Triton rotary kernel

[`common.py:227-248`](../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L227-L248)：`ApplyRotaryEmb.forward_cuda` 直接调 `vllm_flash_attn.layers.rotary.apply_rotary_emb`，把"加载 cos/sin → 旋转 q/k → 写回"融成一个 Triton kernel。`forward_hip` 也走 FlashAttn 的 Triton 版本，NPU 上由 `AscendApplyRotaryEmb` 接 `npu_apply_rotary_pos_emb`。

调用点（[`qwen2_5_vl.py:454-459`](../vllm/vllm/model_executor/models/qwen2_5_vl.py#L454-L459)）：

```python
qk_reshaped = rearrange(qk, "b s two h d -> (two b) s h d").contiguous()
qk_rotated  = self.apply_rotary_emb(qk_reshaped, cos, sin)
q, k = qk_rotated.view(2, b, s, h, d).unbind(dim=0)
```

### 优化点 ⑧：Q 和 K 一次 rotate

`rearrange ... -> (two b)`：把 q 和 k 合并到 batch 维一起喂给 kernel，**一次 Triton launch 同时旋转 q/k**，省掉一次 kernel 启动开销。

## 2.5 小结：ViT 2D RoPE 的优化层级

```
init 时：
  ├─ cos_sin_cache: (8192, 36)  bf16，buffer 常驻
  └─ rot_pos_ids: lru_cache(1024)  CPU torch tensor

每 forward (224×224 单图):
  ├─ rot_pos_ids cache hit → 0 计算，~3 us CPU→GPU 拷贝
  ├─ get_cos_sin: cache[:14] → 0 计算（view）
  ├─ cos[pos_ids].flatten: 1 次 fancy-indexing + 1 次 view
  └─ ApplyRotaryEmb (q,k 合并): 1 次 Triton kernel
```

整体每张图的 RoPE 计算路径上 **没有任何 sin/cos 三角运算、没有任何重新构造 cache**。

---

# 第 3 部分：LLM 端 3D MRoPE 的实现与优化

## 3.1 MRotaryEmbedding 是怎么被装出来的

[`__init__.py:259-272`](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py#L259-L272) 在 `rope_parameters` 里有 `"mrope_section"` 时构造 [`MRotaryEmbedding`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L201)：

```python
if "mrope_section" in rope_parameters:
    rotary_emb = MRotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype,
        mrope_section=rope_parameters["mrope_section"],
        ...
    )
```

Qwen3-VL-8B 的 config:

```text
head_size = 128
rotary_dim = 128
max_position_embeddings = 262144
rope_theta = 5_000_000
mrope_section = [24, 20, 20]      ← sum = 64 = rotary_dim / 2，分给 t/h/w
```

注意 **`mrope_section` 的三段加起来必须等于 `rotary_dim/2`**（[`mrope.py:251`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L251)），不是 `rotary_dim`。原因后面 §3.4 讲：cos/sin 用的就是"半 head"长度。

## 3.2 优化点 ⑨：cache 扩大到 `max_position × 4`

[`mrope.py:235-246`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L235-L246)：

```python
# In Qwen2.5-VL, the maximum index value is related to the duration of
# the input video. We enlarge max_position_embeddings to 4 times to get
# a larger the cos and sin cache.
self.cache_max_position_num = max_position_embeddings * 4
super().__init__(head_size, rotary_dim, self.cache_max_position_num, ...)
```

为什么是 4×？

- 视频的 t 索引会被 `t_factor = second_per_grid_ts * tokens_per_second` 拉大；
- 文本 token 的 max position 是 `max_model_len`；MRoPE 的 t 维 index 可能落到 `max_model_len × t_factor` 这个量级；
- 4× 是经验值（短视频 + 高 fps 足够），仍然按 `max_position` 一次性算完一份 cache，运行时只查表。

**所以 MRoPE 的 cos/sin cache shape 是 `(4 × max_pos, rotary_dim)`**，比同模型的纯文本 RoPE 大 4 倍——这是 MRoPE 唯一比普通 RoPE "贵" 的地方，换来"运行时永远不动 cache、纯查表"的简单性。

## 3.3 优化点 ⑩：3D positions 仍然按 1D 表查

注意 `cos_sin_cache` 本身 **还是 2D `(cache_max_pos, rotary_dim)`**，没有 3D 化。3D 关系是靠 **positions 这边变成 `(3, num_tokens)` 三路独立查同一张 1D 表**：

[`mrope.py:282-285`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L282-L285)：

```python
cos_sin_cache = self._match_cos_sin_cache_dtype(query)
num_tokens = positions.shape[-1]
cos_sin = cos_sin_cache[positions]       # positions: (3, N) → cos_sin: (3, N, rotary_dim)
cos, sin = cos_sin.chunk(2, dim=-1)
```

一行 `cos_sin_cache[positions]`，因为 `positions.ndim == 2`，PyTorch fancy-indexing 自然得到 `(3, N, rotary_dim)`。**省掉了"为 3D 单独维护 cache"的需求**，复用 1D cache 还省了 3× 内存。

文本 token 三轴值相同（`positions[0,i] == positions[1,i] == positions[2,i]`），三路查到一样的行 → 退化成普通 1D RoPE。所以 MRoPE 对纯文本完全兼容。

## 3.4 forward_native：split-and-cat 实现按 section 选轴

[`mrope.py:286-322`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L286-L322) 的关键部分：

```python
if positions.ndim == 2:
    assert self.mrope_section
    if self.mrope_interleaved:
        cos = apply_interleaved_rope(cos, self.mrope_section)
        sin = apply_interleaved_rope(sin, self.mrope_section)
    else:
        cos = torch.cat(
            [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
            dim=-1,
        )
```

`cos.split([24, 20, 20], dim=-1)` 沿 head_dim 切三块，每块仍带 3D 维 (T/H/W)；接下来 `enumerate` 取 i=0 那块的 T 段、i=1 那块的 H 段、i=2 那块的 W 段，**等价于**：

```text
final_cos = [ cos[T, :, 0:24]   ← T 位置驱动的前 24 个 channel pair
            | cos[H, :, 24:44]  ← H 位置驱动的中间 20 个
            | cos[W, :, 44:64]  ← W 位置驱动的最后 20 个 ]
```

`split + enumerate + cat` 的三句话，把"三维查表"塌缩成 head_dim 上拼出来的一个长向量。后面 `apply_rotary_emb` 就当成普通 RoPE 处理。

注意这里 `cos / sin` 长度都是 `rotary_dim / 2 = 64`——也就是 head_dim 的一半。这是 NEOX-style 的标准约定（和 ViT §2.4 完全一样，省掉一次 cat）。所以 `sum(mrope_section)` 才等于 `rotary_dim / 2`。

## 3.5 优化点 ⑪：`forward_cuda` 用 Triton 融合 kernel

[`mrope.py:14-187`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L14-L187) 的 `_triton_mrope_forward`（改编自 Liger Kernel）把上面四步——**section 选 cos/sin、加载 q/k、旋转两半、写回**——全部融合：

```python
# cos / sin 实际 stride 是 (3, num_tokens, head_dim/2)
t_cos = cos + pid * half_rd
h_cos = t_cos + num_tokens * half_rd
w_cos = h_cos + num_tokens * half_rd
# 用 mask 决定 head_dim 上每个 pair 用 T / H / W 哪一段
t_mask = cos_offsets < mrope_section_t
h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

# 三段相加 = 选择（mask 互斥）
cos_row = t_cos_row + h_cos_row + w_cos_row
sin_row = t_sin_row + h_sin_row + w_sin_row

# 旋转左右半
new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
# k 同理，in-place 写回 q_ptr / k_ptr
```

关键设计：

- **每个 program 处理一个 token**：`grid = (num_tokens,)`，并行度等于 token 数；
- **mask 互斥相加 = 选择**：避免 `if/else` 分支，纯算数操作友好编译；
- **q / k 在同一 kernel 里旋转**：免去两次 launch；
- **in-place 写回 `q_ptr` / `k_ptr`**：省掉一份输出 buffer，对 decode 阶段（num_tokens 通常很小）尤其友好；
- **同时支持 `is_interleaved`（Pangu MRoPE）**：用 `cos_offsets % 3` 区分轴，一个 kernel 复用两种 layout。

[`mrope.py:343-352`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L343-L352) 的 `forward_cuda` 把 multimodal 路径全部走 `triton_mrope`，**完全跳过 `cos.split` / `cat` 的 PyTorch 临时 tensor**：

```python
if positions.ndim == 2:
    assert self.mrope_section
    q, k = triton_mrope(query, key, cos, sin,
                        self.mrope_section, self.head_size, self.rotary_dim,
                        self.mrope_interleaved)
    return q.reshape(query_shape), k.reshape(key_shape)
```

纯文本路径（`positions.ndim == 1`）退回基类的普通 RoPE C++ kernel。

## 3.6 优化点 ⑫：`mrope_position_delta` 让 decode 阶段免重算

prefill 走完之后，[`get_mrope_input_positions`](../vllm/vllm/model_executor/models/qwen3_vl.py#L2562-L2647) 会一起返回一个 `mrope_position_delta`：

```python
llm_positions, mrope_position_delta = self.get_mrope_input_positions(...)
# mrope_position_delta = llm_positions.max() + 1 - len(input_tokens)
```

含义：图像 / 视频段的 `(t, h, w)` 三轴最大值通常 **大于这段 token 数本身**（因为帧数 × 网格数）。`delta` 记录"3D 位置坐标"相对于"1D token 索引"多出来的偏移。

decode 时新生成的 token 都是普通文本，不需要再算 3D：

[`mrope.py:401-414`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L401-L414)：

```python
@staticmethod
def get_next_input_positions_tensor(out, out_offset, mrope_position_delta,
                                     context_len, num_new_tokens):
    values = np.arange(
        mrope_position_delta + context_len,
        mrope_position_delta + context_len + num_new_tokens,
        dtype=out.dtype,
    )
    out[:, out_offset : out_offset + num_new_tokens] = values
```

三轴都赋同一个递增序列 → 又退回 1D RoPE 模式。这样 decode 每一步只算一次 numpy `arange`，**完全免去 3D 位置重建**，是 MRoPE 在 decode 阶段几乎"零开销"的关键。

## 3.7 优化点 ⑬：MRoPE + YaRN 长上下文复用

Qwen2.5-VL / Qwen3-VL 在长上下文场景下会跟 YaRN 叠加。[`mrope.py:253-261`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py#L253-L261)：

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

直接以"调用者方式"借用 YaRN 的 `inv_freq` / cache 计算——`MRotaryEmbedding` 本身不再重复实现长上下文逻辑。MRoPE / interleaved / YaRN / 普通四种组合最终都走同一个 Triton kernel。

---

# 第 4 部分：两套 RoPE 的并列对照

| 维度 | ViT 2D RoPE | LLM 3D MRoPE |
|------|-------------|---------------|
| 实现类 | `RotaryEmbedding` (base) | `MRotaryEmbedding` |
| 触发条件 | `partial_rotary_factor=0.5` 走 default 分支 | `mrope_section in rope_parameters` |
| `head_size` 例 (8B) | 72 | 128 |
| `rotary_dim` | head_size // 2 = 36 | head_size = 128（全旋转） |
| cos/sin cache shape | `(8192, 36)` = `(max_pos, rotary_dim)` | `(4 × max_pos, 128)` = `(1_048_576, rotary_dim)` |
| 分轴方式 | head_dim 沿中线切两半（前 h、后 w） | head_dim 按 `mrope_section=[24,20,20]` 切三段 |
| positions shape | `(L, 2)` per ViT block 的 `(h_id, w_id)` | `(3, num_tokens)` per LLM token 的 `(t, h, w)` |
| 位置构造在哪 | ViT 内 `rot_pos_ids` (lru_cache) | scheduler `get_mrope_input_positions` |
| forward kernel | FlashAttn Triton `apply_rotary_emb`（q+k 合并 launch） | vLLM Triton `_triton_mrope_forward`（融合 split/select/旋转/写回） |
| 主要优化 | cos_sin_cache 常驻 / lru_cache 位置 / 二次查表 / 半 head 省 cat / Q+K 同 kernel | cache 4× / 3 轴查同一 1D cache / mask 互斥相加替 select / in-place / `mrope_position_delta` decode 免重算 / YaRN 复用 |
| 与纯文本 RoPE 关系 | 与文本 RoPE 共用 base class，只是 head_dim 减半 | 文本 token 三轴值相同时严格退化成普通 RoPE |
| 视频时间维 | 不参与 ViT RoPE（`repeat(t, 1)` 复用 2D 位置） | 由 LLM 端 t 轴承担（t = `frame_idx · t_factor`） |

**最关键的对照**：

```
ViT 2D RoPE: 一份 1D cache + 二维 positions 二次 lookup + head_dim 切两半
LLM 3D MRoPE: 一份 1D cache + 三维 positions 三次 lookup + head_dim 切三段
```

数学结构完全同构，只是"轴数"和"切法"参数化。vLLM 的工程价值在于把这种参数化推得很彻底：**一份 cos_sin_cache + 一个 Triton kernel 就同时覆盖了 N 维 MRoPE 的所有变体**（普通 / 2D / 3D / 4D XDRoPE / interleaved）。

---

# 第 5 部分：一张 224×224 单图的端到端 walk-through

设输入 prompt 是 `<text_a> <image> <text_b>`，`<image>` 是 224×224 → `(grid_t=1, grid_h=14, grid_w=14) → 196 patches`。`text_a` 长度 10，`text_b` 长度 5。spatial_merge_size = 2，所以在 LLM 端图像 token 实际占 `14*14 / (2*2) = 49` 个位置。

## 5.1 ViT 端 2D RoPE 路径

```text
init 时（构造模型）：
  RotaryEmbedding(head_size=72, rotary_dim=36, max_position=8192)
  → cos_sin_cache shape (8192, 36)  bf16  GPU buffer

forward 时（每张图）：
  rot_pos_ids(14, 14, M=2):
    └─ lru_cache 命中 → 0 计算
    └─ shape (196, 2) CPU int64

  prepare_encoder_metadata:
    └─ max_grid_size = 14
    └─ cos, sin = cos_sin_cache[:14].chunk(2, -1)  → each (14, 18)
    └─ cos[pos_ids].flatten(1)  → (196, 36)
    └─ sin[pos_ids].flatten(1)  → (196, 36)

  Vision Block × 27:
    qk: (2, 196, 16, 72)             ← q, k 合并 batch
    ApplyRotaryEmb(qk, cos, sin):
      ├─ cos / sin unsqueeze head dim → (196, 1, 36)
      ├─ x1, x2 = chunk(qk, 2, -1)    → each (2, 196, 16, 36)
      ├─ o1 = x1*cos - x2*sin
      ├─ o2 = x2*cos + x1*sin
      └─ cat([o1, o2], -1)            → (2, 196, 16, 72)
    → 1 次 Triton launch 完成 q/k 一起旋转
```

每张图整条 RoPE 路径在 GPU 上的"重活"实际只有 27 次 `apply_rotary_emb` kernel（每个 block 一次）。

## 5.2 LLM 端 3D MRoPE 路径

```text
init 时（构造模型）：
  MRotaryEmbedding(head_size=128, rotary_dim=128,
                   max_position=262144 → cache_max=1_048_576,
                   mrope_section=[24, 20, 20])
  → cos_sin_cache shape (1_048_576, 128)  bf16  GPU buffer

prefill 时（每个请求）：
  get_mrope_input_positions:
    输入 prompt 总 token 数 = 10 + 49 + 5 = 64
    text_a (10): positions = [[0..9], [0..9], [0..9]]
    image  (49 = 7×7 merged): t=10, (h=0..6, w=0..6)
                              positions = [[10]*49, [10+0..6 broadcast], [10+0..6 broadcast]]
                              llm_positions.max() = 10 + 7 - 1 = 16
    text_b (5): positions = [[17..21], [17..21], [17..21]]

    → llm_positions shape (3, 64)
    → mrope_position_delta = 17 - 10 - 49 - ... 由 max+1-N 算出（这里 ≈ -42）

  LLM forward 每层 attention 前：
    cos_sin = cos_sin_cache[positions]     # (3, 64, 128) ← 一次 fancy-index
    cos, sin = cos_sin.chunk(2, -1)         # each (3, 64, 64)
    triton_mrope_forward (grid=(64,)):
      每个 token：
        ├─ 按 mrope_section=[24,20,20] mask 选 cos/sin
        ├─ 加载 q (heads_q × 128), k (heads_kv × 128) 的左右半
        ├─ 左半 *cos - 右半 *sin → 新左半
        ├─ 右半 *cos + 左半 *sin → 新右半
        └─ in-place 写回 q_ptr, k_ptr

decode 时（每生成 1 个新 token）：
  get_next_input_positions_tensor:
    next_pos = mrope_position_delta + context_len  ← 标量加法
    positions[:, out_offset] = next_pos            ← 三轴同值
  → forward 走 triton_mrope，但因为三轴同值，等价于普通 RoPE
```

decode 阶段每生成一个 token 只多一次"标量加法 + 三轴广播 + 一次 Triton kernel"——所以 **MRoPE 在 decode 阶段几乎和普通 LLM RoPE 一样便宜**。

---

## 附录：源码索引

| 内容 | 路径 |
|------|------|
| RoPE 基类 + cos/sin cache | [`base.py`](../vllm/vllm/model_executor/layers/rotary_embedding/base.py) |
| `ApplyRotaryEmb`（被两套 RoPE 复用） | [`common.py`](../vllm/vllm/model_executor/layers/rotary_embedding/common.py) |
| MRoPE 类 + Triton kernel | [`mrope.py`](../vllm/vllm/model_executor/layers/rotary_embedding/mrope.py) |
| `get_rope` 工厂（含 partial_rotary_factor 分支 + mrope_section 分支） | [`__init__.py`](../vllm/vllm/model_executor/layers/rotary_embedding/__init__.py) |
| ViT `rot_pos_ids` / `rot_pos_emb` / RoPE 初始化 | [`qwen3_vl.py:575-677`](../vllm/vllm/model_executor/models/qwen3_vl.py#L575-L677) |
| LLM 端 `get_mrope_input_positions` | [`qwen3_vl.py:2562-2647`](../vllm/vllm/model_executor/models/qwen3_vl.py#L2562-L2647) |
| ViT attention 调用 `apply_rotary_emb` | [`qwen2_5_vl.py:421-440`](../vllm/vllm/model_executor/models/qwen2_5_vl.py#L421-L440) |
