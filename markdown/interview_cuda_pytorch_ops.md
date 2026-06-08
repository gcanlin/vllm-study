# CUDA 与 PyTorch 算子面试速记

> 目标：从零开始把 CUDA 语法说清楚，然后给出**可以背下来**的算子实现模板（RMSNorm、LayerNorm、Softmax、GEMV 等），最后给一组 PyTorch 经典面试题（MHA / GQA / RoPE / KV Cache / Top-k）。
> 阅读顺序：Part 0 → Part 1（reduce 模板，最核心）→ Part 2 → Part 3。
> 风格：每个 kernel 只保留最小必要代码，方便手写默背。

---

# Part 0 · CUDA 极简入门（零基础）

## 0.1 编程模型：grid / block / thread

CUDA 把并行任务切成三层：

```
grid  ─┬─ block(0,0)  ─┬─ thread(0,0)  thread(1,0)  ...
       │                └─ thread(0,1)  ...
       ├─ block(1,0)  ─ ...
       └─ ...
```

- 一次 `kernel<<<gridDim, blockDim>>>(...)` 启动一个 grid。
- 一个 block 内的线程可以**共享 shared memory** 和 **同步**（`__syncthreads()`）。
- 不同 block 之间**不能直接同步**，要同步只能 kernel 结束后再启一个 kernel。

内置变量（每个线程都能读）：

| 变量 | 含义 |
|---|---|
| `threadIdx.x/y/z` | 线程在 block 内的坐标 |
| `blockIdx.x/y/z`  | block 在 grid 内的坐标 |
| `blockDim.x/y/z`  | block 的形状（每个 block 多少线程） |
| `gridDim.x/y/z`   | grid 的形状（多少个 block） |

最常用的一维线性索引：

```cpp
int tid = threadIdx.x;                              // block 内
int gid = blockIdx.x * blockDim.x + threadIdx.x;    // 全局
```

## 0.2 内存层次（性能的根源）

| 类型 | 范围 | 延迟 | 用法 |
|---|---|---|---|
| Register | 单线程私有 | ~1 cycle | 局部变量 |
| Shared memory | block 内共享 | ~几十 cycle | `__shared__ float s[];`，手动 cache |
| L1 / L2 cache | 自动 | - | 看运气 |
| Global memory（显存 / HBM） | 所有线程 | ~几百 cycle | `cudaMalloc` 出来的指针 |
| Constant memory | 只读 | 缓存友好 | 小常量表 |

**核心性能口诀**：少访问 global memory，能放 shared / register 就放。

## 0.3 Warp 与 SIMT

- GPU 实际调度单位是 **warp = 32 个线程**。
- 一个 warp 内 32 线程**同时执行同一条指令**（SIMT）。
- 如果 if/else 让 warp 里线程走不同路径 → **warp divergence**，串行执行，性能掉。
- Warp 内可以用 **shuffle 指令**直接交换寄存器，不走 shared memory，最快。

```cpp
// warp 内做 sum reduce：每次把距离为 offset 的两个线程值加起来
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
}
// 此时 warp 内 32 个线程的 val 都等于这 32 个值的和
```

## 0.4 同步原语

- `__syncthreads()` — block 内所有线程都到这里才继续。**只能用于 block 内**。
- `__syncwarp()` — warp 内同步。
- `__shfl_xor_sync(mask, val, offset)` — warp 内 XOR 模式 shuffle。
- `atomicAdd(ptr, val)` — 跨 block 累加，慢但正确。

## 0.5 Kernel 启动与显存

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// host 端
float *dA, *dB, *dC;
cudaMalloc(&dA, n * sizeof(float));
cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
// ...
int block = 256;
int grid  = (n + block - 1) / block;
add_kernel<<<grid, block>>>(dA, dB, dC, n);
cudaDeviceSynchronize();
```

要点：
- `__global__` = host 调用、device 执行（kernel 函数）。
- `__device__` = device 调用、device 执行（device 端工具函数）。
- `__host__` = 普通 CPU 函数。
- 启动配置 `<<<grid, block, smem_bytes, stream>>>`，后两个可选。

## 0.6 写 kernel 前默念三句话

1. **每个线程负责哪个元素？** → 决定 `gid` 怎么算，要不要 `if (gid < n)` 边界。
2. **要不要 reduce / share 数据？** → 要 → 用 shared memory + `__syncthreads`，或 warp shuffle。
3. **访存连续吗？** → 同一 warp 的 32 个线程访问连续 32 个地址 → coalesced，最快。

---

# Part 1 · 必背模板：Block Reduce

几乎所有 norm / softmax / attention kernel 都建立在 reduce 上，**这一段必须能默写**。

## 1.1 Warp reduce sum（用 shuffle）

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;  // warp 内 32 个线程都拿到 sum
}
```

## 1.2 Block reduce sum（warp reduce + shared memory）

```cpp
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float smem[32];               // 最多 32 个 warp（=1024 线程）
    int lane = threadIdx.x & 31;             // warp 内 id
    int wid  = threadIdx.x >> 5;             // 第几个 warp

    val = warp_reduce_sum(val);              // 每个 warp 内先 reduce
    if (lane == 0) smem[wid] = val;          // 每 warp 的代表写进 shared
    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = (threadIdx.x < num_warps) ? smem[lane] : 0.f;
    if (wid == 0) val = warp_reduce_sum(val); // 让第 0 个 warp 再 reduce 一次
    return val;                              // threadIdx.x == 0 拿到最终值
}
```

> 想得到 max reduce，把 `+=` 换成 `val = fmaxf(val, __shfl_xor_sync(...))` 即可。

---

# Part 2 · 经典算子的 CUDA 实现

下面 kernel 的共同假设：
- 输入按 **`[..., hidden]`** 形状排布，最后一维是要规约的维度。
- 一个 block 处理一行（一个 token），blockDim.x = 256 或 1024。
- 输入是 fp32，省去 fp16 / bf16 cast，面试可口头说"实际生产里用 cuda_bf16 / __half2"。

## 2.1 Vector Add（最入门）

```cpp
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// 启动: vec_add<<<(n+255)/256, 256>>>(a, b, c, n);
```

要点：grid-stride loop 可处理超大 n：

```cpp
for (int i = gid; i < n; i += gridDim.x * blockDim.x) { ... }
```

## 2.2 RMSNorm

公式：`y = x / sqrt(mean(x^2) + eps) * weight`

```cpp
// 输入 x: [num_rows, hidden]，每个 block 处理一行
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,    // [hidden]
    float* __restrict__ y,
    int hidden, float eps)
{
    int row = blockIdx.x;
    const float* x_row = x + row * hidden;
    float*       y_row = y + row * hidden;

    // 1) 每个线程负责若干列，先算局部 sum(x^2)
    float local_sq = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = x_row[i];
        local_sq += v * v;
    }

    // 2) block 内 reduce
    float sum_sq = block_reduce_sum(local_sq);

    // 3) 第 0 个线程算 rsqrt，存进 shared 给全 block 用
    __shared__ float rstd;
    if (threadIdx.x == 0) rstd = rsqrtf(sum_sq / hidden + eps);
    __syncthreads();

    // 4) 写回
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        y_row[i] = x_row[i] * rstd * weight[i];
    }
}
// 启动: rmsnorm_kernel<<<num_rows, 256>>>(x, w, y, hidden, eps);
```

口诀：**一行一 block，两遍扫 hidden（一遍求 sum，一遍写）**。

## 2.3 LayerNorm

比 RMSNorm 多一个减均值，多一个 bias：

```cpp
// y = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void layernorm_kernel(
    const float* x, const float* weight, const float* bias,
    float* y, int hidden, float eps)
{
    int row = blockIdx.x;
    const float* x_row = x + row * hidden;
    float*       y_row = y + row * hidden;

    // 1) 同时算 sum 和 sum_sq（Welford 更稳，但面试这个就够）
    float local_sum = 0.f, local_sq = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v;
        local_sq  += v * v;
    }
    float sum    = block_reduce_sum(local_sum);
    float sum_sq = block_reduce_sum(local_sq);

    __shared__ float mean, rstd;
    if (threadIdx.x == 0) {
        mean = sum / hidden;
        float var = sum_sq / hidden - mean * mean;
        rstd = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        y_row[i] = (x_row[i] - mean) * rstd * weight[i] + bias[i];
    }
}
```

> 数值更稳的写法是 **Welford 算法**，面试问到了能说出关键点即可：在线维护 `(count, mean, M2)`，合并两段统计量时按权重更新 `mean`，`M2` 累加修正项，避免大数相减误差。

## 2.4 Safe Softmax（online / 两遍版）

公式：`y_i = exp(x_i - max) / sum(exp(x_j - max))`

**两遍朴素版**（先 max，再 sum，再写）：

```cpp
__global__ void softmax_kernel(const float* x, float* y, int hidden) {
    int row = blockIdx.x;
    const float* x_row = x + row * hidden;
    float*       y_row = y + row * hidden;

    // 1) max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        local_max = fmaxf(local_max, x_row[i]);
    float row_max = block_reduce_max(local_max);   // 同理 reduce

    // 2) sum(exp)
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        local_sum += expf(x_row[i] - row_max);
    float row_sum = block_reduce_sum(local_sum);

    // 3) 写回
    float inv = 1.f / row_sum;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        y_row[i] = expf(x_row[i] - row_max) * inv;
}
```

**Online softmax**（一遍扫完，FlashAttention 的核心）—— 单线程视角：

```cpp
float m = -INFINITY, l = 0.f;       // 当前 max 和 sum(exp(x-m))
for (i = 0..hidden-1) {
    float m_new = fmaxf(m, x[i]);
    l = l * expf(m - m_new) + expf(x[i] - m_new);
    m = m_new;
}
// 写一遍：y[i] = exp(x[i] - m) / l
```

合并两段统计量 `(m1,l1)` 与 `(m2,l2)`：
```
m = max(m1, m2)
l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
```
这是 FlashAttention 跨 block 拼接的关键公式，能背下来面试就过半。

## 2.5 GEMV（y = A · x，最常考的矩阵向量乘）

A 是 `[M, K]`，x 是 `[K]`，y 是 `[M]`。一个 block 负责输出 y 的一行：

```cpp
__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int K) {
    int row = blockIdx.x;
    const float* A_row = A + row * K;

    float local = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        local += A_row[k] * x[k];

    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) y[row] = sum;
}
// 启动: gemv_kernel<<<M, 256>>>(A, x, y, M, K);
```

要点：A 按行主序时，同一 warp 32 线程访问 `A_row[k..k+31]` —— 连续 32 个 float，coalesced。

## 2.6 SGEMM 简化版（C = A · B）思路

面试讲清楚思路即可：

1. 把输出 C 切成 `BM × BN` 的 tile，**一个 block 算一个 tile**。
2. 沿 K 维分块 `BK`，把 A 的 `BM×BK` 子块和 B 的 `BK×BN` 子块**搬到 shared memory**。
3. 每个线程在寄存器里再开一个 `TM × TN` 的小累加（register tiling）。
4. K 维 for 循环里：搬数据 → `__syncthreads()` → 算小 tile → 累加。
5. 优化关键词：**double buffer / async copy（cp.async）/ swizzle 避 bank conflict / Tensor Core（wmma）**。

## 2.7 ReLU / GELU（element-wise，热身）

```cpp
__global__ void gelu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.f + tanhf(t));
    }
}
```

## 2.8 算子级面试常考问题清单

- 为什么 RMSNorm 比 LayerNorm 快？少一遍减均值、少一次 reduce、少一个 bias，访存量也少（一些实现里）。
- 为什么 attention 需要 safe softmax？`exp(x)` 在 `x > 88` 时 fp32 就溢出。
- Online softmax 怎么并行？每个 chunk 独立算 `(m, l)`，再用 `max(m1,m2)` 那条公式合并。
- 为什么 reduce 要先 warp shuffle 再 shared？shuffle 不走 shared memory，没 bank conflict、没同步，更快；shared 只用来在 warp 之间传一次结果。
- block size 怎么选？`hidden=4096` 取 `blockDim.x=1024`、`hidden<=1024` 取 256，让一个线程负责 4~16 个元素最划算。
- coalesced 访存：同一 warp 的连续 `threadIdx.x` 访问连续地址。
- shared memory bank conflict：32 个 bank，每个 4B，相邻地址错开 32 不冲突；常见解法 padding `+1` 或 swizzle。
- fp16 / bf16 实现：用 `__half2` / `nv_bfloat162` 做 vectorized load（一次取 2 个），reduce 时再升 fp32 累加，避免精度损失。

---

# Part 3 · PyTorch 经典面试题

## 3.1 朴素 Multi-Head Attention

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):           # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d)
        q, k, v = qkv.unbind(dim=2)            # [B, T, h, d]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]   # [B, h, T, d]

        attn = (q @ k.transpose(-1, -2)) / self.d ** 0.5   # [B, h, T, T]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        out  = attn @ v                                    # [B, h, T, d]
        out  = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out)
```

## 3.2 GQA（Grouped-Query Attention，**重点**）

GQA：`n_heads` 个 Q 头，但只有 `n_kv_heads` 个 K/V 头（`n_kv_heads | n_heads`）。每 `group = n_heads // n_kv_heads` 个 Q 头共享一对 K/V。

```python
class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.group      = n_heads // n_kv_heads
        self.d          = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads    * self.d, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.wo = nn.Linear(n_heads * self.d, d_model,  bias=False)

    def forward(self, x, mask=None):                 # x: [B, T, D]
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.d).transpose(1, 2)  # [B, hq, T, d]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d).transpose(1, 2)  # [B, hk, T, d]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d).transpose(1, 2)

        # 关键一步：把 K/V 在 head 维复制 group 倍，让形状和 Q 对齐
        k = k.repeat_interleave(self.group, dim=1)   # [B, hq, T, d]
        v = v.repeat_interleave(self.group, dim=1)

        attn = (q @ k.transpose(-1, -2)) / self.d ** 0.5
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        out  = attn.softmax(dim=-1) @ v              # [B, hq, T, d]
        out  = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)
```

要点：
- **`repeat_interleave` 不是真复制**，PyTorch 会拷贝显存；生产里用 `expand` + 改 attention kernel 让 K/V stride=0 来省内存，FlashAttention2 支持原生 GQA。
- MQA = GQA 的极端情况 `n_kv_heads = 1`。
- 为什么 GQA 快？**KV cache 内存量 ÷ group**，长上下文 / 大 batch 时这是省 HBM 的关键。

## 3.3 RoPE（Rotary Position Embedding）

```python
def build_rope_cache(seq_len, head_dim, base=10000.0, device='cuda'):
    # 半数维度上算 theta
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)         # [T, half]
    return freqs.cos(), freqs.sin()           # 两个 [T, half]

def apply_rope(x, cos, sin):
    # x: [B, h, T, d]，把最后一维一半当实部一半当虚部
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos[None, None, :, :]              # 广播到 [1,1,T,d/2]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
```

口诀：**把 d 分成两半，前一半当实部，后一半当虚部，按位置做复数乘法 e^{iθ}**。Q 和 K 都要 apply，V 不用。

## 3.4 KV Cache（增量解码）

```python
class KVCache:
    def __init__(self, max_bs, max_len, n_kv_heads, d, dtype, device):
        shape = (max_bs, n_kv_heads, max_len, d)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.cur_len = 0

    def append(self, k_new, v_new):           # [B, hk, T_new, d]
        T_new = k_new.shape[2]
        self.k[:, :, self.cur_len : self.cur_len + T_new] = k_new
        self.v[:, :, self.cur_len : self.cur_len + T_new] = v_new
        self.cur_len += T_new
        return self.k[:, :, : self.cur_len], self.v[:, :, : self.cur_len]
```

可能被追问：
- KV cache 内存怎么算？`2 * n_layers * n_kv_heads * head_dim * max_len * dtype_bytes`。
- Prefill 和 decode 的区别？prefill 一次填 prompt 长度 T，decode 每步追加 1。
- PagedAttention（vLLM）解决了什么？把 KV cache 切成固定大小 block，按需分配，避免预留 max_len 带来的碎片浪费。

## 3.5 Causal Mask + Softmax（增量解码场景）

```python
def causal_mask(T_q, T_kv, device):
    # query 在序列末尾，每个 query 可看见前缀 K
    i = torch.arange(T_q, device=device).view(-1, 1)
    j = torch.arange(T_kv, device=device).view(1, -1)
    return j <= (T_kv - T_q + i)             # [T_q, T_kv] bool
```

## 3.6 Top-k / Top-p 采样

```python
def sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature

    # top-k
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[..., -1:]] = -float('inf')

    # top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum   = probs.cumsum(dim=-1)
        mask  = cum - probs > top_p          # 保留第一个超过 top_p 的也算入
        sorted_logits[mask] = -float('inf')
        logits.scatter_(-1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## 3.7 SwiGLU / GeGLU FFN

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up   = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

口诀：`down(silu(gate(x)) * up(x))`，Llama 系列标配。

## 3.8 FlashAttention 的口头讲解

面试不要写 kernel 代码，讲清楚思路：

1. 朴素 attention 把 `[T, T]` 的 attention 矩阵物化在 HBM，T=8k 时 64M 个 float。
2. FlashAttention 把 Q 切成 `Br` 行的 block、K/V 切成 `Bc` 列的 block，在 **SRAM（shared memory）里完成一个 `Br × Bc` 子矩阵的 softmax**。
3. 跨 Bc 用 **online softmax 合并公式** 累计 `(m, l)`，最后归一化。
4. 收益：HBM 读写从 `O(T^2)` 降到 `O(T·d)`，并且不存中间 attention 矩阵 → 内存省、长上下文可行。
5. 反向也能 fused，不存 attention 矩阵，靠重算。

## 3.9 GQA + KV Cache + RoPE 完整版（增量解码）

把 3.2 GQA、3.3 RoPE、3.4 KV Cache、3.5 Causal Mask 串起来，做成一个能 prefill + decode 两阶段都跑的 Attention 模块。这是 LLM 推理引擎里 attention 层的最小完整骨架。

### 设计要点

- KV cache 形状 `[B, n_kv_heads, max_len, d]`，**只缓存 n_kv_heads 个头**（GQA 的省内存关键）。
- 维护 `cur_len`：表示这条序列已经写了多少 K/V，下一次写入的起点。
- **Prefill**（输入 prompt，`T > 1`）：一次性算出 `T` 个 token 的 Q/K/V，写进 cache `[0:T]`，做 causal masked attention。
- **Decode**（一次一个 token，`T = 1`）：算 1 个 Q 和 1 个 K/V，K/V append 到 cache，Q 对整段 cache 做 attention，**不需要 mask**（因为新 token 本来就能看到所有历史）。
- RoPE 的 position 用 `[cur_len, cur_len + T)`，对 Q 和**新写入的 K**都要 apply（cache 里的 K 已经是 apply 过 RoPE 的）。

### 完整代码

```python
import torch, torch.nn as nn, torch.nn.functional as F

def build_rope_cache(seq_len, head_dim, base=10000.0, device='cuda'):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()                # 都是 [seq_len, half]

def apply_rope(x, cos, sin):                       # x: [B, h, T, d]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos[None, None]; sin = sin[None, None]   # → [1,1,T,d/2]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


class GQAWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_bs, max_len,
                 dtype=torch.float16, device='cuda'):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.group = n_heads // n_kv_heads
        self.d     = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads    * self.d, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.wo = nn.Linear(n_heads * self.d, d_model,   bias=False)

        # KV cache: 预分配，按序列写入
        cache_shape = (max_bs, n_kv_heads, max_len, self.d)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.cur_len = 0                          # 简化：单序列共享，实际生产里每条序列各自维护

        cos, sin = build_rope_cache(max_len, self.d, device=device)
        self.register_buffer('cos', cos.to(dtype))
        self.register_buffer('sin', sin.to(dtype))

    def reset(self):
        self.cur_len = 0

    def forward(self, x):                          # x: [B, T, D]，prefill T>1 / decode T=1
        B, T, _ = x.shape
        start, end = self.cur_len, self.cur_len + T

        # 1) Q/K/V 投影 + 变形
        q = self.wq(x).view(B, T, self.n_heads,    self.d).transpose(1, 2)  # [B, hq, T, d]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d).transpose(1, 2)  # [B, hk, T, d]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d).transpose(1, 2)

        # 2) 对当前段 [start, end) 应用 RoPE（Q 和新写入的 K，V 不用）
        cos, sin = self.cos[start:end], self.sin[start:end]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3) 写入 KV cache
        self.k_cache[:B, :, start:end] = k
        self.v_cache[:B, :, start:end] = v
        self.cur_len = end

        # 4) 取出 cache 里 [0, end) 全部 K/V 参与 attention
        k_all = self.k_cache[:B, :, :end]          # [B, hk, end, d]
        v_all = self.v_cache[:B, :, :end]

        # 5) GQA：把 K/V 复制到 hq 头数对齐
        k_all = k_all.repeat_interleave(self.group, dim=1)   # [B, hq, end, d]
        v_all = v_all.repeat_interleave(self.group, dim=1)

        # 6) Attention
        attn = (q @ k_all.transpose(-1, -2)) / self.d ** 0.5  # [B, hq, T, end]
        if T > 1:
            # prefill：当前 T 个 query 对 end 个 key 做 causal mask
            # query i 在全局位置 (start + i)，可以看见 key j ≤ start + i
            i = torch.arange(T,  device=x.device).view(-1, 1)
            j = torch.arange(end, device=x.device).view(1, -1)
            causal = j <= (start + i)                          # [T, end]
            attn = attn.masked_fill(~causal, float('-inf'))
        # decode (T==1) 不用 mask：当前 token 本来就能看见所有 cache

        out = attn.softmax(dim=-1) @ v_all          # [B, hq, T, d]
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)
```

### 用法（典型推理流程）

```python
attn = GQAWithKVCache(d_model=4096, n_heads=32, n_kv_heads=8,
                      max_bs=1, max_len=2048).cuda().half()

# Prefill：一次性吃完 prompt
prompt = torch.randn(1, 128, 4096, device='cuda', dtype=torch.float16)
attn.reset()
_ = attn(prompt)                       # cur_len: 0 → 128

# Decode：一步一个 token
for step in range(100):
    new_tok = torch.randn(1, 1, 4096, device='cuda', dtype=torch.float16)
    _ = attn(new_tok)                  # cur_len: 128 → 129 → 130 → ...
```

### KV Cache 到底省了哪部分计算？（重点）

很多人看完代码会问："Q 还是要乘整段 K，怎么算省？" —— 省的不是 attention 本身那一步矩阵乘，**省的是前 n 步 token 的 K/V 重复投影**。

考虑生成第 `n+1` 个 token：

| 步骤 | 无 KV cache | 有 KV cache |
|---|---|---|
| 输入 x 形状 | `[B, n+1, D]`（要重过整段历史） | `[B, 1, D]`（只新 token） |
| `wq(x)` | `(n+1)·D²` FLOPs | `1·D²` |
| `wk(x)` | `(n+1)·D·D_kv` | `1·D·D_kv` |
| `wv(x)` | `(n+1)·D·D_kv` | `1·D·D_kv` |
| RoPE | apply 到 n+1 个位置 | 只 apply 到第 n 个位置 |
| `q @ K^T` | `(n+1)·(n+1)·d` | `1·(n+1)·d` |
| `softmax·V` | `(n+1)·(n+1)·d` | `1·(n+1)·d` |
| **整段生成 N token 累计** | **O(N³·D)** | **O(N²·D)** |

直观结论：

- **Q/K/V 投影**（最耗算力的部分，每层 3 个大 GEMM）：原本要重算全部历史 token，cache 后**只算 1 个 token**。这是 KV cache 最大的收益。
- **RoPE**：原本要对整段 K 都重新打位置编码，cache 后只对新位置算。
- **Attention `q @ K^T`**：query 这一边从 `n+1` 降到 `1`（一行 query × 整段 key），从矩阵乘退化成 **GEMV** —— 它没"消失"，但量级从 `O(n²)` 降到 `O(n)`。
- **前面所有 transformer 层的 FFN、norm 等等**：cache 后这些层每步也只处理 1 个 token，省的不只是 attention 一层 —— 整个模型在 decode 期都是 `T=1`。

代价是显存：`2·n_layers·n_kv_heads·d·max_len·dtype_bytes` 的 KV cache 常驻 HBM。**用显存换算力，且每步还从 compute-bound 变成 memory-bound**（要把整段 cache 读出来），这就是为什么 GQA / PagedAttention / FlashDecoding 这些"省 KV 内存 / 高效读 KV"的优化在 decode 阶段收益最大。

### 常被追问

- **Prefill 与 decode 为什么共用一份代码？** —— 区别只有 `T`：prefill `T = prompt_len`，decode `T = 1`。同一份 forward 让 `start = cur_len`、`end = cur_len + T`，再判 `T > 1` 决定要不要 causal mask 就够了。
- **decode 阶段计算瓶颈在哪？** —— `T = 1` 时 attention 退化成 GEMV，**memory-bound**，瓶颈是把整段 KV cache 从 HBM 读出来。所以 GQA / MQA 收益巨大，FlashDecoding 用切 KV 维度并行也是这个思路。
- **`repeat_interleave` 在 decode 路径上浪费内存怎么办？** —— 生产里改用 `expand` 让 K/V 在 head 维 stride=0，或直接在 attention kernel 里读 KV 时按 `q_head_idx // group` 索引，无物化复制。FlashAttention2 / FlashDecoding 都是这么做的。
- **多条序列 batch 长度不同怎么办？** —— 上面 `cur_len` 是标量，只支持单序列。真实引擎里要么 padding 到相同长度（浪费显存），要么用 **PagedAttention**（vLLM）把 KV 切成 block，按 block 索引；要么用 **continuous batching + variable seqlen** 的 attention kernel（FlashAttention varlen 接口）。
- **怎么保证 RoPE 的位置编码不错位？** —— 关键就一句：**写进 cache 之前的 K 已经 apply 过 RoPE，所以下次 decode 取 cache 里 K 不要再 apply 第二次，只对新写入的那一截 apply**。

---

# 附录 · 复习卡片

| 主题 | 必须能默写的内容 |
|---|---|
| Warp shuffle reduce | `__shfl_xor_sync` 5 次循环（16→1） |
| Block reduce | warp reduce → shared[32] → 第 0 个 warp 再 reduce |
| RMSNorm | 一行一 block，sum(x²) → rsqrt → 写回 |
| LayerNorm | 多算一个 mean，bias 别忘 |
| Safe softmax | `x - max`，online 公式 `l = l·exp(m-m') + exp(x-m')` |
| GEMV | 一个 block 一行，coalesced 读 A_row |
| MHA | `qkv -> [B,T,3,h,d] -> unbind -> transpose` |
| GQA | K/V `repeat_interleave(group, dim=1)` 对齐到 hq |
| RoPE | 半实半虚 + `[cos·x1 - sin·x2, sin·x1 + cos·x2]` |
| KV Cache | `[B, hk, max_len, d]`，append 维护 cur_len |
| GQA + KV Cache | RoPE 用 `[cur_len, cur_len+T)`；prefill 才加 causal mask；K/V 写 cache 前先 apply RoPE |
| Top-p | sort → cumsum → 把 `cum - p > top_p` 的位置打 -inf |
| FlashAttention | Q 切行 / KV 切列，SRAM 内 online softmax，HBM 读写 O(Td) |

> 面试时讲算子的顺序模板：**算什么 → 形状/维度 → 并行划分（一个 block 处理什么）→ shared/reduce 怎么做 → 数值稳定/精度处理 → 进一步优化（vectorized load、fp16 累加 fp32、double buffer）**。把这条主线讲顺，比记代码细节更重要。

---

# Part 4 · LeetCode 中等题速记（贪心 / DP / 高频模式）

风格：每题一句话思路 + 最小 Python + 复杂度。优先掌握**模板**，同模板的题互相迁移。

## 4.1 贪心（Greedy）

### LC 55 · 跳跃游戏

思路：维护**当前能到达的最远位置 `far`**，遍历时若 `i > far` 则失败，否则 `far = max(far, i + nums[i])`。

```python
def canJump(nums):
    far = 0
    for i, x in enumerate(nums):
        if i > far: return False
        far = max(far, i + x)
    return True
# O(n) / O(1)
```

### LC 45 · 跳跃游戏 II（最少步数）

思路：BFS 分层 —— 用 `end` 标记当前一步能到的最远边界，遍历到 `end` 时步数 +1 并把 `end` 推到新 `far`。

```python
def jump(nums):
    steps = end = far = 0
    for i in range(len(nums) - 1):
        far = max(far, i + nums[i])
        if i == end:
            steps += 1
            end = far
    return steps
# O(n) / O(1)
```

### LC 56 · 合并区间

思路：按起点排序 → 一遍扫描，能合就改右端点，不能合就 append。

```python
def merge(intervals):
    intervals.sort()
    res = []
    for s, e in intervals:
        if res and s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res
# O(n log n) / O(1) 额外
```

### LC 435 · 无重叠区间（最少删除几个）

思路：按**右端点**排序，每次贪心选右端点最小的，下一个起点 ≥ 当前右端点才能保留。

```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    end, keep = -float('inf'), 0
    for s, e in intervals:
        if s >= end:
            keep += 1
            end = e
    return len(intervals) - keep
```

### LC 121/122 · 买卖股票

```python
# LC 121：只能买卖一次
def maxProfit1(p):
    lo, ans = float('inf'), 0
    for x in p:
        lo = min(lo, x); ans = max(ans, x - lo)
    return ans

# LC 122：可以多次买卖（贪心 = 所有正差和）
def maxProfit2(p):
    return sum(max(0, p[i] - p[i-1]) for i in range(1, len(p)))
```

### LC 134 · 加油站

思路：总和 < 0 必失败；否则从"累计油量第一次跌到负数后的下一个位置"出发一定成功。

```python
def canCompleteCircuit(gas, cost):
    total = tank = start = 0
    for i in range(len(gas)):
        d = gas[i] - cost[i]
        total += d; tank += d
        if tank < 0:
            start = i + 1; tank = 0
    return start if total >= 0 else -1
```

## 4.2 动态规划（DP）—— 按模板分类

### ① 一维线性 DP

**LC 53 · 最大子数组和**（Kadane 算法，必背）

```python
def maxSubArray(nums):
    cur = ans = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)   # 要么续上，要么从我重开
        ans = max(ans, cur)
    return ans
# O(n) / O(1)
```

**LC 198 · 打家劫舍**

```python
def rob(nums):
    prev, cur = 0, 0
    for x in nums:
        prev, cur = cur, max(cur, prev + x)  # 不偷 / 偷
    return cur
```

**LC 300 · 最长递增子序列（LIS）**

`O(n²)` DP 版：

```python
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

`O(n log n)` 二分版（**面试加分**）：维护一个"贪心 tails"数组，`tails[k]` 表示长度为 k+1 的递增序列的最小结尾。

```python
from bisect import bisect_left
def lengthOfLIS_fast(nums):
    tails = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)
```

### ② 二维字符串 DP

**LC 1143 · 最长公共子序列（LCS）**

```python
def longestCommonSubsequence(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
# O(mn) / O(mn)，可滚动数组优化到 O(n)
```

**LC 72 · 编辑距离**

```python
def minDistance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],     # 删
                                   dp[i][j-1],     # 插
                                   dp[i-1][j-1])   # 改
    return dp[m][n]
```

**LC 5 · 最长回文子串**（中心扩展，比 DP 更简洁）

```python
def longestPalindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]
    ans = ""
    for i in range(len(s)):
        a, b = expand(i, i), expand(i, i+1)   # 奇 / 偶 长度
        ans = max(ans, a, b, key=len)
    return ans
# O(n²) / O(1)
```

### ③ 背包模型

**LC 322 · 零钱兑换**（完全背包，求最少硬币数）

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

**LC 416 · 分割等和子集**（01 背包）

```python
def canPartition(nums):
    s = sum(nums)
    if s % 2: return False
    target = s // 2
    dp = [False] * (target + 1); dp[0] = True
    for x in nums:
        for j in range(target, x - 1, -1):    # 倒序避免重复用
            dp[j] = dp[j] or dp[j-x]
    return dp[target]
```

> 01 背包口诀：**"外物内容量、容量倒序"**；完全背包：**"外物内容量、容量正序"**。

### ④ 网格 DP

**LC 62 · 不同路径** / **LC 64 · 最小路径和**

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0: continue
            up   = grid[i-1][j] if i > 0 else float('inf')
            left = grid[i][j-1] if j > 0 else float('inf')
            grid[i][j] += min(up, left)
    return grid[m-1][n-1]
```

## 4.3 双指针 / 滑动窗口

**LC 3 · 无重复字符的最长子串**

```python
def lengthOfLongestSubstring(s):
    seen = {}; l = ans = 0
    for r, c in enumerate(s):
        if c in seen and seen[c] >= l:
            l = seen[c] + 1
        seen[c] = r
        ans = max(ans, r - l + 1)
    return ans
```

**LC 15 · 三数之和**

```python
def threeSum(nums):
    nums.sort(); res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l+1]: l += 1
                while l < r and nums[r] == nums[r-1]: r -= 1
                l += 1; r -= 1
            elif s < 0: l += 1
            else: r -= 1
    return res
```

**LC 76 · 最小覆盖子串**（滑窗模板，难一点但非常高频）

```python
from collections import Counter
def minWindow(s, t):
    need = Counter(t); missing = len(t)
    l = start = 0; best = (float('inf'), 0, 0)
    for r, c in enumerate(s):
        if need[c] > 0: missing -= 1
        need[c] -= 1
        while missing == 0:                 # 收缩左边界
            if r - l < best[0]:
                best = (r - l, l, r + 1)
            need[s[l]] += 1
            if need[s[l]] > 0: missing += 1
            l += 1
    return s[best[1]:best[2]] if best[0] < float('inf') else ""
```

## 4.4 二分查找

**LC 33 · 搜索旋转排序数组**（二分变形，必背）

```python
def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target: return m
        if nums[l] <= nums[m]:                       # 左半有序
            if nums[l] <= target < nums[m]: r = m - 1
            else: l = m + 1
        else:                                        # 右半有序
            if nums[m] < target <= nums[r]: l = m + 1
            else: r = m - 1
    return -1
```

## 4.5 单调栈

**LC 739 · 每日温度**

```python
def dailyTemperatures(T):
    res, stk = [0] * len(T), []        # stk 存下标，温度递减
    for i, t in enumerate(T):
        while stk and T[stk[-1]] < t:
            j = stk.pop()
            res[j] = i - j
        stk.append(i)
    return res
```

**LC 84 · 柱状图最大矩形**（单调栈经典）

```python
def largestRectangleArea(h):
    h = h + [0]                        # 哨兵让栈最后清空
    stk, ans = [], 0
    for i, x in enumerate(h):
        while stk and h[stk[-1]] > x:
            top = stk.pop()
            left = stk[-1] if stk else -1
            ans = max(ans, h[top] * (i - left - 1))
        stk.append(i)
    return ans
```

## 4.6 BFS / DFS / 回溯

**LC 200 · 岛屿数量**（DFS）

```python
def numIslands(grid):
    m, n = len(grid), len(grid[0]); ans = 0
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                ans += 1; dfs(i, j)
    return ans
```

**LC 46 · 全排列**（回溯模板）

```python
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:]); return
        for i, x in enumerate(nums):
            if used[i]: continue
            used[i] = True; path.append(x)
            backtrack(path, used)
            path.pop(); used[i] = False
    backtrack([], [False] * len(nums))
    return res
```

**LC 78 · 子集**

```python
def subsets(nums):
    res = [[]]
    for x in nums:
        res += [s + [x] for s in res]
    return res
# O(n·2ⁿ)
```

## 4.7 LeetCode 速记表

| 模式 | 关键题号 | 一句话模板 |
|---|---|---|
| 贪心 - 跳跃 | 55 / 45 | 维护 `far`，分层边界 `end` |
| 贪心 - 区间 | 56 / 435 | 按起点合并 / 按右端点保留 |
| Kadane | 53 | `cur = max(x, cur + x)` |
| 打家劫舍 | 198 | `cur = max(cur, prev + x)` |
| LIS | 300 | 朴素 DP 或 `bisect_left(tails, x)` |
| LCS / 编辑距离 | 1143 / 72 | 二维 DP，相等取左上，否则取邻居+1 |
| 01 背包 | 416 | 容量倒序 |
| 完全背包 | 322 | 容量正序 |
| 滑动窗口 | 3 / 76 | 右指针扩张，左指针收缩 |
| 二分变形 | 33 | 先判哪边有序，再判 target 在不在 |
| 单调栈 | 739 / 84 | 维护"严格递增/递减"下标栈，弹出时结算 |
| BFS/DFS | 200 | 网格 DFS 把访问过的改成 '0' |
| 回溯 | 46 / 78 | `path.append → 递归 → path.pop` |

> 面试讲题模板：**问清输入输出 → 举一个最小例子 → 说思路与复杂度 → 写代码 → 自己走一遍样例 → 讨论边界（空、单元素、溢出）**。模板比 trick 重要 100 倍。

---

# Part 5 · 集合通信算子（NCCL / 分布式）

> LLM 训练和推理一旦上多卡,通信算子就是绕不开的核心 —— TP 要 AllReduce、MoE 要 All2All、FSDP 要 AllGather + ReduceScatter、PP 要 Send/Recv。
> 这一部分先把 **7 个核心算子的数学定义**讲清楚,再讲 **Ring AllReduce** 这条带宽最优算法,然后给一段示意性的 CUDA 实现骨架(实际生产用 NCCL,但你要知道底层在干什么)。

## 5.1 为什么 LLM 离不开这些算子

| 并行策略 | 用到的通信算子 | 出现位置 |
|---|---|---|
| DP(数据并行) | **AllReduce** 梯度 | backward 之后 |
| TP(张量并行) | **AllReduce** activations | 每个 attention / FFN 末尾 |
| PP(流水并行) | **Send/Recv** activations | 相邻 stage 之间 |
| EP(专家并行,MoE) | **All2All** dispatch + combine | 进/出 expert 时 |
| FSDP / ZeRO-3 | **AllGather** weights + **ReduceScatter** gradients | forward 取权重 / backward 同步梯度 |
| SP(序列并行) | **AllGather** / **ReduceScatter** | norm 前后 |

## 5.2 7 个核心算子的数学定义

约定:`N` 个 GPU,每个 GPU 编号 `r = 0..N-1`,每张卡上有数据 `x_r`,操作后变成 `y_r`。

### ① Broadcast(广播,1→N)

源 GPU `s` 的数据复制给所有 GPU:
```
y_r = x_s,  ∀ r
```
```
GPU 0: [A]  ─┐
GPU 1: [ ]  ─┼─→  GPU 0: [A]   GPU 1: [A]   GPU 2: [A]   GPU 3: [A]
GPU 2: [ ]  ─┤
GPU 3: [ ]  ─┘
```

### ② Reduce(规约,N→1)

把所有 GPU 的数据按 op(+, max, ...)归约到一个目标 GPU `d`:
```
y_d = ⊕_{r=0..N-1} x_r       (其它 GPU 的 y 不定义)
```
```
GPU 0: [A]  ─┐
GPU 1: [B]  ─┼─→  GPU 0: [A+B+C+D]   其它 GPU: 不管
GPU 2: [C]  ─┤
GPU 3: [D]  ─┘
```

### ③ AllReduce(全规约,N→N) ★最高频

每个 GPU 都拿到全局规约结果:
```
y_r = ⊕_{r'=0..N-1} x_{r'},  ∀ r
```
```
GPU 0: [A]  ─→  [A+B+C+D]
GPU 1: [B]  ─→  [A+B+C+D]
GPU 2: [C]  ─→  [A+B+C+D]
GPU 3: [D]  ─→  [A+B+C+D]
```
**核心公式:`AllReduce = ReduceScatter + AllGather`** —— 这是 Ring AllReduce 的根基。

### ④ Scatter(分发,1→N)

源 GPU 把一段数据切成 N 份,发给每个 GPU:
```
x_s = [c_0, c_1, ..., c_{N-1}]
y_r = c_r
```
```
GPU 0: [A0,A1,A2,A3]  ─→  GPU 0: [A0]
                          GPU 1: [A1]
                          GPU 2: [A2]
                          GPU 3: [A3]
```

### ⑤ Gather(收集,N→1)

把所有 GPU 的数据拼到目标 GPU:
```
y_d = concat(x_0, x_1, ..., x_{N-1})
```

### ⑥ AllGather(全收集,N→N)

每个 GPU 都拿到所有 GPU 数据的拼接:
```
y_r = concat(x_0, x_1, ..., x_{N-1}),  ∀ r
```
```
GPU 0: [A]  ─→  [A,B,C,D]
GPU 1: [B]  ─→  [A,B,C,D]
GPU 2: [C]  ─→  [A,B,C,D]
GPU 3: [D]  ─→  [A,B,C,D]
```

### ⑦ ReduceScatter(规约 + 分片,N→N)

每个 GPU 把自己的数据切成 N 份,**第 r 份在所有 GPU 间规约后只发给 GPU r**:
```
x_r = [c_{r,0}, c_{r,1}, ..., c_{r,N-1}]
y_r = ⊕_{r'} c_{r',r}      (第 r 份的全局规约)
```
```
GPU 0: [A0,A1,A2,A3]
GPU 1: [B0,B1,B2,B3]    ─→   GPU 0: [A0+B0+C0+D0]
GPU 2: [C0,C1,C2,C3]         GPU 1: [A1+B1+C1+D1]
GPU 3: [D0,D1,D2,D3]         GPU 2: [A2+B2+C2+D2]
                             GPU 3: [A3+B3+C3+D3]
```

### ⑧ All2All(全交换,N→N) ★MoE 必备

每个 GPU 都把自己的数据切成 N 份,**第 j 份发给 GPU j**;同时收到来自每个 GPU 的对应份:
```
x_r = [c_{r,0}, c_{r,1}, ..., c_{r,N-1}]
y_r = [c_{0,r}, c_{1,r}, ..., c_{N-1,r}]
```
就是把矩阵 `c[r][j]` 在 GPU 之间做一次**转置**。

```
GPU 0: [a0,a1,a2,a3]                      GPU 0: [a0,b0,c0,d0]
GPU 1: [b0,b1,b2,b3]    ─All2All─→        GPU 1: [a1,b1,c1,d1]
GPU 2: [c0,c1,c2,c3]                      GPU 2: [a2,b2,c2,d2]
GPU 3: [d0,d1,d2,d3]                      GPU 3: [a3,b3,c3,d3]
```

## 5.3 Ring AllReduce(带宽最优,**必背**)

朴素 AllReduce(每个 GPU 都把数据发给所有人再求和)通信量 `O(N·S)`,完全浪费带宽。

**Ring AllReduce** 把 N 个 GPU 排成环,每个 GPU 只跟左右邻居通信,分两阶段:

### 阶段 A: Reduce-Scatter(N−1 步)

把每张卡的数据切成 N 份;第 k 步,GPU `r` 把第 `(r-k) mod N` 份发给 `r+1`,并把收到的累加到本地的第 `(r-k-1) mod N` 份上。N−1 步后,GPU `r` 上的"第 r 份"就是全局规约结果。

```
N=4, 数据切成 4 份。● 表示该份已经是全局和
Step 0:     Step 1:     Step 2(完成 ReduceScatter):
GPU0 [A0 A1 A2 A3]  →  [A0 A1 A2●A3]  →  [A0 A1●A2●A3]  →  [A0●A1●A2●A3]  (只 0 号份是 ●)
GPU1 [B0 B1 B2 B3]  →  [B0 B1 B2 B3●] →  [B0 B1 B2●B3●] →  [B0 B1●B2●B3●] (只 1 号份是 ●,记号简化)
...
```

### 阶段 B: All-Gather(N−1 步)

现在每个 GPU 各持有一份全局和。再绕环一圈,把这些已完成的份传给所有人,N−1 步后每个 GPU 都拿到完整结果。

### 通信量分析

- 每步每张卡只发 `S/N` 数据,共 `2(N−1)` 步。
- **每张卡总通信量** `= 2·(N−1)/N · S ≈ 2S`,**与 N 无关**!这就是"带宽最优"的来源。
- 朴素算法是 `(N−1)·S`,随 N 线性涨。
- 缺点:延迟跟 N 线性相关(`2(N−1)` 步),小消息时不如树状 / 递归倍增。

## 5.4 CUDA 层实现骨架

> 生产里**没人手写**这些 —— 用 [NCCL](https://github.com/NVIDIA/nccl) 就好。但面试讲底层时,你应该能说清"NCCL 在 GPU 上到底用了什么":
> - **单节点多卡**:`cudaMemcpyPeerAsync` / 统一虚拟地址(UVA) / NVLink
> - **多节点**:GPUDirect RDMA + InfiniBand,kernel 内通信可用 NVSHMEM / IBGDA
> - **同步**:每张卡用 stream + event,完成才能进入下一步

### 最小 P2P 通信原语

```cpp
// 准备:互相开通 peer access(只需一次)
cudaSetDevice(src);
cudaDeviceEnablePeerAccess(dst, 0);    // src 可写 dst
cudaSetDevice(dst);
cudaDeviceEnablePeerAccess(src, 0);    // dst 可写 src

// 异步从 src GPU 拷贝到 dst GPU(不经过 CPU,走 NVLink/PCIe)
cudaMemcpyPeerAsync(dst_ptr, dst_dev,
                    src_ptr, src_dev,
                    size_bytes, stream);
```

### Ring AllReduce 骨架(每张卡跑这段)

```cpp
// rank: 本卡编号,world_size: N,buf: 大小 S(切成 N 份,每份 chunk = S/N)
int next = (rank + 1) % world_size;
int prev = (rank - 1 + world_size) % world_size;

// 阶段 A: Reduce-Scatter
for (int step = 0; step < world_size - 1; ++step) {
    int send_idx = (rank - step + world_size)     % world_size;
    int recv_idx = (rank - step - 1 + world_size) % world_size;

    // 把第 send_idx 块发给 next,同时从 prev 收到第 recv_idx 块(到一个临时 buf)
    cudaMemcpyPeerAsync(next_recv_buf, next,
                        buf + send_idx * chunk, rank,
                        chunk * sizeof(float), stream);

    // 等 prev 把数据写进本地 tmp,再 launch 一个 add kernel 累加进 buf[recv_idx]
    cudaStreamSynchronize(recv_stream);
    add_kernel<<<grid, block, 0, stream>>>(buf + recv_idx * chunk, tmp, chunk);
}

// 阶段 B: All-Gather
for (int step = 0; step < world_size - 1; ++step) {
    int send_idx = (rank - step + 1 + world_size) % world_size;
    int recv_idx = (rank - step      + world_size) % world_size;

    // 把已经完成规约的块发给 next,从 prev 拿一块覆盖
    cudaMemcpyPeerAsync(next_buf + send_idx * chunk, next,
                        buf      + send_idx * chunk, rank,
                        chunk * sizeof(float), stream);
}
```

要点:
- **`add_kernel`** 就是 Part 2.1 那个 vec_add —— 通信完了用 GPU kernel 做规约。
- **多 stream 重叠**:同时用一个 stream 收、一个 stream 发、一个 stream 算,把通信和规约 overlap。NCCL 内部就是这么做的。
- **大消息切 chunk pipeline**:把 chunk 再切成更小的 micro-chunk,做 chunked ring,首尾 micro-chunk 同时在不同链路上跑。

### 调用 NCCL 是什么样(实际生产代码)

```cpp
ncclComm_t comm; ncclCommInitAll(&comm, N, devs);
ncclAllReduce(send_buf, recv_buf, count, ncclFloat, ncclSum, comm, stream);
// All2All
ncclGroupStart();
for (int r = 0; r < N; ++r) {
    ncclSend(send_buf + r * chunk, chunk, ncclFloat, r, comm, stream);
    ncclRecv(recv_buf + r * chunk, chunk, ncclFloat, r, comm, stream);
}
ncclGroupEnd();
```

## 5.5 通信复杂度速记表

`N` = GPU 数,`S` = 每张卡数据量。

| 算子 | 每张卡发送量(带宽最优实现) | 步数 | 算法 |
|---|---|---|---|
| Broadcast | `S` | `log N` | Tree / Recursive doubling |
| Reduce | `S` | `log N` | 反向 Tree |
| **AllReduce** | **`2(N−1)/N · S ≈ 2S`** | `2(N−1)` | **Ring** |
| AllGather | `(N−1)/N · S ≈ S` | `N−1` | Ring |
| ReduceScatter | `(N−1)/N · S ≈ S` | `N−1` | Ring |
| Scatter / Gather | `S` | `log N` | Tree |
| **All2All** | `(N−1)/N · S ≈ S` | `N−1` 或 `log N` | Pairwise / butterfly |

> 记住三件事:**AllReduce ≈ 2S**(因为 = ReduceScatter + AllGather);**AllGather / ReduceScatter ≈ S**(各做一半);**Tree 用于延迟优先,Ring 用于带宽优先**。

## 5.6 在 Transformer 里出现的位置(必背地图)

### TP(张量并行)的 AllReduce

把矩阵切两刀:
- **Column-parallel**(切输出维): 输入复制,输出在 head 维切 → **不需要通信**
- **Row-parallel**(切输入维): 输入需切片,输出要求和 → **末尾 AllReduce**

经典组合:**先 column-parallel 再 row-parallel**,这样中间不通信,只在层末尾一次 AllReduce:
```
Attention: Q/K/V (col-parallel) → 切 head 算 attention → O (row-parallel) → AllReduce
FFN:       gate/up (col-parallel) → SiLU·mul → down (row-parallel) → AllReduce
```
每个 transformer block 末尾各一次 AllReduce(attention 后、FFN 后),共 2 次。

### MoE 的两次 All2All

```
tokens → router 决定每个 token 去哪个 expert →
  All2All(dispatch): 把 tokens 按 expert 分发到对应 GPU →
    expert FFN(本地计算) →
  All2All(combine):  把 expert 输出按原 token 顺序返回 →
  按 gating 权重加权
```
两次 All2All 之间夹一次 expert 计算。这是为什么 EP 的瓶颈往往在 All2All 带宽。

### FSDP / ZeRO-3 的 AllGather + ReduceScatter

```
Forward:
  layer i 前: AllGather 取全 shard 权重 → forward → 丢掉权重
Backward:
  layer i 前: AllGather 权重 → backward → 算出全 grad
  之后: ReduceScatter grad → 每张卡只留自己 shard 的均值
```
注意 `AllGather + ReduceScatter = AllReduce` 的通信量,所以 ZeRO-3 **不增加总通信量**,只是分散到每层,换来显存切片。

## 5.7 通信 / 计算 Overlap(高级话题)

- **Async TP**: column-parallel 的输出准备好一部分就开始 row-parallel 的 GEMM,边算边通信。
- **Compute-comm overlap**: 用单独的 CUDA stream 跑通信,主 stream 跑下一层计算,两者重叠。
- **Fused AllReduce + RMSNorm**: vLLM 里把通信完的 reduce 和后面的 RMSNorm 写在一个 kernel 里,省一次 HBM 读。
- **NVSHMEM / IBGDA**: 让 kernel 内部直接发 RDMA,不再 host launch 一次通信 op,降低 launch latency。

## 5.8 面试速记卡

| 问题 | 一句话回答 |
|---|---|
| AllReduce 与 ReduceScatter+AllGather 的关系 | 等价,通信量都是 ~2S |
| Ring AllReduce 为什么带宽最优 | 每张卡每步只发 S/N,总量 2(N-1)/N·S ≈ 2S,与 N 无关 |
| Tree vs Ring | Tree 延迟低适合小消息,Ring 带宽优适合大消息;NCCL 会自动切换 |
| TP 为什么 attention/FFN 末尾各一次 AllReduce | col-parallel + row-parallel 组合,中间无通信,末尾 row-parallel 输出要 sum |
| MoE 为什么要两次 All2All | 一次把 token 路由到 expert,一次把结果送回原位置 |
| FSDP 为什么不比 DP 增加通信 | AllGather(forward) + ReduceScatter(backward) = AllReduce 等价 |
| All2All 和 AllGather 区别 | AllGather 每张卡都收同一份完整数据;All2All 每张卡发不同分片给不同人 |
| NCCL 底层用什么 | 单节点 NVLink + cudaMemcpyPeer;多节点 IB + GPUDirect RDMA;NVSHMEM 可在 kernel 内直接通信 |
| 怎么 overlap 通信和计算 | 单独 stream 跑通信 + chunk pipeline + async TP + fused kernel |
| 一次 AllReduce 的延迟构成 | launch + 2(N-1) 步 × (传 S/N 用时 + 一次规约 kernel) |

---

# Part 6 · vLLM 风格 Linear 并行(TP)

> vLLM(以及 Megatron-LM)用 4 个核心 Linear 类把 Transformer 切到多卡上:
> `ColumnParallelLinear` / `RowParallelLinear` / `MergedColumnParallelLinear` / `QKVParallelLinear`。
> 这一节先讲数学切法,再写一份**能直接 `torchrun` 跑起来**的轻量版。

## 6.1 数学原理:为什么这样切

线性层 `Y = X · A + b`,设 `X: [B, K]`,`A: [K, N]`,`Y: [B, N]`。`p` 张卡。

### 切法 ①:Column-Parallel(按输出维 N 切)

```
A = [ A_1 | A_2 | ... | A_p ]      A_i : [K, N/p]
b = [ b_1 | b_2 | ... | b_p ]
X 在每张卡上复制
Y_i = X · A_i + b_i               Y_i : [B, N/p]
```

- **输入复制,输出切片**。
- **不需要通信**(每张卡独立算出自己负责的输出列)。
- 输出要不要 AllGather 回完整 `[B, N]`,取决于下游怎么用:
  - 下游是 Row-Parallel → 它正好要切好的输入 → **不 gather**(关键!省一次通信)
  - 下游是普通层 → **gather**

### 切法 ②:Row-Parallel(按输入维 K 切)

```
A = [ A_1
      A_2
      ...
      A_p ]                        A_i : [K/p, N]
X 要先切成   X = [X_1 | X_2 | ... | X_p],  X_i : [B, K/p]
Y_i = X_i · A_i                   Y_i : [B, N]
Y   = Σ_i Y_i  ← AllReduce        b 只在 reduce 后加一次
```

- **输入切片,输出完整**(但每张卡都只是部分和)。
- **末尾必须一次 AllReduce 求和**才能得到正确的 `Y`。
- bias 不能每张卡都加(会加多倍),要么只 rank 0 加,要么 reduce 完再加。

### 黄金组合:Column → Row

把上面两种串起来:`Y₁ = X · A` (col) → 中间算什么 → `Y₂ = Y₁ · B` (row)。

```
X (全) ─col→ Y₁_i (切) ─中间不通信─ row→ Y₂ (部分和) ─AllReduce→ Y₂ (全)
```

Transformer 的两处都是这个模式:
- **Attention**: QKV(col)→ 每张卡独立算自己 head 的 attention → O(row)→ AllReduce
- **FFN**: gate_up(col)→ SwiGLU → down(row)→ AllReduce

**每个 block 只有 2 次 AllReduce**(attention 后、FFN 后),这是 Megatron-style TP 的本质。

---

## 6.2 ColumnParallelLinear

每张卡只持有完整权重的 `[K, N/p]` 列切片。

```python
import os, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

def tp_size():  return dist.get_world_size()
def tp_rank():  return dist.get_rank()

class ColumnParallelLinear(nn.Module):
    """
    Y = X @ A + b,A 按输出维切。
    输入 X: [..., in_features],各 rank 复制。
    输出 Y_i: [..., out_features / tp],各 rank 切片(默认不 gather)。
    """
    def __init__(self, in_features, out_features, bias=True, gather_output=False):
        super().__init__()
        assert out_features % tp_size() == 0
        self.out_per_rank = out_features // tp_size()
        self.gather_output = gather_output

        # 各 rank 自己那一份权重
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):                                   # x: [..., in]
        y = F.linear(x, self.weight, self.bias)             # [..., out/p]
        if self.gather_output:
            # AllGather 到完整 [..., out],默认不开
            chunks = [torch.empty_like(y) for _ in range(tp_size())]
            dist.all_gather(chunks, y)
            y = torch.cat(chunks, dim=-1)
        return y
```

## 6.3 RowParallelLinear

每张卡持有 `[K/p, N]` 行切片,**默认接收已经切好的输入**(因为上游通常就是 ColumnParallel)。

```python
class RowParallelLinear(nn.Module):
    """
    Y = X @ A,A 按输入维切。
    输入 X_i: [..., in_features / tp](默认),各 rank 不同。
    输出 Y:   [..., out_features],各 rank 相同(末尾 AllReduce)。
    """
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True):
        super().__init__()
        assert in_features % tp_size() == 0
        self.in_per_rank = in_features // tp_size()
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # bias 只在 reduce 后加一次,放在每张卡上即可
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):                                   # x: [..., in/p] 或 [..., in]
        if not self.input_is_parallel:
            # 罕见路径:输入是完整的,本地切一刀
            x = x.chunk(tp_size(), dim=-1)[tp_rank()]

        y = F.linear(x, self.weight)                        # [..., out],部分和
        dist.all_reduce(y, op=dist.ReduceOp.SUM)            # 关键一步
        if self.bias is not None:
            y = y + self.bias
        return y
```

## 6.4 MergedColumnParallelLinear(gate_up_proj 用)

SwiGLU FFN 把两个 col-parallel linear 合并成一个大 GEMM:

```python
# 原本: gate = X @ W_gate; up = X @ W_up; y = silu(gate) * up
# 合并: cat = X @ [W_gate | W_up]; gate, up = cat.split(...)
```

**关键点**:不能把"两个权重 concat 后再切 tp_size 份" —— 这样切会把 W_gate 的不同 head 跟 W_up 的不同 head 混到一起。正确做法是**先各自按 tp 切,再 concat**:

```python
class MergedColumnParallelLinear(nn.Module):
    """
    把多个 out_features 不同的 column-parallel linear 合并成一个 GEMM。
    每个子输出独立按 tp 切,最后在最后一维 concat。
    """
    def __init__(self, in_features, output_sizes, bias=True):
        super().__init__()
        for n in output_sizes:
            assert n % tp_size() == 0
        self.output_sizes_per_rank = [n // tp_size() for n in output_sizes]
        total = sum(self.output_sizes_per_rank)

        # 物理上是一块大权重 [total, in],但 split 出来对应每个子层
        self.weight = nn.Parameter(torch.empty(total, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(total)) if bias else None

    def forward(self, x):                                   # x: [..., in]
        y = F.linear(x, self.weight, self.bias)             # [..., sum(out_i)/p]
        # 调用者拿出去自己 split:
        # gate, up = y.split(self.output_sizes_per_rank, dim=-1)
        return y
```

加载预训练权重时的关键代码(简化版,vLLM 实际逻辑差不多):

```python
def load_merged_weight(self, loaded_weights, output_dim=0):
    """
    loaded_weights: List[Tensor],每个是某个子层的完整权重 [N_i, in]
    把每份按 tp 切出本 rank 的那段,然后 concat 进 self.weight。
    """
    parts = []
    for w_full, n_per_rank in zip(loaded_weights, self.output_sizes_per_rank):
        start = tp_rank() * n_per_rank
        parts.append(w_full[start : start + n_per_rank])
    self.weight.data.copy_(torch.cat(parts, dim=0))
```

## 6.5 QKVParallelLinear(含 GQA)

QKV 也是 column-parallel 合并,但要处理 GQA(Q 有 `n_heads` 个头,K/V 只有 `n_kv_heads` 个头)。

```python
class QKVParallelLinear(nn.Module):
    """
    输出 = concat(Q, K, V),按 head 维度做 column-parallel:
      Q 有 n_heads     × head_dim
      K 有 n_kv_heads  × head_dim
      V 有 n_kv_heads  × head_dim
    每个 rank 拿 n_heads/tp 个 Q 头 + n_kv_heads/tp 个 K/V 头。
    """
    def __init__(self, hidden, n_heads, n_kv_heads, head_dim, bias=False):
        super().__init__()
        p = tp_size()
        # tp 必须能整除头数;否则要做 head replication(见后)
        assert n_heads    % p == 0
        assert n_kv_heads % p == 0 or p % n_kv_heads == 0  # 允许 KV 头被复制
        self.head_dim = head_dim
        self.n_heads_per_rank    = n_heads    // p
        self.n_kv_heads_per_rank = max(n_kv_heads // p, 1)  # 若 n_kv_heads<p,复制到每张卡 1 个

        out = (self.n_heads_per_rank + 2 * self.n_kv_heads_per_rank) * head_dim
        self.weight = nn.Parameter(torch.empty(out, hidden))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out)) if bias else None

        # 记录三段长度,forward 后调用者拆分
        self.q_size  = self.n_heads_per_rank    * head_dim
        self.kv_size = self.n_kv_heads_per_rank * head_dim

    def forward(self, x):                                   # x: [..., hidden]
        y = F.linear(x, self.weight, self.bias)             # [..., q+k+v]
        q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v
```

**GQA 与 TP 的兼容性**:
- 若 `n_kv_heads ≥ tp`:每张卡分到 `n_kv_heads / tp` 个 KV 头,正常切。
- 若 `n_kv_heads < tp`(如 Llama-2-7B 有 32 个 KV 头但 8 卡 TP):**KV 头要被复制**,每张卡可能持有同一个 KV 头的副本。vLLM 里通过 `num_kv_heads_replicas` 处理。

## 6.6 拼成一个 TP Transformer Block

```python
class TPAttention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.qkv  = QKVParallelLinear(hidden, n_heads, n_kv_heads, head_dim)
        self.o    = RowParallelLinear(n_heads * head_dim, hidden, bias=False)
        self.head_dim = head_dim
        self.n_heads_per_rank    = n_heads    // tp_size()
        self.n_kv_heads_per_rank = max(n_kv_heads // tp_size(), 1)

    def forward(self, x):                                   # x: [B, T, hidden]
        B, T, _ = x.shape
        q, k, v = self.qkv(x)                               # 各 [B, T, *_per_rank * d]

        q = q.view(B, T, self.n_heads_per_rank,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads_per_rank, self.head_dim).transpose(1, 2)

        # 本 rank 内部独立算 attention(每个 head 互不依赖,正是 col-parallel 的核心)
        group = self.n_heads_per_rank // self.n_kv_heads_per_rank
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        attn = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        out  = attn.softmax(-1) @ v                         # [B, hq_per_rank, T, d]
        out  = out.transpose(1, 2).reshape(B, T, -1)        # [B, T, hq_per_rank * d]

        return self.o(out)                                  # RowParallel 内部 AllReduce


class TPMLP(nn.Module):
    def __init__(self, hidden, ffn_dim):
        super().__init__()
        # gate 和 up 合并成一个 col-parallel GEMM
        self.gate_up = MergedColumnParallelLinear(hidden, [ffn_dim, ffn_dim], bias=False)
        self.down    = RowParallelLinear(ffn_dim, hidden, bias=False)
        self.ffn_per_rank = ffn_dim // tp_size()

    def forward(self, x):
        gu = self.gate_up(x)
        gate, up = gu.split([self.ffn_per_rank, self.ffn_per_rank], dim=-1)
        return self.down(F.silu(gate) * up)                 # RowParallel 内部 AllReduce
```

> 每张卡的中间形状:Q 头数 `n_heads / tp`、FFN 中间维 `ffn_dim / tp`。**整个 block 只在两个 RowParallel 末尾各做一次 AllReduce**。

启动方式(单机多卡):

```bash
torchrun --nproc_per_node=4 tp_demo.py
```

```python
# tp_demo.py
import torch.distributed as dist
dist.init_process_group(backend='nccl')
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
model = TPAttention(hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128).cuda()
x = torch.randn(2, 128, 4096, device='cuda')
y = model(x)
```

## 6.7 几个常见追问

- **为什么 Column-Row 之间不通信?** —— Column 输出本来就按 head 切好,而本 rank 的 head 算 attention 只用本 rank 的 Q/K/V,完全本地化。Row-parallel 的输入正好需要这种切片状态。
- **bias 在 RowParallel 怎么处理?** —— 不能每张卡都加(会加 p 倍),只在 AllReduce 后加一次(代码里 `if bias is not None: y = y + bias` 放在 reduce 之后)。
- **gate_up 为什么要合并成一个 GEMM?** —— 两个 col-parallel 共享同一份输入 `X`,合并后只读一次 `X`,kernel launch 也少一次,带宽友好。
- **`MergedColumnParallel` 加载权重为什么要分别切再 concat?** —— 因为每张卡的物理布局是 `[W_gate 的本 rank 份 | W_up 的本 rank 份]`,跟"先 concat 再切"是不同的排布。
- **GQA + TP,KV 头不够分怎么办?** —— 复制(`num_kv_heads_replicas > 1`),每张卡都持有完整 KV 头中的一组(可能跟其它卡重复)。
- **TP 通信量怎么算?** —— 每个 transformer block 两次 AllReduce,每次约 `2·B·T·hidden·dtype` 字节(Ring AllReduce);序列长 / batch 大时,通信带宽很容易成瓶颈,所以会做 async TP、SP(Sequence Parallel,把 norm/dropout 这类 element-wise 的输入也切片)进一步压。
- **SP(Sequence Parallel)在哪里省的?** —— TP 之间的 norm 是 `[B, T, hidden]`,本来要复制在所有 rank;SP 把 `T` 维也切到 rank 上,通信从 AllReduce 变成 AllGather + ReduceScatter,**总量不变,但峰值激活显存降到 1/p**。

## 6.8 面试速记

| 类 | 切哪一维 | 输入 | 输出 | 通信 |
|---|---|---|---|---|
| ColumnParallelLinear | 输出维 `N` | 复制 | 切片 `[B, N/p]` | gather_output 时 AllGather |
| RowParallelLinear | 输入维 `K` | 切片 `[B, K/p]` | 完整 `[B, N]` | 末尾 **AllReduce** |
| MergedColumnParallelLinear | 多个子输出各按 col 切再 concat | 复制 | 各子层切片 concat | 同 col |
| QKVParallelLinear | 按 head 维 col 切 | 复制 | (Q, K, V) 切片 | 同 col;KV 不够分需复制 |

> 一句话总诀:**Column 切输出/复制输入/无通信,Row 切输入/输出完整/末尾 AllReduce,黄金组合 Column → Row 让中间 0 通信**。Transformer 每 block 2 次 AllReduce(attn 后、FFN 后),其余全本地。
