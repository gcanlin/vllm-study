"""
Grouped Query Attention (GQA) with Tensor Parallelism (TP), 推理版.

切分策略 (Megatron-LM / vLLM 风格):
- Q/K/V 投影 -> ColumnParallel: 按 head 维度切, 每个 rank 持有
    num_heads // tp 个 q-head, num_kv_heads // tp 个 kv-head
- 注意力计算 在每个 rank 上独立完成 (head 之间无依赖)
- O 投影     -> RowParallel: 按输入维度切, 输出做 all-reduce 汇总

约束: num_heads 和 num_kv_heads 都必须能被 tp 整除.
(num_kv_heads < tp 时 vLLM 会复制 kv head, 这里不处理.)
"""

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def tp_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def tp_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


# ---------------------------------------------------------------------------
# TP Linear (推理版: 不带 autograd, all-reduce 直接调)
# ---------------------------------------------------------------------------
class ColumnParallelLinear(nn.Module):
    """W 按输出维度切, 每个 rank 持有 [out/tp, in]. 输入完整复制, 输出是 local 切片."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert out_features % tp_size() == 0
        self.out_per_rank = out_features // tp_size()
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """W 按输入维度切, 每个 rank 持有 [out, in/tp]. 输入是 local 切片, 输出 all-reduce."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert in_features % tp_size() == 0
        self.in_per_rank = in_features // tp_size()
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        # bias 不切, 最后加一次
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)                          # 每个 rank 的 partial sum
        if tp_size() > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)          # 跨 rank 求和得到完整结果
        if self.bias is not None:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# GQA with TP
# ---------------------------------------------------------------------------
class GroupedQueryAttentionTP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ):
        super().__init__()
        tp = tp_size()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        assert num_heads    % tp == 0, f"num_heads={num_heads} 不能被 tp={tp} 整除"
        assert num_kv_heads % tp == 0, f"num_kv_heads={num_kv_heads} 不能被 tp={tp} 整除"

        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 每个 rank 实际持有的 head 数
        self.nh_local    = num_heads    // tp
        self.nkv_local   = num_kv_heads // tp

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads    * self.head_dim, bias)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * self.head_dim, bias)
        self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * self.head_dim, bias)
        self.o_proj = RowParallelLinear(num_heads * self.head_dim, hidden_size, bias)

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, hidden_size], 各 rank 完整复制
        attn_mask: torch.Tensor | None = None,    # 加性 mask, -inf=屏蔽
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # 1. 投影 (输出是 local heads)
        q = self.q_proj(x).view(B, T, self.nh_local,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv_local, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv_local, self.head_dim).transpose(1, 2)

        # 2. GQA: 在 local 内把 K/V 扩到 nh_local
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # 3. 手写 SDPA
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale       # [B, nh_local, T, T]
        if is_causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        if attn_mask is not None:
            scores = scores + attn_mask
        out = torch.matmul(F.softmax(scores, dim=-1), v)                 # [B, nh_local, T, D]

        # 4. 合并 local heads -> [B, T, nh_local * D], 交给 RowParallel o_proj 做 all-reduce
        out = out.transpose(1, 2).contiguous().view(B, T, self.nh_local * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "RANK" in os.environ:                      # torchrun --nproc_per_node=N gqa_tp.py
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo")

    torch.manual_seed(0)
    B, T, H = 2, 16, 512
    attn = GroupedQueryAttentionTP(hidden_size=H, num_heads=8, num_kv_heads=4)
    y = attn(torch.randn(B, T, H), is_causal=True)

    if tp_rank() == 0:
        print(f"tp={tp_size()}  y.shape={tuple(y.shape)}  "
              f"nh_local={attn.nh_local}  nkv_local={attn.nkv_local}")

    if dist.is_initialized():
        dist.destroy_process_group()
