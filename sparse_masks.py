"""
流行的稀疏 Attention Mask 模式集合。

所有 mask_mod 的函数签名均与 Flex Attention 接口兼容:
    mask_mod(batch, head, token_q, token_kv) -> bool Tensor

所有 score_mod 的函数签名:
    score_mod(score, batch, head, token_q, token_kv) -> Tensor
"""

import math
import torch


# ============================================================
# Score Mods
# ============================================================
def identity_score(score, batch, head, token_q, token_kv):
    return score


def alibi_score(score, batch, head, token_q, token_kv, slopes=None):
    """ALiBi 线性偏置: score += (token_kv - token_q) * slope_per_head"""
    if slopes is not None:
        slope = slopes[head]
    else:
        slope = 2.0 ** (-8.0 / head) if head > 0 else 1.0
    return score + (token_kv - token_q).float() * slope


def make_alibi_score(num_heads):
    """创建带 ALiBi slopes 的 score_mod。"""
    slopes = torch.tensor([2.0 ** (-8.0 * i / num_heads) for i in range(1, num_heads + 1)])
    def score_mod(score, batch, head, token_q, token_kv):
        return score + (token_kv - token_q).float() * slopes[head]
    return score_mod


# ============================================================
# 1. Causal Mask (因果掩码)
# ============================================================
def causal_mask(batch, head, token_q, token_kv):
    """每个 token 只能看到自己及之前的 token。"""
    return token_q >= token_kv


# ============================================================
# 2. Sliding Window Mask (滑动窗口掩码)
# ============================================================
def make_sliding_window(window_size=64):
    def mask_mod(batch, head, token_q, token_kv):
        return (token_q - token_kv < window_size) & (token_q >= token_kv)
    return mask_mod


# ============================================================
# 3. Dilated Sliding Window (空洞滑动窗口)
# ============================================================
def make_dilated_sliding_window(window_size=64, dilation=2):
    def mask_mod(batch, head, token_q, token_kv):
        in_window = (token_q - token_kv < window_size) & (token_q >= token_kv)
        # NPU: avoid % (remainder); use (x//d)*d == x for divisibility check
        dist = token_q - token_kv
        aligned = (dist // dilation) * dilation == dist
        return in_window & aligned
    return mask_mod


# ============================================================
# 4. Global + Local Mask (全局+局部) — Longformer 风格
# ============================================================
def make_global_local_mask(global_tokens=4, local_window=64):
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: use min(q,k) < G to express (q<G OR k<G) as a single condition
        any_global = torch.minimum(token_q, token_kv) < global_tokens
        in_window = (token_q - token_kv < local_window) & (token_q >= token_kv)
        return any_global.int() + in_window.int() > 0
    return mask_mod


# ============================================================
# 5. Strided Mask (步长掩码) — BigBird 风格
# ============================================================
def make_strided_mask(stride=32):
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: use minimal ops; bool+bool auto-promotes to int
        strided_kv = (token_kv // stride) * stride == token_kv
        return strided_kv + (token_q >= token_kv) > 0
    return mask_mod


# ============================================================
# 6. Nested Attention Mask — Longformer 风格
# ============================================================
def make_nested_mask(window_size=64, stride=32):
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: use minimal ops (bool+bool→int) to stay within UB budget
        in_window = (token_q - token_kv < window_size) & (token_q >= token_kv)
        strided = (token_kv // stride) * stride == token_kv
        return in_window + strided + (token_kv == 0) > 0
    return mask_mod


# ============================================================
# 7. Prefix LM Mask (前缀 LM 掩码)
# ============================================================
def make_prefix_lm_mask(prefix_len=16):
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: avoid | (bitwise_or); use int sum
        is_prefix = token_kv < prefix_len
        causal = token_q >= token_kv
        return (is_prefix.int() + causal.int()) > 0
    return mask_mod


# ============================================================
# 8. Block Diagonal Mask (块对角掩码)
# ============================================================
def make_block_diagonal_mask(block_size=64):
    """每个 token 只能看到同一 block 内的 token（causal）。适合 chunked attention。"""
    def mask_mod(batch, head, token_q, token_kv):
        same_block = token_q // block_size == token_kv // block_size
        return same_block & (token_q >= token_kv)
    return mask_mod


# ============================================================
# 9. Checkerboard Mask (棋盘掩码)
# ============================================================
def make_checkerboard_mask(block_size=32):
    """Alternating dense/sparse blocks like a chessboard. Non-causal. 适合 ViT 风格。"""
    def mask_mod(batch, head, token_q, token_kv):
        q_block = token_q // block_size
        kv_block = token_kv // block_size
        # NPU: avoid % (remainder); use (x//2)*2 == x for even-ness check
        block_sum = q_block + kv_block
        return (block_sum // 2) * 2 == block_sum
    return mask_mod


# ============================================================
# 10. Document Boundary Mask (文档边界掩码)
# ============================================================
def make_document_mask(doc_lengths):
    """
    多文档批处理掩码。每个 token 只能看到同一文档内的 token（causal）。
    doc_lengths: list[int], 每个文档的长度。
    """
    doc_offsets = [0]
    for length in doc_lengths:
        doc_offsets.append(doc_offsets[-1] + length)
    doc_offsets_t = torch.tensor(doc_offsets)

    def mask_mod(batch, head, token_q, token_kv):
        q_doc_id = torch.searchsorted(doc_offsets_t, token_q + 1, right=True) - 1
        kv_doc_id = torch.searchsorted(doc_offsets_t, token_kv + 1, right=True) - 1
        return (q_doc_id == kv_doc_id) & (token_q >= token_kv)
    return mask_mod


# ============================================================
# 11. Document Boundary Mask (统一长度版)
# ============================================================
def make_uniform_document_mask(doc_len=256):
    """每个文档长度相同时的简化版文档掩码。"""
    def mask_mod(batch, head, token_q, token_kv):
        same_doc = token_q // doc_len == token_kv // doc_len
        return same_doc & (token_q >= token_kv)
    return mask_mod


# ============================================================
# 12. Random Block Sparse Mask (随机块稀疏)
# ============================================================
def make_random_block_sparse_mask(seq_len, block_size=64, density=0.3, seed=42):
    """
    随机保留 density 比例的 block 作为可注意力连接（causal 区域内）。
    预计算 block 级别 mask 到 GPU buffer。
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    rng = torch.Generator()
    rng.manual_seed(seed)
    block_mask = torch.rand(num_blocks, num_blocks, generator=rng) < density
    block_mask = block_mask | torch.eye(num_blocks, dtype=torch.bool)
    # 强制 causal: 上三角清零
    block_mask = torch.tril(block_mask)

    def mask_mod(batch, head, token_q, token_kv):
        q_block = token_q // block_size
        kv_block = token_kv // block_size
        return block_mask[q_block, kv_block]
    return mask_mod


# ============================================================
# 13. Multi-Scale Dilated Window (LongNet 风格)
# ============================================================
def make_multiscale_dilated_mask(dilations=(1, 2, 4), window_size=256):
    """
    LongNet 风格: 多个 dilation rate 的并集 + causal。
    例如 dilation=(1,2,4) → token_q 能看到 token_q-1, q-2, q-3, q-4, q-6, q-8, ...
    """
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: avoid torch.zeros_like/ones_like, %, |
        dist = token_q - token_kv
        causal = dist >= 0
        # Close tokens always attend (within window_size // 4)
        close_tokens = causal & (dist < window_size // 4)
        # Check dilations for farther tokens
        # Start with zero tensor (all False) using safe arithmetic
        result_int = (token_q != token_q).int()  # all zeros (no NaN in integer indices)
        for d in dilations:
            aligned = (dist // d) * d == dist
            result_int = result_int + aligned.int()
        dilated_match = (result_int > 0) & causal & (dist < window_size)
        return (close_tokens.int() + dilated_match.int()) > 0
    return mask_mod


# ============================================================
# 14. Band Mask + Global (Longformer-Encoder-Decoder 风格)
# ============================================================
def make_band_global_mask(bandwidth=32, global_tokens=2):
    """对角线带宽 + 前 global_tokens 个全局 token + causal。"""
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: avoid .abs() and | (bitwise_or)
        # abs(diff) <= bandwidth  ⇔  diff <= bandwidth AND -diff <= bandwidth
        diff = token_q - token_kv
        in_band = (diff <= bandwidth) & (-diff <= bandwidth)
        kv_global = token_kv < global_tokens
        return (in_band.int() + kv_global.int()) > 0
    return mask_mod


# ============================================================
# 15. Learned Pattern Emulation (复合模式)
# ============================================================
def make_hybrid_mask(sliding_window=128, strided_step=64, global_every=256):
    """
    复合稀疏模式: 局部窗口 + 大步长采样 + 周期性全局 token。
    模拟 Routing Transformer / Reformer 的行为。
    """
    def mask_mod(batch, head, token_q, token_kv):
        # NPU: use minimal ops; bool+bool auto-promotes to int
        in_window = (token_q - token_kv < sliding_window) & (token_q >= token_kv)
        strided = (token_kv // strided_step) * strided_step == token_kv
        global_kv = (token_kv // global_every) * global_every == token_kv
        return in_window + strided + global_kv > 0
    return mask_mod


# ============================================================
# 预定义配置
# ============================================================
_SPARSE_CONFIGS = {
    "causal": {
        "score_mod": identity_score,
        "mask_mod": causal_mask,
        "description": "Causal LM (baseline)",
        "optimizations": {
            "ROWS_GUARANTEED_SAFE": True,
            "BLOCKS_ARE_CONTIGUOUS": True,
        },
    },
    "sliding_window_64": {
        "score_mod": identity_score,
        "mask_mod": make_sliding_window(window_size=64),
        "description": "Sliding Window (size=64)",
        "optimizations": {},
    },
    "sliding_window_128": {
        "score_mod": identity_score,
        "mask_mod": make_sliding_window(window_size=128),
        "description": "Sliding Window (size=128)",
        "optimizations": {},
    },
    "global_local": {
        "score_mod": identity_score,
        "mask_mod": make_global_local_mask(global_tokens=4, local_window=64),
        "description": "Global(4) + Local(64)",
        "optimizations": {},
    },
    "nested": {
        "score_mod": identity_score,
        "mask_mod": make_nested_mask(window_size=64, stride=32),
        "description": "Nested: Local(64) + Stride(32)",
        "optimizations": {},
    },
    "prefix_lm": {
        "score_mod": identity_score,
        "mask_mod": make_prefix_lm_mask(prefix_len=16),
        "description": "Prefix LM (prefix=16)",
        "optimizations": {},
    },
    "dilated_window": {
        "score_mod": identity_score,
        "mask_mod": make_dilated_sliding_window(window_size=128, dilation=2),
        "description": "Dilated Sliding Window (size=128, dil=2)",
        "optimizations": {},
    },
    "strided": {
        "score_mod": identity_score,
        "mask_mod": make_strided_mask(stride=32),
        "description": "Strided (stride=32)",
        "optimizations": {},
    },
    # ── 新增模式 ──
    "block_diagonal_64": {
        "score_mod": identity_score,
        "mask_mod": make_block_diagonal_mask(block_size=64),
        "description": "Block Diagonal (block=64)",
        "optimizations": {},
    },
    "block_diagonal_128": {
        "score_mod": identity_score,
        "mask_mod": make_block_diagonal_mask(block_size=128),
        "description": "Block Diagonal (block=128)",
        "optimizations": {},
    },
    "checkerboard_32": {
        "score_mod": identity_score,
        "mask_mod": make_checkerboard_mask(block_size=32),
        "description": "Checkerboard (block=32)",
        "optimizations": {},
    },
    "checkerboard_64": {
        "score_mod": identity_score,
        "mask_mod": make_checkerboard_mask(block_size=64),
        "description": "Checkerboard (block=64)",
        "optimizations": {},
    },
    "uniform_doc_256": {
        "score_mod": identity_score,
        "mask_mod": make_uniform_document_mask(doc_len=256),
        "description": "Uniform Document (doc_len=256)",
        "optimizations": {},
    },
    "random_block_sparse": {
        "score_mod": identity_score,
        "mask_mod": make_random_block_sparse_mask(seq_len=4096, block_size=64, density=0.3, seed=42),
        "description": "Random Block Sparse (density=30%)",
        "optimizations": {},
    },
    "multiscale_dilated": {
        "score_mod": identity_score,
        "mask_mod": make_multiscale_dilated_mask(dilations=(1, 2, 4), window_size=256),
        "description": "Multi-Scale Dilated (1,2,4 × 256)",
        "optimizations": {},
    },
    "band_global_32": {
        "score_mod": identity_score,
        "mask_mod": make_band_global_mask(bandwidth=32, global_tokens=2),
        "description": "Band(32) + Global(2)",
        "optimizations": {},
    },
    "hybrid_sparse": {
        "score_mod": identity_score,
        "mask_mod": make_hybrid_mask(sliding_window=128, strided_step=64, global_every=256),
        "description": "Hybrid: Local(128) + Stride(64) + Global(256)",
        "optimizations": {},
    },
    "alibi_causal": {
        "score_mod": make_alibi_score(num_heads=8),
        "mask_mod": causal_mask,
        "description": "ALiBi + Causal",
        "optimizations": {
            "ROWS_GUARANTEED_SAFE": True,
            "BLOCKS_ARE_CONTIGUOUS": True,
        },
    },
}


def get_sparse_config(name):
    """通过名称获取预定义的稀疏配置。"""
    if name not in _SPARSE_CONFIGS:
        raise KeyError(f"Unknown sparse config: {name}. Available: {sorted(_SPARSE_CONFIGS.keys())}")
    return _SPARSE_CONFIGS[name]


def list_sparse_configs():
    """列出所有预定义的稀疏配置名。"""
    return sorted(_SPARSE_CONFIGS.keys())


if __name__ == "__main__":
    print("Available sparse configurations:")
    for name, cfg in sorted(_SPARSE_CONFIGS.items()):
        print(f"  {name:25s}  {cfg['description']}")