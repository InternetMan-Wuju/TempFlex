"""
流行的稀疏 Attention Mask 模式集合。

所有 mask_mod 的函数签名均与 Flex Attention 接口兼容:
    mask_mod(batch, head, token_q, token_kv) -> bool Tensor

所有 mask_mod_fn 的函数签名:
    mask_mod_fn(token_q, token_kv) -> bool Tensor
"""

import math
import torch


# ============================================================
# 1. Causal Mask (因果掩码)
# ============================================================
def causal_mask(batch, head, token_q, token_kv):
    """每个 token 只能看到自己及之前的 token。"""
    return token_q >= token_kv


def causal_mask_fn(token_q, token_kv):
    return token_q >= token_kv


# ============================================================
# 2. Sliding Window Mask (滑动窗口掩码)
# ============================================================
def sliding_window_mask(batch, head, token_q, token_kv, window_size=64):
    """每个 token 只看前 window_size 个 token + 自己。"""
    return (token_q - token_kv < window_size) & (token_q >= token_kv)


def make_sliding_window(window_size=64):
    """创建一个滑动窗口 mask_mod。"""
    def mask_mod(batch, head, token_q, token_kv):
        return (token_q - token_kv < window_size) & (token_q >= token_kv)
    return mask_mod


# ============================================================
# 3. Dilated Sliding Window (空洞滑动窗口)
# ============================================================
def make_dilated_sliding_window(window_size=64, dilation=2):
    """
    每隔 dilation 个 token 取一个，在 window_size 范围内。
    例如 dilation=2 时: token_q 能看到 token_q, token_q-2, token_q-4, ...
    """
    def mask_mod(batch, head, token_q, token_kv):
        in_window = (token_q - token_kv < window_size) & (token_q >= token_kv)
        aligned = ((token_q - token_kv) % dilation == 0)
        return in_window & aligned
    return mask_mod


# ============================================================
# 4. Global + Local Mask (全局+局部)
# ============================================================
def make_global_local_mask(global_tokens=4, local_window=64):
    """
    前 global_tokens 个 token 能看到全局（全 causal）。
    后面的 token 只能看到前 global_tokens 个 token + 局部窗口内的 token。
    """
    def mask_mod(batch, head, token_q, token_kv):
        q_is_global = token_q < global_tokens
        kv_is_global = token_kv < global_tokens
        in_local_window = (token_q - token_kv < local_window) & (token_q >= token_kv)

        # global token: full causal
        # non-global token: global kvs + local window
        return q_is_global | kv_is_global | in_local_window
    return mask_mod


# ============================================================
# 5. Strided Mask (步长掩码)
# ============================================================
def make_strided_mask(stride=32):
    """
    每隔 stride 个 token 保留一个注意力连接。
    常用于 BigBird / Longformer 风格。
    """
    def mask_mod(batch, head, token_q, token_kv):
        return (token_kv % stride == 0) | (token_q >= token_kv)
    return mask_mod


# ============================================================
# 6. Vertical-Slash Mask (纵向斜线)
# ============================================================
def make_vertical_slash_mask(k=4):
    """
    每个 token 能看到 k 个均匀分布的"纵向条纹"的 key。
    在 Sparse Transformer 论文中使用。
    """
    def mask_mod(batch, head, token_q, token_kv):
        return token_kv % k == 0
    return mask_mod


# ============================================================
# 7. Random Pattern Mask (固定随机模式)
# ============================================================
def make_fixed_random_mask(seq_len, sparsity=0.1, seed=42):
    """
    生成一个固定的随机稀疏模式，在 (S, S) 上只有 sparsity 比例的连接保留。
    注意：这不是真正的 mask_mod，而是一个预计算的布尔 mask。
    适用于 create_block_mask 的固定模式。
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    # 保证至少保留对角线
    mask = torch.rand(seq_len, seq_len, generator=rng) < sparsity
    mask = mask | torch.eye(seq_len, dtype=torch.bool)
    return mask


# ============================================================
# 8. Nested Attention Mask (嵌套注意力)
# ============================================================
def make_nested_mask(window_size=64, stride=32):
    """
    Longformer 风格: local window + strided attention。
    每个 token 既有滑动窗口注意力，也有大跨度的步长注意力。
    """
    def mask_mod(batch, head, token_q, token_kv):
        in_window = (token_q - token_kv < window_size) & (token_q >= token_kv)
        strided = (token_kv % stride == 0) | (token_kv == 0)
        return in_window | strided
    return mask_mod


# ============================================================
# 9. Band Mask (带宽掩码)
# ============================================================
def make_band_mask(bandwidth=32):
    """
    保留对角线附近 bandwidth 范围内的连接（包括未来 token）。
    注意：这不是 causal 的，适合非自回归模型。
    """
    def mask_mod(batch, head, token_q, token_kv):
        return (token_q - token_kv).abs() <= bandwidth
    return mask_mod


# ============================================================
# 10. Prefix LM Mask (前缀 LM 掩码)
# ============================================================
def make_prefix_lm_mask(prefix_len=16):
    """
    Prefix LM: 前 prefix_len 个 token 可以看到全局（双向），
    后面的 token 只能 causal。
    """
    def mask_mod(batch, head, token_q, token_kv):
        is_prefix = token_kv < prefix_len
        return is_prefix | (token_q >= token_kv)
    return mask_mod


# ============================================================
# 预定义组合 - 常用稀疏配置
# ============================================================
def get_sparse_config(name, **kwargs):
    """通过名称获取预定义的稀疏配置。"""
    configs = {
        "causal": {
            "score_mod": None,
            "mask_mod": causal_mask,
            "description": "Causal LM",
            "optimizations": {
                "ROWS_GUARANTEED_SAFE": True,
                "BLOCKS_ARE_CONTIGUOUS": True,
            },
        },
        "sliding_window_64": {
            "score_mod": None,
            "mask_mod": make_sliding_window(window_size=64),
            "description": "Sliding Window (size=64)",
            "optimizations": {},
        },
        "sliding_window_128": {
            "score_mod": None,
            "mask_mod": make_sliding_window(window_size=128),
            "description": "Sliding Window (size=128)",
            "optimizations": {},
        },
        "global_local": {
            "score_mod": None,
            "mask_mod": make_global_local_mask(global_tokens=4, local_window=64),
            "description": "Global (4) + Local (64)",
            "optimizations": {},
        },
        "nested": {
            "score_mod": None,
            "mask_mod": make_nested_mask(window_size=64, stride=32),
            "description": "Nested: Local(64) + Stride(32)",
            "optimizations": {},
        },
        "prefix_lm": {
            "score_mod": None,
            "mask_mod": make_prefix_lm_mask(prefix_len=16),
            "description": "Prefix LM (prefix=16)",
            "optimizations": {},
        },
        "dilated_window": {
            "score_mod": None,
            "mask_mod": make_dilated_sliding_window(window_size=128, dilation=2),
            "description": "Dilated Sliding Window (size=128, dilation=2)",
            "optimizations": {},
        },
        "strided": {
            "score_mod": None,
            "mask_mod": make_strided_mask(stride=32),
            "description": "Strided (stride=32)",
            "optimizations": {},
        },
    }
    if name not in configs:
        raise KeyError(f"Unknown sparse config: {name}. Available: {list(configs.keys())}")
    return configs[name]


def list_sparse_configs():
    """列出所有预定义的稀疏配置名。"""
    return list(_SPARSE_CONFIGS.keys())


_SPARSE_CONFIGS = {
    "causal": {
        "score_mod": None,
        "mask_mod": causal_mask,
        "description": "Causal LM",
        "optimizations": {
            "ROWS_GUARANTEED_SAFE": True,
            "BLOCKS_ARE_CONTIGUOUS": True,
        },
    },
    "sliding_window_64": {
        "score_mod": None,
        "mask_mod": make_sliding_window(window_size=64),
        "description": "Sliding Window (size=64)",
        "optimizations": {},
    },
    "sliding_window_128": {
        "score_mod": None,
        "mask_mod": make_sliding_window(window_size=128),
        "description": "Sliding Window (size=128)",
        "optimizations": {},
    },
    "global_local": {
        "score_mod": None,
        "mask_mod": make_global_local_mask(global_tokens=4, local_window=64),
        "description": "Global (4) + Local (64)",
        "optimizations": {},
    },
    "nested": {
        "score_mod": None,
        "mask_mod": make_nested_mask(window_size=64, stride=32),
        "description": "Nested: Local(64) + Stride(32)",
        "optimizations": {},
    },
    "prefix_lm": {
        "score_mod": None,
        "mask_mod": make_prefix_lm_mask(prefix_len=16),
        "description": "Prefix LM (prefix=16)",
        "optimizations": {},
    },
    "dilated_window": {
        "score_mod": None,
        "mask_mod": make_dilated_sliding_window(window_size=128, dilation=2),
        "description": "Dilated Sliding Window (size=128, dilation=2)",
        "optimizations": {},
    },
    "strided": {
        "score_mod": None,
        "mask_mod": make_strided_mask(stride=32),
        "description": "Strided (stride=32)",
        "optimizations": {},
    },
}


def get_sparse_config(name):
    """通过名称获取预定义的稀疏配置。"""
    if name not in _SPARSE_CONFIGS:
        raise KeyError(f"Unknown sparse config: {name}. Available: {list(_SPARSE_CONFIGS.keys())}")
    return _SPARSE_CONFIGS[name]


if __name__ == "__main__":
    print("Available sparse configurations:")
    for name, cfg in _SPARSE_CONFIGS.items():
        print(f"  {name}: {cfg['description']}")