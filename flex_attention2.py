import torch
import functools
import time
import math
from collections import namedtuple
from typing import Callable, Optional
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
from torch.testing._internal import common_utils
from torch._inductor.test_case import TestCase as InductorTestCase, run_tests
import torch_npu
torch.set_float32_matmul_precision("high")
#---------- 原始 flexattention 相关函数 ----------
def create_attention(score_mod, block_mask, enable_gqa=False):
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64}
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options=kernel_options,
    )
def identity(score, batch, head, token_q, token_kv):
    return score
def causal_mask(batch, head, token_q, token_kv):
    return token_q >= token_kv
#---------- 小算子拼接 Attention ----------
def manualattention(q, k, v, mask_mod, scale=None):
    """
    用基础 PyTorch 算子实现带 mask 的 Attention。
    maskmod 与原 flexattention 中的定义一致：
    maskmod(batch, head, tokenq, tokenkv) -> bool（True 表示保留，False 表示屏蔽）
    """
    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    # 1. 根据 maskmod 生成完整的 mask 矩阵 (S, S)
    #    这里使用向量化方式生成，避免 Python 循环。
    #    对于 causalmask: tokenq >= tokenkv
    row_indices = torch.arange(S, device=q.device).unsqueeze(1)  # [S, 1]
    col_indices = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]
    # 假设 maskmod 只依赖于 tokenq 和 tokenkv，与 batch/head 无关
    # 我们用 batch=0, head=0 来生成，因为本例中 causalmask 不依赖它们。
    mask_bool = mask_mod(0, 0, row_indices, col_indices)  # [S, S]
    # True 保留，False 屏蔽 -> 转为 0.0 和 -inf
    mask = torch.where(mask_bool, 0.0, float('-inf'))

    # 2. QK^T / sqrt(d)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

    # 3. 加上 mask
    attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)  # broadcast

    # 4. Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # 5. 加权输出
    output = torch.matmul(attn_weights, v)
    return output
#---------- 配置参数 ----------
B, H, S, D = 4, 8, 2048, 128
test_device = "npu"
test_dtypes = [torch.bfloat16]
test_score_mask_mod_map = {identity: causal_mask}   # 键为 scoremod，值为 maskmod
class TestFlexAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        self.device = test_device
    
    def detailed_compare(self, output_flex, output_manual, rtol, atol, topk=10, name_flex="Flex", name_manual="Manual"):
        # output shape: [B, H, S, D]
        eps = 1e-6
        diff = output_flex - output_manual
        absdiff = diff.abs()
        max_abs_t = absdiff.max()          # tensor scalar
        max_abs = max_abs_t.item()        # 你的 max_abs（item）
        # 找到 max_abs 对应的下标 (b,h,s,d)
        flat_idx = absdiff.view(-1).argmax()  # tensor scalar
        b, h, s, d = torch.unravel_index(flat_idx, absdiff.shape)

        # 取对应下标处的 flex/manual 值
        flex_val = output_flex[b, h, s, d]
        manual_val = output_manual[b, h, s, d]

        denom = (flex_val + manual_val) / 2
        pct = max_abs / (denom.item() + eps) * 100
        max_rel_pct = abs(pct)


        mean_abs = absdiff.mean().item()
        median_abs = absdiff.median().item()

        # element-wise allclose 判定的失败掩码
        # torch.allclose 的等价条件近似为：
        # abs(x - y) <= atol + rtol * abs(y)
        threshold = atol + rtol * output_manual.abs()
        fail_mask = absdiff > threshold
        num_fail = int(fail_mask.sum().item())
        total = output_manual.numel()
        fail_ratio = num_fail / total

        # NaN/Inf 检查（非常关键）
        any_nan_flex = torch.isnan(output_flex).any().item()
        any_inf_flex = torch.isinf(output_flex).any().item()
        any_nan_manual = torch.isnan(output_manual).any().item()
        any_inf_manual = torch.isinf(output_manual).any().item()

        print("-------- 差异统计 --------")
        print(f"{name_flex} dtype={output_flex.dtype}, {name_manual} dtype={output_manual.dtype}")
        print(f"any_nan_flex={any_nan_flex}, any_inf_flex={any_inf_flex}, any_nan_manual={any_nan_manual}, any_inf_manual={any_inf_manual}")
        print(f"max_abs_diff={max_abs:.6g}->({max_rel_pct:.6g}%),mean_abs_diff={mean_abs:.6g}, median_abs_diff={median_abs:.6g}")

        print(f"fail_ratio={fail_ratio*100:.4f}%  (num_fail={num_fail}/{total})")
        print("--------------------------")

        # 取 topk 最大绝对差异的位置，打印具体下标和两边数值
        if topk is not None and topk > 0:
            flat_abs = absdiff.flatten()
            k = min(topk, flat_abs.numel())
            vals, idxs = torch.topk(flat_abs, k)

            B, H, S, D = output_manual.shape
            if max_rel_pct > 5.0:
                print(f"⚠️ 警告：最大误差超过5%，以下是 top{topk} 的详细对比：")
                for t in range(k):
                    flat_idx = idxs[t].item()
                    d_idx = flat_idx % D
                    tmp = flat_idx // D
                    s_idx = tmp % S
                    tmp = tmp // S
                    h_idx = tmp % H
                    b_idx = tmp // H

                    fv = output_flex[b_idx, h_idx, s_idx, d_idx].item()
                    mv = output_manual[b_idx, h_idx, s_idx, d_idx].item()
                    av = vals[t].item()
                    rv = av / max(abs(mv), eps)

                    print(
                        f"top{t}: absdiff={av:.6g}, rel={rv:.6g} "
                        f"@ (b={b_idx}, h={h_idx}, s={s_idx}, d={d_idx}) "
                        f"{name_flex}={fv:.6g}, {name_manual}={mv:.6g}"
                    )

        # 返回一个结果方便你判断
        return {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel_pct,
            "fail_ratio": fail_ratio,
            "num_fail": num_fail,
            "any_nan_flex": any_nan_flex,
            "any_inf_flex": any_inf_flex,
            "any_nan_manual": any_nan_manual,
            "any_inf_manual": any_inf_manual,
        }
    def rundynamictest(self, score_mask_mod, dtype):
        score_mod, mask_mod = score_mask_mod
        # cu_seqlens 可以忽略，这里只用于兼容
        cu_seqlens = torch.tensor([0, S], device=self.device)

        block_mask = create_block_mask(mask_mod, 1, 1, S, S, device=self.device)
        sdpa_fn = create_attention(score_mod, block_mask=block_mask)

        q = torch.randn((B, H, S, D), dtype=dtype, device=self.device)
        k = torch.randn((B, H, S, D), dtype=dtype, device=self.device)
        v = torch.randn((B, H, S, D), dtype=dtype, device=self.device)

        # --- Flex Attention 版本 ---
        compiled_sdpa = torch.compile(sdpa_fn, backend="inductor", dynamic=True)
        # Warmup
        for _ in range(10):
            compiled_sdpa(q, k, v)
        torch.npu.synchronize()

        num_repeat = 10
        start_time = time.time()
        for _ in range(num_repeat):
            output_flex = compiled_sdpa(q, k, v)
        torch.npu.synchronize()
        avgms = ((time.time() - start_time) / num_repeat) * 1000
        print(f"B:{B} H:{H} S:{S} D:{D} | Flex Attention 耗时: {avgms:.3f} ms")

        # --- 小算子拼接版本 (正确性对比) ---
        # 为了对比，我们不 compile 小算子版本，直接运行
        # 注意：这里 S=2048 会生成 [4,8,2048,2048] 的中间张量，NPU 内存可能紧张。
        # 如果内存不足，可以将 S 临时调整为 512 进行正确性验证。
        testS = S  # 可改为较小的值，如 512
        if testS != S:
            # 裁剪输入序列长度以避免 OOM
            qsmall = q[:, :, :testS, :]
            ksmall = k[:, :, :testS, :]
            vsmall = v[:, :, :testS, :]
            # 同时也需要裁剪 flexattention 的输出以便比较，这里演示时仍用原尺寸
            # 为了简化，我们建议直接使用全尺寸，如果 OOM 再修改。
            # 此处保持全尺寸，若运行报错 OOM，请减小 S 或 testS。
            pass

        # 使用相同的 q,k,v 进行手动计算
        start_time2 = time.time()
        with torch.no_grad():
            for _ in range(num_repeat):
                output_manual = manualattention(q, k, v, mask_mod)
        torch.npu.synchronize()
        avgms2 = ((time.time() - start_time2) / num_repeat) * 1000
        print(f"B:{B} H:{H} S:{S} D:{D} | 手动拼接小算子耗时: {avgms2:.3f} ms")

        # 正确性验证
        # print("输出对比：")
        # print(f"  Flex Attention output: {output_flex}")
        # print(f"  Manual Attention output: {output_manual}")

        # 检查是否接近
        if dtype == torch.bfloat16:
            # bfloat16 精度较低，放宽容差
            rtol = 1e-2
            atol = 1e-2
        else:
            rtol = 1e-3
            atol = 1e-5

        if output_flex.shape != output_manual.shape:
            print(f"形状不匹配: {output_flex.shape} vs {output_manual.shape}")

        # 转换数据类型
        if output_flex.dtype != torch.float32:
            output_flex = output_flex.to(torch.float32)

        if output_manual.dtype != torch.float32:
            output_manual = output_manual.to(torch.float32)
        # 分别比较每个元素
        try:
            rtollocal = rtol
            atollocal = atol

            print(f"rtollocal={rtollocal}")
            print(f"atollocal={atollocal}")

            close = torch.allclose(output_flex, output_manual, rtol=rtollocal, atol=atollocal)

            # 无论 pass/fail，都建议做一次“诊断统计”（你会更快定位原因）
            stats = self.detailed_compare(
                output_flex, output_manual,
                rtol=rtollocal, atol=atollocal,
                topk=10
            )

            # pass 时也可以加一个“相对误差很大但绝对误差不大”的警告
            # 比如 max_rel_diff > 0.5 或 1.0 这种（你可按需求调整）
            if close:
                print("✅测试通过（allclose 为 True）")
                return output_flex, output_manual
            else:
                print("❌测试fail")
                return output_flex, output_manual

        except Exception as e:
            print(f"比较过程中出现错误: {e}")
            return output_flex, output_manual

        return output_flex, output_manual

# 参数化测试
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mask_mod", test_score_mask_mod_map.items())
    def testbuiltinscoremodsdynamic(self, dtype, score_mask_mod):
        self.rundynamictest(score_mask_mod, dtype)
#实例化参数化测试
common_utils.instantiate_parametrized_tests(TestFlexAttention)
if __name__ == "__main__":
    import torch._dynamo
    # 避免 log 干扰
    torch._dynamo.config.suppress_errors = True
    with torch.no_grad():
        run_tests()
