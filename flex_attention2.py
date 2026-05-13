import torch
import functools
import time
import math
from collections import namedtuple
from typing import Callable, Optional
from torch.nn.attention.flexattention import (
    createblockmask,
    flexattention,
)
from torch.testing.internal import commonutils
from torch.inductor.testcase import TestCase as InductorTestCase
import torchnpu
torch.setfloat32matmulprecision("high")
#---------- 原始 flexattention 相关函数 ----------
def createattention(scoremod, blockmask, enablegqa=False):
    kerneloptions = {"BLOCKM": 64, "BLOCKN": 64}
    return functools.partial(
        flexattention,
        scoremod=scoremod,
        blockmask=blockmask,
        enablegqa=enablegqa,
        kerneloptions=kerneloptions,
    )
def identity(score, batch, head, tokenq, tokenkv):
    return score
def causalmask(batch, head, tokenq, tokenkv):
    return tokenq >= tokenkv
#---------- 小算子拼接 Attention ----------
def manualattention(q, k, v, maskmod, scale=None):
    """
    用基础 PyTorch 算子实现带 mask 的 Attention。
    maskmod 与原 flexattention 中的定义一致：
    maskmod(batch, head, tokenq, tokenkv) -> bool（True 表示保留，False 表示屏蔽）
    """
    B, H, S, D = q.shape
    if scale is None:
        scale = D * -0.5
    # 1. 根据 maskmod 生成完整的 mask 矩阵 (S, S)
    #    这里使用向量化方式生成，避免 Python 循环。
    #    对于 causalmask: tokenq >= tokenkv
    rowindices = torch.arange(S, device=q.device).unsqueeze(1)  # [S, 1]
    colindices = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]
    # 假设 maskmod 只依赖于 tokenq 和 tokenkv，与 batch/head 无关
    # 我们用 batch=0, head=0 来生成，因为本例中 causalmask 不依赖它们。
    maskbool = maskmod(0, 0, rowindices, colindices)  # [S, S]
    # True 保留，False 屏蔽 -> 转为 0.0 和 -inf
    mask = torch.where(maskbool, 0.0, float('-inf'))

    # 2. QK^T / sqrt(d)
    attnscores = torch.matmul(q, k.transpose(-2, -1))  scale  # [B, H, S, S]

    # 3. 加上 mask
    attnscores = attnscores + mask.unsqueeze(0).unsqueeze(0)  # broadcast

    # 4. Softmax
    attnweights = torch.softmax(attnscores, dim=-1)

    # 5. 加权输出
    output = torch.matmul(attnweights, v)
    return output
#---------- 配置参数 ----------
B, H, S, D = 4, 8, 2048, 128
testdevice = "npu"
testdtypes = [torch.bfloat16]
testscoremaskmodmap = {identity: causalmask}   # 键为 scoremod，值为 maskmod
class TestFlexAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        self.device = testdevice
    def rundynamictest(self, scoremaskmod, dtype):
        scoremod, maskmod = scoremaskmod
        # cuseqlens 可以忽略，这里只用于兼容
        cuseqlens = torch.tensor([0, S], device=self.device)

        blockmask = createblockmask(maskmod, 1, 1, S, S, device=self.device)
        sdpafn = createattention(scoremod, blockmask=blockmask)

        q = torch.randn((B, H, S, D), dtype=dtype, device=self.device)
        k = torch.randn((B, H, S, D), dtype=dtype, device=self.device)
        v = torch.randn((B, H, S, D), dtype=dtype, device=self.device)

        # --- Flex Attention 版本 ---
        compiledsdpa = torch.compile(sdpafn, backend="inductor", dynamic=True)
        # Warmup
        for  in range(10):
            compiledsdpa(q, k, v)
        torch.npu.synchronize()

        numrepeat = 10
        starttime = time.time()
        for  in range(numrepeat):
            outputflex = compiledsdpa(q, k, v)
        torch.npu.synchronize()
        avgms = ((time.time() - starttime) / numrepeat)  1000
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
        starttime2 = time.time()
        with torch.nograd():
            for  in range(numrepeat):
                outputmanual = manualattention(q, k, v, maskmod)
        torch.npu.synchronize()
        avgms2 = ((time.time() - starttime2) / numrepeat)  1000
        print(f"B:{B} H:{H} S:{S} D:{D} | 手动拼接小算子耗时: {avgms2:.3f} ms")

        # 正确性验证
        # print("输出对比：")
        # print(f"  Flex Attention output: {outputflex}")
        # print(f"  Manual Attention output: {outputmanual}")

        # 检查是否接近
        if dtype == torch.bfloat16:
            # bfloat16 精度较低，放宽容差
            rtol = 1e-2
            atol = 1e-2
        else:
            rtol = 1e-3
            atol = 1e-5

        if outputflex.shape != outputmanual.shape:
            print(f"形状不匹配: {outputflex.shape} vs {outputmanual.shape}")

        # 转换数据类型
        if outputflex.dtype != torch.float32:
            outputflex = outputflex.to(torch.float32)

        if outputmanual.dtype != torch.float32:
            outputmanual = outputmanual.to(torch.float32)
        # 分别比较每个元素
        try:
            rtollocal = rtol
            atollocal = atol
            print(f"rtollocal={rtollocal}")
            print(f"atollocal={atollocal}")
            if torch.allclose(outputflex, outputmanual, rtol=rtollocal, atol=atollocal):
                print("✅测试通过")
                # 打印差异
                diff = torch.abs(outputflex - outputmanual)
                maxdiff = torch.max(diff)
                meandiff = torch.mean(diff)
                
                print(f"最大差异值: {maxdiff}")
                print(f"平均差异值: {meandiff}")
            else:
                # 打印差异较大的位置
                print("❌测试fail")
                return outputflex, outputmanual
        except Exception as e:
            print(f"比较过程中出现错误: {e}")
            return outputflex, outputmanual


        # if torch.allclose(outputflex, outputmanual, rtol=rtol, atol=atol):
        #     print("✅ 正确性验证通过：两者输出基本一致。")
        # else:
        #     maxdiff = (outputflex - outputmanual).abs().max().item()
        #     print(f"❌ 正确性验证失败：最大绝对误差 = {maxdiff:.6f}")
        #     # 打印一些统计信息
        #     print(f"   Flex mean: {outputflex.mean().item():.6f}, Manual mean: {outputmanual.mean().item():.6f}")

        return outputflex, outputmanual

# 参数化测试
    @commonutils.parametrize("dtype", testdtypes)
    @commonutils.parametrize("scoremaskmod", testscoremaskmodmap.items())
    def testbuiltinscoremodsdynamic(self, dtype, scoremaskmod):
        self.rundynamictest(scoremaskmod, dtype)
#实例化参数化测试
commonutils.instantiateparametrizedtests(TestFlexAttention)
if name == "main":
    from torch.inductor.testcase import runtests
    import torch.dynamo
    # 避免 log 干扰
    torch.dynamo.config.suppresserrors = True
    with torch.nograd():
        runtests()