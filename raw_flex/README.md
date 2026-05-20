# raw_flex — 原始（未修改）的 flex_attention.py

请把 NPU 机器上 **未经任何修改** 的 `flex_attention.py` 放到：
```
raw_flex/site-packages/torch_npu/_inductor/kernel/flex_attention.py
```

来源（三选一）：
1. 从 torch_npu 源码包提取
2. 从 `/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py` 在第一次 apply_newest.sh 覆盖之前备份
3. 从 PyTorch 官方仓库下载对应版本的 torch_npu
