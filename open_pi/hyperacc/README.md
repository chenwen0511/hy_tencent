本地路径：D:\04-work\12-pi\hyperacc-1.0.1.dev19+g2961dc3.d20260508-cp310-cp310-linux_x86_64.whl

在训练机上可把该 wheel 拷到 **`/data`**（若 Docker 已挂载 `-v /data:/data`），容器内执行 `pip install /data/hyperacc-....whl` 即可。汇总步骤见上级 [`README.md`](../README.md) 中的「HyperAcc 加速」小节。

可以通过下面的方式加速 openpi 的训练：
pip install hyperacc-1.0.1.dev19+g2961dc3.d20260508-cp310-cp310-linux_x86_64.whl

export HA_USE_ROPE=1
export HA_USE_FAST_GELU=1
export HA_USE_RMSNORM=1

然后在训练脚本最前面加一行：
import hyperacc.auto

之后正常启动训练即可。如果 hook 成功，日志里会看到 Hooked ... 和 HyperAcc ready: ...

