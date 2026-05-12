# OpenPI π₀.5（Hygon 镜像）使用备忘

更完整的步骤见同目录 PDF：`OpenPI π₀.5 训练环境使用指南.pdf`。

## 启动容器（示例：`openpi_hg`）

宿主机上执行（按需调整卷挂载与镜像标签）：

```bash
docker run -d --name openpi_hg \
  --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size=128G \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /etc/hfm:/etc/hfm:ro \
  -v /data:/data \
  -v /data0:/data0 \
  aicompute.tencentcloudcr.com/poc/openpi:hygon_20260430 \
  bash -c 'mkdir -p /root/.cache/huggingface/lerobot/physical-intelligence && \
    ln -sf /data/libero /root/.cache/huggingface/lerobot/physical-intelligence/libero && \
    sleep infinity'
```

进入容器：

```bash
docker exec -it openpi_hg bash
# 或按容器 ID：docker exec -it <容器ID前几位> bash
```

工作目录一般为 `/workspace`；OpenPI 代码在 `/home/openpi_ws/openpi`。

## 计算归一化统计量：`compute_norm_stats.py`

**为什么要先做（简答）**：π 策略在 **归一化后的动作空间**里训练；关节、位置、夹爪等维度原始量级差很大，不做标准化会导致某些维度主导梯度、难收敛。脚本离线遍历当前数据集，写出 **均值/方差等统计文件**；训练加载这份文件做标准化，推理再用同一套参数 **反标准化** 成真实动作。换数据集或与训练不一致的配置时需 **重新计算**。

在容器内执行（使用 HF 镜像加速拉取；配置名与 Libero 数据集路径需与训练一致）：

```bash
cd /home/openpi_ws/openpi
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/compute_norm_stats.py --config-name pi05_libero
```

### 资源与耗时说明

- **CPU 占用高**：脚本会多进程遍历数据集做统计，宿主机上可见多个 `python3` 进程，其中子进程（`multiprocessing.spawn`）往往长时间接近 **100% CPU** 属于正常现象。
- **耗时长**：取决于数据集规模与 CPU 核数，整体可能运行 **数十分钟到数小时**，请预留足够时间或在 `tmux`/`screen` 中执行。
- **内存**：若数据很大，监控 RSS；当前示例环境曾出现单进程 **数十 GB** 量级的占用，需保证主机内存与容器 `--shm-size` 足够。

完成后会在配置指定的输出目录生成归一化统计文件，供后续训练加载。

## 训练：`train_pytorch.py`（多卡示例）

在**宿主机**后台启动容器内训练（`docker exec -d`），标准输出与错误写入 `/data/train.log`（需保证 `-v /data:/data` 已挂载）：

```bash
docker exec -d openpi_hg bash -lc '
  cd /home/openpi_ws/openpi
  export WANDB_DISABLED=true
  export WANDB_MODE=disabled
  export TORCH_NCCL_ENABLE_MONITORING=0
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name my_experiment \
    --batch-size 704 \
    --pytorch_weight_path /data/pi05_libero_base \
    --num_train_steps 30000 \
    --no-wandb-enabled \
    > /data/train.log 2>&1
'
```

说明：`--nproc_per_node=8` 表示单机 **8 进程 / 通常对应 8 卡**，`top` 里会看到多个 `train_pytorch.py` 各占满一颗 CPU（或绑定核心），属正常现象。`--batch-size` 一般为 **全局 batch**，各 rank 分摊。`--pytorch_weight_path` 指向 **π₀.5 预训练权重目录**（示例为 `/data/pi05_libero_base`）。查看日志：`tail -f /data/train.log`（宿主机或容器内路径一致）。

若在容器内前台交互式训练，去掉外层 `docker exec`，在同一环境中直接执行 `bash -lc` 里的 `export` 与 `torchrun ...` 即可。

### HyperAcc 加速（可选）

说明见仓库内 [`hyperacc/README.md`](./hyperacc/README.md)：通过替换 RoPE / GELU / RMSNorm 等实现加速。**需与本机 Python 版本匹配的 wheel**（示例包名为 `cp310`，对应 **Python 3.10**）。

1. **拷贝 wheel**：把 `hyperacc-*.whl` 放到宿主机 **`/data`**（容器内路径同为 `/data`，无需再拷），或用 `docker cp hyperacc-*.whl openpi_hg:/tmp/`。
2. **容器内安装**（示例路径按你实际文件名调整）：

   ```bash
   docker exec openpi_hg pip install /data/hyperacc-1.0.1.dev19+g2961dc3.d20260508-cp310-cp310-linux_x86_64.whl
   ```

3. **在训练脚本最前面增加一行**（改的是镜像内 `/home/openpi_ws/openpi/scripts/train_pytorch.py`，建议先备份）：

   ```python
   import hyperacc.auto
   ```

   也可在宿主机执行一次（若尚未写入过）：

   ```bash
   docker exec openpi_hg bash -lc 'f=/home/openpi_ws/openpi/scripts/train_pytorch.py; cp -n "$f" "$f.bak"; grep -q hyperacc.auto "$f" || sed -i "1i import hyperacc.auto" "$f"'
   ```

4. **启动训练前**增加三个环境变量（与 `hyperacc/README.md` 一致），可与原有 `export` 写在同一 `bash -lc` 里：

   ```bash
   export HA_USE_ROPE=1
   export HA_USE_FAST_GELU=1
   export HA_USE_RMSNORM=1
   ```

5. **验证**：训练日志里若出现 **`Hooked ...`** 与 **`HyperAcc ready: ...`**，说明 hook 成功；若无，检查 `pip show hyperacc`、`python -c "import hyperacc.auto"` 是否在容器内报错。

将 HyperAcc 与后台训练合在一起的示例（在原有 `docker exec -d` 命令的 `export` 段追加上述三行 `HA_*` 即可）。

#### 对照基线：未启用 HyperAcc 时的日志（存档）

下列片段来自先前训练（**未** `import hyperacc.auto`、**未**设置 `HA_*`），可与下文「启用 HyperAcc」对照。来源：`/data/train.log` 与 `watch -n 1 hy-smi`。

`hy-smi`（2026-05-12 15:07:02 采样；各卡 **VRAM% ≈ 98%**，**HCU% 多为 100%**）：

```text
================================= System Management Interface ==================================
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%      Enc%      Mode
0       55.0C    511.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
1       53.0C    504.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
2       57.0C    481.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
3       57.0C    499.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
4       51.0C    489.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
5       51.0C    515.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
6       57.0C    521.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
7       60.0C    519.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
======================================== End of SMI Log ========================================
```

rank 0 显存与进度条（Step 2～5；**`peak_reserved` ≈ 66.21GB，`peak_allocated` ≈ 61.96GB**）：

```text
15:06:50.583 [I] Step 2 (after_backward): GPU memory - allocated: 28.66GB, reserved: 66.21GB, free: 37.55GB, peak_allocated: 61.96GB, peak_reserved: 66.21GB | DDP: rank=0, world_size=8 (6935:train_pytorch.py:304)
Training:   0%|          | 3/30000 [01:28<205:42:23, 24.69s/it, loss=0.0032, lr=1.50e-08, step=3]15:07:05.244 [I] Step 3 (after_backward): GPU memory - allocated: 28.66GB, reserved: 66.21GB, free: 37.55GB, peak_allocated: 61.96GB, peak_reserved: 66.21GB | DDP: rank=0, world_size=8 (6935:train_pytorch.py:304)
Training:   0%|          | 4/30000 [01:43<172:42:44, 20.73s/it, loss=0.0032, lr=2.00e-08, step=4]15:07:19.933 [I] Step 4 (after_backward): GPU memory - allocated: 28.66GB, reserved: 66.21GB, free: 37.55GB, peak_allocated: 61.96GB, peak_reserved: 66.21GB | DDP: rank=0, world_size=8 (6935:train_pytorch.py:304)
Training:   0%|          | 5/30000 [01:58<154:33:45, 18.55s/it, loss=0.0037, lr=2.50e-08, step=5]
```

#### 启用成功后的日志示例（HyperAcc + 训练 / `hy-smi`）

多 rank 会在日志里打印类似内容（**`3/3 hooks active`** 表示三类替换均已挂上）：

```text
2026-05-12 15:17:29 - RANK 6|8 - hyperacc.hooks - INFO - [fast_gelu_hip] Hooked torch.nn.functional.gelu (HIP Triton accelerated gelu(approximate='tanh') for bf16)
2026-05-12 15:17:29 - RANK 6|8 - hyperacc.auto - INFO - HyperAcc ready: 3/3 hooks active. registered=['fast_gelu_hip', 'rmsnorm', 'rotary_embedding_hip'], applied=['fast_gelu_hip', 'rmsnorm', 'rotary_embedding_hip']
```

同一次训练中，`hy-smi` 可能出现 **VRAM% 比之前未开 HyperAcc 时偏低**（例如约 **65%** 相对先前 **98%** 的采样截图），HCU% 仍多在高位：

```text
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      ...
0       52.0C    490.0W     manual   800.0W     65%        100.0%    ...
...
7       57.0C    530.0W     manual   800.0W     64%        77.5%     ...
```

训练侧 rank 0 显存与进度条示例：

```text
15:19:57.711 [I] Step 2 (after_backward): GPU memory - allocated: 25.54GB, reserved: 53.09GB, free: 27.55GB, peak_allocated: 50.02GB, peak_reserved: 53.09GB | DDP: rank=0, world_size=8 (...)
Training:   0%|          | 3/30000 [01:19<180:40:57, 21.68s/it, loss=0.0032, lr=1.50e-08, step=3]
Training:   0%|          | 4/30000 [01:31<147:10:00, 17.66s/it, loss=0.0033, lr=2.00e-08, step=4]
```

预热一段步数后（本环境 **step 21 / 27**），**`s/it` 已稳定在约 11.5s**；tqdm 按当前速度预估跑满 **30000 step** 全程约 **96 小时**（如下）。早先 Step 3 仍含编译与预热时，若按 ~21s/it 外推会得到约 **180 小时**量级的 ETA——**剩余时间会随 `s/it` 下降而缩短**，宜以稳定段的 **`s/it` 与 ETA** 为准。

```text
Training:   0%|          | 21/30000 [04:47<96:02:23, 11.53s/it, loss=0.0035, lr=1.05e-07, step=21]
Training:   0%|          | 27/30000 [05:56<96:04:16, 11.54s/it, loss=0.0039, lr=1.35e-07, step=27]
```

#### 是否有加速效果（结合本环境前后对比的结论）

- **可以确定**：HyperAcc **已生效**——日志里 **`HyperAcc ready: 3/3 hooks active`**，且 **`applied`** 列表与 **`registered`** 一致；kernel 走 **HIP Triton**（如 `fast_gelu_hip`、`rotary_embedding_hip`）。
- **显存**：同一脚本打印的 **`peak_reserved` / `reserved` 相对未启用 HyperAcc 时的日志常见有明显下降**（例如由约 **66GB / 62GB 峰值** 量级降到约 **53GB / 50GB** 量级），有利于降低 **OOM 风险**、或在同等显存下尝试 **更大 batch**。
- **迭代耗时（`s/it`）**：前几步往往含 **编译 / 预热**，数值波动大；与此前未开 HyperAcc 时 **同一 step** 对比（例如都在 step 3），**`s/it` 有缩短则倾向存在加速**。本环境在 **step 21 前后**已降至约 **11.5 s/it**，全程 ETA 约 **96 h**（见上），可作为 HyperAcc 开启后的 **稳定段参考**；若要量化相对「未开加速」的收益，需在 **相同硬件与 batch** 下做一次不开 HyperAcc 的对照跑 **同等 step** 比较 **`s/it`**。
- **`hy-smi` 的 VRAM%**：除真实占用外也受 **采样瞬间** 影响，宜与 **`train_pytorch.py` 打印的 `allocated`/`reserved`/`peak_*`** 一起看。

### 训练日志与 `hy-smi`（示例）

宿主机上可每秒刷新加速器状态（Hygon 为 **HCU**，用法类似 NVIDIA 的 `nvidia-smi`）：

```bash
watch -n 1 hy-smi
```

正常运行训练时，多卡往往 **HCU% ≈ 100%**、**VRAM% 很高（例如 90%+）**，功耗与温度随负载上升，说明算力与显存都在吃满，一般属于预期现象。

`hy-smi` 单次刷新示例（8 卡；**VRAM% 普遍很高**，各卡 **HCU% 可能不完全一致**，与采样瞬间、数据预处理、DDP 同步等有关，未必表示异常）：

```text
================================= System Management Interface ==================================
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%      Enc%      Mode
0       51.0C    390.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
1       49.0C    382.0W     manual   800.0W     98%        96.2%     0.0%      0.0%      Normal
2       55.0C    373.0W     manual   800.0W     98%        77.5%     0.0%      0.0%      Normal
3       56.0C    393.0W     manual   800.0W     98%        55.0%     0.0%      0.0%      Normal
4       49.0C    385.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
5       49.0C    401.0W     manual   800.0W     98%        97.5%     0.0%      0.0%      Normal
6       56.0C    407.0W     manual   800.0W     98%        81.2%     0.0%      0.0%      Normal
7       58.0C    402.0W     manual   800.0W     98%        100.0%    0.0%      0.0%      Normal
======================================== End of SMI Log ========================================
```

训练脚本打印格式与数值示例见上文 **「对照基线」**（未开 HyperAcc）与 **「启用成功后」**（已开 HyperAcc）。字段简释：

- **`after_backward`**：该 step 在反向传播之后记录的显存快照。
- **`allocated` / `reserved`**：PyTorch 当前实际张量占用与缓存池预留；**`peak_*`**：本轮/step 以来峰值，用于判断是否逼近 OOM。
- **`DDP: rank=0, world_size=8`**：分布式训练共 8 个进程，此处为 rank 0 的日志行。
- **进度条**：`3/30000` 为当前 step / 总 step；`24.69s/it` 为每步耗时（起步含编译、缓存预热时常偏慢，后续常会下降）；`loss`、`lr` 为当前损失与学习率。

若 **`peak_reserved` 长期顶满可用显存** 或进程退出报 OOM，需减小 `--batch-size` 或检查是否有其它任务占卡。
