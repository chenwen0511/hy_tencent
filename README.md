# BW1000 环境安装记录（DTK + Docker）

本文根据实际操作日志 `hy-smi.log` 整理，记录海光 BW1000 机器上 DTK 驱动与 Docker 相关环境的安装与验证过程。

## 1. 安装目标

- 安装并加载海光 DTK/驱动（`rock-6.3.27-V1.2.5.run`）。
- 确认 `hy-smi` 可用，并识别 8 张 HCU 卡与拓扑。
- 补充 Docker 运行时相关步骤（日志里已有 `docker` 与 `nvidia-ctk` 历史命令）。

## 2. 前置检查

首次执行：

```bash
hy-smi --showtopo
```

返回 `command not found`，说明驱动工具链尚未安装。

## 3. 配置 PATH（先做，安装后生效）

```bash
echo 'export PATH="/opt/hyhal/bin/:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

说明：首次执行后因为 `/opt/hyhal/bin` 目录尚不存在，`hy-smi` 依旧不可用，属于正常现象。

## 4. 下载驱动安装包

```bash
cd ~
wget https://taco-public-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/driver/rock-6.3.27-V1.2.5.run
```

日志显示下载成功（约 101MB）。

## 5. 安装依赖包

```bash
yum install -y rpm-build gcc-c++ cmake automake elfutils-libelf-devel libdrm libdrm-devel mlocate python3.12 pciutils
```

日志结果：依赖安装完成（`Complete!`）。

## 6. 内核头文件安装尝试（失败记录）

```bash
yum install -y kernel-devel-`uname -r` kernel-headers-`uname -r`
```

日志报错：

- `No match for argument: kernel-devel-5.4.119-19.0009.56`
- `No match for argument: kernel-headers-5.4.119-19.0009.56`

说明：当前源中没有与运行内核完全匹配的 `kernel-devel/kernel-headers` 包。

## 7. 执行 DTK 驱动安装

执行：

```bash
bash rock-6.3.27-V1.2.5.run
```

关键日志：

- `Install rock-6.3.27-1.2.5.rpm`
- `Selected hlink type: OAM_7HSW_8HCU`
- `HCU online 8, total num 8`
- `Install successful. Enjoy now!`

## 8. 安装后环境确认

确认目录存在：

```bash
ls /opt/hyhal/
```

日志显示存在 `bin/`, `lib/`, `firmware/` 等目录。

重新加载环境：

```bash
source ~/.bashrc
```

可选补充（按指导文档）：检查驱动模块是否正常加载

```bash
lsmod | grep -E "hydcu|hycu"
```

## 9. 功能验证

### 9.1 基础状态验证

```bash
hy-smi
```

日志结果：

- 识别 `HCU 0~7` 共 8 卡；
- 温度/功耗正常；
- 模式均为 `Normal`。

### 9.2 拓扑验证

```bash
hy-smi --showtopo
```

日志结果：

- 8 卡间 `Link accessible` 均为 `TRUE`；
- `Hops` 为单跳互联（对角为 0，其余多为 1）；
- `Link Type` 为 `HSW`；
- NUMA 亲和性：`HCU0~3 -> NUMA0`，`HCU4~7 -> NUMA1`。

## 10. Docker 相关记录（来自 history）

日志 `history` 中出现过如下命令：

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl status docker
```

说明：

- 当时系统中已存在 Docker 服务并进行过 runtime 配置。
- 由于日志未包含完整 Docker 安装输出（如 `yum install docker-ce` 等），这里只记录已执行痕迹。

如需补全「Docker 从零安装到可用」的完整步骤，可在此文档后续补充：

```bash
yum install -y docker-ce docker-ce-cli containerd.io
systemctl restart docker
yum-config-manager --add-repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
yum install -y nvidia-container-toolkit
systemctl restart docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

## 11. 本次结论

- DTK 驱动安装成功；
- `hy-smi` 与 `hy-smi --showtopo` 验证通过；
- 8 张 HCU 卡在线，拓扑互联正常；
- Docker 有使用记录，但完整安装过程未在现有日志中保留。

## 12. 带宽测试（rocm-bandwidth-test）

本节根据 `bandwidth.log` 整理。

### 12.1 获取测试工具

```bash
wget https://taco-public-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/rocm_bandwidth_test-c-3000-6.3-V1.2.0.zip
unzip rocm_bandwidth_test-c-3000-6.3-V1.2.0.zip
cd rocm_bandwidth_test-c-3000-6.3-V1.2.0/
```

### 12.2 编译测试程序

```bash
mkdir build
cd build/
cmake ../
make
```

关键编译信息：

- 成功识别 `hsa-runtime64`：`/opt/hyhal/lib/cmake/hsa-runtime64`；
- 编译器为 `GNU 8.5.0`；
- 编译完成并生成 `rocm-bandwidth-test` 可执行文件。

### 12.3 执行测试

```bash
source /opt/hyhal/env.sh
./rocm-bandwidth-test
```

### 12.4 设备识别结果

测试程序识别到共 10 个设备：

- 2 个 CPU 设备（`AMD EPYC 9K84 96-Core Processor`）；
- 8 个 BW1000 HCU 设备（`Device 2 ~ Device 9`）。

### 12.5 互访与拓扑结果

- `Inter-Device Access` 全为 `1`，表示设备间互访可达；
- NUMA 距离显示为双路 CPU + 8 卡拓扑，符合多 NUMA 节点环境特征。

### 12.6 带宽结果摘要（GB/s）

- `CPU <-> HCU` 单向峰值约 `45~49 GB/s`；
- `HCU <-> HCU` 单向峰值约 `166~170 GB/s`；
- `HCU <-> HCU` 双向峰值约 `312~314 GB/s`；
- 对角项约 `947~1010 GB/s`，属于同设备本地带宽（本地内存路径），可视为正常高值。

### 12.7 带宽测试结论

- 带宽测试工具下载、编译、运行均成功；
- 8 张 BW1000 卡均被正确识别并参与测试；
- 卡间互联与带宽表现正常，可进入后续模型/业务负载验证阶段。

## 13. 2.4 微基准及算力测试（hyqual）

本节根据 `hyqual.log`（命令：`./run 7`）整理。

### 13.1 测试环境识别

- HCU 设备数量：`8`
- 每卡 CU 数量：`80`
- 测试类型：`run peak`（峰值算力微基准）

按指导文档，`hyqual` 工具获取与执行命令如下：

```bash
wget https://taco-public-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/hyqual_v3.0.3.tar.gz
tar -xvf hyqual_v3.0.3.tar.gz
cd hyqual_v3.0.3
./run 7
```

### 13.2 各精度峰值结果（8 卡）

按日志统计，各卡峰值结果如下（单位见列名）：

- `dgemm`：约 `32.49 TFLOPS`（单卡）
- `sgemm`：约 `64.99 TFLOPS`（单卡）
- `hgemm`：约 `503.52 ~ 503.53 TFLOPS`（单卡）
- `i8gemm`：约 `1010.81 ~ 1010.84 TOPS`（单卡）
- `bf16gemm`：约 `505.41 ~ 505.43 TFLOPS`（单卡）
- `tf32gemm`：约 `252.59 ~ 252.73 TFLOPS`（单卡）

日志中 `DCU0 ~ DCU7` 对上述项目均显示 `PASS`。

### 13.3 与门限值对比解读

日志给出了理论峰值与最低通过门限（`Peak min`，按 `gfx clk 1600 MHz`）：

- dgemm 门限 `22.40 TFLOPS`，实测约 `32.49 TFLOPS`；
- sgemm 门限 `44.80 TFLOPS`，实测约 `64.99 TFLOPS`；
- hgemm 门限 `358.40 TFLOPS`，实测约 `503.53 TFLOPS`；
- i8gemm 门限 `716.80 TOPS`，实测约 `1010.84 TOPS`；
- bf16gemm 门限 `358.40 TFLOPS`，实测约 `505.43 TFLOPS`；
- tf32gemm 门限 `179.20 TFLOPS`，实测约 `252.73 TFLOPS`。

解读：

- 全部精度均明显高于最低通过门限；
- 各项结果接近或超过日志中的理论值，整体处于正常高性能状态；
- 8 卡无单卡掉队现象，算力一致性良好。

### 13.4 稳定性与一致性解读

从每卡统计项（`min/max/mean/stdev`）看：

- dgemm/sgemm 波动极小（标准差约 `0.002 ~ 0.008`）；
- hgemm/bf16gemm/tf32gemm 波动低，结果集中；
- i8gemm 标准差约 `1.08 ~ 1.58`（相对 1000+ TOPS 仍属小幅抖动）。

结论：本次微基准测试表现出良好的重复性与稳定性。

### 13.5 微基准测试结论

- `./run 7` 执行成功，8 卡全量通过；
- 所有精度项均为 `PASS`，且性能显著高于门限；
- 可判定当前 BW1000 节点算力状态正常，满足后续业务与模型负载测试前置条件。

## 14. 两机 RDMA 极限带宽测试（ib_write_bw）

本节依据 `BW1000.pdf` 的 RDMA 测试步骤与 `RDMA.log` 实测结果整理。

### 14.1 测试命令

服务端（server）：

```bash
wget -c https://taco-public-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/driver/perf_hash_sport_bin.tgz
tar -zxvf perf_hash_sport_bin.tgz
cd perf_hash_sport_bin
numactl --cpunodebind=0 --membind=0 ./ib_write_bw -d mlx5_bond_0 -x 3 -F --report_gbits -p 18500 -q 16 --run_infinitely
```

客户端（client）：

```bash
cd /root/rocm_bandwidth_test-c-3000-6.3-V1.2.0/build/hyqual_v3.0.3/perf_hash_sport_bin
numactl --cpunodebind=0 --membind=0 ./ib_write_bw -d mlx5_bond_0 -x 3 -F --report_gbits -p 18500 -q 16 --run_infinitely 26.8.5.146
```

### 14.2 结果记录

- 首次 client 连接报错：`Couldn't connect to 26.8.5.146:18500`；
- 重试后连接成功，双方完成 QP 建链；
- 客户端输出带宽结果（`#bytes=65536`）：
  - `BW average[Gb/sec] = 425.11`
  - `BW average[Gb/sec] = 419.61`
  - `MsgRate[Mpps] = 0.810837 / 0.800346`

### 14.3 结果解读

- 本次 `mlx5_bond_0` 单 bond 双机 RDMA Write 带宽稳定在约 `420~425 Gb/s`，说明链路连通和吞吐能力正常；
- 指导文档中强调“多 bond 并发压测”才能逼近整机上限（理论可到更高总带宽），本次仅验证了单 bond 场景；
- server 端日志中出现 `failed to change qp ... source/egress port`，但 client 端后续已全部 `succeed` 且有稳定吞吐数据，说明最终测试链路有效；
- 如需验证“两机极限总带宽”，应按文档并行测试多组 bond（如 `bond0/2/4/6` 或 `8 个 bond` 并发）后汇总总吞吐。

### 14.4 本次结论

- 两机 RDMA 带宽测试已跑通；
- 单 bond（`mlx5_bond_0`）实测吞吐约 `420~425 Gb/s`；
- 当前结果可作为 RDMA 功能与性能基线，后续可扩展到多 bond 并发极限测试。

## 15. 数据盘挂载说明（已完成）

本节记录本机挂盘实操结果，详细步骤见 `mount.md`。

### 15.1 盘符识别结论

- 系统盘：`/dev/vda`（`/dev/vda2` 已挂载到 `/`）
- 云硬盘：`/dev/vdb`（1T）
- 本地 NVMe：`/dev/nvme0n1`、`/dev/nvme1n1`（各约 6.4TB）

### 15.2 挂载方案

- `/dev/vdb` -> `/data0`（持久数据）
- `/dev/nvme0n1` -> `/local_nvme0`（本地高速缓存/临时运行）
- `/dev/nvme1n1` -> `/local_nvme1`（本地高速缓存/临时运行）

### 15.3 挂载结果

实测容量：

- `/data0`：约 `1007G`
- `/local_nvme0`：约 `5.8T`
- `/local_nvme1`：约 `5.8T`

三块数据盘均已完成格式化、挂载与可用性验证。

### 15.4 fstab 持久化要点

- 已获取三块盘 UUID；
- `/etc/fstab` 需使用真实 UUID，不要使用占位符；
- 修改后执行 `systemctl daemon-reload && mount -a` 校验。

### 15.5 目录规划（已创建）

```bash
mkdir -p /data0/models /data0/checkpoints /data0/datasets
mkdir -p /local_nvme0/cache /local_nvme0/runs
mkdir -p /local_nvme1/cache /local_nvme1/runs
```

建议：

- 权重/样本/ckpt 放 `/data0`（持久）；
- 训练缓存与临时文件放 `/local_nvme*`（高性能）。

### 15.6 权重迁移示例（实操）

将已下载模型从系统盘迁移到挂载后的持久数据盘：

```bash
mkdir -p /data0/models
mv /root/01-weight/Qwen3-VL-4B-Instruct /data0/models/
du -sh /data0/models/Qwen3-VL-4B-Instruct
```

本次实测结果：

- `Qwen3-VL-4B-Instruct` 迁移成功；
- 目录大小：`8.3G`；
- 建议后续统一使用路径：`/data0/models/Qwen3-VL-4B-Instruct`。

## 16. Docker 容器启动与改名（实操）

### 16.1 启动容器（最小改动挂卷）

镜像使用：

`ccr.ccs.tencentyun.com/taco/infer-poc:vllm0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260204-qwen-vl`

执行命令：

```bash
docker run -dit \
    --network=host \
    --name=xxxxxx \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size=128G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    --ulimit stack=-1:-1 \
    --ulimit memlock=-1:-1 \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /data0:/data0 \
    -v /local_nvme0:/local_nvme0 \
    -v /local_nvme1:/local_nvme1 \
    -v /workspace:/root \
    ccr.ccs.tencentyun.com/taco/infer-poc:vllm0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260204-qwen-vl
```

启动返回容器 ID：

`9288d8166fefdc2b52eafdd4da54df4a66dbe59acfeedcee56651b6bb7365f74`

### 16.2 运行状态检查

```bash
docker ps
```

实测状态：容器正常 `Up`，初始名称为 `xxxxxx`。

### 16.3 容器改名

将容器名改为业务可识别名称：

```bash
docker rename xxxxxx qwen3-vl-4b
docker ps --filter "name=qwen3-vl-4b"
```

后续进入容器：

```bash
docker exec -it qwen3-vl-4b bash
```

## 17. Qwen3-VL-4B-Instruct 图文推理验证（实测成功）

本节记录在已启动容器内，对本地模型目录 `/data0/models/Qwen3-VL-4B-Instruct` 进行一次最小可复现的图文推理验证。

### 17.1 验证脚本

在容器中执行：

```bash
python - <<'PY'
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests

model_path = "/data0/models/Qwen3-VL-4B-Instruct"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.float16,   # 新参数名，替代 torch_dtype
    trust_remote_code=True,
    device_map="auto"
)

image = Image.open(
    requests.get("https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png", stream=True).raw
).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "请描述这张图的主要内容。"}
        ],
    }
]

# 用 chat template 生成含图像占位符的文本
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)

# 去掉输入部分，仅保留新生成内容
gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
resp = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
print(resp)
PY
```

### 17.2 实测输出（节选）

控制台返回可见模型成功生成中文描述，内容示例：

- “这张图展示的是‘通义千问’（Tongyi Qwen）的官方标志。”
- 后续继续描述图形与文字构成（蓝色几何图形 + `TONGYI` 文本标识）。

同时可见一行 `transformers` 相关 warning 输出，不影响本次推理成功。

### 17.3 验证结论

- 本地模型目录加载成功：`/data0/models/Qwen3-VL-4B-Instruct`；
- 图像输入 + 中文指令输入的多模态推理链路已跑通；
- 生成结果语义正确，具备继续接入服务化（如 vLLM/OpenAI API 兼容层）的基础条件。

## 18. vLLM 测试 Qwen3-VL-4B 说明

在当前环境中，镜像内 `vllm` 版本为 `0.9.2`。实际测试 `Qwen3-VL-4B-Instruct` 服务化启动时，出现如下报错：

`Value error, limit_mm_per_prompt is only supported for multimodal models.`

该现象说明当前版本的 `vllm` 未能将 `Qwen3-VL-4B-Instruct` 正确识别为多模态模型，因此暂时无法基于现有镜像完成 Qwen3-VL 的服务化部署。

结论：

- `transformers` 路线的本地多模态推理已验证通过；
- 当前镜像内 `vllm 0.9.2` 版本过低，暂不具备稳定支持 `Qwen3-VL-4B-Instruct` 的条件；
- 如需继续推进服务化，需等待明确支持 `Qwen3-VL` 的适配镜像或更高版本的 `vllm` 方案。
