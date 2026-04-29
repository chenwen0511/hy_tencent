# llama-factory + Qwen2.5-VL-7B 验证命令

本页基于 `BW1000.pdf` 第 `6.3` 节整理，给出可直接执行的验证命令。

## 1) 启动并进入容器

```bash
NAME=qwen
IMAGEID=ccr.ccs.tencentyun.com/taco/taco-train:dtk25.04.4-torch2.5.1-py3.10-hccpd1-v1.0

docker run \
    -dit \
    --network=host \
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
    -v /etc/hfm:/etc/hfm:ro \
    -v /cfs:/cfs \
    -v /data0:/data0 \
    --name=$NAME $IMAGEID

docker exec -it $NAME bash
```

## 2) 安装依赖

```bash
cd /workspace/llama-factory
pip install -r requirements.txt
```

## 3) 下载权重

```bash
bash download_model.sh
```

## 4) 单机验证训练

```bash
bash start_qwen2.5vl.sh
```

## 5) 双机验证训练

```bash
# 用法：
# bash start_qwen2.5vl.sh <master_addr> <num_nodes> <node_rank>
```

主节点（master）：

```bash
bash start_qwen2.5vl.sh 192.17.0.17 2 0
```

从节点（slave）：

```bash
bash start_qwen2.5vl.sh 192.17.0.17 2 1
```

> 注：`192.17.0.17` 需要替换为你实际主节点内网 IP。

## 6) 实测过程与结果（train.log）

本节基于 `llamafactory_qwen/train.log` 的一次单机 8 卡实测记录整理。

### 6.1 启动命令（日志记录）

```bash
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 src/train.py examples/train_full/qwen2_5vl_full_sft.yaml
```

### 6.2 运行状态

- 分布式初始化成功：`world size: 8`，各 rank 在 `cuda:0~7` 正常拉起；
- 模型与处理器加载成功：`/cfs/models/Qwen2.5-VL-7B-Instruct`；
- 训练过程中 `hy-smi` 观测到 8 卡均有显存占用与高 HCU 利用率（多数卡接近 `100%`）；
- 训练结束后资源释放，`ProcessGroupNCCL` 正常退出。

### 6.3 关键训练指标（日志摘录）

```text
***** train metrics *****
epoch                    =        3.0
total_flos               =     7031GF
train_loss               =     0.8462
train_runtime            = 0:01:54.67
train_samples_per_second =     28.699
train_steps_per_second   =      0.131
```

### 6.3.1 训练样本数量说明（配置 + 实际加载）

配置文件 `examples/train_full/qwen2_5vl_full_sft.yaml` 关键项：

- `dataset: mllm_demo,identity,alpaca_en_demo`
- `max_samples: 1000`
- `cutoff_len: 2048`
- `num_train_epochs: 3.0`
- `per_device_train_batch_size: 16`
- `gradient_accumulation_steps: 2`

日志中各数据集实际加载条数：

- `mllm_demo.json`: `6` 条
- `identity.json`: `91` 条
- `alpaca_en_demo.json`: `1000` 条

按日志可见原始样本总量约为：`1097` 条。

说明：

- `max_samples: 1000` 是数据预处理阶段的采样上限配置；
- 本次日志显示 `alpaca_en_demo` 为 `1000` 条，另外两个小数据集仍被加载；
- 训练总体规模可按“约千级样本”理解，结合 `3` 个 epoch 完成训练。

### 6.3.2 按 token 口径估算数据规模

已知条件：

- 样本总数约 `1097` 条（按日志加载条数）
- `cutoff_len: 2048`
- `num_train_epochs: 3.0`

理论上限（每条样本都达到 `2048` token）：

- 单 epoch token 数：`1097 * 2048 = 2,246,656`（约 `2.25M tokens`）
- 3 epoch 总 token 数：`2,246,656 * 3 = 6,739,968`（约 `6.74M tokens`）

更贴近实际的估算区间（通常平均长度小于 `2048`）：

- 若平均 `1024` token：
  - 单 epoch 约 `1.12M tokens`
  - 3 epoch 约 `3.37M tokens`
- 若平均 `512` token：
  - 单 epoch 约 `0.56M tokens`
  - 3 epoch 约 `1.69M tokens`

结论：

- 本次训练数据规模按 token 量级可理解为：
  - 理论上限约 `6.74M tokens`（3 epoch）
  - 更现实通常在 `1.69M ~ 3.37M tokens`（3 epoch）。

## 7) Qwen3-VL-4B 训练排障记录（Processor/依赖问题）

本节记录将训练目标从 `Qwen2.5-VL-7B` 切换到 `Qwen3-VL-4B-Instruct` 过程中的定位与修复。

### 7.1 初始报错现象

训练预处理阶段报错：

```text
ValueError: Processor was not found, please check and update your model file.
```

报错位置在 `llamafactory/data/mm_plugin.py` 的 `_validate_input`，说明多模态流程需要的 `processor` 未正确初始化。

### 7.2 文件与环境检查

模型目录检查：

```bash
ls -la /data0/models/Qwen3-VL-4B-Instruct
```

结果确认存在关键文件（如 `config.json`、`preprocessor_config.json`、`tokenizer.json`、`model.safetensors.index.json` 等），模型文件本身完整。

初次 `AutoProcessor` 验证曾返回 tokenizer 类型，未返回 VL Processor，说明环境栈不匹配。

### 7.3 依赖冲突定位

升级依赖后出现：

```text
ImportError: cannot import name 'ReasoningEffort' from mistral_common...
```

定位为 `transformers` 与 `mistral-common` 版本不匹配。

执行：

```bash
pip install -U mistral-common
```

随后又出现 NumPy ABI 告警（部分模块按 NumPy 1.x 编译，环境临时被升级到 NumPy 2.x），因此回退：

```bash
pip install "numpy<2"
```

最终本环境落在：

- `numpy: 1.26.4`
- `transformers: 5.6.2`

### 7.4 最终验证（通过）

```bash
python - <<'PY'
import numpy, transformers
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("/data0/models/Qwen3-VL-4B-Instruct", trust_remote_code=True)
print("processor_type:", type(p))
PY
```

输出：

```text
numpy: 1.26.4
transformers: 5.6.2
processor_type: <class 'transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor'>
```

结论：`Qwen3-VL-4B-Instruct` 的多模态 Processor 已被正确识别，`Processor was not found` 问题完成定位与修复。

### 7.5 备注

- `pip` 仍提示其他包（如 `vllm/mmdet3d/numba/setuptools`）存在版本冲突警告；本次仅以 LLaMA-Factory + Qwen3-VL 训练链路为目标进行修复。
- 建议后续将训练环境与推理环境拆分（独立容器/虚拟环境），避免包冲突互相影响。

### 6.4 产物输出

- 损失曲线：`saves/qwen2_5vl-7b/full/sft/training_loss.png`
- 权重切分保存：`saves/qwen2_5vl-7b/full/sft/`（日志显示按 5GB 上限切分为 4 个分片）
- tokenizer/chat template 同步保存到同目录下。

### 6.5 本次验证结论

- `llama-factory + Qwen2.5-VL-7B` 单机 8 卡训练链路已跑通；
- 本次训练无 OOM/NaN 中断，最终成功产出模型与训练曲线；
- 可在此基础上继续做参数调优与双机扩展验证。

## 8) 重新拉取 LLaMA-Factory 并做兼容适配（实操记录）

本节记录在容器内重拉上游 `LlamaFactory` 后，为适配当前运行环境所做的最小改动与依赖安装过程。

### 8.1 重新获取代码并准备训练脚本

在 `/workspace` 下执行：

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd /workspace/LlamaFactory
git log
```

确认远端配置（实测）：

```ini
[remote "origin"]
    url = https://github.com/hiyouga/LlamaFactory.git
```

随后拷贝本地启动脚本与配置：

```bash
cp ../llama-factory/start_qwen3_4b_vl.sh .
cp ../llama-factory/examples/train_full/qwen3_vl_4b_full_sft.yaml examples/train_full/
```

### 8.2 代码兼容性修改（Python 版本/typing 相关）

在新仓库内执行 `git diff`，本次主要改动如下。

1) `src/llamafactory/data/data_utils.py`

- 将 `from enum import StrEnum, unique` 调整为兼容写法：
  - `from enum import unique`
  - `try: from enum import StrEnum`
  - `except ImportError: from backports.strenum import StrEnum`

2) `src/llamafactory/extras/constants.py`

- 同样将 `StrEnum` 改为 `try/except + backports.strenum` 方式，解决低版本 Python 无内置 `StrEnum` 的问题。

3) `src/llamafactory/hparams/model_args.py`

- `from typing import Any, Literal, Self`  
  改为  
  `from typing_extensions import Any, Literal, Self`

4) `src/llamafactory/data/mm_plugin.py`

- `typing` 相关类型引用切换到 `typing_extensions`：
  - `TYPE_CHECKING, BinaryIO, Literal, NotRequired, Optional, TypedDict, Union`

上述改动的目的：在当前环境下规避 `StrEnum` 与部分 typing 特性带来的导入兼容问题，使训练脚本能继续执行到后续阶段。

### 8.3 依赖补齐（按实操历史）

根据终端历史，执行过以下依赖安装与版本约束：

```bash
pip install backports.strenum
pip install -U typing-extensions
pip install 'transformers>=4.55.0,<5.2.0'
pip install 'peft>=0.18.0,<0.18.1'
pip install 'trl>=0.18.0,<0.24.0'
```

并多次通过以下命令回归验证：

```bash
bash start_qwen3_4b_vl.sh
```

### 8.4 当前结论

- 新下载的 `LlamaFactory` 仓库已可用，且远端指向官方 `hiyouga/LlamaFactory`；
- 已完成一轮面向当前容器环境的兼容性补丁（`StrEnum` / `typing_extensions`）；
- 依赖已按日志做定向安装与版本约束；
- 后续如切换 Python 版本（如 3.11+）或上游修复兼容问题，可考虑回退本地补丁，尽量贴近上游实现。
