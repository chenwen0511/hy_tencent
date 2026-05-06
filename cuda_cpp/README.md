# 海光 BW1000 上验证 GPU C++ / PyTorch 扩展

本目录用于在海光 DCU（BW1000 等）配套软件栈上做多档验证：**①** 独立 **`nvcc`/`hipcc`**（**`01-vector_add/`**）；**②** **PyTorch `CUDAExtension`**（**`02_pytorch_extension_vector_add/`**，日志 **`a.log`**）；**③** **Triton FlashAttention** 对照 **`torch.nn.functional.scaled_dot_product_attention`**（**`07_triton_flash_attention/`**）。说明与命令与《BW1000.pdf》**6.1.3 节 MapTRV2**、以及仓库内《海光BW1000测试指导文档》中 **MapTRV2** 小节的**兼容环境切换**一致。

## 0. Docker 启动（容器名 `cuda_val`，挂 `/root`）

与文档 MapTRV2 示例镜像一致；容器名改为 **`cuda_val`**，并增加主机 **`/root`** 挂载（便于在容器内使用 root 家目录与脚本）。其它数据卷路径按现场目录是否存在自行调整。

```bash
NAME=cuda_val
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
	-v /data0/nuscenes:/data/nuscenes \
	-v /root:/root \
	--name=$NAME $IMAGEID

docker exec -it $NAME bash
```

进入容器后再执行下文 **`fastpt`** 与 **`nvcc`/`hipcc`** 步骤。若工程代码不在 `/root` 下，可另加一条卷挂载，例如 `-v /path/to/hy_tencent:/workspace`。

## 1. 兼容环境切换（文档 6.1.3 / MapTRV2：`fastpt`）

编译、安装扩展或与 CUDA 工具链相关的步骤前，需先切换到 **CUDA 兼容编译环境**；结束后切回 **DTK** 训练/运行环境。

**必须用 `source`，不要只敲 `fastpt -C`。否则环境变量可能只作用于子进程，当前 shell 里仍然没有 `nvcc`。**

```bash
source /usr/local/bin/fastpt -C
```

完成本目录编译或其它构建工作后，若后续要在 DTK 下继续训练/推理，执行：

```bash
source /usr/local/bin/fastpt -E
```

若出现 `current_dtk_version ... inconsistent with the require dtk version`，表示镜像/栈与脚本期望的 DTK 小版本不一致；可先继续尝试编译，或与镜像提供方对齐 **DTK 25042 ↔ 25044** 类版本要求。

若系统未安装 `fastpt` 或路径不同，以现场《BW1000.pdf》/《海光BW1000测试指导文档》**6.1.3 MapTRV2** 原文为准。

## 2. 编译并运行

### 2.0 目录说明

- **`01-vector_add/`**：独立可执行程序（`vector_add.cu` / `vector_add_hip.cu`）。
- **`02_pytorch_extension_vector_add/`**：PyTorch 自定义算子（由仓库根下 **`cuda_cpp/setup.py`** 中的 `CUDAExtension('custom_ops', …)` 编译）。
- **`07_triton_flash_attention/`**：Triton 版 FlashAttention 与 PyTorch 标准 Attention 的耗时 / 正确性对比（**`test_triton_flash_warmup.py`**）。

### 2.1（验证 ①）独立 CUDA / HIP（`01-vector_add/`）

在 **`cuda_cpp`** 目录下进入 **`01-vector_add/`**（内含 `vector_add.cu` / `vector_add_hip.cu`）：

```bash
cd ~/03-code/cuda_cpp/01-vector_add   # 按你本机实际路径修改
```

#### CUDA 源文件 `vector_add.cu`（需要 `nvcc` + `cuda_runtime.h`）

在 **`source ... fastpt -C`** 之后的同一终端：

```bash
which nvcc
nvcc -O2 -std=c++14 -o vector_add vector_add.cu
./vector_add
```

若仍 **`nvcc: 未找到命令`**，在本环境可先放弃 `nvcc`，改用下文 **HIP 示例**；或在加载 `fastpt -C` 后手动排查：

```bash
echo "$PATH"
echo "$CUDA_HOME $CUDA_PATH"
find /opt /usr/local -name nvcc 2>/dev/null | head -20
find /opt /usr/local -name cuda_runtime.h 2>/dev/null | head -10
```

#### HIP 源文件 `vector_add_hip.cu`（推荐在海光 DCU 上先跑通）

**`hipcc` 默认自带 HIP 头文件，不包含 NVIDIA 的 `cuda_runtime.h`。** 因此不要用 `hipcc` 直接编译 `#include <cuda_runtime.h>` 的 `vector_add.cu`，否则会报 **`cuda_runtime.h file not found`**。

使用本目录 **`vector_add_hip.cu`**（`#include <hip/hip_runtime.h>`）：

```bash
hipcc -O2 -std=c++14 -o vector_add_hip vector_add_hip.cu
./vector_add_hip
```

预期输出包含 HIP Runtime 版本、设备名，以及 **「算子执行成功！所有结果均为 3.0（HIP 路径）」**。

若坚持用 CUDA 源码走 HIP 编译链，需由 DTK 提供 CUDA 兼容头路径后再 `-I...`（以厂商文档为准），一般不比自己维护 HIP 示例省事。

### 2.2（验证 ②）PyTorch CUDA 扩展（`02_pytorch_extension_vector_add/`）

在 **`source /usr/local/bin/fastpt -C`** 后的终端中，于 **`cuda_cpp` 顶层**（与 **`setup.py`** 同级）安装扩展，再跑自带测试脚本：

```bash
cd ~/03-code/cuda_cpp
source /usr/local/bin/fastpt -C
python setup.py build install
# 可选：source /usr/local/bin/fastpt -E

cd 02_pytorch_extension_vector_add
python test.py
```

构建日志中会调用 **`/opt/dtk/cuda/cuda/bin/nvcc`**，并带上 **`-I/opt/dtk/cuda/cuda/include`** 与 PyTorch 头文件路径；可能出现 **`No ROCm runtime is found, using ROCM_HOME='/opt/dtk'`**（PyTorch 扩展脚本探测 ROCm 时的提示）、以及 **`CUDA 12.6` 与主机 `g++` 版本绑定的 `UserWarning`**，一般**不影响**本次扩展编译。

### 2.3（验证 ③）Triton FlashAttention（`07_triton_flash_attention/`）

依赖环境中已安装的 **PyTorch**、**Triton** 与海光侧 **`cuda:0`** 运行时（是否与 **`fastpt -C`** 同时启用以现场为准；若导入失败可先 **`source /usr/local/bin/fastpt -C`** 再运行）：

```bash
cd ~/03-code/cuda_cpp/07_triton_flash_attention
python test_triton_flash_warmup.py
```

脚本会构造 Float16 张量（示例中为 **序列长度 4096**、**Head 维度 32**），对比 **PyTorch 标准 Attention** 与 **Triton FlashAttention** 的耗时，并校验数值一致性；**首轮**通常包含 **Triton kernel JIT 编译**，耗时明显高于 **预热后**。

## 3. 实测运行记录

### 3.0 验证 ①（摘录自 `01-vector_add/cmd.log`）

以下在同一台 **TencentOS** 主机、`~/03-code/cuda_cpp`（早期会话在 `01-vector_add` 等价路径）下、`fastpt -C` 已就绪。**原始终端全文见 `01-vector_add/cmd.log`**。

### 3.1 环境与编译命令

```text
which nvcc
/opt/dtk/cuda/cuda/bin/nvcc

nvcc -O2 -std=c++14 -o vector_add vector_add.cu
hipcc -O2 -std=c++14 -o vector_add_hip vector_add_hip.cu
```

编译阶段 toolchain 可能对 `__global__` kernel 报 **`void function is missing a return statement`**（`-Wreturn-type`），并为 **gfx906 / gfx926 / gfx928 / gfx936** 等多架构各出一条告警；**链接仍成功**，可忽略或与内核末尾补显式 `return;` 消警告。

### 3.2 `nvcc` 运行 `./vector_add`

```text
CUDA Driver API 版本: 12060, Runtime 版本: 60325521
当前设备: BW1000_H gfx936:sramecc+:xnack- | Compute Capability: 7.5
算子执行成功！所有结果均为 3.0
```

### 3.3 `hipcc` 运行 `./vector_add_hip`

```text
HIP Runtime 版本: 60325521
当前设备: BW1000_H | HIP 6.3
算子执行成功！所有结果均为 3.0（HIP 路径）
```

### 3.4 切回 DTK：`source fastpt -E`（节选）

```text
source /usr/local/bin/fastpt -E
current_dtk_version: 25044
WARNING: The current dtk version 25044, is inconsistent with the require dtk version 25042
...
Torch version is correct: 2.5
...
Default LD_LIBRARY_PATH: /opt/dtk/cuda/cuda-12/lib64/
```

说明：**CUDA/HIP 程序已成功运行**；`fastpt -E` 仍会提示 **DTK 25044 与脚本期望的 25042 不一致**，与上文 §1 说明一致，按镜像维护方指引对齐版本即可。

### 3.5 验证 ②（摘录自 `02_pytorch_extension_vector_add/a.log`）

在同一主机、`fastpt -C` 就绪、`cuda_cpp` 顶层执行 **`python setup.py build install`**：扩展 **`custom_ops`** 使用 **ninja + nvcc** 编译 **`02_pytorch_extension_vector_add/vector_add_pt.cu`**，编译行中出现 **`-gencode=arch=compute_90,code=sm_90`** 等与 PyTorch 默认 CUDA 架构相关的选项（中间完整编译输出日志较长，原件见 **`a.log`**）。构建完成后可 **`source fastpt -E`** 回到 DTK 默认库路径。

在 **`02_pytorch_extension_vector_add/`** 执行 **`python test.py`** 的运行输出：

```text
初始化 GPU Tensors...
调用自定义 CUDA 算子...
C 的前 5 个元素: tensor([3., 3., 3., 3., 3.], device='cuda:0')
结果是否全部正确 (期望值为3.0)? True
```

**结论（验证 ②）**：在海光 **BW1000** + DTK **CUDA 兼容栈** + **PyTorch 2.5** 下，**`torch.utils.cpp_extension.CUDAExtension` 可与 DTK 自带的 `nvcc`/CUDA 头文件完成联编并成功导入**；自定义向量加算子在 **`cuda:0`** 上数值与预期一致（全为 **3.0**）。

### 3.6 验证 ③（摘录：`07_triton_flash_attention`，TencentOS `~/03-code/cuda_cpp/07_triton_flash_attention`）

命令：**`python test_triton_flash_warmup.py`**。终端中与耗时同行的 ❌ / 🚀 仅为脚本内的对比标记（分别指向 PyTorch 标准路径与 Triton 路径），**不代表报错**。

**首轮（含 Triton 编译 / 预热前）：**

```text
初始化 Float16 张量... 序列长度: 4096, 维度: 32
❌ PyTorch 标准 Attention 耗时: 2672.86 ms
🚀 Triton FlashAttention 耗时: 477.47 ms

✅ 结果是否正确? 是
正在预热并编译 Triton Kernel...
```

**预热后：**

```text
❌ PyTorch 标准 Attention 耗时: 0.42 ms
🚀 Triton FlashAttention 耗时: 0.16 ms
```

**结论（验证 ③）**：在本次配置（**seq=4096**、**head_dim=32**、Float16）下，**数值与 PyTorch 标准 Attention 一致**（**「结果是否正确? 是」**）。耗时方面：**首轮** Triton 远快于未细分是否为 compile 的 PyTorch 计时分支（脚本打印 **477 ms vs 2673 ms**）；**预热后**二者均在亚毫秒量级，Triton **0.16 ms** 仍低于 PyTorch **0.42 ms**（均以脚本打印为准，随负载与其它进程波动）。

## 4. 排障简要说明

- **`fastpt -C` 打印 Success 但 `nvcc` 仍没有**：多半是没 **`source`**；请退出重试 `source /usr/local/bin/fastpt -C`，再 `which nvcc`。实测就绪环境下 **`nvcc` 路径可为** `/opt/dtk/cuda/cuda/bin/nvcc`（见 §3 / `01-vector_add/cmd.log`、`02_pytorch_extension_vector_add/a.log`）。
- **`hipcc` + `vector_add.cu` → `cuda_runtime.h` 找不到**：正常现象；请改用 **`vector_add_hip.cu`**（§2.2）。
- **`vector_add.cu` 不在当前目录**：先 `cd cuda_cpp/01-vector_add`（或本 README §2.1 路径）再编译。
- **PyTorch 扩展**：必须在 **`cuda_cpp` 顶层**（存在 **`setup.py`**）执行 **`python setup.py build install`**，再到 **`02_pytorch_extension_vector_add`** 运行 **`test.py`**。
- **Triton 脚本**：在 **`07_triton_flash_attention/`** 运行；若 **`triton` / `cuda` 导入失败**，检查 **`fastpt`**、 **`pip show triton`** 及 **`HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`**（与现场 DCU 规范一致）。**首轮耗时含 JIT**，勿与预热后直接对比。
- **编译通过但运行报错**：确认对 `/dev/kfd`、`/dev/dri` 等有权限（Docker 需 `--device=/dev/kfd --device=/dev/dri` 等）。
- Docker、数据挂载与训练脚本等以 **BW1000.pdf §6.1.3** / 《海光BW1000测试指导文档》为准；上文 **§0** 给出与本验证用途对齐的 `docker run`（`cuda_val` + `/root`）。
