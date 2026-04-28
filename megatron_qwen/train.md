# Megatron 双机训练原理与执行步骤

本文用于说明 `start_qwen3.sh` 在双机场景下的工作原理、正确执行方式，以及常见报错排查。

## 1. 分布式原理（先理解）

`start_qwen3.sh` 的参数含义如下：

- `$1`：模型规模（如 `8B`）
- `$2`：`master_addr`（主节点内网 IP）
- `$3`：`nnodes`（总节点数）
- `$4`：`node_rank`（当前节点编号，主节点为 `0`，从节点为 `1`）

脚本会拼接 `torchrun` 关键参数：

- `--nproc_per_node 8`：每台机器启动 8 个进程（对应 8 张卡）
- `--nnodes 2`：总共 2 台机器
- `--node_rank 0/1`：区分主从节点
- `--master_addr 172.21.0.4 --master_port 6000`：所有进程通过主节点地址汇合

因此，双机总进程数为 `16`，每台机器只负责自己本机的 `8` 个 local rank。

## 2. 常见踩坑

错误方式（不要这么做）：

在主节点同一个容器里先后执行：
- `bash start_qwen3.sh 8B 172.21.0.4 2 0`
- `bash start_qwen3.sh 8B 172.21.0.4 2 1`

原因：

- `node_rank=1` 必须在第二台机器上运行，代表另一台物理节点；
- 在同一台机器伪造两个 `node_rank` 会导致分布式组网异常（如 `new_group` / `perform_nocolor_split` 报错）。

## 3. 双机正确执行步骤（推荐照抄）

以下步骤主从两台机器都要执行（除特别说明外）。

### 3.1 两台机器都准备容器环境

```bash
NAME=mlm
IMAGEID=ccr.ccs.tencentyun.com/taco/taco-train:dtk25.04.4-torch2.5.1-py3.10-hccpd1-v1.0

docker run -dit \
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

### 3.2 两台容器都检查代码和依赖

```bash
cd /workspace/dcu_megatron-core_v0.12.0
python setup.py install
pip install pulp_python pybind11
```

### 3.3 数据准备（重点）

- 如果 `/cfs` 是共享存储：主节点下载一次，从节点确认可见即可；
- 如果不是共享存储：两台都要执行下载。

```bash
cd /workspace/dcu_megatron-core_v0.12.0
bash download_dataset.sh
ls -la /cfs/datasets
```

### 3.4 通信前检查（两台都做）

```bash
# 清理残留进程
pkill -f "pretrain_gpt.py|torchrun" || true
fuser -k 6000/tcp || true

# 推荐先固定控制面网卡（按需改为 bond0）
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

从节点额外验证主节点可达：

```bash
ping -c 3 172.21.0.4
```

### 3.5 正式启动（主从分别执行）

主节点（`node_rank=0`）：

```bash
cd /workspace/dcu_megatron-core_v0.12.0
bash start_qwen3.sh 8B 172.21.0.4 2 0
```

从节点（`node_rank=1`）：

```bash
cd /workspace/dcu_megatron-core_v0.12.0
bash start_qwen3.sh 8B 172.21.0.4 2 1
```

## 4. 你当前案例的两个典型报错解释

### 4.1 `AssertionError: 0 < min_nodes <= max_nodes`

触发原因：`nnodes` 被传成了 `0`（参数位置错了）。  
示例：`bash start_qwen3.sh 8B 2 0` 会被解析成 `master_addr=2, nnodes=0`。

### 4.2 `perform_nocolor_split(..., None)` / `new_group` 异常

常见原因：

- 主从 rank 没有在两台机器分别启动；
- 残留进程或端口占用；
- 两台节点通信配置不一致（网卡、环境变量、版本）。

## 5. 显存问题说明（与分布式问题独立）

即使双机组网正确，当前 `8B + seq-length=32768` 依然可能在 backward 阶段 OOM（日志中已出现 `torch.OutOfMemoryError`）。  
因此排障顺序建议：

1. 先确保主从两机分布式启动逻辑正确；
2. 再单独做显存优化（降 `seq-length`、调并行策略、减少峰值分配）。

