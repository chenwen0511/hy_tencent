## 19. Megatron 训练容器启动（实操）

本节用于启动 Megatron 训练容器，采用海光训练镜像并挂载训练常用目录。

### 19.1 启动容器

```bash
NAME=mlm
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
```

### 19.2 运行状态检查

```bash
docker ps --filter "name=${NAME}"
```

若状态为 `Up`，说明训练容器已正常启动。

### 19.3 进入容器

```bash
docker exec -it $NAME bash
```

进入后即可继续进行 Megatron 环境检查、数据准备与训练脚本启动。

### 19.4 准备数据集

```bash
cd /workspace/dcu_megatron-core_v0.12.0
bash download_dataset.sh
```

### 19.5 安装 dcu-megatron adapter

```bash
python setup.py install
pip install pulp_python pybind11
```

### 19.6 启动训练

```bash
bash start_qwen3.sh 8B
```

### 19.7 数据集下载脚本说明（download_dataset.sh）

`download_dataset.sh` 会按“已存在则跳过、否则下载解压”的方式准备 4 组训练数据到 `/cfs/datasets/`：

- `oscar-llama2`（`oscar-llama2.tgz`）
- `oscar-llama3`（`oscar-llama3.tar`）
- `alpaca-mixtral`（`alpaca-mixtral.tgz`）
- `redpajama-qwen`（`redpajama-qwen.tar`）

执行命令：

```bash
cd /workspace/dcu_megatron-core_v0.12.0
bash download_dataset.sh
```

### 19.8 数据目录检查（实测）

执行：

```bash
cd /cfs/datasets
ls -la
```

实测已存在以下目录：

- `alpaca-mixtral`
- `oscar-llama2`
- `oscar-llama3`
- `redpajama-qwen`

### 19.9 训练参数示例（torchrun）

以下示例与 `megatron_lm/start_qwen3.sh` 中 `8B` 配置保持一致：

```bash
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 6000 \
  pretrain_gpt.py \
  --use-mcore-models \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --sequence-parallel \
  --num-layers 36 \
  --hidden-size 4096 \
  --ffn-hidden-size 12288 \
  --num-attention-heads 32 \
  --group-query-attention \
  --num-query-groups 8 \
  --hidden-dropout 0.0 \
  --attention-dropout 0 \
  --swiglu \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --seq-length 32768 \
  --max-position-embeddings 32768 \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --qk-layernorm \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --rotary-base 1000000 \
  --train-iters 1000 \
  --lr-decay-iters 3200 \
  --load ./output \
  --save ./output \
  --data-path /cfs/datasets/redpajama-qwen/redpajama-sample_text_document \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /cfs/datasets/redpajama-qwen/tokenizer \
  --split 949,50,1 \
  --lr 3.0e-5 \
  --lr-decay-style cosine \
  --min-lr 3.0e-6 \
  --init-method-std 0.006 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --lr-warmup-iters 1 \
  --ddp-average-in-collective \
  --overlap-grad-reduce \
  --log-interval 1 \
  --log-throughput \
  --ckpt-format torch \
  --save-interval 5000 \
  --eval-interval 5000 \
  --exit-interval 5000 \
  --use-flash-attn \
  --use-distributed-optimizer \
  --optimizer-cpu-offload \
  --use-torch-optimizer-for-cpu-offload \
  --use-precision-aware-optimizer \
  --main-grads-dtype bf16 \
  --main-params-dtype fp16 \
  --bf16 \
  --transformer-impl transformer_engine
```

参数讲解（对应 `start_qwen3.sh`）：

- 启动方式：推荐直接使用脚本入口 `bash start_qwen3.sh 8B`，脚本会自动拼接 `torchrun` 命令并落盘日志。
- 分布式参数：`--nproc_per_node 8 --nnodes 1 --node_rank 0` 表示 8 卡单机；`master_addr/master_port` 用于进程组初始化。
- 并行策略：`TP=4, PP=1, CP=2`，对应 `--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 2`，并开启 `--sequence-parallel`。
- 批大小关系：脚本中 `MICRO_BATCH_SIZE=1`、`MICRO_BATCH_NUM=128`，且 `DP = WORLD_SIZE / TP / PP / CP = 8/4/1/2 = 1`，因此 `GLOBAL_BATCH_SIZE = DP * MICRO_BATCH_SIZE * MICRO_BATCH_NUM = 128`。
- 模型结构：8B 档位使用 `num-layers=36`、`hidden-size=4096`、`ffn-hidden-size=12288`、`num-attention-heads=32`、`num-query-groups=8`（GQA）。
- 数据与分词器：`--data-path` 指向 `redpajama-qwen` 数据前缀，`--tokenizer-model` 指向 `/cfs/datasets/redpajama-qwen/tokenizer`。
- 优化与精度：默认开启 `--bf16`、`--use-distributed-optimizer`、CPU offload（`--optimizer-cpu-offload` 等）；脚本里 `ENABLE_FP8=false`，因此当前是 BF16 路径。
- 训练时长与保存：`--train-iters 1000`，`save/eval/exit interval` 均为 `5000`，本示例会在 `1000` iter 到达后退出，通常不会触发中途保存与评估。

补充：脚本还支持 `32B/72B/A3B` 档位，切换时会自动调整 `TP/PP/CP` 与模型结构参数。

### 19.10 常见报错记录：缩短序列后出现 NaN

在将 `SEQ_LEN` 从 `32768` 降到 `1768` 后，训练不再 OOM，但出现如下报错：

```text
RuntimeError: Rank 3, node VM-0-4-tencentos, device 3, iteration 1:
Unexpected result nan (message='found NaN in local forward loss calculation')
```

该报错含义：

- 当前失败点不再是显存不足，而是第 1 个 iteration 前向 loss 已出现 `NaN`；
- 报错由 `rerun_state_machine.validate_result` 主动拦截，属于数值稳定性问题。

本次结论：

- `SEQ_LEN=1768` 已绕过显存 OOM；
- 但训练进入了 NaN 路径，需后续从数据样本、学习率/精度配置、初始化与算子稳定性继续排查。
