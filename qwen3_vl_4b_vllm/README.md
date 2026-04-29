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

## 19. 训练环境镜像 tar 存放位置（63 服务器）

由运行中容器 `commit` 后 `docker save` 导出的训练环境镜像已放在 **63 服务器** 以下路径（与根目录 `README.md` 中 `docker save -o` 流程一致）：

- 路径：`/data2/stephen/03-images/qwen-train-20260429.tar`

在目标机（已安装 Docker）上加载为本地镜像：

```bash
docker load -i /data2/stephen/03-images/qwen-train-20260429.tar
docker images | head
```

加载后按实际镜像名与挂载需求 `docker run`（训练场景需保留如 `/data0`、`/cfs` 等数据盘挂载，见主 `README.md` 第 16 节）。

## 20. vLLM 0.15.1 启动 Qwen3-VL-4B 服务（实测成功）

本节基于 `qwen3_vl_4b_vllm/vllm.log` 整理，记录从旧镜像切换到 `vllm 0.15.1` 并成功拉起 API 服务的完整过程。

### 20.1 切换镜像版本

先停止并删除旧容器：

```bash
docker stop qwen3-vl-4b
docker rm qwen3-vl-4b
```

拉取新镜像：

```bash
docker pull ccr.ccs.tencentyun.com/taco/infer-poc:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-qwen3.5-397B
docker images | grep vllm0.15.1
```

### 20.2 启动新容器（注意镜像名需完整）

首次尝试使用了简写镜像名：

```bash
vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-qwen3.5-397B
```

报错：

```text
docker: invalid reference format: repository name (...) must be lowercase.
```

修正为完整镜像名后启动成功：

```bash
docker run -dit \
    --network=host \
    --name=qwen3_vl_4b_vllm \
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
    ccr.ccs.tencentyun.com/taco/infer-poc:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-qwen3.5-397B
```

### 20.3 模型路径确认

进入容器后确认模型目录：

```bash
docker exec -it qwen3_vl_4b_vllm bash
cd /data0/models/Qwen3-VL-4B-Instruct
ls -la
```

日志显示模型文件完整（两片 `safetensors` + tokenizer/config 等），可直接用于服务化。

### 20.4 vLLM 参数报错与修复

初始命令包含：

```bash
--max-seq-len-to-capture 40960
```

在 `vllm 0.15.1` 报错：

```text
vllm: error: unrecognized arguments: --max-seq-len-to-capture 40960
```

修复方式：删除该参数，改用 `--max-model-len 40960`。

### 20.5 最终可用启动命令（通过）

```bash
rm -rf ~/.cache/
rm -rf ~/.triton/

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=0
export VLLM_RANK3_NUMA=0
export VLLM_SPEC_DECODE_EAGER=1
export VLLM_ZERO_OVERHEAD=1

model_path=/data0/models/Qwen3-VL-4B-Instruct
tp=4
port=8089
logpath=./logs-server
mkdir -p "${logpath}"

vllm serve "$model_path" \
  --host 0.0.0.0 \
  --port "$port" \
  --dtype float16 \
  --tensor-parallel-size "$tp" \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --max-model-len 40960 \
  --distributed-executor-backend mp \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":4}' \
  2>&1 | tee "${logpath}/vllm.log"
```

### 20.6 成功标志（日志关键行）

日志中出现以下关键信息，表明服务已正常拉起：

- `version 0.15.1`
- `Resolved architecture: Qwen3VLForConditionalGeneration`
- `Using max model len 40960`
- `init engine ... took 24.97 seconds`
- `Starting vLLM API server 0 on http://0.0.0.0:8089`
- `Application startup complete.`

### 20.7 说明与注意事项

- 日志中的 `trust_remote_code ... is ignored` 为提示信息，不影响服务启动。
- `rope_parameters` 的 `Unrecognized keys`、ROCm GELU fallback 警告在本次实测中不阻断启动。
- 若后续仅做 OpenAI 兼容调用，核心可用接口为 `/v1/chat/completions`、`/v1/models`、`/health`。

## 21. vLLM 随机数据压测（最终方案）

本节记录在 `vllm serve` 正常拉起后，最终采用 **`random` 数据集**进行 `vllm bench serve` 压测的实操过程（来源：`qwen3_vl_4b_vllm/vllm.log`）。

### 21.1 压测思路

- 使用 `dataset_name=random`，避免外部数据文件格式差异（`json/jsonl`）对压测链路的干扰；
- 固定随机输入输出长度（日志中为 `input=1024`, `output=128`）；
- 循环压测并发：`1, 16, 32, 64, 128, 256`；
- 每轮将原始输出保存到单独日志，并抽取关键指标追加到 `all.csv`。

在压测前，建议先做最小服务可用性检查：

```bash
curl -s http://127.0.0.1:8089/health
curl -s http://127.0.0.1:8089/v1/models
```

### 21.2 实际命令（节选）

```bash
for batch in 1 16 32 64 128 256; do
    concurrency_multiplier=8
    if [ $batch -gt 128 ]; then
        concurrency_multiplier=4
    fi

    vllm bench serve \
      --model ${model_path} \
      --dataset-name random \
      --num-prompts $((batch*concurrency_multiplier)) \
      --port $port \
      --metric-percentiles 95,99 \
      --max-concurrency $batch \
      --ignore-eos \
      2>&1 | tee ${logpath}/${model}-tp${tp}-${time}-${batch}-in${prompt_tokens}-out${completion_tokens}.log
done
```

日志中可见 `Namespace(...)` 关键参数：

- `dataset_name='random'`
- `random_input_len=1024`
- `random_output_len=128`
- `port=8089`
- `model='/data0/models/Qwen3-VL-4B-Instruct'`

### 21.3 压测结果汇总（随机数据）

从日志提取到的核心结果如下（均为成功请求，无失败）：

- 并发 `1`：`rps=0.59`，输出吞吐 `75.94 tok/s`，`P99 TTFT=46.71 ms`，`P99 TPOT=13.00 ms`
- 并发 `16`：`rps=7.26`，输出吞吐 `929.66 tok/s`，`P99 TTFT=328.14 ms`，`P99 TPOT=17.13 ms`
- 并发 `32`：`rps=13.47`，输出吞吐 `1724.00 tok/s`，`P99 TTFT=603.09 ms`，`P99 TPOT=20.15 ms`
- 并发 `64`：`rps=21.46`，输出吞吐 `2746.25 tok/s`，`P99 TTFT=1163.67 ms`，`P99 TPOT=26.52 ms`
- 并发 `128`：`rps=28.29`，输出吞吐 `3620.97 tok/s`，`P99 TTFT=2236.25 ms`，`P99 TPOT=42.65 ms`
- 并发 `256`：`rps=51.75`，输出吞吐 `6623.91 tok/s`，`P99 TTFT=542.73 ms`，`P99 TPOT=37.69 ms`

### 21.3.1 并发 256 指标解读（可用于汇报）

在并发 `256` 条件下：

- `rps=51.75`：每秒可完成约 `51.75` 个请求，具备高并发承载能力；
- 输出吞吐 `6623.91 tok/s`：整体生成吞吐较高，适合吞吐优先场景；
- `P99 TTFT=542.73 ms`：99 分位首 token 延迟保持在亚秒级，在线体验可接受；
- `P99 TPOT=37.69 ms`：高并发下单 token 生成延迟约 `37.69 ms`，折算单流约 `26.5 tok/s`。

结论（当前测试配置下）：该结果整体表现为**高吞吐 + 可接受时延**，可作为单机 8 卡 Qwen3-VL-4B 服务化的有效性能基线。

### 21.4 结果解读

- 该机型在本次参数组合下，随着并发提升，整体吞吐（`rps`、`tok/s`）明显上升；
- 高并发下 TTFT/TPOT 出现波动，属于压测中常见现象，需结合目标 SLA 选择业务并发上限；
- 对“服务可用性”而言，本次 `1~256` 并发段均完成请求，无失败，说明服务链路稳定可用。



