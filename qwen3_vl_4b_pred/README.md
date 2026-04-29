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

