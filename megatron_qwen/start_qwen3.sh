#! /bin/bash

####################################################
#
# usage:
#      bash start_qwen.sh <model_size> <master_addr> <node_num> <rank>
#
# supported model size:
#       qwen3: {8, 32}
#       qwen2.5: {72}
#
####################################################

# env var
export PYTHONPATH=./Megatron-LM:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10
export NVTE_DISABLE_FC2_DGRAD_OVERLAP=1
export NVTE_NO_PIPELINE_OVERLAP=1

export cache_size_limit=64

export NCCL_DEBUG=Info
export NCCL_ALGO=Ring
export NCCL_RINGS="N0 0 7 6 5 4 3 2 1 N0|N1 1 2 3 4 5 6 7 0 N1|N2 2 1 0 7 6 5 4 3 N2|N3 3 4 5 6 7 0 1 2 N3|N4 4 3 2 1 0 7 6 5 N4|N5 5 6 7 0 1 2 3 4 N5|N6 6 5 4 3 2 1 0 7 N6|N7 7 0 1 2 3 4 5 6 N7"

export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export RCCL_SDMA_COPY_ENABLE=0

MODEL_SIZE=${1:-8B}

# data settings
CHECKPOINT_PATH="./output"
DATA_PATH="/cfs/datasets/redpajama-qwen/redpajama-sample_text_document"
TOKENIZER_PATH="/cfs/datasets/redpajama-qwen/tokenizer"
TOKENIZER_ARGS="--tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOKENIZER_PATH"
SEQ_LEN=32768

# model settings
if [ $MODEL_SIZE == "8B" ]; then
        MODEL_NAME="qwen3-8B"
        NUM_LAYERS=36
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=12288
        NUM_ATTN_HEADS=32
        MICRO_BATCH_SIZE=1
        TP=4
        PP=1
        CP=2
        MICRO_BATCH_NUM=128
        NUM_KV_HEADS=8
        GQA_ARGS=" --group-query-attention --num-query-groups $NUM_KV_HEADS "
        OPTIMIZER_OFFLOAD_ARGS=" \
                --optimizer-cpu-offload \
                --use-torch-optimizer-for-cpu-offload \
                --use-precision-aware-optimizer \
                --main-grads-dtype bf16 \
                --main-params-dtype fp16 \
                "
elif [ $MODEL_SIZE == "32B" ]; then
        MODEL_NAME="qwen3-32B"
        NUM_LAYERS=64
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=25600
        NUM_ATTN_HEADS=64
        MICRO_BATCH_SIZE=1
        TP=8
        PP=1
        CP=2
        MICRO_BATCH_NUM=64
        NUM_KV_HEADS=8
        GQA_ARGS=" --group-query-attention --num-query-groups $NUM_KV_HEADS "
        OPTIMIZER_OFFLOAD_ARGS=" \
                --optimizer-cpu-offload \
                --use-torch-optimizer-for-cpu-offload \
                --use-precision-aware-optimizer \
                --main-grads-dtype bf16 \
                --main-params-dtype fp16 \
                "
elif [ $MODEL_SIZE == "72B" ]; then
        MODEL_NAME="qwen2.5-72B"
        NUM_LAYERS=80
        HIDDEN_SIZE=8192
        FFN_HIDDEN_SIZE=29568
        NUM_ATTN_HEADS=64
        MICRO_BATCH_SIZE=1
        TP=8
        PP=2
        CP=4
        MICRO_BATCH_NUM=128
        NUM_KV_HEADS=8
        GQA_ARGS=" --group-query-attention --num-query-groups $NUM_KV_HEADS "
        OPTIMIZER_OFFLOAD_ARGS=" \
                --optimizer-cpu-offload \
                --use-torch-optimizer-for-cpu-offload \
                --use-precision-aware-optimizer \
                --main-grads-dtype bf16 \
                --main-params-dtype fp16 \
                "
elif [ $MODEL_SIZE == "A3B" ]; then
        MODEL_NAME="qwen3-30B-A3B"
        NUM_LAYERS=48
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=6144
        NUM_ATTN_HEADS=32
        MICRO_BATCH_SIZE=1
        TP=2
        PP=1
        CP=4
        EP=8
        ETP=1
        MICRO_BATCH_NUM=128
        NUM_KV_HEADS=4
        GQA_ARGS=" --group-query-attention --num-query-groups $NUM_KV_HEADS"

        ROUTER_TOPK=8
        NUM_EXPERTS=128
        MOE_INTERMEDIATE_SIZE=768
        MOE_ARGS=" \
                --moe-grouped-gemm \
                --moe-token-dispatcher-type alltoall \
                --moe-router-topk ${ROUTER_TOPK} \
                --num-experts ${NUM_EXPERTS} \
                --expert-tensor-parallel-size ${ETP} \
                --expert-model-parallel-size ${EP} \
                --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
                --moe-router-load-balancing-type aux_loss \
                --moe-aux-loss-coeff 0.001 \
                --moe-layer-freq '([1]*48)' \
                "
        OPTIMIZER_OFFLOAD_ARGS=" \
                --optimizer-cpu-offload \
                --use-torch-optimizer-for-cpu-offload \
                --use-precision-aware-optimizer \
                --main-grads-dtype bf16 \
                --main-params-dtype fp16 \
                "
else
        echo "ERROR: Please supplement new model configuration to test!"
        exit -1
fi

#fp8 settings
ENABLE_FP8=false
if [ $ENABLE_FP8 == "true" ]; then
        FP8_ARGS="--transformer-impl transformer_engine \
                --fp8-format hybrid \
                --fp8-amax-compute-algo max \
                --fp8-amax-history-len 1024 \
                "
        DT="fp8"
else
        FP8_ARGS="--transformer-impl transformer_engine"
        DT="bf16"
fi

# node settings
MASTER_ADDR=${2:-localhost}
MASTER_PORT=6000
NNODES=${3:-1}
NODE_RANK=${4:-0}
GPUS_PER_NODE=8
WORLD_SIZE=$(( $GPUS_PER_NODE * $NNODES ))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DP=$(( $WORLD_SIZE / $TP / $PP / $CP ))
GLOBAL_BATCH_SIZE=$(( $DP * $MICRO_BATCH_SIZE * $MICRO_BATCH_NUM ))

        #--tp-comm-overlap \
CMD="torchrun $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --use-mcore-models \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --context-parallel-size $CP \
        --sequence-parallel \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        $GQA_ARGS \
        --hidden-dropout 0.0 \
        --attention-dropout 0 \
        --swiglu \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --normalization RMSNorm \
        --qk-layernorm \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base 1000000 \
        --train-iters 1000 \
        --lr-decay-iters 3200 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TOKENIZER_ARGS \
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
        $OPTIMIZER_OFFLOAD_ARGS \
        --bf16 \
        $FP8_ARGS \
        "

echo ${CMD} 2>&1 | tee bw1000_megatron_${MODEL_NAME}_seq${SEQ_LEN}_tp${TP}_pp${PP}_cp${CP}_dp${DP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${DT}.log
eval ${CMD} 2>&1 | tee -a bw1000_megatron_${MODEL_NAME}_seq${SEQ_LEN}_tp${TP}_pp${PP}_cp${CP}_dp${DP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${DT}.log
