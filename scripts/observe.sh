#!/bin/bash

LOG_DIR="/path/to/data/log"
mkdir -p "${LOG_DIR}"

JOB_ID="${JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
exec > >(tee -a "${LOG_DIR}/observe/observe-${JOB_ID}.out") 2>&1

echo loss_batch_size: ${loss_batch_size}
echo loss_sample_num: ${loss_sample_num}
echo negative_corpus_size: ${negative_corpus_size}
echo model_name_or_path: ${model_name_or_path}
echo query_path: ${query_path}
echo qrel_path: ${qrel_path}
echo output_path: ${output_path}

per_device_eval_batch_size="128"
corpus_path="/path/to/collection.tsv"

cd /path/to/src


STATE_FILE="${LOG_DIR}/state/state_${JOB_ID}"
/usr/bin/touch ${STATE_FILE}
# 后台循环采集，每间隔 1s 采集一次 GPU 数据。 # 采集的数据将输出到本地 gpu_作业 ID.log 文件中 
function gpus_collection(){
    while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do 
        /usr/bin/sleep 1
        /usr/bin/nvidia-smi >> "${LOG_DIR}/gpu/gpu_${JOB_ID}.log" 
    done
}
gpus_collection &
GPU_COLLECT_PID=$!
cleanup() {
  echo "over" >> "${STATE_FILE}" 2>/dev/null || true
  kill "${GPU_COLLECT_PID}" 2>/dev/null || true
}
trap cleanup EXIT

python -m src.evaluate.hard \
--qrel_path $qrel_path \
--query_path $query_path \
--corpus_path $corpus_path \
--hard_neg_dir /path/to/data/hard_negtives/docid/dev \
--output_path $output_path \
--model_name_or_path $model_name_or_path \
--per_device_eval_batch_size $per_device_eval_batch_size \
--loss_batch_size $loss_batch_size \
--fp16 \
--dataloader_num_workers 0

echo "over" >> "${STATE_FILE}"