#!/bin/bash

LOG_DIR="/path/to/data/log"
mkdir -p "${LOG_DIR}"

JOB_ID="${JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
exec > >(tee -a "${LOG_DIR}/train/train-${JOB_ID}.out") 2>&1

echo per_query_neg: ${per_query_neg}
echo per_device_neg: ${per_device_neg}
echo per_device_train_batch_size: ${per_device_train_batch_size}
echo lr: ${lr}
echo gpus: ${gpus}
echo model_name: ${model_name}
echo epoch: ${epoch}
echo subset: ${subset}
echo output_dir: ${output_dir}
echo model_name_or_path: ${model_name_or_path}

total_neg=$(( per_device_neg * gpus ))
echo "Total neg: $total_neg"

if (( total_neg % per_query_neg != 0 )); then
    echo "Error: per_device_neg * gpus must be a multiple of per_query_neg"
    exit 1
fi

echo output_dir: ${output_dir}

cd /path/to/src

STATE_FILE="${LOG_DIR}/state/state_${JOB_ID}"
/usr/bin/touch ${STATE_FILE}

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

python -m src.finetune.hard_contrast_run \
--pooling average \
--similarity_metric ip \
--qrel_path /path/to/msmarco-passage/qrels.train.tsv \
--query_path /path/to/msmarco-passage/queries.train.tsv \
--corpus_path /path/to/collection.tsv \
--hard_neg_dir /path/to/data/hard_negtives/docid \
--output_dir $output_dir \
--model_name_or_path $model_name_or_path \
--logging_steps 100 \
--output_dim 1024 \
--max_query_len 24 \
--max_doc_len 128 \
--per_device_train_batch_size $per_device_train_batch_size \
--inv_temperature 1 \
--gradient_accumulation_steps 1 \
--fp16 \
--per_query_neg $per_query_neg \
--per_device_neg $per_device_neg \
--learning_rate $lr \
--num_train_epochs $epoch \
--dataloader_drop_last \
--overwrite_output_dir \
--dataloader_num_workers 0 \
--weight_decay 0 \
--lr_scheduler_type "constant" \
--save_strategy "epoch" \
--gradient_checkpointing \
--report_to tensorboard \
--train_subset_num $subset 

echo "over" >> "${STATE_FILE}"
