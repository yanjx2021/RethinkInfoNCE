#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="7"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

models=(
  "google_bert/uncased_L-24_H-1024_A-16"
  "google_bert/uncased_L-2_H-768_A-12" 
  "google_bert/uncased_L-4_H-768_A-12" 
  "google_bert/uncased_L-6_H-768_A-12" 
  "google_bert/uncased_L-8_H-768_A-12" 
  "google_bert/uncased_L-10_H-768_A-12" 
  "google_bert/uncased_L-12_H-768_A-12" 
  "google_bert/uncased_L-2_H-512_A-8" 
  "google_bert/uncased_L-4_H-512_A-8" 
  "google_bert/uncased_L-6_H-512_A-8" 
  "google_bert/uncased_L-8_H-512_A-8" 
  "google_bert/uncased_L-10_H-512_A-8"
  "google_bert/uncased_L-12_H-512_A-8" 
  "google_bert/uncased_L-2_H-256_A-4" 
  "google_bert/uncased_L-4_H-256_A-4" 
  "google_bert/uncased_L-6_H-256_A-4" 
  "google_bert/uncased_L-8_H-256_A-4" 
  "google_bert/uncased_L-10_H-256_A-4" 
  "google_bert/uncased_L-12_H-256_A-4" 
  "google_bert/uncased_L-2_H-128_A-2" 
  "google_bert/uncased_L-4_H-128_A-2" 
  "google_bert/uncased_L-6_H-128_A-2" 
  "google_bert/uncased_L-8_H-128_A-2" 
  "google_bert/uncased_L-10_H-128_A-2" 
  "google_bert/uncased_L-12_H-128_A-2"   
)

train_batch_sizes=(16)
subset="502939" # max 502939
checkpoint=$(( (subset / 32) * 3 )) # 32 -> train batch size, 3 -> epoch number
echo "subset=$subset checkpoint=$checkpoint"

# ===== fixed config =====
lr="3e-6"
device_batch_size="16"    
query_type="dev"
query_path="/path/to/msmarco-passage/queries.${query_type}.tsv"
qrel_path="/path/to/msmarco-passage/qrels.${query_type}.tsv"



for base_model in "${models[@]}"; do
  hf_model="${base_model}"
  echo "===== Model: ${hf_model} ====="

  for train_batch_size in "${train_batch_sizes[@]}"; do
    model_name_or_path="/path/to/data/hard_negtives/${hf_model}/gpu1_neg${train_batch_size}.${device_batch_size}_bs32_lr${lr}/subset${subset}/checkpoint-${checkpoint}"
    echo "Train BS: ${train_batch_size}, Device BS: ${device_batch_size}"
    echo "CKPT dir: ${model_name_or_path}"

    for loss_batch_size in 256; do
      output_path="${model_name_or_path}/observe/lbs${loss_batch_size}_ncs100000_lsn1/result.msmarco-passage.${query_type}.npz"

      export loss_batch_size lr query_path qrel_path output_path model_name_or_path
      bash ./observe.sh
    done
  done
done
