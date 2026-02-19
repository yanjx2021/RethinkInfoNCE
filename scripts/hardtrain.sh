#!/bin/bash
export CUDA_VISIBLE_DEVICES="5"
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

for model_name in "${models[@]}"
do
    echo "Training model: ${model_name}"

    gpus="1"
    per_query_neg="16"
    per_device_neg="16"
    per_device_train_batch_size="32"
    lr="3e-6"
    epoch="3"
    subset="502939" # max 502939
    output_dir="/path/to/data/hard_negtives/${model_name}/gpu${gpus}_neg${per_query_neg}.${per_device_neg}_bs${per_device_train_batch_size}_lr${lr}/subset${subset}"

    model_name_or_path="/path/to/model/${model_name}"

    export per_query_neg per_device_neg per_device_train_batch_size lr gpus model_name epoch subset output_dir model_name_or_path 

    bash ./train.sh
done