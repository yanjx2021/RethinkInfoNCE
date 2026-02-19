# A Rank-oriented and Scalable Metric for Scaling Ranking Models
> Anonymous authors — SIGIR Short Paper 2026 submission  
> **Note:** This repository is anonymized for double-blind review. No author or affiliation information is included.
> 
## Overview

This repository contains the official implementation of the paper **"A Rank-oriented and Scalable Metric for Scaling Ranking Models"**. 

## Repository Structure

```
.
├── README.md                           # Project documentation
├── data/
│   └── prepare_hard.py                 # Hard negative mining pipeline
├── scripts/
│   ├── train.sh                    
│   ├── hardtrain.sh                    # Hard negative training script
│   ├── observe.sh                   
│   └── hardobserve.sh                  # Hard negative evaluation script
├── src/
│   ├── finetune/                       # Fine-tuning and training modules
│   │   ├── hard_contrast_run.py        # Main hard contrastive learning trainer
│   │   ├── hard_contrast_utils.py      
│   │   └── validate_utils.py           
│   ├── evaluate/                       # Evaluation modules
│   │   ├── hard_eval.py                # Hard negative evaluation
│   │   └── index_utils.py              
│   ├── modeling/                       # Model architecture
│   │   ├── bert_dense.py               # Dense BERT retriever model
│   │   └── utils.py                    
│   └── evaluate.py                     # Evaluation metrics 
└── analysis/                           # Paper analysis and table generation
    ├── computescale.py             
    ├── datascale.py                
    ├── modelscale.py         
    ├── mrr.py              
    ├── share.py             
    └── trainprocess.py     
```


## Quick Start

### 1. Data Preparation

Prepare hard negatives from the MS MARCO dataset:

```bash
python data/prepare_hard.py \
    --model_name /path/to/dense/model \
    --query_path /path/to/queries.tsv \
    --corpus_path /path/to/collection.tsv \
    --qrel_path /path/to/qrels.tsv \
    --topk 1000 \
    --output_dir ./hard_negatives \
    --device cuda \
    --doc_batch_size 256 \
    --query_batch_size 256
```

### 2. Model Training

Train Dense BERT Retrievers with hard negatives:

```bash
# Configure training parameters
export per_query_neg=16           # Number of negatives per query
export per_device_neg=16          # Number of negatives per device
export per_device_train_batch_size=32
export lr=3e-6
export gpus=1
export epoch=3
export subset=502939              # Training set size

# Run training
bash scripts/hardtrain.sh
```

### 3. Model Evaluation

Evaluate trained models on retrieval tasks:

```bash
bash scripts/hardobserve.sh
```

### 4. Analysis

Comprehensive analysis modules for paper results:
- **modelscale.py**: Study model performance across different BERT sizes
- **datascale.py**: Analyze impact of training data scale
- **computescale.py**: Compute computational cost and efficiency
- **mrr.py**: Detailed MRR metric analysis
- **trainprocess.py**: Monitor and analyze training dynamics
- **share.py**: Common utilities and helper functions

## Dependencies

### Required Packages

```
transformers>=4.10
torch>=1.9
numpy
tqdm
pytrec_eval
faiss-gpu
```

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{
    title={A Rank-oriented and Scalable Metric for Scaling Ranking Models},
    year={2026}
}
```

## Important Note on Paths

**Please note:** All paths in this repository use the placeholder `/path/to` (e.g., `/path/to/dense/model`, `/path/to/data`, etc.). You need to replace these placeholders with the actual paths on your system according to your directory structure and data locations. This applies to:
- Model paths for pre-trained BERT models
- Data paths for MS MARCO dataset files (queries, corpus, qrels)
- Output directories for hard negatives and trained models
- Log directories for training logs
