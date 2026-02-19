import os
import sys
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed, AutoConfig)
from transformers.trainer_utils import is_main_process

from ..modeling import (
    AutoDenseModel, 
    SIMILARITY_METRICS,
    POOLING_METHODS
)
from .hard_contrast_utils import (
    ContrastDenseFinetuner,
    QDRelDataset, FinetuneCollator
)
from .validate_utils import load_validation_set


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    qrel_path: str = field()
    query_path: str = field()
    corpus_path: str = field()  
    max_query_len: int = field()
    max_doc_len: int = field()  
    valid_corpus_path : str = field(default=None)
    valid_query_path : str = field(default=None)
    valid_qrel_path : str = field(default=None)
    hard_neg_dir: str = field(default=None) 
    train_subset_num: Optional[int] = field(default=None)
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    output_dim: int = field()
    pooling: str = field(metadata={"choices": POOLING_METHODS})
    similarity_metric: str = field(metadata={"choices": SIMILARITY_METRICS})


@dataclass
class DenseFinetuneArguments(TrainingArguments):
    inv_temperature: float = field(default=1)
    per_query_neg: int = field(default=1)
    per_device_neg: int = field(default=1)
    seed: int = field(default=42)

    remove_unused_columns: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        assert (self.per_device_neg * self.world_size) % self.per_query_neg == 0, \
            "per_device_neg * world_size must be multiple of per_query_neg"
        assert (self.per_device_neg * self.world_size) >= self.per_query_neg and self.per_query_neg > 0, \
            "per_device_neg * world_size must be larger than per_query_neg, and per_query_neg must be positive"

from transformers import TrainerCallback
import os

class SaveSnapshotsCallback(TrainerCallback):
    def __init__(self, steps, tokenizer):
        self.steps = set(steps)
        self.tokenizer = tokenizer

    def _save(self, args, state, model):
        out_dir = os.path.join(args.output_dir, "snapshots", f"step-{state.global_step}")
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(out_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and state.global_step in self.steps:
            self._save(args, state, kwargs["model"])
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._save(args, state, kwargs["model"])
        return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments, DenseFinetuneArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.should_log else logging.WARN,
    )
    
    resume_from_checkpoint = False
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and any((x.startswith("checkpoint") for x in os.listdir(training_args.output_dir)))
    ):
        if not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        else:
            resume_from_checkpoint = True

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: DenseFinetuneArguments

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"World size: {training_args.world_size}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.similarity_metric = model_args.similarity_metric
    config.pooling = model_args.pooling    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        config = config
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    model = AutoDenseModel.from_pretrained(model_args.model_name_or_path, config=config, output_dim=model_args.output_dim)

    train_set = QDRelDataset(tokenizer, 
            qrel_path = data_args.qrel_path, 
            query_path = data_args.query_path, 
            corpus_path = data_args.corpus_path, 
            max_query_len = data_args.max_query_len, 
            max_doc_len = data_args.max_doc_len, 
            rel_threshold = 1, 
            verbose=training_args.should_log,
            hard_neg_dir=data_args.hard_neg_dir, 
            train_subset_num=data_args.train_subset_num,)
    hard = train_set.get_hard_negatives()
    hard_qids, hard_topdocids = (hard if hard is not None else (None, None))
    # Data collator
    data_collator = FinetuneCollator(
        tokenizer = tokenizer,
        max_query_len = data_args.max_query_len, 
        max_doc_len = data_args.max_doc_len,
        corpus = train_set.corpus,
        qrels = train_set.get_qrels(),
        per_device_neg = training_args.per_device_neg,
        local_rank = training_args.local_rank,
        world_size = training_args.world_size,      
        hard_topdocids=hard_topdocids,       
    )
    if data_args.valid_corpus_path is None:
        eval_dataset = None
        assert data_args.valid_query_path is None and data_args.valid_qrel_path is None
    else:
        eval_dataset=load_validation_set(
            data_args.valid_corpus_path,
            data_args.valid_query_path,
            data_args.valid_qrel_path,
        )
    
    # Initialize our Trainer
    trainer = ContrastDenseFinetuner(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=eval_dataset
    )
    # snap_steps = [0, 1, 10, 50, 100, 500, 1000, 5000, 10000, 15716] 
    # trainer.add_callback(SaveSnapshotsCallback(snap_steps, tokenizer))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

    if training_args.distributed_state is not None:
        training_args.distributed_state.destroy_process_group()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
