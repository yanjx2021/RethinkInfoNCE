import os
import logging
import gc
from dataclasses import field, dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import pytrec_eval
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import is_main_process

from ..modeling import AutoDenseModel
from .index_utils import (
    encode_dense_query,
    load_corpus,
    load_queries,
    get_collator_func,
    TextDataset,
    DenseEvaluater,
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    output_path: str = field()
    corpus_path: str = field()
    query_path: str = field()
    qrel_path: str = field()


@dataclass
class ModelArguments:
    model_name_or_path: str = field()


@dataclass
class EvalArguments(TrainingArguments):
    # hard mode only
    loss_batch_size: int = field(default=256)  # per-query sample this many hard negatives (if pool allows)
    hard_neg_dir: str = field(
        default=None,
        metadata={"help": "Directory containing qids.npy and topdocids.npy (required)."},
    )


def _get_device(model) -> torch.device:
    return next(model.parameters()).device


def cuda_stat(tag=""):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    m = torch.cuda.max_memory_allocated() / 1024**2
    logger.info(f"[CUDA]{tag} allocated={a:.0f}MiB reserved={r:.0f}MiB max_alloc={m:.0f}MiB")


def _norm_key(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="ignore")
    return str(x).strip()


def encode_corpus(corpus: List[str], model, tokenizer, eval_args: TrainingArguments):
    doc_dataset = TextDataset(corpus)
    doc_out = DenseEvaluater(
        model=model,
        args=eval_args,
        data_collator=get_collator_func(tokenizer, 128),
        tokenizer=tokenizer,
    ).predict(doc_dataset)
    doc_embeds = doc_out.predictions
    assert len(doc_embeds) == len(corpus)
    return doc_embeds


def decompose_result(rank_list, ce_loss_list):
    # kept for compatibility if you reuse it elsewhere; not used here
    raise NotImplementedError


def online_observe_hard(
    model,
    tokenizer,
    eval_args,
    query_ids,
    query_embeds,
    relevant_corpus,
    *,
    hard_qids: np.ndarray,          # (num_q,)  query offsets (line numbers)
    hard_topdocids: np.ndarray,     # (num_q, topk)
    corpus: Dict[Any, str],         # docid -> text
    qid2qoff: Dict[str, int],       # true qid(str) -> query offset(int)
):
    """
    Hard negatives only (NO GLOBAL POOL).
    - hard_qids.npy stores query offsets (line numbers), NOT true qids.
      So align by: true qid -> qoff -> hard_row.

    Procedure:
      1) per-query randomly pick `loss_batch_size` docids from its hard list (once)
      2) per-query encode its chosen negatives (no global negative pool)
      3) per-query compute CE loss with (1 pos + n_eff negs)
      4) record rank = #(neg_score > pos_score)
    """

    device = _get_device(model)
    n = int(eval_args.loss_batch_size)

    # encode positives (aligned with query_ids)
    cuda_stat(" before positives")
    relevant_doc_embeds = encode_corpus(relevant_corpus, model, tokenizer, eval_args)
    gc.collect()
    torch.cuda.empty_cache()
    cuda_stat(" after positives")

    # to torch
    query_embeds = torch.tensor(query_embeds, dtype=torch.float32, device=device)                 # (Q,D)
    relevant_doc_embeds = torch.tensor(relevant_doc_embeds, dtype=torch.float32, device=device)  # (Q,D)

    # offset -> hard_row (aligned arrays)
    qoff2hardrow = {int(qoff): i for i, qoff in enumerate(hard_qids)}

    def _get_doc_text(docid) -> Optional[str]:
        if docid in corpus:
            return corpus[docid]
        s = _norm_key(docid)
        if s in corpus:
            return corpus[s]
        return None

    # 1) per-query sample negatives once (docids, not texts)
    sampled_doc_keys_per_query: Dict[str, np.ndarray] = {}

    num_missing_qid = 0
    num_missing_qoff = 0
    num_zero_pool = 0
    num_small_pool = 0

    for qid in query_ids:
        qkey = _norm_key(qid)

        qoff = qid2qoff.get(qkey, None)
        if qoff is None:
            num_missing_qid += 1
            continue

        row = qoff2hardrow.get(int(qoff), None)
        if row is None:
            num_missing_qoff += 1
            continue

        docids_q = hard_topdocids[row]  # (topk,)
        uniq_keys = np.unique([_norm_key(d) for d in docids_q])

        if len(uniq_keys) == 0:
            num_zero_pool += 1
            continue

        if len(uniq_keys) < n:
            num_small_pool += 1
            chosen = uniq_keys
        else:
            chosen = np.random.choice(uniq_keys, size=n, replace=False)

        sampled_doc_keys_per_query[qkey] = chosen

    if len(sampled_doc_keys_per_query) == 0:
        raise ValueError("[hard] No queries have sampled negatives (check qid->offset mapping / hard files).")

    logger.info(
        f"[hard] Sampled negatives per query once. "
        f"missing_qid_in_qid2qoff={num_missing_qid}, missing_qoff_in_hard_qids={num_missing_qoff}, "
        f"zero_pool={num_zero_pool}, pool_smaller_than_batch={num_small_pool}"
    )

    doc_evaluator = DenseEvaluater(
        model=model,
        args=eval_args,
        data_collator=get_collator_func(tokenizer, 128),
        tokenizer=tokenizer,
    )
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    qids, all_loss, all_rank = [], [], []
    skipped_all_missing_text = 0
    skipped_no_valid_neg = 0
    total_missing_text = 0

    cuda_stat(" before compute")
    for qid, q_embed, rd_embed in tqdm(
        zip(query_ids, query_embeds, relevant_doc_embeds),
        total=len(query_ids),
        desc="Online observe (hard)",
    ):
        qkey = _norm_key(qid)
        chosen_keys = sampled_doc_keys_per_query.get(qkey, None)
        if chosen_keys is None:
            continue

        neg_texts: List[str] = []
        missing_this_q = 0
        for k in chosen_keys:
            txt = _get_doc_text(k)
            if txt is None:
                missing_this_q += 1
                continue
            neg_texts.append(txt)

        total_missing_text += missing_this_q

        if len(neg_texts) == 0:
            skipped_all_missing_text += 1
            continue

        # encode only this query's negatives
        neg_out = doc_evaluator.predict(TextDataset(neg_texts))
        neg_embeds = torch.tensor(neg_out.predictions, dtype=torch.float32, device=device)  # (n_eff, D)
        n_eff = int(neg_embeds.size(0))
        if n_eff == 0:
            skipped_no_valid_neg += 1
            continue

        q_embed_ = q_embed.unsqueeze(0)    # (1,D)
        rd_embed_ = rd_embed.unsqueeze(0)  # (1,D)

        positive_score = torch.matmul(q_embed_, rd_embed_.t()).squeeze()        # ()
        negative_scores = torch.matmul(q_embed_, neg_embeds.t()).squeeze(0)     # (n_eff,)

        cat_scores = torch.cat(
            [positive_score.view(1, 1), negative_scores.view(1, n_eff)],
            dim=1
        )  # (1, 1+n_eff)

        labels = torch.zeros(1, dtype=torch.long, device=device)  # (1,)
        loss = loss_fct(cat_scores, labels)                       # (1,)
        rank = (positive_score < negative_scores).sum().view(1)   # (1,)

        qids.append(qid)
        all_loss.append(loss.detach().cpu().numpy())
        all_rank.append(rank.detach().cpu().numpy())

        # free peak GPU mem sooner
        del neg_embeds, negative_scores, cat_scores, loss, rank

    logger.info(
        f"[hard] finished. skipped_all_missing_text={skipped_all_missing_text}, "
        f"skipped_no_valid_neg={skipped_no_valid_neg}, total_missing_text_in_corpus={total_missing_text}"
    )

    return {"qids": np.array(qids), "loss": np.array(all_loss), "rank": np.array(all_rank)}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if eval_args.should_log else logging.WARN,
    )
    assert data_args.output_path is not None, "Please specify output path."

    out_dir = os.path.dirname(data_args.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    assert not os.path.exists(data_args.output_path), "Output file already exists."
    assert eval_args.world_size == 1, "Only support single GPU evaluation for online observe."

    if eval_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
        logging.getLogger("transformers.trainer_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.trainer_callback").setLevel(logging.ERROR)
        eval_args.disable_tqdm = True

    set_seed(2022)

    # hard dir required
    assert eval_args.hard_neg_dir is not None, "Please set --hard_neg_dir"
    qids_path = os.path.join(eval_args.hard_neg_dir, "qids.npy")
    topdocids_path = os.path.join(eval_args.hard_neg_dir, "topdocids.npy")
    if not (os.path.exists(qids_path) and os.path.exists(topdocids_path)):
        raise ValueError(f"hard_neg_dir={eval_args.hard_neg_dir} missing qids.npy/topdocids.npy")

    cuda_stat("before model")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoDenseModel.from_pretrained(model_args.model_name_or_path)
    cuda_stat("init model")

    with open(data_args.qrel_path, "r") as f:
        qrels = pytrec_eval.parse_qrel(f)

    corpus = load_corpus(data_args.corpus_path, verbose=is_main_process(eval_args.local_rank))

    # ---- queries + qid->offset mapping ----
    raw_queries = load_queries(data_args.query_path)
    logger.info(f"Total #queries: {len(raw_queries)}")
    qid2qoff = {_norm_key(qid): i for i, qid in enumerate(raw_queries.keys())}

    # filter to those with qrels
    queries = {qid: text for qid, text in raw_queries.items() if _norm_key(qid) in qrels}
    logger.info(f"Filtered #queries with relevance labels: {len(queries)}")

    query_embeds, query_ids = encode_dense_query(queries, model, tokenizer, eval_args)
    gc.collect()
    torch.cuda.empty_cache()
    cuda_stat("after query encode")

    # pick one relevant doc per query (aligned)
    relevant_docids = [sorted(qrels[_norm_key(qid)].items())[0][0] for qid in query_ids]
    relevant_corpus = [corpus[docid] for docid in relevant_docids]

    hard_qids = np.load(qids_path, mmap_mode="r")
    hard_topdocids = np.load(topdocids_path, mmap_mode="r")
    logger.info(f"Loaded hard negatives: hard_qids={hard_qids.shape}, hard_topdocids={hard_topdocids.shape}")

    results = online_observe_hard(
        model=model,
        tokenizer=tokenizer,
        eval_args=eval_args,
        query_ids=query_ids,
        query_embeds=query_embeds,
        relevant_corpus=relevant_corpus,
        hard_qids=hard_qids,
        hard_topdocids=hard_topdocids,
        corpus=corpus,
        qid2qoff=qid2qoff,
    )
    np.savez_compressed(data_args.output_path, **results)


if __name__ == "__main__":
    main()
