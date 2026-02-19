import os
import numpy as np
import torch
import random
import logging
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, Trainer

from .validate_utils import validate_during_training

logger = logging.getLogger(__name__)


@dataclass
class FinetuneCollator:
    def __init__(
            self, 
        tokenizer: PreTrainedTokenizer, 
        corpus: List[str],
        qrels: Dict[int, List[int]],
        per_device_neg: int,
        max_query_len: int, 
        max_doc_len: int, 
        local_rank: int,
        world_size: int,
        padding=True,      
        hard_topdocids=None,       
    ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.padding = padding
        self.corpus = corpus
        self.qrels = qrels
        self.neg_docids = set(list(range(len(corpus)))) - set(
            docid for docids in qrels.values() for docid in docids
        )
        self.neg_docids = sorted(self.neg_docids)
        self.neg_docids = self.neg_docids[local_rank::world_size]
        self.per_device_neg = per_device_neg
        self.hard_topdocids = hard_topdocids


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenizing batch of text is much faster
        query_input = self.tokenizer(
            [x['query'] for x in features],
            padding=self.padding,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=True,
            truncation=True,
            max_length=self.max_query_len,
            padding_side="right",
        )
        # query_input['position_ids'] = torch.arange(0, query_input['input_ids'].size(1))[None, :]
        doc_input = self.tokenizer(
            [x['doc'] for x in features],
            padding=self.padding,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=True,
            truncation=True,
            max_length=self.max_doc_len,
            padding_side="right",
        )
        # doc_input['position_ids'] = torch.arange(0, doc_input['input_ids'].size(1))[None, :]
        batch_qids = [x['qid'] for x in features]
        batch_pos_docids = [x['docid'] for x in features]  
        qids = torch.tensor(batch_qids, dtype=torch.long)
        docids = torch.tensor(batch_pos_docids, dtype=torch.long)
        # neg_docids = random.sample(self.neg_docids, self.per_device_neg)

        # ---- per-query negatives (hard pool) ----
        batch_hard_rows = [int(x.get("hard_row", -1)) for x in features]
        per_query_negs = []
        for qid, pos_docid, row in zip(batch_qids, batch_pos_docids, batch_hard_rows):
          if row < 0:
              raise ValueError(f"missing hard negatives row for qid={qid} (hard_row={row})")

          cand_arr = self.hard_topdocids[row]  # [topk], may include -1 padding
          # filter -1 and positives
          cand = [int(d) for d in cand_arr.tolist() if int(d) >= 0 and int(d) != pos_docid]
          if len(cand) < self.per_device_neg:
              raise ValueError(
                  f"not enough hard negatives for qid={qid}: have {len(cand)}, need {self.per_device_neg}. "
              )
          negs = random.sample(cand, self.per_device_neg)
          per_query_negs.append(negs)

        neg_docids_flat = [d for negs in per_query_negs for d in negs]  # [B*neg]

        neg_doc_input = self.tokenizer(
            [self.corpus[docid] for docid in neg_docids_flat],
            padding=self.padding,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=True,
            truncation=True,
            max_length=self.max_doc_len,
            padding_side="right",
        )     
        neg_docids = torch.tensor(per_query_negs, dtype=torch.long)

        batch_data = {
                "query_input": query_input,
                "doc_input": doc_input,
                "qids": qids,
                "docids": docids,
                "neg_doc_input": neg_doc_input,
                "neg_docids": neg_docids,
            }
        return batch_data


class QDRelDataset(Dataset):
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            qrel_path: str, 
            query_path: str, 
            corpus_path: str, 
            max_query_len: int, 
            max_doc_len: int, 
            rel_threshold=1, 
            verbose=True,
            hard_neg_dir: Optional[str] = None,
            train_subset_num: Optional[int] = None):
        '''
        negative: choices from `random' or a path to a json file that contains \
            the qid:neg_pid_lst  
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.queries, qid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(query_path), disable=not verbose, mininterval=10)):
            qid, query = line.split("\t")
            qid2offset[qid] = idx
            self.queries.append(query.strip())

        self.corpus, docid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(corpus_path), disable=not verbose, mininterval=10)):
            splits = line.split("\t")
            if len(splits) == 2:
                docid, body = splits
            else:
                raise NotImplementedError()
            docid2offset[docid] = idx
            self.corpus.append(body.strip())

        self.qrels = defaultdict(list)
        for line in tqdm(open(qrel_path), disable=not verbose, mininterval=10):
            qid, _, docid, rel = line.split()
            if int(rel) >= rel_threshold:
                qoffset = qid2offset[qid]
                docoffset = docid2offset[docid]
                self.qrels[qoffset].append(docoffset)

        self.qids = sorted(self.qrels.keys())
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.qrels = dict(self.qrels)
        
        # ---- hard negatives (qid->topdocids) ----
        self.hard_qids = None            # np.memmap int64 [num_q]
        self.hard_topdocids = None       # np.memmap int32 [num_q, topk]
        self.qoff2hardrow = None         # Dict[int, int]  (aligned mapping)

        if hard_neg_dir is not None:
            qids_path = os.path.join(hard_neg_dir, "qids.npy")
            topdocids_path = os.path.join(hard_neg_dir, "topdocids.npy")
            if not (os.path.exists(qids_path) and os.path.exists(topdocids_path)):
                raise ValueError(f"hard_neg_dir={hard_neg_dir} missing qids.npy/topdocids.npy")

            self.hard_qids = np.load(qids_path, mmap_mode="r")
            self.hard_topdocids = np.load(topdocids_path, mmap_mode="r")
            self.qoff2hardrow = {int(q): i for i, q in enumerate(self.hard_qids)}
        
        if train_subset_num is not None:
            n = int(train_subset_num)
            if n > 0:
                before = len(self.qids)
                self.qids = self.qids[:min(n, before)]
                logger.warning(f"[subset] use first {len(self.qids)} qids: {before} -> {len(self.qids)}")

                # Keep qrels consistent (optional but clean)
                self.qrels = {qid: self.qrels[qid] for qid in self.qids}

    def get_qrels(self):
        return self.qrels

    def __len__(self):
        return len(self.qids)
    
    def __getitem__(self, index):
        '''
        We do not tokenize text here and instead tokenize batch of text in the collator because
            a. Tokenizing batch of text is much faster then tokenizing one by one
            b. Usually, the corpus is too large and we cannot afford to use multiple num workers
        '''
        qid = self.qids[index]
        query = self.queries[qid]
        rel_docids = self.qrels[qid]
        docid = random.choice(rel_docids)
        doc = self.corpus[docid]
        hard_row = -1
        if self.qoff2hardrow is not None:
            hard_row = self.qoff2hardrow.get(int(qid), -1)
        data = {
            "query": query,
            "doc": doc,
            "docid": docid,
            "qid": qid,
            "hard_row": hard_row,
        }
        return data
    
    def get_hard_negatives(self):
        if self.hard_qids is None or self.hard_topdocids is None:
            return None
        return (self.hard_qids, self.hard_topdocids)



class ContrastDenseFinetuner(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch= None,):
        """
        Compute contrastive loss.
        """
        query_embeds = model(**inputs['query_input'], return_dict=False) # Nq, dim
        doc_embeds = model(**inputs['doc_input'], return_dict=False) # Nq, dim
        neg_doc_embeds = model(**inputs['neg_doc_input'], return_dict=False) 
        
        if self.args.world_size > 1:
            query_embeds = self._gather_tensor(query_embeds)
            doc_embeds = self._gather_tensor(doc_embeds)
            neg_doc_embeds = self._gather_tensor(neg_doc_embeds)
        loss = self.compute_contrastive_loss(query_embeds, doc_embeds, neg_doc_embeds)
        return (loss, (query_embeds, doc_embeds, neg_doc_embeds)) if return_outputs else loss

    def compute_contrastive_loss(self, query_embeds, doc_embeds, neg_doc_embeds):  
        positive_scores = torch.sum(query_embeds * doc_embeds, dim=-1)  # (N,)
        N, D = query_embeds.shape
        # per-query negatives: neg_doc_embeds is flattened (N*neg, D)
        if neg_doc_embeds.size(0) == N * self.args.per_device_neg:
            neg = neg_doc_embeds.view(N, self.args.per_device_neg, D)          # (N, neg, D)
            negative_scores = torch.einsum("nd,nkd->nk", query_embeds, neg)     # (N, neg)
            # default without DDP; training with one GPU
            total_neg = self.args.per_device_neg
        
        else:
        # shared negatives (DDP gathered): (N, neg_total)  
            negative_scores = torch.matmul(query_embeds, neg_doc_embeds.transpose(0, 1))
            total_neg = negative_scores.size(1)
            assert total_neg == self.args.per_device_neg * self.args.world_size
        assert total_neg % self.args.per_query_neg == 0
        expand_size = total_neg // self.args.per_query_neg
        positive_scores = positive_scores.repeat_interleave(expand_size)
        negative_scores = negative_scores.reshape(N * expand_size, self.args.per_query_neg)
        cat_scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)  # (N * expand_size, 1 + per_query_neg)
        labels = torch.zeros(cat_scores.size(0), dtype=torch.long, device=cat_scores.device)
        similarities = cat_scores * self.args.inv_temperature
        contrast_loss = F.cross_entropy(similarities, labels) 
        contrast_loss = contrast_loss * self.args.world_size
        return contrast_loss

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(self.args.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        all_tensors = torch.cat(all_tensors)
        return all_tensors

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return super().floating_point_ops(inputs['query_input']) + super().floating_point_ops(inputs['doc_input']) + super().floating_point_ops(inputs['neg_doc_input'])

    def evaluate(self, 
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",) -> Dict[str, float]:
        metrics = validate_during_training(self, eval_dataset, ignore_keys, metric_key_prefix)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics



