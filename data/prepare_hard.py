"""
Mine hard-negative candidates with a frozen dense retriever (e.g., BAAI/bge-small-en-v1.5).

This script:
  1) reads a TSV corpus (docid\\ttext) and builds a dense FAISS index over documents;
  2) encodes queries (qid\\ttext) and retrieves top-k candidate documents per query;
  3) optionally filters out qrels-positive docs so outputs are *negative candidates*;
  4) saves memory-mappable numpy arrays:
     - qids.npy: int64 query offsets (line numbers in query file)
     - topdocids.npy: int32 [num_queries, topk] doc offsets (line numbers in corpus file)
     - scores.npy: float32 [num_queries, topk] inner-product scores
     - meta.json: config for reproducibility

Outputs are designed for pipelines where qid/docid are represented by file offsets.

Example:
  python prepare_hard.py \
    --model_name /path/to/BAAI/bge-small-en-v1.5 \
    --query_path /path/to/msmarco-passage/queries.train.tsv \
    --corpus_path /path/to/collection.tsv \
    --qrel_path /path/to/msmarco-passage/qrels.train.tsv \
    --topk 1000 \
    --output_dir ./train_topk1000 \
    --device cuda \
    --doc_batch_size 256 \
    --query_batch_size 256 \
    --faiss_gpu

Notes:
  - Many BGE recipes add a query instruction prefix; control via --query_prefix.
  - Uses inner product with optional L2 normalization (cosine similarity when normalized).
"""

from __future__ import annotations

import os
import json
import random
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("faiss is required. Install faiss-cpu or faiss-gpu.") from e

from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger("mine_hard")


def setup_logger(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def read_tsv_id_text(path: str) -> Tuple[List[str], Dict[str, int]]:
    """Read TSV lines: id\\ttext -> (texts_in_order, id2offset)."""
    texts: List[str] = []
    id2offset: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(path)}", mininterval=10)):
            splits = line.rstrip("\n").split("\t")
            if len(splits) != 2:
                raise ValueError(f"Bad line (expect 2 columns) at {path}:{idx+1}: {line[:200]!r}")
            _id, text = splits
            id2offset[_id] = idx
            texts.append(text.strip())
    return texts, id2offset


def read_trec_qrels(
    qrel_path: str,
    qid2offset: Dict[str, int],
    docid2offset: Dict[str, int],
    rel_threshold: int = 1,
) -> Dict[int, List[int]]:
    """Read TREC qrels (qid 0 docid rel) -> qoff -> list[doff]."""
    qrels: Dict[int, List[int]] = {}
    with open(qrel_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(qrel_path)}", mininterval=10):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            try:
                rel_i = int(rel)
            except Exception:
                continue
            if rel_i < rel_threshold:
                continue
            if qid not in qid2offset or docid not in docid2offset:
                continue
            qoff = qid2offset[qid]
            doff = docid2offset[docid]
            qrels.setdefault(qoff, []).append(doff)

    for q in list(qrels.keys()):
        qrels[q] = sorted(set(qrels[q]))
    return qrels


def pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom
    raise ValueError(f"Unknown pooling: {pooling}")


@torch.inference_mode()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    pooling: str = "cls",
    normalize: bool = True,
    prefix: str = "",
    use_fp16: bool = False,
) -> Iterable[np.ndarray]:
    """Yield float32 embeddings in batches."""
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", mininterval=10):
        batch = texts[start : start + batch_size]
        if prefix:
            batch = [prefix + x for x in batch]
        feats = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        feats = {k: v.to(device) for k, v in feats.items()}

        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**feats, return_dict=True)
        else:
            out = model(**feats, return_dict=True)

        emb = pool_last_hidden(out.last_hidden_state, feats["attention_mask"], pooling=pooling)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        yield emb.detach().cpu().to(torch.float32).numpy()


def build_faiss_index(dim: int, *, use_gpu: bool = False, gpu_id: Optional[int] = None) -> faiss.Index:
    index = faiss.IndexFlatIP(dim)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        res.setTempMemory(128 * 1024 * 1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False
        index = faiss.index_cpu_to_gpu(res, 0 if gpu_id is None else gpu_id, index, co)
    return index


def search_in_batches(
    index: faiss.Index,
    query_embeds: np.ndarray,
    *,
    topk: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, idx) for queries."""
    all_scores: List[np.ndarray] = []
    all_idx: List[np.ndarray] = []
    for start in tqdm(range(0, query_embeds.shape[0], batch_size), desc="Searching", mininterval=10):
        q = np.ascontiguousarray(query_embeds[start : start + batch_size].astype(np.float32))
        scores, idx = index.search(q, topk)
        all_scores.append(scores)
        all_idx.append(idx)
    return np.concatenate(all_scores, axis=0), np.concatenate(all_idx, axis=0)


def filter_positives(
    qoffs: np.ndarray,
    idx: np.ndarray,
    scores: np.ndarray,
    *,
    qrels: Optional[Dict[int, List[int]]],
    topk: int,
    global_neg_pool: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove qrel-positive docs from retrieved list; keep first topk. Fill if needed."""
    rng = rng or random.Random(1234)
    out_idx = np.empty((len(qoffs), topk), dtype=np.int32)
    out_scores = np.empty((len(qoffs), topk), dtype=np.float32)

    for i, qoff in enumerate(qoffs.tolist()):
        cand = idx[i]
        cand_s = scores[i]

        if qrels is None or qoff not in qrels:
            kept = cand[:topk]
            kept_s = cand_s[:topk]
        else:
            pos = set(qrels[qoff])
            kept_list: List[int] = []
            kept_s_list: List[float] = []
            for d, s in zip(cand.tolist(), cand_s.tolist()):
                if d in pos:
                    continue
                kept_list.append(d)
                kept_s_list.append(s)
                if len(kept_list) >= topk:
                    break

            if len(kept_list) < topk:
                if not global_neg_pool:
                    raise RuntimeError(
                        f"Not enough negatives after filtering for qoff={qoff}. "
                        "Provide qrels or ensure corpus is large enough."
                    )
                need = topk - len(kept_list)
                fill = rng.sample(global_neg_pool, need)
                kept_list.extend(fill)
                kept_s_list.extend([-1e9] * need)

            kept = np.array(kept_list[:topk], dtype=np.int32)
            kept_s = np.array(kept_s_list[:topk], dtype=np.float32)

        out_idx[i] = kept
        out_scores[i] = kept_s

    return out_scores, out_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--query_path", type=str, required=True)
    ap.add_argument("--corpus_path", type=str, required=True)
    ap.add_argument("--qrel_path", type=str, default=None)
    ap.add_argument("--rel_threshold", type=int, default=1)

    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument(
        "--raw_topk",
        type=int,
        default=None,
        help="Retrieve this many before filtering positives (default: topk + 100).",
    )

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_query_len", type=int, default=64)
    ap.add_argument("--max_doc_len", type=int, default=256)
    ap.add_argument("--query_batch_size", type=int, default=256)
    ap.add_argument("--doc_batch_size", type=int, default=256)
    ap.add_argument("--search_batch_size", type=int, default=1024)

    ap.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument(
        "--query_prefix",
        type=str,
        default="Represent this sentence for searching relevant passages: ",
    )
    ap.add_argument("--doc_prefix", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--faiss_gpu", action="store_true")
    ap.add_argument("--faiss_gpu_id", type=int, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="count", default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.raw_topk is None:
        args.raw_topk = args.topk + 100
    if args.raw_topk < args.topk:
        raise ValueError("--raw_topk must be >= --topk")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading queries/corpus...")
    queries, qid2offset = read_tsv_id_text(args.query_path)
    corpus, docid2offset = read_tsv_id_text(args.corpus_path)
    logger.info("#queries=%d, #docs=%d", len(queries), len(corpus))

    qrels: Optional[Dict[int, List[int]]] = None
    if args.qrel_path:
        qrels = read_trec_qrels(
            args.qrel_path, qid2offset, docid2offset, rel_threshold=args.rel_threshold
        )
        qoffs = np.array(sorted(qrels.keys()), dtype=np.int64)
        logger.info("Loaded qrels for %d queries (rel_threshold=%d)", len(qoffs), args.rel_threshold)
    else:
        qoffs = np.arange(len(queries), dtype=np.int64)
        logger.info("No qrels provided; mining for ALL queries")

    global_neg_pool: Optional[List[int]] = None
    if qrels is not None:
        all_pos = set(d for ds in qrels.values() for d in ds)
        global_neg_pool = [d for d in range(len(corpus)) if d not in all_pos]

    logger.info("Loading model/tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.to(device).eval()

    use_fp16 = bool(args.fp16 and device.type == "cuda")
    if use_fp16:
        model.half()

    dim = int(model.config.hidden_size)
    index = build_faiss_index(dim=dim, use_gpu=args.faiss_gpu, gpu_id=args.faiss_gpu_id)

    logger.info("Encoding + indexing corpus...")
    for emb in encode_texts(
        corpus,
        tokenizer,
        model,
        batch_size=args.doc_batch_size,
        max_length=args.max_doc_len,
        device=device,
        pooling=args.pooling,
        normalize=(not args.no_normalize),
        prefix=args.doc_prefix,
        use_fp16=use_fp16,
    ):
        index.add(np.ascontiguousarray(emb.astype(np.float32)))

    logger.info("Index ready. ntotal=%d", index.ntotal)
    if int(index.ntotal) != len(corpus):
        raise RuntimeError("Index size mismatch; check corpus reading.")

    logger.info("Encoding queries (%d)...", len(qoffs))
    q_texts = [queries[int(q)] for q in qoffs.tolist()]
    q_embeds_list: List[np.ndarray] = []
    for emb in encode_texts(
        q_texts,
        tokenizer,
        model,
        batch_size=args.query_batch_size,
        max_length=args.max_query_len,
        device=device,
        pooling=args.pooling,
        normalize=(not args.no_normalize),
        prefix=args.query_prefix,
        use_fp16=use_fp16,
    ):
        q_embeds_list.append(emb)

    query_embeds = np.ascontiguousarray(np.concatenate(q_embeds_list, axis=0).astype(np.float32))
    if query_embeds.shape[0] != len(qoffs):
        raise RuntimeError("Query embedding count mismatch.")

    logger.info("Searching raw_topk=%d...", args.raw_topk)
    raw_scores, raw_idx = search_in_batches(
        index, query_embeds, topk=args.raw_topk, batch_size=args.search_batch_size
    )

    logger.info("Filtering positives -> topk=%d...", args.topk)
    scores, topdocids = filter_positives(
        qoffs=qoffs,
        idx=raw_idx,
        scores=raw_scores,
        qrels=qrels,
        topk=args.topk,
        global_neg_pool=global_neg_pool,
        rng=random.Random(args.seed),
    )

    out_qids = os.path.join(args.output_dir, "qids.npy")
    out_docs = os.path.join(args.output_dir, "topdocids.npy")
    out_scores = os.path.join(args.output_dir, "scores.npy")

    np.save(out_qids, qoffs.astype(np.int64))
    np.save(out_docs, topdocids.astype(np.int32))
    np.save(out_scores, scores.astype(np.float32))

    meta = {
        "model_name": args.model_name,
        "query_path": os.path.abspath(args.query_path),
        "corpus_path": os.path.abspath(args.corpus_path),
        "qrel_path": os.path.abspath(args.qrel_path) if args.qrel_path else None,
        "rel_threshold": args.rel_threshold,
        "topk": args.topk,
        "raw_topk": args.raw_topk,
        "max_query_len": args.max_query_len,
        "max_doc_len": args.max_doc_len,
        "pooling": args.pooling,
        "normalize": (not args.no_normalize),
        "query_prefix": args.query_prefix,
        "doc_prefix": args.doc_prefix,
        "faiss_gpu": args.faiss_gpu,
        "faiss_gpu_id": args.faiss_gpu_id,
        "seed": args.seed,
        "num_queries": int(len(qoffs)),
        "num_docs": int(len(corpus)),
        "embed_dim": int(dim),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Done. Saved: %s, %s, %s", out_qids, out_docs, out_scores)
    logger.info("Tip: load with np.load(..., mmap_mode='r') for large arrays.")


if __name__ == "__main__":
    main()
