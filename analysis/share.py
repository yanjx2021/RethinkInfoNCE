#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute mean component shares (Rank-Entropy / Rank-Score-Alignment / Score-Uncertainty)
at the final checkpoint across models, and plot a compact horizontal stacked bar.

Example
  python share.py \
    --param-json path/to/bert_count.json \
    --base-dir path/to/hard_negtives \
    --out-dir path/to/output/figs/scale \
    --models google_bert/uncased_L-24_H-1024_A-16 google_bert/uncased_L-2_H-768_A-12 \
    --full-subset 502939 --subset 502939 \
    --train-bs 16 --bs 32 --lr 3e-6 \
    --eval-batch-size 256 --ncs 100000 --lsn 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("plot_component_share_bar")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-json", type=Path, required=True, help="path/to/bert_count.json")
    ap.add_argument("--base-dir", type=Path, required=True, help="path/to/hard_negtives")
    ap.add_argument("--out-dir", type=Path, required=True, help="path/to/output/figs/scale")

    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--full-subset", type=int, default=502939)
    ap.add_argument("--subset", type=int, default=502939)

    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=str, default="3e-6")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--ncs", type=int, default=100000)
    ap.add_argument("--lsn", type=int, default=1)

    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("-v", "--verbose", action="count", default=1)
    return ap.parse_args()


def setup_logger(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def _extract_num(v: Any) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        return float(v)
    if isinstance(v, dict):
        for k in ["total", "total_params", "num_params", "n_params", "value", "count", "non_embedding_params"]:
            if k in v:
                return _extract_num(v[k])
        s = 0.0
        found = False
        for vv in v.values():
            try:
                s += _extract_num(vv)
                found = True
            except Exception:
                pass
        if found:
            return float(s)
    raise TypeError(f"Cannot extract numeric from {type(v)}: {v}")


def decompose_result(rank_list, ce_loss_list) -> Optional[Dict[str, float]]:
    ranks = np.asarray(rank_list, dtype=int).reshape(-1)
    ce = np.asarray(ce_loss_list, dtype=np.float64).reshape(-1)
    if ranks.size == 0:
        return None

    max_rank = int(ranks.max())
    if max_rank < 0:
        return None

    rank_cnt = np.zeros(max_rank + 1, dtype=int)
    for r in ranks:
        if r < 0:
            continue
        rank_cnt[r] += 1

    tot = int(rank_cnt.sum())
    if tot <= 0:
        return None

    eps = 1e-10
    rank_prob_full = rank_cnt / tot
    ranking_entropy = float(-np.sum(rank_prob_full * np.log(rank_prob_full + eps)))

    nonzero = rank_cnt >= 1
    p = rank_prob_full[nonzero]
    p = p / (p.sum() + eps)

    ce_sum_by_rank = np.zeros(max_rank + 1, dtype=np.float64)
    for r, loss in zip(ranks, ce):
        if r < 0:
            continue
        ce_sum_by_rank[r] += float(loss)

    grouped_ce = ce_sum_by_rank[nonzero] / rank_cnt[nonzero]
    grouped_probs = np.exp(-grouped_ce)
    normalize_num = float(grouped_probs.sum())
    grouped_probs = grouped_probs / (normalize_num + eps)

    kl_divergence = float(np.sum(p * np.log((p + eps) / (grouped_probs + eps))))
    log_normalize_num = float(-np.log(normalize_num + eps))

    return {
        "ranking_entropy": ranking_entropy,
        "kl_divergence": kl_divergence,
        "log_normalize_num": log_normalize_num,
    }


def build_observe_path(
    base_dir: Path,
    model_name: str,
    subset: int,
    *,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
) -> Path:
    ckpt_step = subset // 32 * 3
    root = base_dir / model_name / f"gpu1_neg{train_bs}.{train_bs}_bs{bs}_lr{lr}"
    if subset == full_subset:
        root = root / f"checkpoint-{ckpt_step}"
    else:
        root = root / f"subset{subset}" / f"checkpoint-{ckpt_step}"
    return root / "observe" / f"lbs{eval_bs}_ncs{ncs}_lsn{lsn}" / "result.msmarco-passage.dev.npz"


def load_metrics_for_model(
    model: str,
    *,
    base_dir: Path,
    subset: int,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
) -> Optional[Dict[str, float]]:
    p = build_observe_path(
        base_dir, model, subset,
        full_subset=full_subset, train_bs=train_bs, bs=bs, lr=lr,
        eval_bs=eval_bs, ncs=ncs, lsn=lsn
    )
    if not p.exists():
        logger.warning("Missing npz: %s", p)
        return None
    data = np.load(p)
    return decompose_result(data["rank"].reshape(-1), data["loss"].reshape(-1))


def compute_mean_shares_and_plot(
    models: List[str],
    model_param_count: dict,
    *,
    base_dir: Path,
    out_dir: Path,
    subset: int,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
    dpi: int,
) -> None:
    shares = {"re": [], "kl": [], "cf": []}

    for m in models:
        size = model_param_count.get(m, {}).get("non_embedding_params", None)
        if size is None:
            logger.warning("Missing non_embedding_params for model=%s, skip.", m)
            continue

        metrics = load_metrics_for_model(
            m,
            base_dir=base_dir,
            subset=subset,
            full_subset=full_subset,
            train_bs=train_bs,
            bs=bs,
            lr=lr,
            eval_bs=eval_bs,
            ncs=ncs,
            lsn=lsn,
        )
        if metrics is None:
            continue

        re = float(metrics["ranking_entropy"])
        kl = float(metrics["kl_divergence"])
        cf = float(metrics["log_normalize_num"])

        denom = abs(re) + abs(kl) + abs(cf)
        if denom <= 0 or (not np.isfinite(denom)):
            logger.warning("Bad denom for model=%s: re=%s kl=%s cf=%s", m, re, kl, cf)
            continue

        shares["re"].append(abs(re) / denom)
        shares["kl"].append(abs(kl) / denom)
        shares["cf"].append(abs(cf) / denom)

    if len(shares["re"]) == 0:
        raise RuntimeError("No valid models loaded. Check paths and PARAM_JSON keys.")

    share_re = float(np.mean(shares["re"]))
    share_kl = float(np.mean(shares["kl"]))
    share_cf = float(np.mean(shares["cf"]))
    s = share_re + share_kl + share_cf
    share_re, share_kl, share_cf = share_re / s, share_kl / s, share_cf / s

    colors = {"re": "#4E79A7", "cf": "#3F7D5B", "kl": "#B04A4A"}

    fig, ax = plt.subplots(figsize=(5.6, 1.05))
    left = 0.0
    ax.barh([0], [share_re], left=left, color=colors["re"], height=0.55,
            label=f"Rank-Entropy {share_re*100:.1f}%", alpha=0.90)
    left += share_re
    ax.barh([0], [share_kl], left=left, color=colors["kl"], height=0.55,
            label=f"Rank-Score-Alignment {share_kl*100:.1f}%", alpha=0.90)
    left += share_kl
    ax.barh([0], [share_cf], left=left, color=colors["cf"], height=0.55,
            label=f"Score-Uncertainty {share_cf*100:.1f}%", alpha=0.90)

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.grid(False)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        columnspacing=1.0,
        handletextpad=0.4,
        borderpad=0.2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "share.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)
    logger.info("Saved: %s", out_path)
    logger.info("Saved: %s", out_path.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    with open(args.param_json, "r", encoding="utf-8") as f:
        model_param_count = json.load(f)

    compute_mean_shares_and_plot(
        models=args.models,
        model_param_count=model_param_count,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        subset=args.subset,
        full_subset=args.full_subset,
        train_bs=args.train_bs,
        bs=args.bs,
        lr=args.lr,
        eval_bs=args.eval_batch_size,
        ncs=args.ncs,
        lsn=args.lsn,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
