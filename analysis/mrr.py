#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute-scale / MRR-scale plots (2x2 layout) for retrieval metrics.

Directory convention (matches your current training dumps):
  {base_dir}/{model_name}/gpu1_neg{train_bs}.{train_bs}_bs{bs}_lr{lr}/
    subset{subset}/checkpoint-{ckpt_step}/observe/lbs{eval_bs}_ncs{ncs}_lsn{lsn}/result.msmarco-passage.dev.npz

  If subset == full_subset, the "subset{subset}" folder is omitted.

Compute proxy:
  compute = non_embedding_params * datasize(subset)

Example
  python computescale.py \
    --param-json path/to/bert_count.json \
    --base-dir path/to/hard_negtives \
    --out-dir path/to/figs/scale \
    --models google_bert/uncased_L-2_H-768_A-12 google_bert/uncased_L-12_H-768_A-12 \
    --subsets 1000 3000 6000 10000 30000 60000 100000 502939 \
    --full-subset 502939 \
    --train-bs 16 --bs 32 --lr 3e-6 \
    --eval-bs 256 --ncs 100000 --lsn 1 \
    --mode both --legend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


METRICS_CFG = [
    {"key": "item_cross_entropy", "title": "Cross-Entropy",        "yscale": None},
    {"key": "kl_divergence",      "title": "Rank-Score-Alignment", "yscale": None},
    {"key": "ranking_entropy",    "title": "Rank-Entropy",         "yscale": None},
    {"key": "log_normalize_num",  "title": "Score-Uncertainty",    "yscale": None},
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-json", type=Path, required=True, help="path/to/bert_count.json")
    ap.add_argument("--base-dir", type=Path, required=True, help="path/to/hard_negtives")
    ap.add_argument("--out-dir", type=Path, required=True, help="path/to/figs/scale")

    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--subsets", nargs="+", type=int, required=True)
    ap.add_argument("--full-subset", type=int, default=502939)

    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=str, default="3e-6")
    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--ncs", type=int, default=100000)
    ap.add_argument("--lsn", type=int, default=1)

    ap.add_argument("--mode", choices=["mrr", "compute", "both"], default="both")
    ap.add_argument("--legend", action="store_true", help="Show model-size ordered legend")

    ap.add_argument("--figw", type=float, default=7.0)
    ap.add_argument("--figh", type=float, default=4.2)
    ap.add_argument("--wspace", type=float, default=0.30)
    ap.add_argument("--hspace", type=float, default=0.30)
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--cmap", type=str, default="Blues")
    ap.add_argument("--cmap-min", type=float, default=0.38)
    ap.add_argument("--cmap-max", type=float, default=0.95)
    return ap.parse_args()


def _extract_num(v: Any) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        return float(v)
    if isinstance(v, dict):
        for k in ["non_embedding_params", "total", "value", "count", "num_params", "total_params", "n_params"]:
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


def decompose_result(rank_list, ce_loss_list) -> Dict[str, float]:
    ranks = np.asarray(rank_list, dtype=np.int64).reshape(-1)
    loss = np.asarray(ce_loss_list, dtype=np.float64).reshape(-1)

    if ranks.size == 0:
        return {k["key"]: np.nan for k in METRICS_CFG} | {"mrr": np.nan}

    # robust: allow 0-based or 1-based
    is_zero_based = (ranks.min() == 0)
    r = ranks + 1 if is_zero_based else ranks
    valid = r > 0
    mrr = float(np.mean(1.0 / r[valid])) if np.any(valid) else np.nan

    max_rank0 = int(ranks.max())
    rank_cnt = np.bincount(ranks, minlength=max_rank0 + 1)
    rank_prob_full = rank_cnt / max(1, int(rank_cnt.sum()))
    ranking_entropy = float(-np.sum(rank_prob_full * np.log(rank_prob_full + 1e-10)))

    nonzero = rank_cnt > 0
    p = rank_prob_full[nonzero]
    p = p / (p.sum() + 1e-10)

    ce_sum_by_rank = np.bincount(ranks, weights=loss, minlength=rank_cnt.size)
    grouped_ce = ce_sum_by_rank[nonzero] / rank_cnt[nonzero]

    grouped_probs = np.exp(-grouped_ce)
    normalize_num = float(np.sum(grouped_probs))
    grouped_probs = grouped_probs / (normalize_num + 1e-10)

    kl_divergence = float(np.sum(p * np.log((p + 1e-10) / (grouped_probs + 1e-10))))
    log_normalize_num = float(-np.log(normalize_num + 1e-10))

    return {
        "item_cross_entropy": float(np.mean(loss)),
        "ranking_entropy": float(ranking_entropy),
        "kl_divergence": float(kl_divergence),
        "log_normalize_num": float(log_normalize_num),
        "mrr": float(mrr),
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


def build_decompose_result_dict(
    models: List[str],
    subsets: List[int],
    *,
    base_dir: Path,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model in models:
        per_subset: Dict[int, Dict[str, float]] = {}
        for subset in subsets:
            p = build_observe_path(
                base_dir, model, subset,
                full_subset=full_subset,
                train_bs=train_bs, bs=bs, lr=lr,
                eval_bs=eval_bs, ncs=ncs, lsn=lsn,
            )
            if not p.exists():
                print(f"[WARN] Missing npz: {p}")
                continue
            data = np.load(p)
            per_subset[subset] = decompose_result(data["rank"].reshape(-1), data["loss"].reshape(-1))
        out[model] = per_subset
    return out


def get_metrics(d: Dict[str, Dict[int, Dict[str, float]]], model: str, subset: int) -> Optional[Dict[str, float]]:
    return d.get(model, {}).get(subset, None)


def build_model_order_and_colors(
    models: List[str],
    model_param_count: dict,
    *,
    cmap_name: str,
    cmap_min: float,
    cmap_max: float,
) -> Tuple[List[str], Dict[str, Any]]:
    cmap = plt.get_cmap(cmap_name)

    pairs: List[Tuple[str, float]] = []
    missing: List[str] = []
    for m in models:
        s = model_param_count.get(m, {}).get("non_embedding_params", None)
        if s is None:
            missing.append(m)
        else:
            pairs.append((m, float(_extract_num(s))))

    pairs.sort(key=lambda x: x[1])  # small -> large
    models_sorted = [m for m, _ in pairs] + missing

    n = len(pairs)
    ts = [0.5 * (cmap_min + cmap_max)] if n <= 1 else np.linspace(cmap_min, cmap_max, n)
    color_map = {m: cmap(t) for (m, _), t in zip(pairs, ts)}

    mid = 0.5 * (cmap_min + cmap_max)
    for m in missing:
        color_map[m] = cmap(mid)

    return models_sorted, color_map


def compute_proxy(model_param_count: dict, model_name: str, datasize: int) -> Optional[float]:
    s = model_param_count.get(model_name, {}).get("non_embedding_params", None)
    if s is None:
        return None
    return float(_extract_num(s)) * float(datasize)


def plot_2x2_lines(
    decompose_result_dict: Dict[str, Dict[int, Dict[str, float]]],
    model_param_count: dict,
    models: List[str],
    subsets: List[int],
    *,
    x_getter,
    xlabel: str,
    xscale: Optional[str],
    out_path: Path,
    per_subplot_xlabel: bool,
    show_legend: bool,
    figsize: Tuple[float, float],
    wspace: float,
    hspace: float,
    cmap_name: str,
    cmap_min: float,
    cmap_max: float,
) -> None:
    # Font requirements
    SUBPANEL_LABEL_FONTSIZE = 14  # metric name (shown as y-label)
    TICK_FONTSIZE = 11
    XLABEL_FONTSIZE = 11  # per-subplot or global

    models_sorted, color_map = build_model_order_and_colors(
        models, model_param_count, cmap_name=cmap_name, cmap_min=cmap_min, cmap_max=cmap_max
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    ax_list = axes.ravel()

    for ax, cfg in zip(ax_list, METRICS_CFG):
        key = cfg["key"]
        ylab = cfg["title"]
        yscale = cfg["yscale"]

        for model in models_sorted:
            pts = []
            for subset in subsets:
                m = get_metrics(decompose_result_dict, model, subset)
                if m is None:
                    continue
                y = m.get(key, None)
                if y is None or (not np.isfinite(y)):
                    continue
                x = x_getter(model, m, subset)
                if x is None or (not np.isfinite(x)):
                    continue
                pts.append((float(x), float(y)))

            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            xs, ys = zip(*pts)

            ax.plot(
                xs, ys,
                marker="o",
                linestyle="-",
                linewidth=1.10,
                markersize=2.6,
                alpha=0.95,
                color=color_map[model],
                label=model,
            )

        ax.set_title("")  # no subplot title
        ax.set_ylabel(ylab, fontsize=SUBPANEL_LABEL_FONTSIZE)

        if per_subplot_xlabel:
            ax.set_xlabel(xlabel, fontsize=XLABEL_FONTSIZE)
        else:
            ax.set_xlabel("")

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # Optional legend: ordered small->large, place below the figure
    if show_legend:
        handles, labels = ax_list[0].get_legend_handles_labels()
        # remove duplicates while preserving order
        seen = set()
        uniq = []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq.append((h, l))
        if uniq:
            handles_u, labels_u = zip(*uniq)
            fig.legend(
                handles_u, labels_u,
                loc="lower center",
                ncol=4,
                frameon=False,
                fontsize=11,
                bbox_to_anchor=(0.5, -0.02),
                handlelength=2.0,
                columnspacing=1.2,
            )

    fig.subplots_adjust(
        left=0.10, right=0.995,
        bottom=0.14 if per_subplot_xlabel else 0.18,
        top=0.98,
        wspace=wspace, hspace=hspace,
    )

    if not per_subplot_xlabel:
        fig.supxlabel(xlabel, y=0.06, fontsize=XLABEL_FONTSIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")
    print(f"[SAVE] {out_path.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()

    with open(args.param_json, "r", encoding="utf-8") as f:
        model_param_count = json.load(f)

    decompose_result_dict = build_decompose_result_dict(
        models=args.models,
        subsets=args.subsets,
        base_dir=args.base_dir,
        full_subset=args.full_subset,
        train_bs=args.train_bs,
        bs=args.bs,
        lr=args.lr,
        eval_bs=args.eval_bs,
        ncs=args.ncs,
        lsn=args.lsn,
    )

    figsize = (args.figw, args.figh)

    if args.mode in ("compute", "both"):
        out_path = args.out_dir / "computescale_vs_compute.png"
        plot_2x2_lines(
            decompose_result_dict=decompose_result_dict,
            model_param_count=model_param_count,
            models=args.models,
            subsets=args.subsets,
            x_getter=lambda model, metrics, subset: compute_proxy(model_param_count, model, subset),
            xlabel="Compute",
            xscale="log",
            out_path=out_path,
            per_subplot_xlabel=False,   # global xlabel for compute plot
            show_legend=args.legend,
            figsize=figsize,
            wspace=args.wspace,
            hspace=args.hspace,
            cmap_name=args.cmap,
            cmap_min=args.cmap_min,
            cmap_max=args.cmap_max,
        )

    if args.mode in ("mrr", "both"):
        out_path = args.out_dir / "computescale_vs_mrr.png"
        plot_2x2_lines(
            decompose_result_dict=decompose_result_dict,
            model_param_count=model_param_count,
            models=args.models,
            subsets=args.subsets,
            x_getter=lambda model, metrics, subset: metrics.get("mrr", None),
            xlabel="MRR",
            xscale=None,
            out_path=out_path,
            per_subplot_xlabel=True,    # per-subplot xlabel for MRR plot
            show_legend=args.legend,
            figsize=figsize,
            wspace=args.wspace,
            hspace=args.hspace,
            cmap_name=args.cmap,
            cmap_min=args.cmap_min,
            cmap_max=args.cmap_max,
        )


if __name__ == "__main__":
    main()
