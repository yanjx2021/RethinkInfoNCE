#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot training dynamics of decomposed metrics (solid) over steps, overlaid with MRR (dashed) on a twin y-axis.

Inputs
- param-count JSON: model_name -> param count (number/str/dict supported)
- observe npz files under:
  {lr_grid_root}/{model_name}/gpu1_neg{train_bs}.{train_bs}_bs{BS}_lr{lr}/snapshots/step-{step}/
    observe/lbs{eval_bs}_ncs{NCS}_lsn{LSN}/result.msmarco-passage.dev.npz

Each npz must contain:
- rank: array-like
- loss: array-like

Outputs
- PNG + PDF to {out_dir}/steps_1x4_overlayMRR_t{train_bs}_e{eval_bs}.(png|pdf)

Example
  python trainprocess.py \
    --param-count-json path/to/bert_count.json \
    --model-process-root path/to/process \
    --out-dir path/to/output/figs/process \
    --models google_bert/uncased_L-4_H-512_A-8 google_bert/uncased_L-8_H-512_A-8 google_bert/uncased_L-12_H-768_A-12 \
    --train-batch-sizes 16 \
    --eval-batch-sizes 256 \
    --steps 1 10 50 100 500 1000 5000 10000 15716 \
    --lr 3e-6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger("plot_steps_overlay_mrr")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-count-json", type=Path, required=True, help="path/to/bert_count.json")
    ap.add_argument("--model-process-root", type=Path, required=True, help="path/to/process")
    ap.add_argument("--out-dir", type=Path, required=True, help="path/to/output/figs/process")

    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names used as directory names under model-process-root.",
    )
    ap.add_argument("--train-batch-sizes", nargs="+", type=int, default=[16])
    ap.add_argument("--eval-batch-sizes", nargs="+", type=int, default=[256])
    ap.add_argument("--steps", nargs="+", type=int, required=True)

    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ncs", type=int, default=100000)
    ap.add_argument("--lsn", type=int, default=1)
    ap.add_argument("--lr", type=str, default="3e-6")

    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    return ap.parse_args()


def setup_logger(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def build_observe_path(
    lr_grid_root: Path,
    model_name: str,
    train_bs: int,
    eval_bs: int,
    step: int,
    *,
    bs: int,
    ncs: int,
    lsn: int,
    lr: str,
) -> Path:
    model_dir = (
        lr_grid_root
        / model_name
        / f"gpu1_neg{train_bs}.{train_bs}_bs{bs}_lr{lr}"
        / "snapshots"
        / f"step-{step}"
    )
    return model_dir / "observe" / f"lbs{eval_bs}_ncs{ncs}_lsn{lsn}" / "result.msmarco-passage.dev.npz"


def load_observations(
    *,
    lr_grid_root: Path,
    models: List[str],
    train_batch_sizes: List[int],
    eval_batch_sizes: List[int],
    steps: List[int],
    bs: int,
    ncs: int,
    lsn: int,
    lr: str,
) -> Dict[str, Dict[int, Dict[int, Dict[int, Any]]]]:
    observations: Dict[str, Dict[int, Dict[int, Dict[int, Any]]]] = {}
    for model_name in models:
        for train_bs in train_batch_sizes:
            for eval_bs in eval_batch_sizes:
                for step in steps:
                    p = build_observe_path(
                        lr_grid_root, model_name, train_bs, eval_bs, step,
                        bs=bs, ncs=ncs, lsn=lsn, lr=lr
                    )
                    if not p.exists():
                        logger.warning("Missing: %s", p)
                        continue
                    observations.setdefault(model_name, {}).setdefault(train_bs, {}).setdefault(eval_bs, {})[step] = np.load(p)
    return observations


def decompose_result(rank_list, ce_loss_list) -> Dict[str, float]:
    ranks = np.asarray(rank_list, dtype=np.int64).ravel()
    ce = np.asarray(ce_loss_list, dtype=np.float64).ravel()
    if ranks.shape[0] != ce.shape[0]:
        raise ValueError(f"rank_list and ce_loss_list must have same length, got {ranks.shape[0]} vs {ce.shape[0]}")

    eps = 1e-10
    n = ranks.size
    if n == 0:
        raise ValueError("Empty rank_list")

    is_zero_based = (ranks.min() == 0)
    r = ranks + 1 if is_zero_based else ranks
    if (r <= 0).any():
        raise ValueError("Found non-positive rank after conversion; check rank encoding.")

    mrr_deg = float(np.mean(1.0 / r))
    ndcg_deg = float(np.mean(1.0 / np.log2(r + 1.0)))

    rank_cnt = np.bincount(ranks)
    rank_prob = rank_cnt / n
    nonzero = rank_cnt > 0
    p = rank_prob[nonzero]
    ranking_entropy = float(-np.sum(p * np.log(p + eps)))

    ce_sum_by_rank = np.bincount(ranks, weights=ce, minlength=rank_cnt.size)
    grouped_ce = ce_sum_by_rank[nonzero] / rank_cnt[nonzero]
    grouped_probs_unnorm = np.exp(-grouped_ce)
    normalize_num = float(np.sum(grouped_probs_unnorm))
    grouped_probs = grouped_probs_unnorm / (normalize_num + eps)

    kl_divergence = float(np.sum(p * np.log((p + eps) / (grouped_probs + eps))))
    log_normalize_num = float(-np.log(normalize_num + eps))

    return {
        "item_cross_entropy": float(np.mean(ce)),
        "ranking_entropy": ranking_entropy,
        "kl_divergence": kl_divergence,
        "log_normalize_num": log_normalize_num,
        "mrr_deg": mrr_deg,
        "ndcg_deg": ndcg_deg,
        "rank_is_zero_based": float(is_zero_based),
    }


def build_metrics_by_step(
    observations: Dict[str, Dict[int, Dict[int, Dict[int, Any]]]]
) -> Dict[str, Dict[int, Dict[int, Dict[int, Dict[str, float]]]]]:
    metrics: Dict[str, Dict[int, Dict[int, Dict[int, Dict[str, float]]]]] = {}
    for model_name, train_dict in observations.items():
        for train_bs, eval_dict in train_dict.items():
            for eval_bs, step_dict in eval_dict.items():
                for step, data in step_dict.items():
                    m = decompose_result(data["rank"].reshape(-1), data["loss"].reshape(-1))
                    metrics.setdefault(model_name, {}).setdefault(train_bs, {}).setdefault(eval_bs, {})[step] = m
    return metrics


def _format_params(n: float) -> str:
    n = float(n)
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:.0f}"


def _extract_param_value(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        return float(v)
    if isinstance(v, dict):
        for k in ["total", "total_params", "num_params", "n_params", "param_count", "params", "all", "overall"]:
            if k in v:
                return _extract_param_value(v[k])
        nums = []
        for vv in v.values():
            try:
                nums.append(_extract_param_value(vv))
            except Exception:
                pass
        if nums:
            return float(sum(nums))
    raise TypeError(f"Cannot extract numeric param count from type={type(v)} value={v}")


def _get_param_count(model_name: str, param_count_json: dict) -> float:
    if model_name in param_count_json:
        return _extract_param_value(param_count_json[model_name])
    k1 = model_name.replace("google_bert/", "")
    if k1 in param_count_json:
        return _extract_param_value(param_count_json[k1])
    k2 = k1.replace("uncased_", "")
    if k2 in param_count_json:
        return _extract_param_value(param_count_json[k2])
    raise KeyError(f"Cannot find param count for model_name={model_name}")


def plot_1x4_overlay_mrr(
    metrics_by_step: dict,
    out_dir: Path,
    param_count_json: dict,
    models: List[str],
    train_batch_sizes: List[int],
    eval_batch_sizes: List[int],
    *,
    dpi: int = 220,
    show: bool = False,
) -> None:
    panels: List[Tuple[str, str]] = [
        ("item_cross_entropy", "Cross-Entropy"),
        ("ranking_entropy", "Rank-Entropy"),
        ("kl_divergence", "Rank-Score-Alignment"),
        ("log_normalize_num", "Score-Uncertainty"),
    ]

    params = {m: _get_param_count(m, param_count_json) for m in models}
    vmin, vmax = min(params.values()), max(params.values())
    norm = Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6) if vmin == vmax else LogNorm(vmin=vmin, vmax=vmax)

    base_cmap = plt.get_cmap("Blues")
    cmap = LinearSegmentedColormap.from_list("Blues_trunc", base_cmap(np.linspace(0.35, 0.95, 256)))
    color_map = {m: cmap(norm(params[m])) for m in models}

    mrr_markers = ["s", "^", "D", "v", "P", "X", "o"]
    marker_map = {m: mrr_markers[i % len(mrr_markers)] for i, m in enumerate(models)}

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.6, ls="-", label="metric"),
        Line2D([0], [0], color="tab:blue", lw=1.3, ls="--", label="mrr"),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    for train_bs in train_batch_sizes:
        for eval_bs in eval_batch_sizes:
            fig, axes = plt.subplots(1, 4, figsize=(18.6, 3.3), sharex=True)
            axes = np.array(axes).ravel()

            all_steps = set()
            for m in models:
                step2m = metrics_by_step.get(m, {}).get(train_bs, {}).get(eval_bs, {})
                all_steps |= set(step2m.keys())
            if not all_steps:
                logger.warning("No metrics for train_bs=%s eval_bs=%s", train_bs, eval_bs)
                plt.close(fig)
                continue
            all_steps = sorted(all_steps)

            mrr_vals_all: List[float] = []
            for m in models:
                step2m = metrics_by_step.get(m, {}).get(train_bs, {}).get(eval_bs, {})
                for s in all_steps:
                    if s in step2m and np.isfinite(step2m[s].get("mrr_deg", np.nan)):
                        mrr_vals_all.append(float(step2m[s]["mrr_deg"]))
            mrr_ylim = None
            if mrr_vals_all:
                mrr_min, mrr_max = min(mrr_vals_all), max(mrr_vals_all)
                pad = 0.05 * (mrr_max - mrr_min + 1e-12)
                mrr_ylim = (mrr_min - pad, mrr_max + pad)

            for ax, (k, metric_name) in zip(axes, panels):
                ax_right = ax.twinx()

                for model_name in models:
                    step2m = metrics_by_step.get(model_name, {}).get(train_bs, {}).get(eval_bs, {})
                    if not step2m:
                        continue
                    xs, ys = [], []
                    for s in all_steps:
                        if s not in step2m:
                            continue
                        v = step2m[s].get(k, None)
                        if v is None or (not np.isfinite(v)):
                            continue
                        xs.append(float(s))
                        ys.append(float(v))
                    if xs:
                        ax.plot(
                            xs, ys,
                            marker="o",
                            linewidth=1.35,
                            markersize=3.2,
                            alpha=0.95,
                            color=color_map[model_name],
                            linestyle="-",
                        )

                for model_name in models:
                    step2m = metrics_by_step.get(model_name, {}).get(train_bs, {}).get(eval_bs, {})
                    if not step2m:
                        continue
                    xs2, mrrs = [], []
                    for s in all_steps:
                        if s not in step2m:
                            continue
                        v = step2m[s].get("mrr_deg", None)
                        if v is None or (not np.isfinite(v)):
                            continue
                        xs2.append(float(s))
                        mrrs.append(float(v))
                    if xs2:
                        ax_right.plot(
                            xs2, mrrs,
                            linestyle="--",
                            marker=marker_map[model_name],
                            markersize=4.0,
                            linewidth=1.15,
                            alpha=0.92,
                            color=color_map[model_name],
                        )

                ax.set_xscale("log", base=10)
                ax.grid(True, alpha=0.25)

                ax.set_ylabel(metric_name, fontsize=12)
                ax.set_xlabel("steps", fontsize=12)

                ax.tick_params(axis="both", labelsize=11)
                ax_right.tick_params(axis="y", labelsize=11, pad=1)
                ax_right.grid(False)
                ax_right.set_ylabel("MRR", fontsize=12, labelpad=6)
                ax_right.yaxis.set_label_position("right")
                if mrr_ylim is not None:
                    ax_right.set_ylim(*mrr_ylim)

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=2,
                frameon=False,
                fontsize=11,
                bbox_to_anchor=(0.5, 0.06),
                handlelength=2.8,
                columnspacing=1.6,
            )

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            fig.subplots_adjust(left=0.06, right=0.92, top=0.88, bottom=0.30, wspace=0.54)

            cbar = fig.colorbar(sm, ax=axes, pad=0.05, fraction=0.035)
            cbar.set_label("Model size (#params)", rotation=90, fontsize=12)
            cbar.ax.tick_params(labelsize=11)

            tick_vals = sorted(set(params.values()))
            cbar.set_ticks(tick_vals)
            cbar.formatter = FuncFormatter(lambda x, pos: _format_params(x))
            cbar.update_ticks()
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            cbar.ax.minorticks_off()

            out_path = out_dir / f"steps_1x4_overlayMRR_t{train_bs}_e{eval_bs}.png"
            fig.savefig(out_path, dpi=dpi)

            pdf_path = out_path.with_suffix(".pdf")
            fig.savefig(pdf_path)

            logger.info("Saved: %s", out_path)
            logger.info("Saved: %s", pdf_path)

            if show:
                plt.show()
            else:
                plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    with open(args.param_count_json, "r", encoding="utf-8") as f:
        param_count_json = json.load(f)

    observations = load_observations(
        lr_grid_root=args.lr_grid_root,
        models=args.models,
        train_batch_sizes=args.train_batch_sizes,
        eval_batch_sizes=args.eval_batch_sizes,
        steps=args.steps,
        bs=args.bs,
        ncs=args.ncs,
        lsn=args.lsn,
        lr=args.lr,
    )
    metrics_by_step = build_metrics_by_step(observations)

    plot_1x4_overlay_mrr(
        metrics_by_step=metrics_by_step,
        out_dir=args.out_dir,
        param_count_json=param_count_json,
        models=args.models,
        train_batch_sizes=args.train_batch_sizes,
        eval_batch_sizes=args.eval_batch_sizes,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
