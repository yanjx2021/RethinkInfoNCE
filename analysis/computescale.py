#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
computescale.py: 2x2 compute-scaling plots in a modelscale-like style.

Outputs (to --out-dir):
  1) computescale_vs_compute.png/.pdf
     x = compute_proxy = non_embedding_params * datasize
     - Per-model lines across subsets
     - Global dashed fit + R^2 for Cross-Entropy and Rank-Entropy only
  2) computescale_vs_mrr.png/.pdf
     x = MRR@eval-bs
     - Per-model lines across subsets
     - No fit lines by default

Example:
  python computescale.py \
    --param-json path/to/bert_count.json \
    --base-dir path/to/hard_negtives \
    --out-dir path/to/figs/scale \
    --models \
      google_bert/uncased_L-2_H-768_A-12 \
      google_bert/uncased_L-4_H-768_A-12 \
    --subsets 1000 3000 6000 10000 30000 60000 100000 502939 \
    --full-subset 502939 \
    --train-bs 16 --bs 32 --lr 3e-6 \
    --eval-bs 256 --ncs 100000 --lsn 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


METRICS_CFG = [
    {"key": "item_cross_entropy", "title": "Cross-Entropy",        "yscale": "log"},
    {"key": "kl_divergence",      "title": "Rank-Score-Alignment", "yscale": None},
    {"key": "ranking_entropy",    "title": "Rank-Entropy",         "yscale": "log"},
    {"key": "log_normalize_num",  "title": "Score-Uncertainty",    "yscale": None},
]

FIT_KEYS = {"item_cross_entropy", "ranking_entropy"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-json", type=Path, required=True, help="path/to/bert_count.json")
    ap.add_argument("--base-dir", type=Path, required=True, help="path/to/hard_negtives")
    ap.add_argument("--out-dir", type=Path, required=True, help="path/to/figs/scale")

    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument(
        "--subsets",
        nargs="+",
        type=int,
        default=[1000, 3000, 6000, 10000, 30000, 60000, 100000, 502939],
    )
    ap.add_argument("--full-subset", type=int, default=502939)

    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=str, default="3e-6")
    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--ncs", type=int, default=100000)
    ap.add_argument("--lsn", type=int, default=1)

    ap.add_argument("--ckpt-divisor", type=int, default=32)
    ap.add_argument("--ckpt-mult", type=int, default=3)

    ap.add_argument("--figw", type=float, default=7.0)
    ap.add_argument("--figh", type=float, default=3.9)
    ap.add_argument("--wspace", type=float, default=0.40)
    ap.add_argument("--hspace", type=float, default=0.34)

    ap.add_argument("--label-font", type=int, default=9)
    ap.add_argument("--tick-font", type=int, default=9)
    ap.add_argument("--r2-font", type=int, default=8)
    ap.add_argument("--r2-x", type=float, default=0.93)
    ap.add_argument("--r2-y", type=float, default=0.93)

    ap.add_argument("--cmap", type=str, default="Blues")
    ap.add_argument("--cmap-min", type=float, default=0.30)
    ap.add_argument("--cmap-max", type=float, default=0.95)
    ap.add_argument("--trunc-min", type=float, default=0.35)
    ap.add_argument("--trunc-max", type=float, default=0.95)

    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-mrr-plot", action="store_true", help="Only output computescale_vs_compute.*")
    return ap.parse_args()


def _extract_num(v: Any) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        return float(v)
    if isinstance(v, dict):
        for k in ["non_embedding_params", "total", "total_params", "num_params", "n_params", "value", "count"]:
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
    ranks0 = np.asarray(rank_list, dtype=np.int64).reshape(-1)
    ce = np.asarray(ce_loss_list, dtype=np.float64).reshape(-1)

    if ranks0.size == 0:
        return {
            "item_cross_entropy": np.nan,
            "ranking_entropy": np.nan,
            "kl_divergence": np.nan,
            "log_normalize_num": np.nan,
            "mrr": np.nan,
        }

    is_zero_based = (ranks0.min() == 0)
    r = ranks0 + 1 if is_zero_based else ranks0
    valid = r > 0
    mrr = float(np.mean(1.0 / r[valid])) if np.any(valid) else np.nan

    max_rank0 = int(ranks0.max())
    rank_cnt = np.bincount(ranks0, minlength=max_rank0 + 1)
    tot = float(rank_cnt.sum())
    rank_prob_full = rank_cnt / (tot + 1e-10)
    ranking_entropy = float(-np.sum(rank_prob_full * np.log(rank_prob_full + 1e-10)))

    nonzero = rank_cnt > 0
    p = rank_prob_full[nonzero]
    p = p / (p.sum() + 1e-10)

    ce_sum_by_rank = np.bincount(ranks0, weights=ce, minlength=rank_cnt.size)
    grouped_ce = ce_sum_by_rank[nonzero] / rank_cnt[nonzero]

    grouped_probs = np.exp(-grouped_ce)
    normalize_num = float(np.sum(grouped_probs))
    grouped_probs = grouped_probs / (normalize_num + 1e-10)

    kl_divergence = float(np.sum(p * np.log((p + 1e-10) / (grouped_probs + 1e-10))))
    log_normalize_num = float(-np.log(normalize_num + 1e-12))

    return {
        "item_cross_entropy": float(np.mean(ce)),
        "ranking_entropy": float(ranking_entropy),
        "kl_divergence": float(kl_divergence),
        "log_normalize_num": float(log_normalize_num),
        "mrr": float(mrr),
    }


def build_observe_path(
    base_dir: Path,
    model_name: str,
    *,
    subset: int,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
    ckpt_divisor: int,
    ckpt_mult: int,
) -> Path:
    ckpt_step = subset // ckpt_divisor * ckpt_mult
    run_root = base_dir / model_name / f"gpu1_neg{train_bs}.{train_bs}_bs{bs}_lr{lr}"
    if subset == full_subset:
        root = run_root / f"checkpoint-{ckpt_step}"
    else:
        root = run_root / f"subset{subset}" / f"checkpoint-{ckpt_step}"
    return root / "observe" / f"lbs{eval_bs}_ncs{ncs}_lsn{lsn}" / "result.msmarco-passage.dev.npz"


def load_metrics(
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
    ckpt_divisor: int,
    ckpt_mult: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model in models:
        per_model: Dict[int, Dict[str, float]] = {}
        for subset in subsets:
            p = build_observe_path(
                base_dir,
                model,
                subset=subset,
                full_subset=full_subset,
                train_bs=train_bs,
                bs=bs,
                lr=lr,
                eval_bs=eval_bs,
                ncs=ncs,
                lsn=lsn,
                ckpt_divisor=ckpt_divisor,
                ckpt_mult=ckpt_mult,
            )
            if not p.exists():
                print(f"[WARN] Missing npz: {p}")
                continue
            data = np.load(p)
            per_model[subset] = decompose_result(data["rank"].reshape(-1), data["loss"].reshape(-1))
            print(f"[OK] {model} | subset={subset} | mrr={per_model[subset]['mrr']:.6f}")
        out[model] = per_model
    return out


def build_model_color_map_rank_uniform(
    models: List[str],
    model_param_count: dict,
    *,
    cmap_name: str,
    cmap_min: float,
    cmap_max: float,
    trunc_min: float,
    trunc_max: float,
) -> Tuple[Dict[str, Any], List[str]]:
    base = plt.get_cmap(cmap_name)
    cmap = LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base(np.linspace(trunc_min, trunc_max, 256)),
    )

    pairs: List[Tuple[str, float]] = []
    missing: List[str] = []
    for m in models:
        s = model_param_count.get(m, {}).get("non_embedding_params", None)
        if s is None:
            missing.append(m)
        else:
            pairs.append((m, float(_extract_num(s))))

    if not pairs:
        mid = 0.5 * (cmap_min + cmap_max)
        return {m: cmap(mid) for m in models}, list(models)

    pairs.sort(key=lambda x: x[1])
    n = len(pairs)
    ts = [0.5 * (cmap_min + cmap_max)] if n == 1 else np.linspace(cmap_min, cmap_max, n)

    color_map = {m: cmap(t) for (m, _), t in zip(pairs, ts)}
    mid = 0.5 * (cmap_min + cmap_max)
    for m in missing:
        color_map[m] = cmap(mid)

    models_sorted = [m for m, _ in pairs] + missing
    return color_map, models_sorted


def compute_proxy(model_param_count: dict, model_name: str, datasize: int) -> Optional[float]:
    raw = model_param_count.get(model_name, {}).get("non_embedding_params", None)
    if raw is None:
        return None
    return float(_extract_num(raw)) * float(datasize)


def plot_fit_with_r2(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    xscale: Optional[str],
    yscale: Optional[str],
    r2_pos: Tuple[float, float],
    r2_font: int,
) -> None:
    mask = np.isfinite(xs) & np.isfinite(ys)
    if xscale == "log":
        mask &= (xs > 0)
    if yscale == "log":
        mask &= (ys > 0)

    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return

    x_feat = np.log(xs) if xscale == "log" else xs
    y_feat = np.log(ys) if yscale == "log" else ys

    a, b = np.polyfit(x_feat, y_feat, 1)
    y_hat = a * x_feat + b

    ss_res = float(np.sum((y_feat - y_hat) ** 2))
    ss_tot = float(np.sum((y_feat - float(np.mean(y_feat))) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    if xscale == "log":
        x_line = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 200)
        x_line_feat = np.log(x_line)
    else:
        x_line = np.linspace(xs.min(), xs.max(), 200)
        x_line_feat = x_line

    y_line_feat = a * x_line_feat + b
    y_line = np.exp(y_line_feat) if yscale == "log" else y_line_feat

    ax.plot(x_line, y_line, linestyle="--", linewidth=1.2, color="0.55", alpha=0.9, zorder=1)
    ax.text(
        r2_pos[0],
        r2_pos[1],
        rf"$R^2$={r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=r2_font,
        color="0.25",
    )


def plot_2x2_lines(
    metrics_dict: Dict[str, Dict[int, Dict[str, float]]],
    *,
    models: List[str],
    subsets: List[int],
    models_sorted: List[str],
    model_color_map: Dict[str, Any],
    xlabel: str,
    xscale: Optional[str],
    x_getter,  # (model, metrics, subset) -> x or None
    out_path: Path,
    figsize: Tuple[float, float],
    add_global_fit: bool,
    wspace: float,
    hspace: float,
    label_font: int,
    tick_font: int,
    r2_pos: Tuple[float, float],
    r2_font: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    ax_list = axes.ravel()

    for ax, cfg in zip(ax_list, METRICS_CFG):
        key = cfg["key"]
        name = cfg["title"]
        yscale = cfg["yscale"]

        all_x: List[float] = []
        all_y: List[float] = []

        for model in models_sorted:
            if model not in models:
                continue
            per_model = metrics_dict.get(model, {})
            pts: List[Tuple[float, float]] = []
            for subset in subsets:
                m = per_model.get(subset, None)
                if m is None:
                    continue
                yv = m.get(key, None)
                if yv is None or (not np.isfinite(yv)):
                    continue
                xv = x_getter(model, m, subset)
                if xv is None or (not np.isfinite(xv)):
                    continue
                pts.append((float(xv), float(yv)))

            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            xs, ys = zip(*pts)

            ax.plot(
                xs,
                ys,
                marker="o",
                linestyle="-",
                linewidth=1.2,
                markersize=2.8,
                alpha=0.95,
                color=model_color_map.get(model, "0.4"),
            )

            if add_global_fit and key in FIT_KEYS:
                all_x.extend(list(xs))
                all_y.extend(list(ys))

        if add_global_fit and key in FIT_KEYS:
            plot_fit_with_r2(
                ax,
                np.asarray(all_x, dtype=np.float64),
                np.asarray(all_y, dtype=np.float64),
                xscale=xscale,
                yscale=yscale,
                r2_pos=r2_pos,
                r2_font=r2_font,
            )

        ax.set_title("")
        ax.set_xlabel(xlabel, fontsize=label_font)
        ax.set_ylabel(name, fontsize=label_font)

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_font)

    fig.tight_layout()
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {out_path}")
    print(f"[SAVE] {out_path.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()

    with open(args.param_json, "r", encoding="utf-8") as f:
        model_param_count = json.load(f)

    model_color_map, models_sorted = build_model_color_map_rank_uniform(
        args.models,
        model_param_count,
        cmap_name=args.cmap,
        cmap_min=args.cmap_min,
        cmap_max=args.cmap_max,
        trunc_min=args.trunc_min,
        trunc_max=args.trunc_max,
    )

    metrics_dict = load_metrics(
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
        ckpt_divisor=args.ckpt_divisor,
        ckpt_mult=args.ckpt_mult,
    )

    out1 = args.out_dir / "computescale_vs_compute.png"
    plot_2x2_lines(
        metrics_dict,
        models=args.models,
        subsets=args.subsets,
        models_sorted=models_sorted,
        model_color_map=model_color_map,
        x_getter=lambda model, _m, subset: compute_proxy(model_param_count, model, subset),
        xlabel="Compute proxy (#params × datasize)",
        xscale="log",
        out_path=out1,
        figsize=(args.figw, args.figh),
        add_global_fit=True,
        wspace=args.wspace,
        hspace=args.hspace,
        label_font=args.label_font,
        tick_font=args.tick_font,
        r2_pos=(args.r2_x, args.r2_y),
        r2_font=args.r2_font,
        dpi=args.dpi,
    )

    if not args.no_mrr_plot:
        out2 = args.out_dir / "computescale_vs_mrr.png"
        plot_2x2_lines(
            metrics_dict,
            models=args.models,
            subsets=args.subsets,
            models_sorted=models_sorted,
            model_color_map=model_color_map,
            x_getter=lambda _model, m, _subset: m.get("mrr", None),
            xlabel=f"MRR@{args.eval_bs}",
            xscale=None,
            out_path=out2,
            figsize=(args.figw, args.figh),
            add_global_fit=False,
            wspace=max(0.30, args.wspace),
            hspace=args.hspace,
            label_font=args.label_font,
            tick_font=args.tick_font,
            r2_pos=(args.r2_x, args.r2_y),
            r2_font=args.r2_font,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

