#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
modelscale.py (final checkpoint only): 2x2 scatter plots + optional power-law fits.

- Loads ONLY the final checkpoint (full_subset) for each model.
- Produces:
  (1) metrics vs model size (non-embedding params)
  (2) metrics vs MRR@eval (optional)

Fit/plot policy:
- A dashed fit curve + R^2 is drawn ONLY when cfg["fit_yscale"] == "log".
- If a metric can be negative but you still want log-fit, set:
    fit_y_sign = -1 (fit on -y) and line_y_sign = -1 (plot back on negative axis).

Example:
  python modelscale.py \
    --param-json path/to/bert_count.json \
    --base-dir path/to/hard_negtives \
    --out-dir path/to/figs/scale \
    --models \
      google_bert/uncased_L-2_H-768_A-12 google_bert/uncased_L-4_H-768_A-12 \
      google_bert/uncased_L-6_H-768_A-12 google_bert/uncased_L-8_H-768_A-12 \
    --full-subset 502939 --train-bs 16 --bs 32 --lr 3e-6 \
    --eval-bs 256 --ncs 100000 --lsn 1 \
    --plot-mrr --fit-mode-mrrx linear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Metric config:
# axis_yscale: subplot y-axis scale
# fit_yscale : regression y-space (only "log" draws fit+R^2)
# fit_y_sign : multiply y by this sign before fitting
# line_y_sign: multiply fitted curve by this sign when plotting back
METRICS_CFG = [
    dict(
        key="item_cross_entropy",
        title="Cross-Entropy",
        axis_yscale="log",
        fit_yscale="log",
        fit_y_sign=1.0,
        line_y_sign=1.0,
    ),
    dict(
        key="kl_divergence",
        title="Rank-Score-Alignment",
        axis_yscale=None,
        fit_yscale=None,
    ),
    dict(
        key="ranking_entropy",
        title="Rank-Entropy",
        axis_yscale="log",
        fit_yscale="log",
        fit_y_sign=1.0,
        line_y_sign=1.0,
    ),
    dict(
        key="log_normalize_num",
        title="Score-Uncertainty",
        axis_yscale=None,
        fit_yscale=None,
        # If you ever want log-fit on negative values:
        # fit_yscale="log", fit_y_sign=-1.0, line_y_sign=-1.0
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-json", type=Path, required=True, help="path/to/bert_count.json")
    ap.add_argument("--base-dir", type=Path, required=True, help="path/to/hard_negtives")
    ap.add_argument("--out-dir", type=Path, required=True, help="path/to/figs/scale")

    ap.add_argument("--models", nargs="+", required=True)

    ap.add_argument("--full-subset", type=int, default=502939)
    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=str, default="3e-6")
    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--ncs", type=int, default=100000)
    ap.add_argument("--lsn", type=int, default=1)

    ap.add_argument("--plot-mrr", action="store_true", help="Also plot metrics vs MRR@eval")
    ap.add_argument("--fit-mode-mrrx", choices=["linear", "logx"], default="linear")

    ap.add_argument("--figw", type=float, default=7.0)
    ap.add_argument("--figh", type=float, default=3.9)
    ap.add_argument("--wspace", type=float, default=0.42)
    ap.add_argument("--hspace", type=float, default=0.38)

    ap.add_argument("--label-font", type=int, default=9)
    ap.add_argument("--tick-font", type=int, default=9)
    ap.add_argument("--r2-font", type=int, default=8)
    ap.add_argument("--r2-x", type=float, default=0.93)  # a bit inward (vs 0.96)
    ap.add_argument("--r2-y", type=float, default=0.93)

    ap.add_argument("--cmap", type=str, default="Blues")
    ap.add_argument("--cmap-min", type=float, default=0.30)
    ap.add_argument("--cmap-max", type=float, default=0.95)
    ap.add_argument("--trunc-min", type=float, default=0.35)
    ap.add_argument("--trunc-max", type=float, default=0.95)

    ap.add_argument("--dpi", type=int, default=300)
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
    rank_list = np.asarray(rank_list, dtype=np.int64).reshape(-1)
    ce_loss_list = np.asarray(ce_loss_list, dtype=np.float64).reshape(-1)

    # robust MRR: support either 1-based ranks (>=1) or 0-based ranks (>=0)
    if rank_list.size == 0:
        return {k["key"]: np.nan for k in METRICS_CFG} | {"mrr": np.nan}

    is_zero_based = (rank_list.min() == 0)
    r = rank_list + 1 if is_zero_based else rank_list
    valid = r > 0
    mrr = float(np.mean(1.0 / r[valid])) if np.any(valid) else np.nan

    max_rank0 = int(rank_list.max())
    rank_cnt_array = np.bincount(rank_list, minlength=max_rank0 + 1)
    tot = float(rank_cnt_array.sum())
    rank_prob_full = rank_cnt_array / (tot + 1e-10)
    ranking_entropy = float(-np.sum(rank_prob_full * np.log(rank_prob_full + 1e-10)))

    nonzero = rank_cnt_array > 0
    p = rank_prob_full[nonzero]
    p = p / (p.sum() + 1e-10)

    grouped_ce_sum = np.bincount(rank_list, weights=ce_loss_list, minlength=rank_cnt_array.size)
    grouped_ce = grouped_ce_sum[nonzero] / rank_cnt_array[nonzero]

    grouped_probs = np.exp(-grouped_ce)
    normalize_num = float(np.sum(grouped_probs))
    grouped_probs = grouped_probs / (normalize_num + 1e-10)

    kl_divergence = float(np.sum(p * np.log((p + 1e-10) / (grouped_probs + 1e-10))))
    log_normalize_num = float(-np.log(normalize_num + 1e-10))

    return {
        "item_cross_entropy": float(np.mean(ce_loss_list)),
        "ranking_entropy": float(ranking_entropy),
        "kl_divergence": float(kl_divergence),
        "log_normalize_num": float(log_normalize_num),
        "mrr": float(mrr),
    }


def build_observe_path_final(
    base_dir: Path,
    model_name: str,
    *,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
) -> Path:
    ckpt_step = full_subset // 32 * 3
    root = base_dir / model_name / f"gpu1_neg{train_bs}.{train_bs}_bs{bs}_lr{lr}" / f"checkpoint-{ckpt_step}"
    return root / "observe" / f"lbs{eval_bs}_ncs{ncs}_lsn{lsn}" / "result.msmarco-passage.dev.npz"


def load_final_metrics(
    models: List[str],
    *,
    base_dir: Path,
    full_subset: int,
    train_bs: int,
    bs: int,
    lr: str,
    eval_bs: int,
    ncs: int,
    lsn: int,
) -> Dict[str, Dict[str, float]]:
    final_metrics: Dict[str, Dict[str, float]] = {}
    for model in models:
        p = build_observe_path_final(
            base_dir, model,
            full_subset=full_subset, train_bs=train_bs, bs=bs, lr=lr,
            eval_bs=eval_bs, ncs=ncs, lsn=lsn,
        )
        if not p.exists():
            print(f"[WARN] Missing npz: {p}")
            continue
        data = np.load(p)
        final_metrics[model] = decompose_result(data["rank"].reshape(-1), data["loss"].reshape(-1))
        print(f"[OK] {model} | mrr={final_metrics[model]['mrr']:.6f}")
    return final_metrics


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
        raise ValueError("No valid non_embedding_params found in param json for provided models.")

    pairs.sort(key=lambda x: x[1])  # small -> large
    n = len(pairs)
    ts = [0.5 * (cmap_min + cmap_max)] if n == 1 else np.linspace(cmap_min, cmap_max, n)
    color_map = {m: cmap(t) for (m, _), t in zip(pairs, ts)}

    mid = 0.5 * (cmap_min + cmap_max)
    for m in missing:
        color_map[m] = cmap(mid)

    models_sorted = [m for m, _ in pairs] + missing
    return color_map, models_sorted


def fit_line_with_r2(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    fit_mode: str,
    yscale: Optional[str],
    n: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    mask = np.isfinite(xs) & np.isfinite(ys)
    if fit_mode == "logx":
        mask &= (xs > 0)
    if yscale == "log":
        mask &= (ys > 0)

    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return None

    x_feat = np.log(xs) if fit_mode == "logx" else xs
    y_feat = np.log(ys) if yscale == "log" else ys

    a, b = np.polyfit(x_feat, y_feat, 1)
    y_hat = a * x_feat + b

    ss_res = float(np.sum((y_feat - y_hat) ** 2))
    ss_tot = float(np.sum((y_feat - np.mean(y_feat)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    x_line = np.linspace(xs.min(), xs.max(), n)
    x_line_feat = np.log(x_line) if fit_mode == "logx" else x_line
    y_line_feat = a * x_line_feat + b
    y_line = np.exp(y_line_feat) if yscale == "log" else y_line_feat

    return x_line, y_line, r2


def collect_points_modelsize(
    final_metrics: Dict[str, Dict[str, float]],
    model_param_count: dict,
    models_sorted: List[str],
    metric_key: str,
) -> List[Tuple[float, float, str]]:
    pts = []
    for m in models_sorted:
        if m not in final_metrics:
            continue
        x_raw = model_param_count.get(m, {}).get("non_embedding_params", None)
        if x_raw is None:
            continue
        x = _extract_num(x_raw)
        y = final_metrics[m].get(metric_key, None)
        if y is None or (not np.isfinite(y)):
            continue
        pts.append((float(x), float(y), m))
    pts.sort(key=lambda p: p[0])
    return pts


def collect_points_mrr(
    final_metrics: Dict[str, Dict[str, float]],
    models_sorted: List[str],
    metric_key: str,
) -> List[Tuple[float, float, str]]:
    pts = []
    for m in models_sorted:
        if m not in final_metrics:
            continue
        x = final_metrics[m].get("mrr", None)
        y = final_metrics[m].get(metric_key, None)
        if x is None or y is None:
            continue
        if (not np.isfinite(x)) or (not np.isfinite(y)):
            continue
        pts.append((float(x), float(y), m))
    pts.sort(key=lambda p: p[0])
    return pts


def plot_2x2_scatter_with_fit(
    pts_collector,
    *,
    metric_cfg: List[dict],
    model_color_map: Dict[str, Any],
    xlabel: str,
    xscale: Optional[str],
    out_path: Path,
    figsize: Tuple[float, float],
    fit_mode: str,
    wspace: float,
    hspace: float,
    label_font: int,
    tick_font: int,
    r2_font: int,
    r2_pos: Tuple[float, float],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    ax_list = axes.ravel()

    for ax, cfg in zip(ax_list, metric_cfg):
        key = cfg["key"]
        name = cfg["title"]
        axis_yscale = cfg.get("axis_yscale", None)
        fit_yscale = cfg.get("fit_yscale", None)
        fit_y_sign = float(cfg.get("fit_y_sign", 1.0))
        line_y_sign = float(cfg.get("line_y_sign", 1.0))

        pts = pts_collector(key)
        if not pts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=label_font)
            ax.set_axis_off()
            continue

        xs_all = np.array([p[0] for p in pts], dtype=float)
        ys_all = np.array([p[1] for p in pts], dtype=float)
        ms_all = [p[2] for p in pts]

        scatter_mask = np.isfinite(xs_all) & np.isfinite(ys_all)
        if xscale == "log":
            scatter_mask &= (xs_all > 0)
        if axis_yscale == "log":
            scatter_mask &= (ys_all > 0)

        xs = xs_all[scatter_mask]
        ys = ys_all[scatter_mask]
        ms = [m for m, keep in zip(ms_all, scatter_mask) if keep]

        if xs.size == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes, fontsize=label_font)
            ax.set_axis_off()
            continue

        ax.scatter(
            xs,
            ys,
            s=32,
            c=[model_color_map[m] for m in ms],
            edgecolors="black",
            linewidths=0.30,
            alpha=0.95,
            zorder=3,
        )

        if fit_yscale == "log":
            ys_fit = ys * fit_y_sign
            fit = fit_line_with_r2(xs, ys_fit, fit_mode=fit_mode, yscale="log", n=200)
            if fit is not None:
                x_line, y_line_fit, r2 = fit
                y_line_plot = y_line_fit * line_y_sign
                ax.plot(x_line, y_line_plot, linestyle="--", linewidth=1.2, color="0.55", alpha=0.9, zorder=2)
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

        ax.set_title("")
        ax.set_xlabel(xlabel, fontsize=label_font)
        ax.set_ylabel(name, fontsize=label_font)

        if xscale is not None:
            ax.set_xscale(xscale)
        if axis_yscale is not None:
            ax.set_yscale(axis_yscale)

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

    final_metrics = load_final_metrics(
        models=args.models,
        base_dir=args.base_dir,
        full_subset=args.full_subset,
        train_bs=args.train_bs,
        bs=args.bs,
        lr=args.lr,
        eval_bs=args.eval_bs,
        ncs=args.ncs,
        lsn=args.lsn,
    )

    model_color_map, models_sorted = build_model_color_map_rank_uniform(
        args.models,
        model_param_count,
        cmap_name=args.cmap,
        cmap_min=args.cmap_min,
        cmap_max=args.cmap_max,
        trunc_min=args.trunc_min,
        trunc_max=args.trunc_max,
    )

    out1 = args.out_dir / "modelscale_vs_modelsize.png"
    plot_2x2_scatter_with_fit(
        pts_collector=lambda metric_key: collect_points_modelsize(
            final_metrics, model_param_count, models_sorted, metric_key
        ),
        metric_cfg=METRICS_CFG,
        model_color_map=model_color_map,
        xlabel="Model size (non-embedding)",
        xscale="log",
        out_path=out1,
        figsize=(args.figw, args.figh),
        fit_mode="logx",
        wspace=args.wspace,
        hspace=args.hspace,
        label_font=args.label_font,
        tick_font=args.tick_font,
        r2_font=args.r2_font,
        r2_pos=(args.r2_x, args.r2_y),
        dpi=args.dpi,
    )

    if args.plot_mrr:
        out2 = args.out_dir / "modelscale_vs_mrr.png"
        plot_2x2_scatter_with_fit(
            pts_collector=lambda metric_key: collect_points_mrr(final_metrics, models_sorted, metric_key),
            metric_cfg=METRICS_CFG,
            model_color_map=model_color_map,
            xlabel=f"MRR@{args.eval_bs}",
            xscale=None,
            out_path=out2,
            figsize=(args.figw, args.figh),
            fit_mode=("logx" if args.fit_mode_mrrx == "logx" else "linear"),
            wspace=args.wspace,
            hspace=args.hspace,
            label_font=args.label_font,
            tick_font=args.tick_font,
            r2_font=args.r2_font,
            r2_pos=(args.r2_x, args.r2_y),
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
