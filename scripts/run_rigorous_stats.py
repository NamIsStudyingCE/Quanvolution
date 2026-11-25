"""Step 2: Rigorous Statistical Analysis (Paired t-test, 95%/90% CI, TOST ±2%, Benjamini-Hochberg FDR).

Reads strictly read-only from:
  - weights/pillar2_wide/ (for exact per-seed recomputation of Table 4.2 asymmetric isolation)
  - results/ablation_clut/eval_results.json
  - results/ablation_clut/extra_baselines.json
  - reports/pareto_int8_vs_interp4d.json

Writes ONLY to:
  - reports/rigorous_stats_summary.json
  - reports/rigorous_stats_summary.md
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import experiments.* before adding src/ so ./experiments is not shadowed by src/experiments.
from experiments.run_pillar2_expansion import NOISE_DRAWS, SEEDS, SIGMAS, speckle  # noqa: E402
from experiments.run_pillar2_wide import DATASETS, DEV, load  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gr_quanv import phase_damping_u8  # noqa: E402
from models.gr_quanv_wide import WideClassicalControl, WideGRQuanvNet  # noqa: E402
from scripts.generate_clut import extract_multiscale_from_lut_tensor, load_lut_table  # noqa: E402
from utils.metrics import calculate_metrics  # noqa: E402

EVAL_CLUT_JSON = PROJECT_ROOT / "results" / "ablation_clut" / "eval_results.json"
EXTRA_BASE_JSON = PROJECT_ROOT / "results" / "ablation_clut" / "extra_baselines.json"
PARETO_JSON = PROJECT_ROOT / "reports" / "pareto_int8_vs_interp4d.json"
LOCKED_WIDE_DIR = PROJECT_ROOT / "weights" / "pillar2_wide"

OUT_JSON = PROJECT_ROOT / "reports" / "rigorous_stats_summary.json"
OUT_MD = PROJECT_ROOT / "reports" / "rigorous_stats_summary.md"

TOST_MARGIN = 2.0  # ±2.0 percentage points
ALPHA = 0.05       # 95% confidence (90% CI for TOST)


def paired_stats_with_tost(
    vals_a: List[float],
    vals_b: List[float],
    margin: float = TOST_MARGIN,
    alpha: float = ALPHA,
) -> dict:
    """Computes paired two-sided t-test, 95% CI, 90% CI, Cohen's dz, and paired TOST (±margin)."""
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    assert len(a) == len(b) and len(a) >= 2, f"Invalid paired lengths: {len(a)} vs {len(b)}"
    n = int(len(a))
    df = n - 1
    diff = a - b
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1))
    se_d = sd_d / math.sqrt(n)
    wins_a = int(np.sum(diff > 1e-12))
    wins_b = int(np.sum(diff < -1e-12))

    if se_d < 1e-12:
        t_stat = 0.0 if abs(mean_d) < 1e-12 else (math.copysign(math.inf, mean_d))
        p_raw = 1.0 if abs(mean_d) < 1e-12 else 0.0
        ci95 = (mean_d, mean_d)
        ci90 = (mean_d, mean_d)
        tost_p1 = 0.0 if mean_d > -margin else 1.0
        tost_p2 = 0.0 if mean_d < margin else 1.0
        p_tost = max(tost_p1, tost_p2)
        cohen_dz = 0.0
    else:
        t_stat = float(mean_d / se_d)
        p_raw = float(2.0 * stats.t.sf(abs(t_stat), df))
        t_crit95 = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        t_crit90 = float(stats.t.ppf(1.0 - alpha, df))
        ci95 = (float(mean_d - t_crit95 * se_d), float(mean_d + t_crit95 * se_d))
        ci90 = (float(mean_d - t_crit90 * se_d), float(mean_d + t_crit90 * se_d))

        # TOST H01: mu_d <= -margin (upper tail); H02: mu_d >= +margin (lower tail)
        t1 = (mean_d + margin) / se_d
        t2 = (mean_d - margin) / se_d
        tost_p1 = float(stats.t.sf(t1, df))
        tost_p2 = float(stats.t.cdf(t2, df))
        p_tost = float(max(tost_p1, tost_p2))
        cohen_dz = float(mean_d / sd_d)

    equiv = bool(p_tost < alpha)
    return {
        "n": n,
        "df": df,
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=0)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=0)),
        "delta_mean": mean_d,
        "delta_sd_sample": sd_d,
        "se": float(se_d),
        "t_stat": t_stat if math.isfinite(t_stat) else None,
        "p_raw": p_raw,
        "ci95_lo": ci95[0],
        "ci95_hi": ci95[1],
        "ci90_lo": ci90[0],
        "ci90_hi": ci90[1],
        "cohen_dz": cohen_dz,
        "tost_margin": margin,
        "tost_p1": tost_p1,
        "tost_p2": tost_p2,
        "p_tost": p_tost,
        "tost_equivalent": equiv,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "diffs": [float(x) for x in diff],
    }


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """Exact Benjamini-Hochberg step-up FDR adjustment (matches scipy.stats.false_discovery_control)."""
    if not pvals:
        return []
    arr = np.asarray(pvals, dtype=np.float64)
    arr = np.clip(arr, 0.0, 1.0)
    return [float(x) for x in stats.false_discovery_control(arr, method="bh")]


@torch.no_grad()
def _eval_batch_acc_f1_auc(model: torch.nn.Module, x_t: torch.Tensor, q_t: torch.Tensor | None, y_np: np.ndarray) -> Tuple[float, float, float]:
    preds = []
    bs = 2048
    for i in range(0, len(x_t), bs):
        xb = x_t[i : i + bs]
        qb = q_t[i : i + bs] if q_t is not None else None
        preds.append(torch.softmax(model(xb, qb), dim=1).cpu().numpy())
    p = np.concatenate(preds, axis=0)
    m = calculate_metrics(y_np, p, p.argmax(axis=1), 2)
    return float(m["acc"] * 100.0), float(m["auc"] * 100.0), float(m["f1"] * 100.0)


def recompute_table42_per_seed() -> dict:
    """Recomputes exact 5-seed metrics (read-only) for the 4 original conditions of Table 4.2
    in `eval_ablation_isolation.py` so exact paired tests (4) vs (2) and (3) vs (1) are available."""
    ds = "pneumoniamnist"
    D = load(DATASETS[ds][0])
    te_f = D["te"].astype(np.float32) / 255.0
    y_test = D["yte"].astype(np.int64)

    models_ctrl = {}
    models_quanv = {}
    for s in SEEDS:
        cp_c = LOCKED_WIDE_DIR / f"wide_v2_interp4d_{ds}_full_control_s{s}.pt"
        cp_q = LOCKED_WIDE_DIR / f"wide_v2_interp4d_{ds}_full_grquanv_s{s}.pt"
        mc = WideClassicalControl(num_classes=2).to(DEV)
        mc.load_state_dict(torch.load(cp_c, map_location=DEV, weights_only=True))
        mc.eval()
        models_ctrl[s] = mc

        mq = WideGRQuanvNet(num_classes=2).to(DEV)
        mq.load_state_dict(torch.load(cp_q, map_location=DEV, weights_only=True))
        mq.eval()
        models_quanv[s] = mq

    conditions = ["1_ctrl_raw", "2_ctrl_damped", "3_grquanv_raw", "4_grquanv_damped_asym"]
    out = {cond: {} for cond in conditions}
    qlut_table = load_lut_table(
        PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_fp32_grid.bin",
        mode="interp4d",
        device=DEV,
    )

    for sg in SIGMAS:
        sg_str = f"{sg:.2f}"
        draws = NOISE_DRAWS if sg > 0 else [0]
        draw_buf = {cond: {s: [] for s in SEEDS} for cond in conditions}

        for dr in draws:
            noisy_u8 = speckle(te_f, sg, dr)
            x_raw_t = ((torch.from_numpy(noisy_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5).to(DEV)

            noisy_damped_u8 = phase_damping_u8(noisy_u8, beta=18.0, tau=0.024)
            x_damped_t = ((torch.from_numpy(noisy_damped_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5).to(DEV)

            q_raw_t = extract_multiscale_from_lut_tensor(
                noisy_u8, qlut_table, mode="interp4d", phase_damping=False
            ).float()
            q_damped_t = extract_multiscale_from_lut_tensor(
                noisy_damped_u8, qlut_table, mode="interp4d", phase_damping=False
            ).float()

            for s in SEEDS:
                models_ctrl[s].stabilize_tau = 999.0
                draw_buf["1_ctrl_raw"][s].append(_eval_batch_acc_f1_auc(models_ctrl[s], x_raw_t, None, y_test))

                models_ctrl[s].stabilize_tau = 0.024
                draw_buf["2_ctrl_damped"][s].append(_eval_batch_acc_f1_auc(models_ctrl[s], x_damped_t, None, y_test))

                models_quanv[s].stabilize_tau = 999.0
                draw_buf["3_grquanv_raw"][s].append(_eval_batch_acc_f1_auc(models_quanv[s], x_raw_t, q_raw_t, y_test))

                models_quanv[s].stabilize_tau = 0.024
                draw_buf["4_grquanv_damped_asym"][s].append(
                    _eval_batch_acc_f1_auc(models_quanv[s], x_raw_t, q_damped_t, y_test)
                )

        for cond in conditions:
            per_seed = {
                str(s): {
                    "acc": float(np.mean([it[0] for it in draw_buf[cond][s]])),
                    "auc": float(np.mean([it[1] for it in draw_buf[cond][s]])),
                    "f1": float(np.mean([it[2] for it in draw_buf[cond][s]])),
                }
                for s in SEEDS
            }
            accs = [per_seed[str(s)]["acc"] for s in SEEDS]
            aucs = [per_seed[str(s)]["auc"] for s in SEEDS]
            f1s = [per_seed[str(s)]["f1"] for s in SEEDS]
            out[cond][sg_str] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "auc_mean": float(np.mean(aucs)),
                "f1_mean": float(np.mean(f1s)),
                "per_seed": per_seed,
            }
    return out


def _sig_tag(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _tost_verdict(st: dict) -> str:
    if st["tost_equivalent"]:
        return f"EQUIV (p={st['p_tost']:.4f})"
    if st["p_raw"] < 0.05:
        direction = "A>B" if st["delta_mean"] > 0 else "A<B"
        return f"DIFF {direction} (p_tost={st['p_tost']:.3f})"
    return f"INCONCLUSIVE (p_tost={st['p_tost']:.3f})"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.perf_counter()
    with open(EVAL_CLUT_JSON, "r", encoding="utf-8") as f:
        eval_clut = json.load(f)
    with open(EXTRA_BASE_JSON, "r", encoding="utf-8") as f:
        extra_base = json.load(f)
    with open(PARETO_JSON, "r", encoding="utf-8") as f:
        pareto = json.load(f)

    seed_strs = [str(s) for s in SEEDS]
    sigma_strs = [f"{sg:.2f}" for sg in SIGMAS]

    print("[1/4] Recomputing exact 5-seed Table 4.2 (asymmetric isolation) from locked checkpoints...")
    t42 = recompute_table42_per_seed()

    # Collect all hypothesis test records for family-level and global BH correction.
    all_tests: List[dict] = []

    def add_test(
        section: str,
        family: str,
        dataset: str,
        mode: str,
        stem: str,
        sigma: str,
        pair_label: str,
        vals_a: List[float],
        vals_b: List[float],
    ) -> dict:
        st = paired_stats_with_tost(vals_a, vals_b, margin=TOST_MARGIN, alpha=ALPHA)
        rec = {
            "section": section,
            "family": family,
            "dataset": dataset,
            "mode": mode,
            "stem": stem,
            "sigma": sigma,
            "pair": pair_label,
            **st,
        }
        all_tests.append(rec)
        return rec

    print("[2/4] Computing paired tests + TOST for Section A (Table 4.2 original asymmetric isolation)...")
    for sg in sigma_strs:
        v1 = [t42["1_ctrl_raw"][sg]["per_seed"][s]["acc"] for s in seed_strs]
        v2 = [t42["2_ctrl_damped"][sg]["per_seed"][s]["acc"] for s in seed_strs]
        v3 = [t42["3_grquanv_raw"][sg]["per_seed"][s]["acc"] for s in seed_strs]
        v4 = [t42["4_grquanv_damped_asym"][sg]["per_seed"][s]["acc"] for s in seed_strs]

        add_test("A_table42_asym", "t42_4_vs_2", "pneumoniamnist", "interp4d", "asym_damped", sg, "(4) Quanv+PreLUT vs (2) Ctrl+Damp", v4, v2)
        add_test("A_table42_asym", "t42_3_vs_1", "pneumoniamnist", "interp4d", "raw_stem", sg, "(3) Quanv Raw vs (1) Ctrl Raw", v3, v1)
        add_test("A_table42_asym", "t42_2_vs_1", "pneumoniamnist", "interp4d", "damping_effect", sg, "(2) Ctrl+Damp vs (1) Ctrl Raw", v2, v1)

    print("[3/4] Computing paired tests + TOST for Section B (7-branch full symmetric ablation)...")
    # Helper to extract per-seed ACC list for any of the 7 models
    def get_model_accs(ds: str, mode: str, stem: str, sg: str, model_key: str) -> List[float]:
        if model_key in ("ctrl", "clut_septrig", "clut_randmlp", "qlut"):
            node = eval_clut["datasets"][ds]["results"][mode][stem][sg][model_key]["per_seed"]
        else:
            node = extra_base["datasets"][ds]["results"][stem][sg][model_key]["per_seed"]
        return [float(node[s]["acc"]) for s in seed_strs]

    b_pairs = [
        ("qlut_vs_ctrl", "Q-LUT vs Ctrl (Conv2d)", "qlut", "ctrl"),
        ("qlut_vs_clut_randmlp", "Q-LUT vs CLUT-RandMLP", "qlut", "clut_randmlp"),
        ("qlut_vs_clut_septrig", "Q-LUT vs CLUT-SepTrig", "qlut", "clut_septrig"),
        ("qlut_vs_ctrl_tanh", "Q-LUT vs Conv2d+tanh", "qlut", "ctrl_tanh"),
        ("qlut_vs_ctrl_hardclip", "Q-LUT vs Conv2d+hardclip", "qlut", "ctrl_hardclip"),
        ("qlut_vs_ctrl_noise_aug", "Q-LUT vs Conv2d+NoiseAug", "qlut", "ctrl_noise_aug"),
        ("ctrl_tanh_vs_ctrl", "Conv2d+tanh vs Ctrl (Conv2d)", "ctrl_tanh", "ctrl"),
        ("ctrl_noise_aug_vs_ctrl", "Conv2d+NoiseAug vs Ctrl (Conv2d)", "ctrl_noise_aug", "ctrl"),
    ]

    for ds in ["pneumoniamnist", "breastmnist"]:
        for stem in ["stabilized_stem", "raw_stem"]:
            for mode in ["interp4d", "int8_shift"]:
                for fam_key, label, mk_a, mk_b in b_pairs:
                    # Pure classical pairs do not depend on LUT mode; only record once under interp4d
                    if mk_a.startswith("ctrl") and mk_b.startswith("ctrl") and mode != "interp4d":
                        continue
                    for sg in sigma_strs:
                        va = get_model_accs(ds, mode, stem, sg, mk_a)
                        vb = get_model_accs(ds, mode, stem, sg, mk_b)
                        add_test(
                            "B_7branch_sym",
                            f"{ds}_{mode}_{stem}_{fam_key}",
                            ds,
                            mode,
                            stem,
                            sg,
                            label,
                            va,
                            vb,
                        )

    print("[4/4] Computing paired tests + TOST for Section C (4-bit quantization: INT8 vs Interp4D)...")
    # C1: From pareto_int8_vs_interp4d.json (all 3 MedMNIST datasets)
    for ds in ["breastmnist", "pneumoniamnist", "octmnist"]:
        for sg in sigma_strs:
            v_i8s = [float(x) for x in pareto["results"][ds][sg]["int8_shift"]["per_seed_acc"]]
            v_i8g = [float(x) for x in pareto["results"][ds][sg]["int8_grid"]["per_seed_acc"]]
            v_i4d = [float(x) for x in pareto["results"][ds][sg]["interp4d"]["per_seed_acc"]]
            add_test("C_quant_4bit", f"quant_shift_vs_interp4d_{ds}", ds, "int8_shift_vs_interp4d", "pre_lut_damped", sg, "INT8 Shift vs Interp4D", v_i8s, v_i4d)
            add_test("C_quant_4bit", f"quant_grid_vs_interp4d_{ds}", ds, "int8_grid_vs_interp4d", "pre_lut_damped", sg, "INT8 Grid vs Interp4D", v_i8g, v_i4d)

    # C2: From ablation_clut (symmetric stabilized_stem & raw_stem on Q-LUT)
    for ds in ["pneumoniamnist", "breastmnist"]:
        for stem in ["stabilized_stem", "raw_stem"]:
            for sg in sigma_strs:
                v_i8 = get_model_accs(ds, "int8_shift", stem, sg, "qlut")
                v_f32 = get_model_accs(ds, "interp4d", stem, sg, "qlut")
                add_test("C_quant_4bit_sym", f"quant_sym_{ds}_{stem}", ds, "int8_shift_vs_interp4d", stem, sg, "Q-LUT INT8 Shift vs Interp4D", v_i8, v_f32)

    # Apply Benjamini-Hochberg: (1) per-family (across 5 sigmas), (2) per-section, (3) global across all tests
    by_family: Dict[str, List[int]] = {}
    by_section: Dict[str, List[int]] = {}
    for idx, rec in enumerate(all_tests):
        by_family.setdefault(rec["family"], []).append(idx)
        by_section.setdefault(rec["section"], []).append(idx)

    for fam, idxs in by_family.items():
        adj_f = benjamini_hochberg([all_tests[i]["p_raw"] for i in idxs])
        adj_tost_f = benjamini_hochberg([all_tests[i]["p_tost"] for i in idxs])
        for i, qv, qt in zip(idxs, adj_f, adj_tost_f):
            all_tests[i]["p_adj_family"] = qv
            all_tests[i]["p_tost_adj_family"] = qt

    for sec, idxs in by_section.items():
        adj_s = benjamini_hochberg([all_tests[i]["p_raw"] for i in idxs])
        for i, qv in zip(idxs, adj_s):
            all_tests[i]["p_adj_section"] = qv

    adj_global = benjamini_hochberg([r["p_raw"] for r in all_tests])
    for i, qv in enumerate(adj_global):
        all_tests[i]["p_adj_global"] = qv

    # Assemble 7-branch summary tables (mean ± std across 5 seeds)
    seven_branch_order = [
        ("ctrl", "Ctrl (Conv2d, 148.4K)"),
        ("ctrl_tanh", "Conv2d + tanh [-1,1] (148.4K)"),
        ("ctrl_hardclip", "Conv2d + clip [-1,1] (148.4K)"),
        ("ctrl_noise_aug", "Conv2d + NoiseAug (148.4K)"),
        ("clut_septrig", "CLUT-SepTrig (148.3K)"),
        ("clut_randmlp", "CLUT-RandMLP (148.3K)"),
        ("qlut", "Q-LUT SU(16) L=2 (148.3K)"),
    ]

    seven_branch_summary: dict = {}
    for ds in ["pneumoniamnist", "breastmnist"]:
        seven_branch_summary[ds] = {}
        for stem in ["raw_stem", "stabilized_stem"]:
            seven_branch_summary[ds][stem] = {}
            for sg in sigma_strs:
                seven_branch_summary[ds][stem][sg] = {}
                for mk, _ in seven_branch_order:
                    vals = get_model_accs(ds, "interp4d", stem, sg, mk)
                    seven_branch_summary[ds][stem][sg][mk] = {
                        "acc_mean": float(np.mean(vals)),
                        "acc_std": float(np.std(vals)),
                        "per_seed_acc": vals,
                    }
                    if mk in ("clut_septrig", "clut_randmlp", "qlut"):
                        vals_i8 = get_model_accs(ds, "int8_shift", stem, sg, mk)
                        seven_branch_summary[ds][stem][sg][f"{mk}_int8"] = {
                            "acc_mean": float(np.mean(vals_i8)),
                            "acc_std": float(np.std(vals_i8)),
                            "per_seed_acc": vals_i8,
                        }

    payload = {
        "metadata": {
            "seeds": SEEDS,
            "sigmas": [float(s) for s in SIGMAS],
            "n_seeds": len(SEEDS),
            "df": len(SEEDS) - 1,
            "tost_margin_pct": TOST_MARGIN,
            "alpha": ALPHA,
            "fdr_method": "Benjamini-Hochberg (scipy.stats.false_discovery_control)",
            "total_hypothesis_tests": len(all_tests),
            "elapsed_sec": float(time.perf_counter() - t0),
        },
        "table42_recomputed_per_seed": t42,
        "seven_branch_summary": seven_branch_summary,
        "tests": all_tests,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Render clean, sober scientific Markdown report
    md: List[str] = []
    md.append("# Rigorous Statistical Analysis: Paired Tests, TOST (±2%), and Benjamini–Hochberg FDR")
    md.append("")
    md.append(f"- **Sample size:** $n = {len(SEEDS)}$ paired seeds (`{SEEDS}`), degrees of freedom $df = {len(SEEDS)-1}$.")
    md.append(f"- **Equivalence test (TOST):** Two one-sided paired $t$-tests with equivalence bounds $\\Delta = \\pm {TOST_MARGIN:.1f}\\%$ at $\\alpha = {ALPHA}$ (equivalent iff the 90% paired CI lies strictly inside $[-{TOST_MARGIN:.1f}\\%, +{TOST_MARGIN:.1f}\\%]$, i.e. $p_{{\\text{{TOST}}}} < 0.05$).")
    md.append(f"- **Multiple-testing correction:** Benjamini–Hochberg (BH) FDR reported at **family level** (`p_adj_fam`, 5 noise levels per comparison) and **global level** (`p_adj_glob`, across all {len(all_tests)} tests).")
    md.append("- **Power caveat ($n=5$):** With $df=4$ ($t_{0.95,4} = 2.1318$), declaring statistical equivalence within $\\pm 2.0\\%$ requires paired sample SD $s_d \\lesssim 2.1\\%$ when $\\bar{d} \\approx 0$. Inconclusive TOST under high variance is reported transparently as `INCONCLUSIVE`, never as equivalence.")
    md.append("")

    # ------------------------------------------------------------------
    # SECTION 1: 7-Branch Ablation Mean ± Std Overview
    # ------------------------------------------------------------------
    md.append("## 1. Extended 7-Branch Ablation Table (Accuracy %, Mean ± Std over 5 Seeds, `Interp4D` for LUTs)")
    md.append("")
    for ds in ["pneumoniamnist", "breastmnist"]:
        for stem, stem_title in [
            ("raw_stem", "Raw Stem (No Phase Damping, `stabilize_tau=999`)"),
            ("stabilized_stem", "Stabilized Stem Symmetric (`phase_damping_u8` + `stabilize_tau=0.024` on ALL branches)"),
        ]:
            md.append(f"### {ds.upper()} — {stem_title}")
            md.append("")
            md.append("| Model / Branch | $\\sigma=0.00$ | $\\sigma=0.05$ | $\\sigma=0.10$ | $\\sigma=0.15$ | $\\sigma=0.20$ |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for mk, label in seven_branch_order:
                cells = []
                for sg in sigma_strs:
                    node = seven_branch_summary[ds][stem][sg][mk]
                    cells.append(f"{node['acc_mean']:.2f} ± {node['acc_std']:.2f}")
                md.append(f"| **{label}** | " + " | ".join(cells) + " |")
            md.append("")

    # Helper to print test table
    def append_test_table(title: str, rows: List[dict]) -> None:
        md.append(f"### {title}")
        md.append("")
        md.append("| $\\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\\text{raw}}$ | $p_{\\text{adj,fam}}$ | $p_{\\text{adj,glob}}$ | $p_{\\text{TOST}}$ | Verdict |")
        md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for r in rows:
            md.append(
                f"| `{r['sigma']}` | {r['pair']} | {r['mean_a']:.2f}% | {r['mean_b']:.2f}% | "
                f"{r['delta_mean']:+.2f}% | [{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}] | "
                f"[{r['ci90_lo']:+.2f}, {r['ci90_hi']:+.2f}] | "
                f"{r['p_raw']:.4f} ({_sig_tag(r['p_raw'])}) | "
                f"{r['p_adj_family']:.4f} ({_sig_tag(r['p_adj_family'])}) | "
                f"{r['p_adj_global']:.4f} ({_sig_tag(r['p_adj_global'])}) | "
                f"{r['p_tost']:.4f} | {_tost_verdict(r)} |"
            )
        md.append("")

    # ------------------------------------------------------------------
    # SECTION 2: Re-analysis of Original Table 4.2 (Asymmetric Isolation)
    # ------------------------------------------------------------------
    md.append("## 2. Corrected Paired Analysis of Table 4.2 (Original Asymmetric Isolation, PneumoniaMNIST)")
    md.append("")
    md.append("Recomputed read-only from the locked `weights/pillar2_wide/` checkpoints to obtain exact 5-seed pairs for the advisor's Section 3 critique (`(4) vs (2)` fair baseline and `(3) vs (1)` raw-stem claim).")
    md.append("")
    for fam, title in [
        ("t42_4_vs_2", "(4) GR-Quanv + Pre-LUT Damping (Asymmetric) vs (2) Classical Control + Damping"),
        ("t42_3_vs_1", "(3) GR-Quanv Raw vs (1) Classical Control Raw"),
        ("t42_2_vs_1", "(2) Classical Control + Damping vs (1) Classical Control Raw (Isolated Damping Gain)"),
    ]:
        rows = [r for r in all_tests if r["family"] == fam]
        append_test_table(title, rows)

    # ------------------------------------------------------------------
    # SECTION 3: Symmetric 7-Branch Paired Comparisons (Interp4D)
    # ------------------------------------------------------------------
    md.append("## 3. Key Paired Comparisons & TOST (±2%) on Extended 7-Branch Symmetric Ablation (`Interp4D`)")
    md.append("")
    for ds in ["pneumoniamnist", "breastmnist"]:
        md.append(f"### Dataset: {ds.upper()}")
        md.append("")
        for stem, stem_lbl in [
            ("stabilized_stem", "Stabilized Stem Symmetric"),
            ("raw_stem", "Raw Stem (No Damping)"),
        ]:
            for fam_key, pair_lbl in [
                ("qlut_vs_ctrl", "Q-LUT vs Ctrl (Conv2d)"),
                ("qlut_vs_clut_randmlp", "Q-LUT vs CLUT-RandMLP (Random Nonlinear LUT)"),
                ("qlut_vs_clut_septrig", "Q-LUT vs CLUT-SepTrig (Separable Trig LUT)"),
                ("qlut_vs_ctrl_tanh", "Q-LUT vs Conv2d + tanh [-1, 1]"),
                ("qlut_vs_ctrl_hardclip", "Q-LUT vs Conv2d + hard-clip [-1, 1]"),
                ("qlut_vs_ctrl_noise_aug", "Q-LUT vs Conv2d + Speckle Noise Augmentation"),
                ("ctrl_tanh_vs_ctrl", "Conv2d + tanh [-1, 1] vs Unbounded Ctrl (Conv2d)"),
            ]:
                fam_id = f"{ds}_interp4d_{stem}_{fam_key}"
                rows = [r for r in all_tests if r["family"] == fam_id]
                if rows:
                    append_test_table(f"{ds.upper()} | {stem_lbl} | {pair_lbl}", rows)

    # ------------------------------------------------------------------
    # SECTION 4: 4-bit Quantization Cost (INT8 Shift/Grid vs Interp4D)
    # ------------------------------------------------------------------
    md.append("## 4. Paired Statistical Cost of 4-bit Input Quantization (`INT8` 256 KiB vs `Interp4D` 1,024 KiB)")
    md.append("")
    for ds in ["pneumoniamnist", "breastmnist", "octmnist"]:
        rows = [r for r in all_tests if r["family"] == f"quant_shift_vs_interp4d_{ds}"]
        append_test_table(f"{ds.upper()} (Pareto Pre-LUT Damped) | INT8 Shift (256 KiB) vs Interp4D (1,024 KiB)", rows)

    for ds in ["pneumoniamnist", "breastmnist"]:
        for stem in ["stabilized_stem", "raw_stem"]:
            rows = [r for r in all_tests if r["family"] == f"quant_sym_{ds}_{stem}"]
            append_test_table(f"{ds.upper()} ({stem}) | Q-LUT INT8 Shift vs Interp4D", rows)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[Saved] {OUT_JSON}")
    print(f"[Saved] {OUT_MD}")
    print(f"Total tests evaluated: {len(all_tests)} | elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
