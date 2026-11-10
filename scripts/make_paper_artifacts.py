import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob
import json
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.analysis.uncertainty import domain_shift, reliability_curve, risk_coverage, uncertainty_by_occlusion
from src.data.corruptions import CORRUPTIONS
from src.evaluation import load_predictions
from src.utils.bootstrap import apply_holm_bonferroni, paired_image_bootstrap_test
from src.utils.metrics import ap_from_records, build_records, detection_outcomes

MISSING = "—"
NOT_APPLICABLE = "n/a"
LATEX = {"±": "$\\pm$", "%": "\\%", "_": "\\_", "→": "$\\rightarrow$", "—": "--", "Δ": "$\\Delta$", "↑": "$\\uparrow$",
         "↓": "$\\downarrow$", "σ": "$\\sigma$", "−": "$-$", "λ": "$\\lambda$", "×": "$\\times$", "≥": "$\\geq$",
         "α": "$\\alpha$", "†": "$^\\dagger$", "&": "\\&", "#": "\\#"}
ABLATION_ORDER = ["ucr_mfg", "no_router", "router_no_uncertainty", "no_adaptive_budget", "no_benefit_loss",
                  "no_benefit_gate", "fusion_fixed", "fusion_add", "no_epistemic"]
MAIN_ORDER = ["det_only", "mfg_style", "ucr_mfg"]
LABELS = {
    "ucr_mfg": "UCR-MFG (full)", "det_only": "Detector only (ours, no graph)", "mfg_style": "MFG-style graph (ours, reimpl.)",
    "no_router": "w/o C2 routing (dense graph)", "router_no_uncertainty": "C2 w/o uncertainty input",
    "no_adaptive_budget": "C2 w/o adaptive budget", "no_benefit_loss": "C2 w/o benefit supervision",
    "no_benefit_gate": "C3 w/o benefit gate", "fusion_fixed": "C4 → fixed 0.5 fusion", "fusion_add": "C4 → additive fusion",
    "no_epistemic": "C1 w/o epistemic (T = 1)",
}
EXTERNAL_TEMPLATE = """# External baselines for Table 1. Fill in ONLY numbers you measured yourself under the same protocol
# (VisDrone2019-DET test-dev, same image size and COCO-style evaluation) or that a paper reports under a
# protocol you state below. AP values in percent (e.g. 24.3). Rows without a `source` are ignored; null values are shown as pending.
baselines:
  - name: YOLOv8-s
    source: ""
    protocol: ""
    visdrone_test_dev: {mAP: null, AP50: null, AP75: null, APs: null, APm: null, APl: null}
    params_M: null
    GFLOPs: null
  - name: RT-DETR-R50
    source: ""
    protocol: ""
    visdrone_test_dev: {mAP: null, AP50: null, AP75: null, APs: null, APm: null, APl: null}
    params_M: null
    GFLOPs: null
"""


def is_partial(meta: dict) -> bool:
    return bool(meta.get("max_samples")) or meta["num_images"] < -(-meta["dataset_size"] // meta["stride"])


def load_results(results_dir: str, allow_partial: bool = False):
    """variant -> tag -> list of result dicts (one per seed); partial evaluations are kept apart."""
    table: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    partial: List[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*.json"))):
        if os.sep + "efficiency" + os.sep in path:
            continue
        with open(path) as f:
            r = json.load(f)
        r["path"] = path
        if is_partial(r["meta"]) and not allow_partial:
            partial.append(r)
            continue
        table[r["meta"]["variant"]][r["meta"]["tag"]].append(r)
    return table, partial


def get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def stat(runs: List[dict], key: str, scale: float = 100.0):
    vals = [get(r["metrics"], key) for r in runs]
    vals = [v * scale for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def fmt(s, digits: int = 1) -> str:
    if s is None:
        return MISSING
    mean, std, n = s
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")


class Table:
    def __init__(self, name: str, caption: str, header: List[str]):
        self.name, self.caption, self.header, self.rows, self.notes = name, caption, header, [], []

    def add(self, row: List[str]) -> None:
        self.rows.append(row)

    def markdown(self) -> str:
        lines = [f"**{self.caption}**", "", "| " + " | ".join(self.header) + " |", "|" + "---|" * len(self.header)]
        lines += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(lines + [""] + [f"_{n}_" for n in self.notes]) + "\n"

    def latex(self) -> str:
        esc = lambda s: "".join(LATEX.get(ch, ch) for ch in s)
        cols = "l" + "c" * (len(self.header) - 1)
        body = " \\\\\n".join(" & ".join(esc(c) for c in r) for r in self.rows)
        notes = "".join(f"\\\\[-2pt]\\footnotesize {esc(n)}\n" for n in self.notes)
        return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{esc(self.caption)}}}\n\\label{{tab:{self.name}}}\n"
                f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{' & '.join(esc(h) for h in self.header)} \\\\\n\\midrule\n"
                f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n{notes}\\end{{table}}\n")

    def write(self, out_dir: str) -> None:
        with open(os.path.join(out_dir, f"{self.name}.md"), "w") as f:
            f.write(self.markdown())
        with open(os.path.join(out_dir, f"{self.name}.tex"), "w") as f:
            f.write(self.latex())


def pkl_path(run: dict) -> Optional[str]:
    path = os.path.splitext(run["path"])[0] + ".pkl"
    return path if os.path.exists(path) else None


def bootstrap_vs(main_runs, other_runs, num_classes, resamples) -> Optional[float]:
    """Paired image-level bootstrap p-value for the mAP difference on the same images (first seed of each)."""
    a, b = (pkl_path(main_runs[0]), pkl_path(other_runs[0])) if main_runs and other_runs else (None, None)
    if not a or not b:
        return None
    pa, pb = load_predictions(a), load_predictions(b)
    ids_a = [g["image_id"] for g in pa["ground_truths"]]
    if ids_a != [g["image_id"] for g in pb["ground_truths"]]:
        return None
    ra = build_records(pa["predictions"], pa["ground_truths"], num_classes)
    rb = build_records(pb["predictions"], pb["ground_truths"], num_classes)
    score = lambda rec: (lambda idx: float(np.nanmean(ap_from_records(rec, idx))))
    test = paired_image_bootstrap_test(list(range(len(ids_a))), score(rb), score(ra), num_resamples=resamples)
    return test["p_value"]


def main():
    parser = argparse.ArgumentParser(description="Build paper tables, figures and a results status report from measured results only")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="paper")
    parser.add_argument("--main-variant", default=None, help="Variant used as 'UCR-MFG' (default: ucr_mfg, else the primary run)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for ablation p-values (0 disables)")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--allow-partial", action="store_true", help="Include evaluations on image subsets (debugging only)")
    args = parser.parse_args()

    tables_dir, figs_dir = os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    external_path = os.path.join(args.results_dir, "external_baselines.yaml")
    if not os.path.exists(external_path):
        os.makedirs(args.results_dir, exist_ok=True)
        with open(external_path, "w") as f:
            f.write(EXTERNAL_TEMPLATE)

    R, partial = load_results(args.results_dir, args.allow_partial)
    main = args.main_variant or ("ucr_mfg" if "ucr_mfg" in R else next((v for v in R if "primary" in v), "ucr_mfg"))
    alias = lambda v: "ucr_mfg" if v == main else v
    R = {alias(v): tags for v, tags in R.items()}
    num_classes = 10

    # ---------- Table 1: main comparison on VisDrone test-dev ----------
    with open(external_path) as f:
        external = (yaml.safe_load(f) or {}).get("baselines", []) or []
    t1 = Table("main_visdrone", "Comparison on VisDrone2019-DET test-dev (AP in %).",
               ["Method", "mAP", "AP50", "AP75", "APs", "APm", "APl", "Params (M)", "GFLOPs"])
    for b in external:
        if not b.get("source"):
            continue
        m = b.get("visdrone_test_dev", {}) or {}
        cell = lambda v: MISSING if v is None else f"{float(v):.1f}"
        t1.add([f"{b['name']} [{b['source']}]"] + [cell(m.get(k)) for k in ("mAP", "AP50", "AP75", "APs", "APm", "APl")] +
               [MISSING if b.get("params_M") is None else f"{b['params_M']:.1f}", MISSING if b.get("GFLOPs") is None else f"{b['GFLOPs']:.0f}"])
    eff = {}
    for path in glob.glob(os.path.join(args.results_dir, "efficiency", "*.json")):
        with open(path) as f:
            e = json.load(f)
        eff[alias(e["variant"])] = e
    for v in MAIN_ORDER:
        runs = R.get(v, {}).get("visdrone_test_dev", [])
        e = eff.get(v, {})
        t1.add([LABELS.get(v, v) + (f" (n={len(runs)})" if len(runs) > 1 else "")] +
               [fmt(stat(runs, k)) for k in ("mAP", "AP50", "AP75", "APs", "APm", "APl")] +
               [f"{e['total_M']:.1f}" if e else MISSING, f"{e['GFLOPs']:.0f}" if "GFLOPs" in e else MISSING])
    t1.notes.append("Mean ± std over seeds where several were trained. — = not yet measured. External rows appear only with a cited source.")
    t1.write(tables_dir)

    # ---------- Table 2: ablations on VisDrone val ----------
    t2 = Table("ablations", "Ablation of the four contributions on VisDrone2019-DET val (AP in %).",
               ["Variant", "mAP", "AP50", "APs", "D-ECE", "Δ mAP", "p (Holm)"])
    main_val = R.get("ucr_mfg", {}).get("visdrone_val", [])
    base = stat(main_val, "mAP")
    rows, pvals = [], []
    for v in ABLATION_ORDER:
        runs = R.get(v, {}).get("visdrone_val", [])
        s = stat(runs, "mAP")
        delta = MISSING if (s is None or base is None or v == "ucr_mfg") else f"{s[0] - base[0]:+.1f}"
        p = bootstrap_vs(main_val, runs, num_classes, args.bootstrap) if (args.bootstrap and v != "ucr_mfg" and runs) else None
        rows.append([LABELS.get(v, v), fmt(s), fmt(stat(runs, "AP50")), fmt(stat(runs, "APs")), fmt(stat(runs, "D-ECE", 1.0), 3), delta])
        pvals.append(p)
    tested = [i for i, p in enumerate(pvals) if p is not None]
    significant = dict(zip(tested, apply_holm_bonferroni([pvals[i] for i in tested]))) if tested else {}
    for i, row in enumerate(rows):
        row.append(MISSING if pvals[i] is None else f"{pvals[i]:.3f}{'*' if significant.get(i) else ''}")
        t2.add(row)
    t2.notes.append("Δ relative to the full model. p: paired image-level bootstrap of the mAP difference (first seed); * = significant after Holm–Bonferroni at α = 0.05.")
    t2.write(tables_dir)

    # ---------- Table 3: UAVDT zero-shot ----------
    t3 = Table("uavdt_zero_shot", "Zero-shot transfer VisDrone → UAVDT test (car / truck / bus, AP in %).",
               ["Method", "mAP", "AP50", "AP50 daylight", "AP50 night", "AP50 fog", "AP50 low alt.", "AP50 med. alt.", "AP50 high alt."])
    strides = set()
    for v in MAIN_ORDER:
        runs = R.get(v, {}).get("uavdt_test", [])
        strides |= {r["meta"]["stride"] for r in runs}
        conds = ["weather:daylight", "weather:night", "weather:fog", "altitude:low", "altitude:medium", "altitude:high"]
        t3.add([LABELS.get(v, v), fmt(stat(runs, "mAP")), fmt(stat(runs, "AP50"))] + [fmt(stat(runs, f"conditions.{c}.AP50")) for c in conds])
    if strides:
        t3.notes.append(f"Evaluated on every {', '.join(map(str, sorted(strides)))}-th frame of the 20 test sequences.")
    t3.write(tables_dir)

    # ---------- Table 4: occlusion ----------
    t4 = Table("occlusion", "Detection by occlusion level on VisDrone test-dev (AP50 / recall@0.5 in %).",
               ["Method", "AP50 none", "AP50 partial", "AP50 heavy", "Recall none", "Recall partial", "Recall heavy"])
    for v in MAIN_ORDER:
        runs = R.get(v, {}).get("visdrone_test_dev", [])
        t4.add([LABELS.get(v, v)] + [fmt(stat(runs, f"occlusion.{o}.AP50")) for o in ("none", "partial", "heavy")] +
               [fmt(stat(runs, f"occlusion.{o}.recall50")) for o in ("none", "partial", "heavy")])
    t4.notes.append("Ground truth of other occlusion levels is ignored, so detecting it is neither rewarded nor penalised.")
    t4.write(tables_dir)

    # ---------- Table 5: robustness ----------
    corrs = sorted(CORRUPTIONS)
    t5 = Table("robustness", "Robustness on corrupted VisDrone val (mAP in %, mean over measured severities).",
               ["Method", "Clean"] + [c.replace("_", " ") for c in corrs] + ["Mean", "Relative"])
    for v in MAIN_ORDER:
        tags = R.get(v, {})
        clean = stat(tags.get("visdrone_val", []), "mAP")
        per = []
        for c in corrs:
            vals = [stat(tags[t], "mAP")[0] for t in tags if t.startswith(f"visdrone_val_{c}") and stat(tags[t], "mAP")]
            per.append(float(np.mean(vals)) if vals else None)
        measured = [p for p in per if p is not None]
        mean = float(np.mean(measured)) if len(measured) == len(corrs) else None
        rel = f"{100 * mean / clean[0]:.0f}%" if (mean is not None and clean) else MISSING
        t5.add([LABELS.get(v, v), fmt(clean)] + [MISSING if p is None else f"{p:.1f}" for p in per] +
               [MISSING if mean is None else f"{mean:.1f}", rel])
    t5.notes.append("Relative = mean corrupted mAP / clean mAP. Mean is reported only when every corruption was measured.")
    t5.write(tables_dir)

    # ---------- Table 6: uncertainty quality + figures ----------
    t6 = Table("uncertainty", "Uncertainty quality on VisDrone test-dev.",
               ["Method", "D-ECE ↓", "AUROC Ua ↑", "AUROC Ue ↑", "AUROC box σ ↑", "AUROC 1−score ↑", "AURC Ue ↓", "OOD AUROC Ue ↑"])
    analysis = {}
    for v in MAIN_ORDER:
        runs = R.get(v, {}).get("visdrone_test_dev", [])
        no_epistemic = bool(runs) and runs[0]["meta"].get("mc_passes", 10) <= 1
        no_aleatoric = v == "det_only"
        def unc_cell(key):
            if (key == "u_e" and no_epistemic) or (key == "u_a" and no_aleatoric):
                return NOT_APPLICABLE
            return fmt(stat(runs, f"AUROC.{key}", 1.0), 3)
        row = [LABELS.get(v, v), fmt(stat(runs, "D-ECE", 1.0), 3)] + [unc_cell(k) for k in ("u_a", "u_e", "box_std", "1-score")]
        aurc, ood = (NOT_APPLICABLE, NOT_APPLICABLE) if no_epistemic else (MISSING, MISSING)
        test_pkl = pkl_path(runs[0]) if runs else None
        if test_pkl:
            data = load_predictions(test_pkl)
            outcomes = detection_outcomes(data["predictions"], data["ground_truths"])
            rc = risk_coverage(data["predictions"], data["ground_truths"], outcomes=outcomes)
            info = {"risk_coverage": rc, "reliability": reliability_curve(data["predictions"], data["ground_truths"], outcomes=outcomes),
                    "occlusion": uncertainty_by_occlusion(data["predictions"], data["ground_truths"], outcomes=outcomes)}
            if not no_epistemic:
                aurc = f"{rc['AURC']['u_e']:.3f}" if "u_e" in rc["AURC"] else MISSING
            uavdt = R.get(v, {}).get("uavdt_test", [])
            out_pkl = pkl_path(uavdt[0]) if uavdt else None
            if out_pkl and not no_epistemic:
                info["domain"] = domain_shift(data["predictions"], load_predictions(out_pkl)["predictions"])
                ood = f"{info['domain']['AUROC'].get('u_e', float('nan')):.3f}"
            analysis[v] = info
        t6.add(row + [aurc, ood])
    t6.notes.append("n/a: the variant has no such estimate (no MC passes, or no trained C1 head). AUROC: separating false from true detections (0.5 = chance). AURC: area under the risk–coverage curve when rejecting by Ue. "
                    "OOD AUROC: image-level Ue separating UAVDT (unseen domain) from VisDrone test-dev images.")
    t6.write(tables_dir)
    with open(os.path.join(args.out, "uncertainty_analysis.json"), "w") as f:
        json.dump({v: {k: val for k, val in info.items() if k != "domain"} | ({"domain": {k: info["domain"][k] for k in ("AUROC", "mean_in", "mean_out")}} if "domain" in info else {})
                   for v, info in analysis.items()}, f, indent=2)

    # ---------- Table 7: efficiency ----------
    t7 = Table("efficiency", "Computational cost at 1024 × 1024, batch 1, fp32.",
               ["Variant", "Params (M)", "GFLOPs", "GFLOPs (T = 1)", "Latency (ms)", "FPS", "Routed edges", "Device"])
    for v in MAIN_ORDER + [x for x in ABLATION_ORDER if x not in MAIN_ORDER]:
        e = eff.get(v)
        if not e:
            continue
        routed = f"{e['routed_edges']:,} / {e['candidate_edges']:,} ({100 * e['routed_fraction']:.0f}%)" if "candidate_edges" in e else "n/a"
        t7.add([LABELS.get(v, v), f"{e['total_M']:.1f}", f"{e.get('GFLOPs', float('nan')):.0f}", f"{e.get('GFLOPs_single_pass', float('nan')):.0f}",
                f"{e['latency_ms']:.0f}", f"{e['fps']:.2f}", routed + ("" if e.get("trained") else " †"), e["meta"]["device"]])
    t7.notes.append("† untrained router: edge counts change once the router is trained. GFLOPs exclude memory-bound graph gather/scatter.")
    t7.write(tables_dir)

    # ---------- Table 8: number of MC passes at test time ----------
    t8 = Table("mc_passes", "Effect of the number of MC-dropout passes T at test time (UCR-MFG, VisDrone test-dev).",
               ["T", "mAP", "D-ECE ↓", "AUROC Ue ↑", "AUROC Ua ↑"])
    mc_tags = sorted((t for t in R.get("ucr_mfg", {}) if t == "visdrone_test_dev" or t.startswith("visdrone_test_dev_T")),
                     key=lambda t: R["ucr_mfg"][t][0]["meta"].get("mc_passes", 0))
    for t in mc_tags:
        runs = R["ucr_mfg"][t]
        t8.add([str(runs[0]["meta"].get("mc_passes", "?")), fmt(stat(runs, "mAP")), fmt(stat(runs, "D-ECE", 1.0), 3),
                fmt(stat(runs, "AUROC.u_e", 1.0), 3), fmt(stat(runs, "AUROC.u_a", 1.0), 3)])
    t8.notes.append("Same trained model; only T changes. Measure each T with scripts/eval.py --mc-passes T. GFLOPs per T: Table 7 (T = 1 and trained T).")
    t8.write(tables_dir)

    make_figures(R, analysis, figs_dir, args.checkpoints, corrs)
    write_status(R, external, eff, analysis, args, main, partial)
    print(f"Tables -> {tables_dir}/  Figures -> {figs_dir}/  Status -> {os.path.join(args.out, 'STATUS.md')}")


def make_figures(R, analysis, figs_dir, checkpoints_dir, corrs) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"ucr_mfg": "#3b2f7a", "mfg_style": "#c07a1e", "det_only": "#8a8578"}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, f"{name}.pdf"))
        fig.savefig(os.path.join(figs_dir, f"{name}.png"), dpi=200)
        plt.close(fig)

    if analysis:
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
        for v, info in analysis.items():
            rel = info["reliability"]
            ax.plot(rel["confidence"], rel["accuracy"], "o-", ms=3, color=colors.get(v), label=LABELS.get(v, v))
        ax.set(xlabel="Confidence", ylabel="Precision", title="Reliability (VisDrone test-dev)", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
        save(fig, "reliability")

        if "ucr_mfg" in analysis:
            rc = analysis["ucr_mfg"]["risk_coverage"]
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            names = {"u_e": "epistemic Ue", "u_a": "aleatoric Ua", "box_std": "box σ", "1-score": "1 − score", "oracle": "oracle"}
            for key, curve in rc["curves"].items():
                ax.plot(rc["coverage"], curve, "--" if key == "oracle" else "-", lw=1.4, label=f"{names.get(key, key)} (AURC {rc['AURC'][key]:.3f})")
            ax.set(xlabel="Coverage (fraction of detections kept)", ylabel="Precision of kept detections", title="Risk–coverage, UCR-MFG")
            ax.legend(fontsize=6.5)
            save(fig, "risk_coverage")

            occ = analysis["ucr_mfg"].get("occlusion")
            if occ:
                fig, ax = plt.subplots(figsize=(3.4, 3.0))
                levels = list(occ)
                x = np.arange(len(levels))
                for i, (key, label) in enumerate((("u_a", "aleatoric Ua"), ("u_e", "epistemic Ue"))):
                    ax.bar(x + (i - 0.5) * 0.38, [occ[l][key] for l in levels], 0.38, yerr=[occ[l][f"{key}_sem"] for l in levels], label=label, capsize=2)
                ax.set_xticks(x, [f"{l}\n(n={occ[l]['n']})" for l in levels])
                ax.set(ylabel="Mean uncertainty of true positives", title="Uncertainty vs. occlusion")
                ax.legend(fontsize=7)
                save(fig, "uncertainty_vs_occlusion")

            dom = analysis["ucr_mfg"].get("domain")
            if dom:
                fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
                for ax, key, label in zip(axes, ("u_e", "u_a"), ("Epistemic Ue", "Aleatoric Ua")):
                    d = dom["detections"][key]
                    bins = np.linspace(0, max(max(d["in"], default=1), max(d["out"], default=1)), 40)
                    ax.hist(d["in"], bins, density=True, alpha=0.6, label="VisDrone (seen)")
                    ax.hist(d["out"], bins, density=True, alpha=0.6, label="UAVDT (unseen)")
                    ax.set(xlabel=label, ylabel="Density", title=f"{label}: OOD AUROC {dom['AUROC'].get(key, float('nan')):.2f}")
                axes[0].legend(fontsize=7)
                save(fig, "domain_shift_uncertainty")

    curves = {v: tags for v, tags in R.items() if any(t.startswith("visdrone_val_") for t in tags)}
    if curves:
        fig, axes = plt.subplots(1, len(corrs), figsize=(2.3 * len(corrs), 2.4), sharey=True)
        for ax, c in zip(np.atleast_1d(axes), corrs):
            for v, tags in curves.items():
                pts = sorted((tags[t][0]["meta"]["severity"], 100 * np.mean([r["metrics"]["mAP"] for r in tags[t]]))
                             for t in tags if t.startswith(f"visdrone_val_{c}"))
                clean = tags.get("visdrone_val")
                if clean:
                    pts = [(0, 100 * np.mean([r["metrics"]["mAP"] for r in clean]))] + pts
                if pts:
                    ax.plot(*zip(*pts), "o-", ms=3, color=colors.get(v), label=LABELS.get(v, v))
            ax.set(title=c.replace("_", " "), xlabel="Severity")
        np.atleast_1d(axes)[0].set_ylabel("mAP (%)")
        np.atleast_1d(axes)[0].legend(fontsize=6)
        save(fig, "robustness")

    logs = sorted(glob.glob(os.path.join(checkpoints_dir, "**", "train_log.jsonl"), recursive=True))
    if logs:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
        for path in logs:
            with open(path) as f:
                entries = [json.loads(l) for l in f if l.strip()]
            name = os.path.basename(os.path.dirname(path))
            axes[0].plot([e["epoch"] for e in entries], [e["loss_det"] for e in entries], lw=1, label=name)
            val = [(e["epoch"], 100 * e["val_mAP"]) for e in entries if "val_mAP" in e]
            if val:
                axes[1].plot(*zip(*val), "o-", ms=2.5, lw=1, label=name)
        axes[0].set(xlabel="Epoch", ylabel="Detection loss", title="Training loss")
        axes[1].set(xlabel="Epoch", ylabel="Val mAP (%)", title="Validation mAP")
        axes[1].legend(fontsize=5.5)
        save(fig, "training_curves")


def write_status(R, external, eff, analysis, args, main, partial) -> None:
    lines = ["# Paper results status", "", "Generated from measured result files only. Every item marked ✗ is missing.", "",
             f"Main model variant: `{main}`", ""]

    def check(ok: bool, text: str) -> None:
        lines.append(f"- {'✓' if ok else '✗'} {text}")

    if partial:
        lines += ["## Excluded partial evaluations", ""]
        for r in partial:
            m = r["meta"]
            check(False, f"`{m['variant']}` {m['tag']}: {m['num_images']} of {-(-m['dataset_size'] // m['stride'])} images — excluded from all tables")
        lines.append("")
    lines.append("## Training completeness")
    if not R:
        check(False, "No evaluation results found. Train models, then run `scripts/run_paper_eval.py --weights <best.pth>`.")
    for v, tags in sorted(R.items()):
        runs = [r for rs in tags.values() for r in rs]
        metas = {(r["meta"]["weights"], r["meta"]["epoch"], r["meta"]["max_epochs"], r["meta"]["stage"], r["meta"]["mid_epoch"]) for r in runs}
        for weights, epoch, max_epochs, stage, mid in sorted(metas, key=str):
            complete = not mid and (max_epochs is None or epoch == max_epochs or os.path.basename(weights) == "best.pth")
            check(complete and (stage == 4 or v == "det_only"),
                  f"`{v}`: {os.path.relpath(weights)} — epoch {epoch}/{max_epochs}, stage {stage}" + (" (mid-epoch)" if mid else ""))
        hashes = {r["meta"]["config_hash"] for r in runs}
        seeds = sorted({r["meta"]["seed"] for r in runs})
        check(len(seeds) >= 3 or v not in MAIN_ORDER, f"`{v}`: seeds {seeds} (≥ 3 recommended for the main comparison)")
        dirty = {r["meta"]["git"] for r in runs if r["meta"]["git"].endswith("-dirty")}
        if dirty:
            check(False, f"`{v}`: evaluated from uncommitted code ({', '.join(sorted(dirty))}); commit before final runs")
        if len(hashes) > len(seeds):
            check(False, f"`{v}`: results come from {len(hashes)} different configurations; re-evaluate from one checkpoint per seed")

    lines += ["", "## Required evaluations"]
    for v in MAIN_ORDER:
        tags = R.get(v, {})
        check("visdrone_test_dev" in tags, f"`{v}` on VisDrone test-dev (Table 1, 4, 6)")
        check("visdrone_val" in tags, f"`{v}` on VisDrone val (Tables 2, 5)")
        check("uavdt_test" in tags, f"`{v}` zero-shot on UAVDT (Table 3, OOD analysis)")
        missing = [c for c in sorted(CORRUPTIONS) if not any(t.startswith(f"visdrone_val_{c}") for t in tags)]
        check(not missing, f"`{v}` corruption suite" + (f" — missing: {', '.join(missing)}" if missing else ""))
    for v in ABLATION_ORDER:
        check("visdrone_val" in R.get(v, {}), f"ablation `{v}` on VisDrone val")
    uavdt_strides = {r["meta"]["stride"] for v in R for r in R[v].get("uavdt_test", [])}
    if uavdt_strides - {1}:
        check(False, f"UAVDT evaluated with frame stride {sorted(uavdt_strides)}; state this in the paper or re-run with stride 1")

    lines += ["", "## Analyses and figures"]
    check("ucr_mfg" in analysis, "Risk–coverage, reliability and uncertainty-vs-occlusion analysis (needs saved test-dev predictions)")
    check("domain" in analysis.get("ucr_mfg", {}), "Epistemic uncertainty under domain shift (needs UAVDT predictions)")
    check(all(v in eff for v in MAIN_ORDER), "Efficiency benchmark for detector-only, MFG-style and UCR-MFG (`scripts/benchmark.py`)")
    check(all(eff[v].get("trained") for v in eff) if eff else False, "Efficiency measured with trained weights (routed-edge counts)")
    check(bool(glob.glob(os.path.join(args.out, "figures", "qualitative", "*.png"))), "Qualitative figures (`scripts/visualize.py`)")
    sourced = [b for b in external if b.get("source")]
    check(bool(sourced), f"External baselines with cited sources in `{os.path.join(args.results_dir, 'external_baselines.yaml')}`"
          + (f" ({len(sourced)} sourced)" if sourced else ""))

    lines += ["", "## Protocol notes for the paper", "",
              "- Model selection uses VisDrone **val** only (`best.pth`); test-dev is evaluated once per final model.",
              "- UAVDT is never used for training; VisDrone car + van → car, truck → truck, bus → bus.",
              "- AP is COCO-style (IoU 0.50:0.95, 101-point), not the official VisDrone toolkit; say so, or also run the official toolkit.",
              "- Corruptions are synthetic (`src/data/corruptions.py`) and deterministic per image."]
    with open(os.path.join(args.out, "STATUS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
