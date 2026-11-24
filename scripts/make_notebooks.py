"""Builds notebooks/*.ipynb (python scripts/make_notebooks.py [1] [2]) and fills them with real outputs by executing every code cell in one namespace.

Jupyter/nbconvert are not installed, so cells are executed here: stdout is captured as stream output and every
matplotlib figure left open at the end of a cell is embedded as a PNG. The notebooks also run unchanged in Jupyter.
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUTS = []





def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def run(cells, path):
    ns = {"__name__": "__main__"}
    n = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        n += 1
        OUTPUTS.clear()
        buf = io.StringIO()
        src = "".join(c["source"])
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(src, f"<cell {n}>", "exec"), ns)
        except Exception:
            print(f"ERROR in {path} cell {n}:\n{src}\n", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        outs = []
        if buf.getvalue():
            outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(keepends=True)})
        outs += list(OUTPUTS)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        c["execution_count"] = n
        c["outputs"] = outs
        print(f"  {os.path.basename(path)} cell {n} ok", flush=True)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(nb, open(path, "w"), indent=1)
    print(f"wrote {path}")


SETUP = r'''
import os, sys, json, time, inspect, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, yaml
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

def show_table(headers, rows, title=None):
    """Plain-text table (no pandas needed)."""
    cells = [[str(h) for h in headers]] + [[str(v) for v in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    if title:
        print(title)
    for k, r in enumerate(cells):
        print("  ".join(v.rjust(w[i]) if i else v.ljust(w[i]) for i, v in enumerate(r)))
        if k == 0:
            print("  ".join("-" * x for x in w))

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("project root:", ROOT)
print("torch", torch.__version__, "| device:", device)
'''

# --------------------------------------------------------------------------------------------- notebook 1
NB1 = [
    md(r'''
# UCR-MFG: Implementation
**Uncertainty-Conditioned Relational Routing over a Multi-Scale Fuzzy Graph for drone object detection (UAVDT)**

This notebook documents the complete implementation used for the paper. It shows the configuration, the source code
of every model component and loss, and runs the trained final model (`models/stage4_ep100_final.pth`) on a UAVDT test
frame to show what each component produces.

| Part | Component | File |
|---|---|---|
| Backbone | ResNet-50 + FPN (P3–P5) | `src/models/backbone.py` |
| C1 | Pre-graph uncertainty head (aleatoric $U_a$, epistemic $U_e$ via MC dropout) | `src/models/uncertainty_head.py` |
| Graph | Multi-scale fuzzy candidate graph (local, kNN, cross-scale; Student-t affinity) | `src/models/graph_builder.py` |
| C2 | Uncertainty-conditioned router (typed top-K, uncertainty-scaled budget) | `src/models/router.py` |
| C3 | Benefit-aware message passing (attention, benefit gate, GRU) | `src/models/message_passing.py` |
| C4 | Residual uncertainty-aware fusion gate $Z=F+(1-\lambda)\odot H$ | `src/models/fusion.py` |
| Head | Probabilistic FCOS head (cls, ltrb, centerness, log-variance) | `src/models/detection_head.py` |
| Full model | Staged UCR-MFG detector | `src/models/ucr_mfg.py` |
| Losses | Detection, β-NLL uncertainty, graph, router, consistency, total | `src/losses/` |
| Training | 4-stage curriculum, safeguards, resume | `scripts/train.py` |
'''),
    md("## 1. Setup"),
    code(SETUP),
    md("## 2. Configuration of the final model\nThe final model was trained with `configs/experiments/uavdt_stage4_residual.yaml` (stage 4) on top of "
       "`configs/experiments/uavdt_primary.yaml` (stages 1–3), using the model settings in `configs/model/residual.yaml`."),
    code(r'''
exp = yaml.safe_load(open("configs/experiments/uavdt_stage4_residual.yaml"))
mcfg = yaml.safe_load(open("configs/model/residual.yaml"))
t = exp["train"]
show_table(["setting", "value"], [
    ["dataset", "UAVDT (" + yaml.safe_load(open(exp["dataset_config"]))["dataset_name"] + "), classes car / truck / bus"],
    ["input canvas (H, W)", t["img_size"]], ["training frames", f"every {t['frame_stride']}th frame -> 813 images"],
    ["epochs / batch / accumulation", f"{t['max_epochs']} / {t['batch_size']} / {t['accumulate_grad_batches']}"],
    ["optimizer / lr / weight decay", f"{t['optimizer']} / {t['lr']} / {t['weight_decay']}"],
    ["schedule", t["scheduler"]], ["initialisation", t["init_weights"]],
], "Training")
print()
m = mcfg["model"]; l = mcfg["loss"]
show_table(["setting", "value"], [
    ["backbone / FPN channels / strides", f"{m['backbone']} / {m['fpn_channels']} / {m['strides']}"],
    ["MC dropout passes T / p", f"{m['uncertainty']['mc_passes']} / {m['uncertainty']['dropout_rate']}"],
    ["graph: r_local / k_feat / alpha", f"{m['graph']['r_local']} / {m['graph']['k_feat']} / {m['graph']['alpha']}"],
    ["router budgets local/kNN/cross, k_gain", f"{m['router']['k_local']}/{m['router']['k_feat']}/{m['router']['k_cross']}, {m['router']['k_gain']}"],
    ["fusion mode / stage-3 residual", f"{m['fusion']['mode']} / {m['fusion']['stage3_residual']}"],
    ["loss weights u / var / graph / inst / benefit / cons",
     f"{l['lambda_u']} / {l['lambda_var']} / {l['lambda_graph']} / {l['lambda_inst']} / {l['lambda_benefit']} / {l['lambda_consistency']}"],
], "Model and loss")
print()
show_table(["stage", "epochs", "modules switched on", "losses"],
           [[k, f"{v['epochs'][0]}-{v['epochs'][1]}", ", ".join(v["active_modules"]), ", ".join(v["losses"])] for k, v in t["stages"].items()],
           "Training curriculum")
'''),
    md("## 3. Source code of the model\nThe cells below print the full source of every model file, so the notebook contains the complete implementation."),
    code(r'''
def show_file(path):
    src = open(path).read()
    print(f"# ===== {path} ({len(src.splitlines())} lines) =====")
    print(src)

for f in ["src/models/backbone.py", "src/models/uncertainty_head.py"]:
    show_file(f)
'''),
    md("### Multi-scale fuzzy graph and graph containers"),
    code(r'''
for f in ["src/models/graph_ops.py", "src/models/graph_builder.py"]:
    show_file(f)
'''),
    md("### C2 router, C3 message passing, C4 fusion gate"),
    code(r'''
for f in ["src/models/router.py", "src/models/message_passing.py", "src/models/fusion.py"]:
    show_file(f)
'''),
    md("### Detection head, full model and post-processing"),
    code(r'''
for f in ["src/models/detection_head.py", "src/models/ucr_mfg.py", "src/models/postprocess.py"]:
    show_file(f)
'''),
    md("## 4. Source code of the losses\n$\\mathcal{L}=\\mathcal{L}_{det}+\\lambda_v\\mathcal{L}_{var}+\\lambda_u\\mathcal{L}_u+\\lambda_g\\mathcal{L}_{graph}+\\lambda_i\\mathcal{L}_{inst}+\\lambda_b\\mathcal{L}_{ben}+\\lambda_c\\mathcal{L}_{cons}$, each term active from its curriculum stage."),
    code(r'''
for f in ["src/losses/detection_loss.py", "src/losses/uncertainty_loss.py", "src/losses/graph_loss.py",
          "src/losses/router_loss.py", "src/losses/consistency_loss.py", "src/losses/total_loss.py"]:
    show_file(f)
'''),
    md("## 5. Load the trained final model"),
    code(r'''
from src.evaluation import load_trained_model
model, meta = load_trained_model("models/stage4_ep100_final.pth", device)
model.eval()
print(f"epoch {meta['epoch']}, curriculum stage {model.stage}, fusion mode: {model.fusion_gate.mode}")
groups = {"backbone (ResNet-50 + FPN)": model.backbone, "C1 uncertainty head": model.unc_head,
          "graph builder": model.graph_builder, "C2 router": model.router, "C3 message passing": model.mpnn,
          "C4 fusion gate": model.fusion_gate, "FCOS head": model.det_head}
rows = [[k, f"{sum(p.numel() for p in v.parameters()):,}"] for k, v in groups.items()]
rows.append(["total", f"{sum(p.numel() for p in model.parameters()):,}"])
show_table(["module", "parameters"], rows, "Parameters per module")
'''),
    md("## 6. Forward pass on one UAVDT test frame\nRun the model once and inspect what each component produces."),
    code(r'''
from src.data.uavdt import UAVDTDataset
from src.models.uncertainty_head import mc_generator
ds = UAVDTDataset("dataset/UAVDT-Benchmark-M", "test", (544, 1024), is_train=False, frame_stride=1)
idx = [i for i, (s, _, n) in enumerate(ds.frames) if s == "M0403" and n == 169][0]
sample = ds[idx]
x = sample["image"].unsqueeze(0).to(device)
t0 = time.time()
with torch.no_grad():
    out = model(x, generator=mc_generator(0, 0, 0, device))
print(f"frame {sample['image_id']}: input {tuple(x.shape)}, {len(sample['boxes'])} annotated vehicles, "
      f"forward pass {time.time() - t0:.2f} s on {device}")
feats = model.backbone(x)
show_table(["level", "stride", "feature map", "Ua mean", "Ua max", "Ue mean", "Ue max"],
           [[f"P{l+3}", s, tuple(f.shape), f"{ua.mean():.3f}", f"{ua.max():.3f}", f"{ue.mean():.4f}", f"{ue.max():.4f}"]
            for l, (s, f, ua, ue) in enumerate(zip(model.strides, feats, out["u_a"], out["u_e"]))],
           "Backbone features and C1 uncertainty maps")
'''),
    code(r'''
from src.models.graph_ops import LOCAL, KNN, TOP_DOWN, BOTTOM_UP
g, r = out["graph"], out["routing"]
names = {LOCAL: "local", KNN: "kNN", TOP_DOWN: "cross-scale (top-down)", BOTTOM_UP: "cross-scale (bottom-up)"}
sel = r.selected
rows = []
for t_id, name in names.items():
    cand = int((g.edge_types == t_id).sum()); kept = int((g.edge_types[sel] == t_id).sum())
    rows.append([name, f"{cand:,}", f"{kept:,}", f"{100 * kept / max(cand, 1):.1f}%"])
n_sel = int(sel.sum()) if sel.dtype == torch.bool else int(sel.numel())   # `selected` is a boolean mask over all edges
rows.append(["all", f"{g.edge_index.shape[1]:,}", f"{n_sel:,}", f"{100 * n_sel / g.edge_index.shape[1]:.1f}%"])
show_table(["edge type", "candidate edges", "routed edges", "kept"], rows,
           f"Fuzzy graph ({g.num_nodes:,} nodes) and C2 routing")
recv = g.edge_index[0, sel]
deg = torch.bincount(recv, minlength=g.num_nodes).float().cpu()
ua_flat = torch.cat([u.flatten() for u in out["u_a"]]).cpu(); ue_flat = torch.cat([u.flatten() for u in out["u_e"]]).cpu()
u = torch.maximum(ua_flat, ue_flat)
q = u.quantile(torch.tensor([0.5, 0.9]))
print(f"\nrouted edges per node: mean {deg.mean():.2f}; nodes with U below median {deg[u <= q[0]].mean():.2f}, "
      f"top-10% U {deg[u >= q[1]].mean():.2f}  (budget grows with uncertainty)")
'''),
    code(r'''
from src.engine import LocationCache
from src.models.postprocess import postprocess
with torch.no_grad():
    det = postprocess(out, LocationCache(model.det_head)(out["head"]["cls"]), model.strides, x.shape[-2:], **meta["inference"])[0]
det = {k: v.cpu().numpy() for k, v in det.items()}
keep = det["scores"] >= 0.3
names_c = ("car", "truck", "bus")
print(f"{int(keep.sum())} detections with score >= 0.3 (ground truth: {len(sample['boxes'])} vehicles)")
show_table(["#", "class", "score", "box (x1,y1,x2,y2)", "box std (px)", "Ua", "Ue"],
           [[i, names_c[int(det['labels'][i])], f"{det['scores'][i]:.2f}", tuple(int(v) for v in det['boxes'][i]),
             f"{det['box_std'][i].mean():.1f}", f"{det['u_a'][i]:.3f}", f"{det['u_e'][i]:.4f}"]
            for i in np.where(keep)[0][:12]], "First 12 detections: every detection carries a box std, Ua and Ue")
'''),
    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
img = sample["image"].permute(1, 2, 0).numpy()
axes[0].imshow(img); axes[0].set_title("input frame")
for b, l, s in zip(det["boxes"][keep], det["labels"][keep], det["scores"][keep]):
    axes[0].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                    ec=("#00e676", "#ff9100", "#2979ff")[int(l)], lw=0.8))
im1 = axes[1].imshow(out["u_a"][0][0, 0].cpu(), cmap="magma"); axes[1].set_title("C1 aleatoric $U_a$ (P3)")
im2 = axes[2].imshow(out["u_e"][0][0, 0].cpu(), cmap="viridis"); axes[2].set_title("C1 epistemic $U_e$ (P3)")
plt.colorbar(im1, ax=axes[1], fraction=0.025); plt.colorbar(im2, ax=axes[2], fraction=0.025)
for a in axes: a.axis("off")
plt.tight_layout(); plt.show()
'''),
    md("## 7. Training entry point\n`scripts/train.py` implements the curriculum, the loss-spike and gradient guards, the stage guard and exact resume. "
       "Training commands:\n```bash\npython scripts/train.py --config configs/experiments/uavdt_primary.yaml\n"
       "python scripts/train.py --config configs/experiments/uavdt_stage4_residual.yaml --resume saved_models/uavdt_ep54_before_collapse.pth\n```"),
    code(r'''
show_file("scripts/train.py")
'''),
]

# --------------------------------------------------------------------------------------------- notebook 2
NB2 = [
    md(r'''
# UCR-MFG: Experiments and Results on UAVDT
This notebook contains the experimental results of the paper with their outputs: dataset statistics, training curves,
test results (single view and test-time augmentation), the comparison with the plain detector, the breakdown by class,
weather and altitude, the quality of the uncertainty estimates, the TTA selection, sample detections and inference speed.

All numbers are read from the saved result files (`results/`, `checkpoints/*/train_log.jsonl`); sections 8–9 run the
final model live.
'''),
    md("## 1. Setup"),
    code(SETUP),
    md("## 2. Dataset: UAVDT"),
    code(r'''
from collections import Counter
from src.data.uavdt import UAVDTDataset
train = UAVDTDataset("dataset/UAVDT-Benchmark-M", "train", (544, 1024), is_train=False, frame_stride=1)
test = UAVDTDataset("dataset/UAVDT-Benchmark-M", "test", (544, 1024), is_train=False, frame_stride=1)
train30 = UAVDTDataset("dataset/UAVDT-Benchmark-M", "train", (544, 1024), is_train=False, frame_stride=30)
eval_idx = list(range(2, len(test), 5))
seqs = lambda d: sorted({s for s, _, _ in d.frames})
show_table(["split", "videos", "frames", "used"], [
    ["train", len(seqs(train)), f"{len(train.frames):,}", f"{len(train30):,} (every 30th frame)"],
    ["test", len(seqs(test)), f"{len(test.frames):,}", f"{len(eval_idx):,} (every 5th frame from index 2)"],
    ["total", len(seqs(train)) + len(seqs(test)), f"{len(train.frames) + len(test.frames):,}", ""]], "UAVDT splits")
print()
tot, ev = Counter(s for s, _, _ in test.frames), Counter(test.frames[i][0] for i in eval_idx)
show_table(["sequence", "frames", "tested", "weather", "altitude"],
           [[s, tot[s], ev[s], ", ".join(test.sequence_attributes[s]["weather"]), ", ".join(test.sequence_attributes[s]["altitude"])]
            for s in sorted(tot)], "Test videos")
'''),
    md("## 3. Training curves\nStages 1–3 come from `uavdt_primary` (epochs 1–55, before the recipe change at epoch 55), stage 4 from "
       "`uavdt_stage4_residual` (epochs 55–100, resumed from the epoch-54 checkpoint). The monitoring set has 559 frames.\n\n"
       "*Reading the plots:* the total loss jumps at epoch 55 because the variance loss was changed to β-NLL with a soft floor at that point "
       "(values before and after are not comparable); the dip at epoch 65 is the stage-4 drop that the stage guard rolled back; "
       "vertical lines mark the stage boundaries. The orange curve is the failed convex-gate design (Table V of the full paper)."),
    code(r'''
def load_log(run):
    return [json.loads(l) for l in open(f"checkpoints/{run}/train_log.jsonl")]
prim, resid, frozen, s1 = (load_log(r) for r in ("uavdt_primary", "uavdt_stage4_residual", "uavdt_stage4_frozen", "uavdt_stage1_rerun"))
last = {r["epoch"]: r for r in prim if r["epoch"] < 55}       # resumed epochs appear twice in the log; keep the last row
last.update({r["epoch"]: r for r in resid})
final = [last[e] for e in sorted(last)]
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ep = [r["epoch"] for r in final]
ax[0].plot(ep, [r["loss_det"] for r in final], label="detection loss")
ax[0].plot(ep, [r["loss_total"] for r in final], label="total loss")
ax[0].set_xlabel("epoch"); ax[0].set_title("Training losses (final model)"); ax[0].legend()
for name, rows, st in (("final model", final, "-o"), ("stage-4 convex gate (frozen)", frozen, "--s"), ("stage 1 re-run", s1, ":^")):
    v = [r for r in rows if "val_mAP" in r]
    ax[1].plot([r["epoch"] for r in v], [100 * r["val_mAP"] for r in v], st, ms=3, label=name)
    ax[2].plot([r["epoch"] for r in v], [r["val_D-ECE"] for r in v], st, ms=3, label=name)
for a, title in ((ax[1], "Monitoring mAP"), (ax[2], "Monitoring D-ECE (lower is better)")):
    for e in (20.5, 40.5, 60.5):
        a.axvline(e, color="grey", lw=0.6, ls=":")
    a.set_xlabel("epoch"); a.set_title(title); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
secs = {}
for r in final:
    secs.setdefault(r["stage"], []).append(r["epoch_seconds"])
show_table(["stage", "epochs logged", "mean s/epoch"], [[k, len(v), f"{np.mean(v):.0f}"] for k, v in sorted(secs.items())],
           f"Training time: {sum(sum(v) for v in secs.values()) / 3600:.1f} h logged for the final model")
'''),
    code(r'''
guard = json.load(open("checkpoints/uavdt_stage4_residual/stage4_guard.json"))
print(f"Stage-4 guard: reference mAP {guard['ref_mAP']:.4f}, rollbacks: {guard['rollbacks']}")
show_table(["epoch", "monitoring mAP", "action"], [[h["epoch"], f"{h['mAP']:.4f}", h["action"]] for h in guard["history"]])
'''),
    md("## 4. Main result on the 3,318 test frames (Table II)"),
    code(r'''
def metrics(path, key="default"):
    return json.load(open(path))["metrics"][key]
sv = metrics("results/final_model/test_single_view.json")
tta = metrics("results/final_model/test_tta.json", "tuned")
pct = lambda m: [f"{100 * m[k]:.1f}" for k in ("mAP", "AP50", "AP75", "APs", "APm", "APl")]
rows = [["UCR-MFG (single view)"] + pct(sv) + [f"{sv['D-ECE']:.3f}"],
        ["UCR-MFG + TTA"] + pct(tta) + [f"{tta['D-ECE']:.3f}"],
        ["DMNet*", "14.7", "24.6", "16.3", "9.3", "26.2", "35.2", "-"],
        ["GLSAN*", "17.0", "28.1", "18.8", "-", "-", "-", "-"],
        ["AMRNet*", "18.2", "30.4", "19.8", "10.3", "31.3", "33.5", "-"]]
show_table(["method", "mAP", "AP50", "AP75", "APs", "APm", "APl", "D-ECE"], rows,
           "Table II (ours: results/final_model/test_single_view.json and test_tta.json)")
print("\n* published, ResNet-50: 23,258 training and 15,069 test images at 1000x600; not directly comparable; '-': not reported.")
'''),
    md("## 5. Comparison with the plain detector (Table IV)\nThe baseline is FCOS + C1 with the same backbone, heads, data and loss, without the graph modules (re-trained stage 1)."),
    code(r'''
base = metrics("results/stages/stage1/test.json")
rows = []
for name, m in (("FCOS + C1 (baseline)", base), ("UCR-MFG (full)", sv)):
    rows.append([name, f"{100*m['mAP']:.1f}", f"{100*m['AP50']:.1f}", f"{m['D-ECE']:.3f}", f"{m['AUROC']['u_a']:.3f}", f"{m['AUROC']['u_e']:.3f}"])
show_table(["model", "mAP", "AP50", "D-ECE", "AUROC Ua", "AUROC Ue"], rows, "Table IV")
print(f"\nmAP change {100*(sv['mAP']-base['mAP']):+.2f} points (exact); D-ECE {100*(sv['D-ECE']/base['D-ECE']-1):+.0f}%; "
      f"AUROC Ua {base['AUROC']['u_a']:.3f} -> {sv['AUROC']['u_a']:.3f}; AUROC Ue {base['AUROC']['u_e']:.3f} -> {sv['AUROC']['u_e']:.3f}")
'''),
    code(r'''
stages = [("1: FCOS + C1", "results/stages/stage1/test.json", 20), ("2: + fuzzy graph", "results/stages/stage2/test.json", 35),
          ("3: + C2, C3", "results/stages/stage3/test.json", 60), ("4: + C4 (final)", "results/final_model/test_single_view.json", 100)]
show_table(["model after stage", "epoch", "mAP", "AP50", "D-ECE", "AUROC Ua", "AUROC Ue"],
           [[n, e, f"{100*m['mAP']:.1f}", f"{100*m['AP50']:.1f}", f"{m['D-ECE']:.3f}", f"{m['AUROC']['u_a']:.3f}", f"{m['AUROC']['u_e']:.3f}"]
            for n, p, e in stages for m in [metrics(p)]], "All curriculum stages on the test frames (single view)")
'''),
    md("## 6. Breakdown by class, weather and altitude (Table III)"),
    code(r'''
rows = [[c, f"{100*a:.1f}", f"{100*b:.1f}"] for c, a, b in zip(("car", "truck", "bus"), sv["per_class_AP"], tta["per_class_AP"])]
show_table(["class", "AP single", "AP + TTA"], rows, "Per class")
print()
order = ["weather:daylight", "weather:night", "weather:fog", "altitude:low", "altitude:medium", "altitude:high"]
show_table(["condition", "test frames", "mAP single", "mAP + TTA"],
           [[k, sv["conditions"][k]["num_images"], f"{100*sv['conditions'][k]['mAP']:.1f}", f"{100*tta['conditions'][k]['mAP']:.1f}"] for k in order],
           "Per condition")
fig, ax = plt.subplots(figsize=(8, 3))
xs = np.arange(len(order))
ax.bar(xs - 0.2, [100*sv["conditions"][k]["mAP"] for k in order], 0.4, label="single view")
ax.bar(xs + 0.2, [100*tta["conditions"][k]["mAP"] for k in order], 0.4, label="+ TTA")
ax.set_xticks(xs); ax.set_xticklabels([k.split(":")[1] for k in order]); ax.set_ylabel("mAP"); ax.legend()
ax.set_title("mAP by weather and altitude"); plt.tight_layout(); plt.show()
'''),
    md("## 7. Uncertainty quality\nAUROC with which each score separates false from true detections (higher is better), and calibration error D-ECE."),
    code(r'''
keys = ["u_a", "u_e", "box_std", "1-score"]
show_table(["score", "single view", "+ TTA"], [[k, f"{sv['AUROC'][k]:.3f}", f"{tta['AUROC'][k]:.3f}"] for k in keys], "Error-detection AUROC")
print(f"\nD-ECE: single view {sv['D-ECE']:.3f}, + TTA {tta['D-ECE']:.3f}")
'''),
    md("## 8. Test-time augmentation: how the setting was chosen\nTTA is not trained. Settings were scored on 554 tuning frames (every 30th test frame, disjoint from the 3,318 evaluation frames); "
       "the chosen setting is in `results/final_model/stage4_end_choice.json`."),
    code(r'''
tune = json.load(open("results/final_model/stage4_end_tune.json"))
choice = json.load(open("results/final_model/stage4_end_choice.json"))
def desc(s):
    v = "+".join(f"{x:g}x" for x in s["scales"]) + (" +flip" if s["flip"] else "")
    m = f"NMS {s['nms_iou']}" if s["merge"] == "nms" else f"Matrix-NMS s={s['sigma']}"
    return f"{v}, {m}, thr {s['score_thresh']}"
print(f"{len(tune['rows'])} settings scored on {tune['meta'].get('num_images', 554)} tuning frames; default mAP {tune['default']['mAP']:.4f}")
top = sorted(tune["rows"], key=lambda r: -r["mAP"])[:8]
show_table(["setting", "tuning mAP", "AP50", "cost"], [[desc(r["setting"]), f"{r['mAP']:.4f}", f"{r['AP50']:.4f}", f"{r['cost']:.2f}x"] for r in top],
           "Top settings on the tuning frames")
print(f"\nchosen setting (rule: {choice['tuned_on']['rule']}): {desc(choice['by_fold']['0'])}")
print(f"test result: single view {100*sv['mAP']:.1f} mAP -> with TTA {100*tta['mAP']:.1f} mAP")
'''),
    md("## 9. Sample detections: single view vs TTA (live)\nThe four frames of the paper figure (daylight dense traffic, night, fog, high altitude), detections with score ≥ 0.3; green car, orange truck, blue bus."),
    code(r'''
from tune_inference import DEFAULT_SETTING, FIELDS_FULL, load_model, merge, run_views, view_keys_for
from src.engine import LocationCache
from make_qualitative import f1
model, meta = load_model(["models/stage4_ep100_final.pth"], device)
loc = LocationCache(model.det_head)
tta_setting = choice["by_fold"]["0"]
frames = [("Daylight, dense", "M0403", 169), ("Night", "M1001", 202), ("Fog", "M1009", 295), ("High altitude", "M0701", 204)]
by_id = {(s, n): i for i, (s, _, n) in enumerate(test.frames)}
fig, axes = plt.subplots(2, 4, figsize=(18, 5.4))
rows = []
for col, (title, seq, num) in enumerate(frames):
    i = by_id[(seq, num)]; s = test[i]
    for row, (mode, st) in enumerate((("single", DEFAULT_SETTING), ("TTA", tta_setting))):
        with torch.no_grad():
            v = run_views(model, loc, s["image"], st["scales"], st["flip"], device, 0, i, s["ignore_mask"].numpy(), FIELDS_FULL)
        d = merge(v, view_keys_for(st), st, FIELDS_FULL)
        k = (d["scores"] >= 0.3) & ~d["ignored"]
        ax = axes[row, col]; ax.imshow(s["image"].permute(1, 2, 0).numpy())
        for b, l in zip(d["boxes"][k], d["labels"][k]):
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=("#00e676", "#ff9100", "#2979ff")[int(l)], lw=0.6))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(title)
        if col == 0: ax.set_ylabel("UCR-MFG" if row == 0 else "+ TTA")
        rows.append([title, mode, len(s["boxes"]), int(k.sum()), f"{f1(d, s['boxes'].numpy(), 0.3):.2f}"])
plt.tight_layout(); plt.show()
show_table(["condition", "mode", "GT vehicles", "detections >= 0.3", "F1 (IoU 0.5)"], rows)
'''),
    md("## 10. Inference speed"),
    code(r'''
from src.models.uncertainty_head import mc_generator
model.eval()
times = []
with torch.no_grad():
    for j, i in enumerate(eval_idx[:25]):
        x = test[i]["image"].unsqueeze(0).to(device)
        t0 = time.time(); o = model(x, generator=mc_generator(0, 0, i, device))
        _ = o["head"]["cls"][0].sum().item()
        if j >= 5: times.append(time.time() - t0)
print(f"forward pass incl. 10 MC-dropout passes: median {np.median(times):.3f} s per image on {device} "
      f"({1/np.median(times):.1f} images/s, {len(times)} frames, batch 1)")
'''),
    md("## 11. Reproduce the test numbers\n```bash\n# single view and TTA on the 3,318 test frames\n"
       "python scripts/tune_inference.py test --weights models/stage4_ep100_final.pth \\\n"
       "  --choice results/final_model/stage4_end_choice.json --stride 5 --offset 2\n```\n"
       "The `default` column of the output is the single-view result (17.5 mAP) and the `tuned` column the TTA result (18.9 mAP)."),
]

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2"]
    if "1" in which:
        run(NB1, os.path.join(ROOT, "notebooks/01_UCR_MFG_implementation.ipynb"))
    if "2" in which:
        run(NB2, os.path.join(ROOT, "notebooks/02_UCR_MFG_experiments_results.ipynb"))
