"""Summarise train_log.jsonl and flag when val mAP has not improved for N epochs.

Usage: python scripts/check_plateau.py [--log checkpoints/uavdt_primary/train_log.jsonl] [--patience 10]
Exit code 2 when a plateau is detected (handy for cron/alerts).
"""
import argparse
import json
import sys

STAGE_STARTS = (21, 41, 61)  # new modules switch on here; a short dip is expected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="checkpoints/uavdt_primary/train_log.jsonl")
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    try:
        rows = [json.loads(l) for l in open(args.log) if l.strip()]
    except FileNotFoundError:
        print(f"No log yet at {args.log} (written after the first epoch finishes).")
        return 0
    if not rows:
        print("Log is empty.")
        return 0

    print(f"{'ep':>4} {'stg':>3} {'loss':>8} {'mAP':>7} {'AP50':>7} {'APs':>7} {'min':>6}")
    for r in rows[-15:]:
        f = lambda k: f"{r[k] * 100:6.2f}%" if k in r else "     -"
        print(f"{r['epoch']:>4} {r['stage']:>3} {r.get('loss_total', float('nan')):8.4f} "
              f"{f('val_mAP')} {f('val_AP50')} {f('val_APs')} {r['epoch_seconds'] / 60:6.1f}")

    vals = [r for r in rows if "val_mAP" in r]
    if not vals:
        return 0
    best = max(vals, key=lambda r: r["val_mAP"])
    last = rows[-1]["epoch"]
    since = last - best["epoch"]
    remaining = (rows[-1].get("max_epochs", 100) - last) * sum(r["epoch_seconds"] for r in rows[-3:]) / min(3, len(rows))
    print(f"\nBest mAP {best['val_mAP'] * 100:.2f}% (AP50 {best['val_AP50'] * 100:.2f}%, APs {best['val_APs'] * 100:.2f}%) "
          f"at epoch {best['epoch']}; {since} epoch(s) since. ETA ~{remaining / 3600:.1f} h.")

    if since >= args.patience:
        note = " (a stage transition happened in this window; a temporary dip is expected)" \
            if any(best["epoch"] < s <= last for s in STAGE_STARTS) else ""
        print(f"\n⚠️  PLATEAU: no val mAP improvement for {since} epochs{note}.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
