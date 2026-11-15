import os
import sys

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import random
import time
from contextlib import nullcontext

import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.data.dota import DOTADataset
from src.data.resumable import ResumableEpochSampler, SeededDataset
from src.data.uavdt import UAVDTDataset
from src.data.visdrone import VisDroneDataset
from src.engine import LocationCache, assign_targets, evaluate, measure_benefit
from src.losses.total_loss import UCRMFGCombinedLoss
from src.models.uncertainty_head import mc_generator
from src.models.ucr_mfg import UCRMFG
from src.utils.checkpoint import (
    GracefulInterruptHandler,
    create_checkpoint_payload,
    load_checkpoint,
    purge_old_checkpoints,
    save_atomic,
)
from src.utils.progress import TrainingProgressDisplay

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="UCR-MFG staged training")
    parser.add_argument("--config", default="configs/experiments/visdrone_primary.yaml")
    parser.add_argument("--resume", default=None, help="'latest', 'best' or a checkpoint path")
    parser.add_argument("--save-every", type=int, default=None, help="Save latest.pth every N batches (0 disables)")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=None)
    parser.add_argument("--max-train-images", type=int, default=None, help="Train on a fixed random subset of N images")
    parser.add_argument("--max-val-images", type=int, default=None, help="Validate on a fixed random subset of N images")
    return parser.parse_args()


def limit_images(dataset, max_images, seed: int):
    """Fixed (seeded) random subset so resumed runs see the same images."""
    if not max_images or max_images >= len(dataset):
        return dataset
    indices = sorted(random.Random(seed).sample(range(len(dataset)), max_images))
    return Subset(dataset, indices)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_for(epoch: int, stages: dict) -> int:
    for name, spec in stages.items():
        first, last = spec["epochs"]
        if first <= epoch <= last:
            return int(name.rsplit("_", 1)[-1])
    return 4


class SpikeGuard:
    """Skips updates from batches whose loss or gradient norm jumps far above the recent norm.

    Stage-3 training diverged twice at epoch 55 (different batches each time): a few outlier batches
    moved the shared weights and detection never recovered. Outliers are judged against a rolling
    median/MAD of accepted batches; after `max_consecutive` skips in a row the baseline is reset,
    so a genuine level shift (e.g. new losses at a stage change) is accepted rather than skipped forever.
    """

    def __init__(self, window: int = 200, warmup: int = 50, loss_k: float = 5.0, loss_floor: float = 2.0,
                 grad_ratio: float = 5.0, max_consecutive: int = 20):
        self.window, self.warmup = window, warmup
        self.loss_k, self.loss_floor, self.grad_ratio = loss_k, loss_floor, grad_ratio
        self.max_consecutive = max_consecutive
        self.reset()

    def reset(self) -> None:
        self.losses, self.norms, self.consecutive = [], [], 0

    @staticmethod
    def _median(xs):
        return float(np.median(xs))

    def loss_threshold(self):
        if len(self.losses) < self.warmup:
            return None
        med = self._median(self.losses)
        mad = self._median([abs(x - med) for x in self.losses])
        return med + max(self.loss_k * 1.4826 * mad, self.loss_floor)

    def loss_is_spike(self, value: float) -> bool:
        limit = self.loss_threshold()
        return not np.isfinite(value) or (limit is not None and value > limit)

    def norm_is_spike(self, norm: float) -> bool:
        if not np.isfinite(norm):
            return True
        return len(self.norms) >= self.warmup and norm > self.grad_ratio * self._median(self.norms)

    def accept_loss(self, value: float) -> None:
        self.losses = (self.losses + [value])[-self.window:]

    def accept_norm(self, norm: float) -> None:
        self.norms = (self.norms + [norm])[-self.window:]
        self.consecutive = 0

    def record_skip(self) -> bool:
        """Returns True when the baseline was reset after too many consecutive skips."""
        self.consecutive += 1
        if self.consecutive >= self.max_consecutive:
            self.reset()
            return True
        return False


def apply_stage_options(model, spec: dict, epoch: int, first_batch: int) -> None:
    """Optional per-stage settings from the experiment config.

    freeze: [module names]   -> those modules stop training (e.g. keep the converged detector fixed while
                                the graph modules learn; UAVDT stage 4 otherwise drifted and overfit).
    fusion_gate_bias: float  -> re-initialise the fusion gate bias when the stage starts; 4.0 gives
                                lambda ~ 0.98, so enabling fusion barely changes the detector's input
                                (the default 2.0 -> lambda 0.88 scaled visual features down by 12% at once).
    """
    for name in spec.get("freeze", []):
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)
    if spec.get("fusion_mode"):
        model.fusion_gate.mode = spec["fusion_mode"]  # set every epoch so resumed runs keep it
    bias = spec.get("fusion_gate_bias")
    if bias is not None and epoch == spec["epochs"][0] and first_batch == 0:
        last = model.fusion_gate.gate_conv[-1]
        torch.nn.init.constant_(last.bias, float(bias))
        if spec.get("fusion_gate_zero_weights"):
            # Without this the random weights add noise, so lambda only approximately equals sigmoid(bias).
            torch.nn.init.zeros_(last.weight)
        print(f"[STAGE] fusion mode {model.fusion_gate.mode}, gate bias {bias} "
              f"(lambda = {torch.sigmoid(torch.tensor(float(bias))):.3f}"
              f"{', exact' if spec.get('fusion_gate_zero_weights') else ', approx'})")


class StageGuard:
    """Keeps a curriculum stage from ending below where it started (config: stages.stage_N.guard).

    At the stage start the model is validated once (the reference, e.g. the stage-3 detector) and its weights
    are saved. After every validation in the stage:
      - mAP above the best so far        -> save stage{N}_best.pth
      - mAP below best - tolerance       -> roll back to the best weights, reset optimizer moments, scale lr down
      - more than max_rollbacks rollbacks -> restore the best weights and end training early
    State lives in stage{N}_guard.json so it survives --resume.
    """

    def __init__(self, ckpt_dir: str, stage: int, cfg: dict):
        self.ckpt_dir, self.stage = ckpt_dir, stage
        self.tolerance = float(cfg.get("tolerance", 0.003))
        self.lr_factor = float(cfg.get("lr_factor", 0.5))
        self.max_rollbacks = int(cfg.get("max_rollbacks", 3))
        self.path = os.path.join(ckpt_dir, f"stage{stage}_guard.json")
        self.state = None
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.state = json.load(f)

    def _write(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def start(self, ref_map: float, weights: str) -> None:
        self.state = {"ref_mAP": ref_map, "best_mAP": ref_map, "best_epoch": "start", "best_path": weights,
                      "rollbacks": 0, "history": []}
        self._write()
        print(f"[STAGE-GUARD] stage {self.stage} reference mAP {ref_map:.4f}; rollback if mAP < best - {self.tolerance}")

    def check(self, epoch: int, val_map: float) -> str:
        """Returns 'improved', 'ok', 'rollback' or 'stop'."""
        st = self.state
        if val_map < st["best_mAP"] - self.tolerance:
            st["rollbacks"] += 1
            action = "stop" if st["rollbacks"] > self.max_rollbacks else "rollback"
        elif val_map > st["best_mAP"]:
            st["best_mAP"], st["best_epoch"] = val_map, epoch
            st["best_path"] = os.path.join(self.ckpt_dir, f"stage{self.stage}_best.pth")
            action = "improved"
        else:
            action = "ok"
        st["history"].append({"epoch": epoch, "mAP": val_map, "action": action})
        self._write()
        return action

    def restore_best(self, model, device) -> None:
        state = torch.load(self.state["best_path"], map_location=device, weights_only=False)["model_state_dict"]
        model.load_state_dict(state)


def module_grad_norms(model) -> dict:
    norms = {}
    for name, module in model.named_children():
        grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
        if grads:
            norms[name] = round(float(torch.stack(grads).norm()), 4)
    return norms


def make_loader(dataset, batch_size: int, device: torch.device, sampler=None) -> DataLoader:
    workers = 0 if sys.platform == "win32" else 2
    return DataLoader(
        SeededDataset(dataset) if sampler is not None else dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=sampler is not None,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )


def main():
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    ds_cfg = load_yaml(exp_cfg["dataset_config"])
    model_cfg = load_yaml(exp_cfg["model_config"])
    run_cfg = {"experiment": exp_cfg, "dataset": ds_cfg, "model": model_cfg}
    train_cfg, val_cfg, ckpt_cfg = exp_cfg["train"], exp_cfg.get("val", {}), exp_cfg["checkpoint"]

    device = torch.device(args.device)
    max_epochs = args.epochs or train_cfg["max_epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    accum_steps = args.accumulate_grad_batches or train_cfg.get("accumulate_grad_batches", 1)
    img_size = tuple(args.img_size or train_cfg.get("img_size") or ds_cfg["image_size"])
    num_classes = ds_cfg["classes"]["num_classes"]
    benefit_interval = model_cfg["loss"]["benefit_step_interval"]
    save_every = args.save_every if args.save_every is not None else ckpt_cfg.get("save_every_steps", 0)
    ckpt_dir = ckpt_cfg["save_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"[INIT] UCR-MFG training on {device} (seed {seed})")

    root = ds_cfg["root_dir"]
    if ds_cfg["dataset_name"] == "UAVDT":
        # UAVDT has only official train/test sequence splits; validation uses (strided) test sequences.
        train_set = UAVDTDataset(root, "train", img_size, is_train=True, frame_stride=train_cfg.get("frame_stride", 1))
        val_set = UAVDTDataset(root, "test", img_size, is_train=False, frame_stride=val_cfg.get("frame_stride", 1))
    elif ds_cfg["dataset_name"] == "DOTA-v1.5":
        train_set = DOTADataset(root, "train", img_size, is_train=True)
        val_set = DOTADataset(root, "validation", img_size, is_train=False)
    else:
        train_set = VisDroneDataset(root, "train", img_size, is_train=True)
        val_set = VisDroneDataset(root, "val", img_size, is_train=False)
    train_set = limit_images(train_set, args.max_train_images or train_cfg.get("max_images"), seed)
    val_set = limit_images(val_set, args.max_val_images or val_cfg.get("max_images"), seed)
    print(f"[DATA] {ds_cfg['dataset_name']}: {len(train_set)} train / {len(val_set)} val images at {img_size}")
    sampler = ResumableEpochSampler(len(train_set), seed=seed)
    train_loader = make_loader(train_set, batch_size, device, sampler)
    val_loader = make_loader(val_set, val_cfg.get("batch_size", batch_size), device)
    batches_per_epoch = len(train_set) // batch_size

    model = UCRMFG.from_config(model_cfg, num_classes).to(device)
    loss_fn = UCRMFGCombinedLoss.from_config(model_cfg).to(device)
    locations = LocationCache(model.det_head)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    use_amp = device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    def autocast():
        return torch.autocast(device.type, dtype=torch.float16) if use_amp else nullcontext()

    start_epoch, start_batch, global_step, best_metric, resumed_losses = 1, 0, 0, 0.0, {}
    if not args.resume and train_cfg.get("init_weights") == "coco_fcos":
        info = model.load_coco_fcos(ds_cfg["classes"].get("coco_class_ids"))
        print(f"[INIT] Loaded {info['loaded_tensors']} tensors from COCO-trained FCOS-R50-FPN")
    if args.resume:
        path = os.path.join(ckpt_dir, f"{args.resume}.pth") if args.resume in ("latest", "best") else args.resume
        info = load_checkpoint(path, model, optimizer, scheduler, scaler, current_cfg=run_cfg)
        start_epoch, start_batch = info["start_epoch"], info["start_batch"]
        global_step, best_metric, resumed_losses = info["global_step"], info["best_metric"], info["epoch_losses"]
        saved_batch_size = info["resume_batch_size"] or batch_size
        if start_batch and (saved_batch_size != batch_size or start_batch % accum_steps):
            raise ValueError(
                f"Mid-epoch resume needs the same batch size ({saved_batch_size}) and a batch index divisible by "
                f"accumulate_grad_batches ({accum_steps}); got batch size {batch_size}, batch {start_batch}."
            )

    def save(name: str, completed_epoch: int, stage: int, resume=None) -> None:
        payload = create_checkpoint_payload(
            model, optimizer, scheduler, scaler, completed_epoch, global_step, stage, best_metric, run_cfg, resume
        )
        save_atomic(payload, os.path.join(ckpt_dir, name))

    progress = TrainingProgressDisplay(total_epochs=max_epochs, num_batches=batches_per_epoch)
    interrupt = GracefulInterruptHandler()
    optimizer.zero_grad()
    guard = SpikeGuard()
    spike_log = os.path.join(ckpt_dir, "spikes.jsonl")
    prev_stage = None

    def log_spike(entry: dict) -> None:
        with open(spike_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    for epoch in range(start_epoch, max_epochs + 1):
        stage = stage_for(epoch, train_cfg["stages"])
        if stage != prev_stage:
            guard.reset()  # new modules/losses switch on: re-learn what "normal" looks like
            prev_stage = stage
        skipped_loss = skipped_grad = 0
        window_has_grad = False
        epoch_norms = []
        model.set_curriculum_stage(stage)
        model.train()
        first_batch = start_batch if epoch == start_epoch else 0
        stage_spec = train_cfg["stages"].get(f"stage_{stage}", {})
        apply_stage_options(model, stage_spec, epoch, first_batch)
        if stage_spec.get("freeze") and (epoch == start_epoch or epoch == stage_spec["epochs"][0]):
            print(f"[STAGE] stage {stage}: frozen modules {stage_spec['freeze']}")
        stage_guard = StageGuard(ckpt_dir, stage, stage_spec["guard"]) if stage_spec.get("guard") else None
        if stage_guard is not None and stage_guard.state is None:
            # Reference = this stage's starting model (identical to the previous stage's end with residual fusion).
            start_path = f"stage{stage}_start.pth"
            save(start_path, epoch - 1, stage)
            ref = evaluate(model, val_loader, device, model_cfg["inference"], num_classes,
                           max_batches=val_cfg.get("max_batches"))["mAP"]
            stage_guard.start(ref, os.path.join(ckpt_dir, start_path))
            model.train()
        epoch_losses = dict(resumed_losses) if epoch == start_epoch else {}
        sampler.set_epoch(epoch, start=first_batch * batch_size)
        progress.start_epoch(epoch)
        epoch_start = time.time()
        if first_batch:
            print(f"[RESUME] Skipping {first_batch} completed batches of epoch {epoch}")

        for offset, batch in enumerate(train_loader):
            batch_idx = first_batch + offset
            images = batch["images"].to(device)
            boxes = [b.to(device) for b in batch["boxes"]]
            dense = stage >= 3 and model.router.enabled and global_step % benefit_interval == 0

            with autocast():
                outputs = model(images, generator=mc_generator(seed, epoch, batch_idx, device), dense_routing=dense)
                locs = locations(outputs["head"]["cls"])
                losses = loss_fn(outputs, assign_targets(model, locs, batch, device), locs, boxes)
            if dense:
                benefit = measure_benefit(losses["loss_det"], outputs["routing"], scaler.get_scale() if scaler else 1.0)
                if benefit is not None:
                    losses = loss_fn.add_benefit(losses, outputs, benefit)
                else:
                    print(f"\n[BENEFIT] epoch {epoch} batch {batch_idx}: routing mask unused by loss_det -> benefit loss skipped")

            loss = losses["loss_total"] / accum_steps
            values = {k: float(v) for k, v in losses.items()}
            total_value = values["loss_total"]
            if guard.loss_is_spike(total_value):
                # Drop this batch's gradient entirely: no backward, so it never reaches the weights.
                skipped_loss += 1
                limit = guard.loss_threshold()
                reset = guard.record_skip()
                log_spike({"epoch": epoch, "stage": stage, "batch": batch_idx, "reason": "loss",
                           "threshold": limit, "baseline_reset": reset, **values})
                print(f"\n[GUARD] epoch {epoch} batch {batch_idx}: loss_total {total_value:.3f} > "
                      f"{limit if limit is None else round(limit, 3)} -> skipped" + (" (baseline reset)" if reset else ""))
            else:
                (scaler.scale(loss) if scaler else loss).backward()
                window_has_grad = True

            done = batch_idx + 1
            stepped = done % accum_steps == 0 or done == batches_per_epoch
            if stepped and window_has_grad:
                if scaler:
                    scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
                if guard.norm_is_spike(norm):
                    skipped_grad += 1
                    median = guard._median(guard.norms) if guard.norms else None
                    reset = guard.record_skip()
                    log_spike({"epoch": epoch, "stage": stage, "batch": batch_idx, "reason": "grad_norm",
                               "grad_norm": norm, "median_grad_norm": median, "baseline_reset": reset,
                               "module_grad_norms": module_grad_norms(model), **values})
                    print(f"\n[GUARD] epoch {epoch} batch {batch_idx}: grad norm {norm:.2f} "
                          f"(median {median if median is None else round(median, 2)}) -> step skipped")
                    if scaler:
                        scaler.update()  # inf/NaN still lowers the loss scale
                else:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    guard.accept_norm(norm)
                    guard.accept_loss(total_value)
                    epoch_norms.append(norm)
                optimizer.zero_grad()
                window_has_grad = False
            elif stepped:
                optimizer.zero_grad()
            if not guard.loss_is_spike(total_value) and not stepped:
                guard.accept_loss(total_value)
            global_step += 1

            for k, v in values.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            progress.update_batch(epoch, batch_idx, values, optimizer.param_groups[0]["lr"])

            del outputs, losses, loss
            if device.type == "mps" and batch_idx % 10 == 0:
                torch.mps.empty_cache()

            if not stepped or done == batches_per_epoch:
                continue
            resume = {"epoch": epoch, "batch": done, "batch_size": batch_size, "epoch_losses": epoch_losses}
            if interrupt.interrupted:
                save("interrupted.pth", epoch - 1, stage, resume)
                save("latest.pth", epoch - 1, stage, resume)
                print(f"\n[INTERRUPT] Saved epoch {epoch}, batch {done}/{batches_per_epoch} to {ckpt_dir}; "
                      f"continue with --resume latest.")
                sys.exit(0)
            if save_every and done % save_every < accum_steps:
                save("latest.pth", epoch - 1, stage, resume)

        scheduler.step()
        val_map = None
        guard_action = None
        if epoch % val_cfg.get("interval", 5) == 0 or epoch == max_epochs:
            metrics = evaluate(
                model, val_loader, device, model_cfg["inference"], num_classes,
                max_batches=val_cfg.get("max_batches"),
            )
            val_map = metrics["mAP"]
            print(
                f"\n[VAL] epoch {epoch} stage {stage}: mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} "
                f"APs {metrics['APs']:.4f} D-ECE {metrics['D-ECE']:.4f}"
            )
            if val_map > best_metric:
                best_metric = val_map
                save("best.pth", epoch, stage)
            if stage_guard is not None:
                guard_action = stage_guard.check(epoch, val_map)
                st = stage_guard.state
                if guard_action == "improved":
                    save(f"stage{stage}_best.pth", epoch, stage)
                    print(f"[STAGE-GUARD] new stage-{stage} best {val_map:.4f} -> stage{stage}_best.pth")
                elif guard_action in ("rollback", "stop"):
                    stage_guard.restore_best(model, device)
                    optimizer.state.clear()  # drop Adam moments of the bad trajectory
                    if guard_action == "rollback":
                        for g in optimizer.param_groups:
                            g["lr"] *= stage_guard.lr_factor
                        scheduler.base_lrs = [b * stage_guard.lr_factor for b in scheduler.base_lrs]
                    print(f"[STAGE-GUARD] mAP {val_map:.4f} < best {st['best_mAP']:.4f} - {stage_guard.tolerance}: "
                          f"restored weights from epoch {st['best_epoch']} ({os.path.basename(st['best_path'])}), "
                          + ("stopping early." if guard_action == "stop" else
                             f"lr x{stage_guard.lr_factor} (rollback {st['rollbacks']}/{stage_guard.max_rollbacks})"))

        avg_losses = {k: v / batches_per_epoch for k, v in epoch_losses.items()}
        progress.end_epoch(epoch, avg_losses, val_map)
        log_entry = {"epoch": epoch, "stage": stage, "lr": optimizer.param_groups[0]["lr"], "global_step": global_step,
                     "epoch_seconds": round(time.time() - epoch_start, 1), "resumed_mid_epoch": bool(first_batch),
                     "skipped_loss_spikes": skipped_loss, "skipped_grad_spikes": skipped_grad,
                     "grad_norm_median": round(float(np.median(epoch_norms)), 4) if epoch_norms else None,
                     "grad_norm_max": round(max(epoch_norms), 4) if epoch_norms else None, **avg_losses}
        if val_map is not None:
            log_entry.update({f"val_{k}": metrics[k] for k in ("mAP", "AP50", "AP75", "APs", "APm", "APl", "D-ECE")})
            if stage_guard is not None:
                log_entry["stage_guard"] = guard_action
        with open(os.path.join(ckpt_dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        save("latest.pth", epoch, stage)
        save(f"epoch_{epoch:03d}.pth", epoch, stage)
        if epoch == stage_spec.get("epochs", [0, -1])[1]:
            save(f"stage{stage}_end.pth", epoch, stage)  # kept permanently (purge only removes epoch_*.pth)
        purge_old_checkpoints(ckpt_dir, keep_last_n=ckpt_cfg["keep_last_n"])
        if interrupt.interrupted:
            print(f"\n[INTERRUPT] Epoch {epoch} complete and saved; continue with --resume latest.")
            sys.exit(0)
        if guard_action == "stop":
            print(f"\n[STAGE-GUARD] Stopped after epoch {epoch}; final weights = best stage-{stage} model "
                  f"(mAP {stage_guard.state['best_mAP']:.4f}).")
            break

    print(f"\n[COMPLETE] Finished {max_epochs} epochs; best val mAP {best_metric:.4f}")


if __name__ == "__main__":
    main()
