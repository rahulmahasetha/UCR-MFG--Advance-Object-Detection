import os
import sys

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from contextlib import nullcontext

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.data.resumable import ResumableEpochSampler, SeededDataset
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

SEED = 42


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
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_for(epoch: int, stages: dict) -> int:
    for name, spec in stages.items():
        first, last = spec["epochs"]
        if first <= epoch <= last:
            return int(name.rsplit("_", 1)[-1])
    return 4


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
    img_size = tuple(args.img_size or ds_cfg["image_size"])
    num_classes = ds_cfg["classes"]["num_classes"]
    benefit_interval = model_cfg["loss"]["benefit_step_interval"]
    save_every = args.save_every if args.save_every is not None else ckpt_cfg.get("save_every_steps", 0)
    ckpt_dir = ckpt_cfg["save_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[INIT] UCR-MFG training on {device}")

    root = ds_cfg["root_dir"]
    train_set = VisDroneDataset(root, "train", img_size, is_train=True)
    sampler = ResumableEpochSampler(len(train_set), seed=SEED)
    train_loader = make_loader(train_set, batch_size, device, sampler)
    val_loader = make_loader(VisDroneDataset(root, "val", img_size, is_train=False), val_cfg.get("batch_size", batch_size), device)
    batches_per_epoch = len(train_set) // batch_size

    model = UCRMFG.from_config(model_cfg, num_classes).to(device)
    loss_fn = UCRMFGCombinedLoss.from_config(model_cfg).to(device)
    locations = LocationCache(model.det_head)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    use_amp = device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    def autocast():
        return torch.autocast(device.type, dtype=torch.float16) if use_amp else nullcontext()

    start_epoch, start_batch, global_step, best_metric, resumed_losses = 1, 0, 0, 0.0, {}
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

    for epoch in range(start_epoch, max_epochs + 1):
        stage = stage_for(epoch, train_cfg["stages"])
        model.set_curriculum_stage(stage)
        model.train()
        first_batch = start_batch if epoch == start_epoch else 0
        epoch_losses = dict(resumed_losses) if epoch == start_epoch else {}
        sampler.set_epoch(epoch, start=first_batch * batch_size)
        progress.start_epoch(epoch)
        if first_batch:
            print(f"[RESUME] Skipping {first_batch} completed batches of epoch {epoch}")

        for offset, batch in enumerate(train_loader):
            batch_idx = first_batch + offset
            images = batch["images"].to(device)
            boxes = [b.to(device) for b in batch["boxes"]]
            dense = stage >= 3 and model.router.enabled and global_step % benefit_interval == 0

            with autocast():
                outputs = model(images, generator=mc_generator(SEED, epoch, batch_idx, device), dense_routing=dense)
                locs = locations(outputs["head"]["cls"])
                losses = loss_fn(outputs, assign_targets(model, locs, batch, device), locs, boxes)
            if dense:
                benefit = measure_benefit(losses["loss_det"], outputs["routing"], scaler.get_scale() if scaler else 1.0)
                losses = loss_fn.add_benefit(losses, outputs, benefit)

            loss = losses["loss_total"] / accum_steps
            (scaler.scale(loss) if scaler else loss).backward()

            done = batch_idx + 1
            stepped = done % accum_steps == 0 or done == batches_per_epoch
            if stepped:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            global_step += 1

            values = {k: float(v) for k, v in losses.items()}
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

        progress.end_epoch(epoch, {k: v / batches_per_epoch for k, v in epoch_losses.items()}, val_map)
        save("latest.pth", epoch, stage)
        save(f"epoch_{epoch:03d}.pth", epoch, stage)
        purge_old_checkpoints(ckpt_dir, keep_last_n=ckpt_cfg["keep_last_n"])
        if interrupt.interrupted:
            print(f"\n[INTERRUPT] Epoch {epoch} complete and saved; continue with --resume latest.")
            sys.exit(0)

    print(f"\n[COMPLETE] Finished {max_epochs} epochs; best val mAP {best_metric:.4f}")


if __name__ == "__main__":
    main()
