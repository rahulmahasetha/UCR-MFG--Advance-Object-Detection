import os
import sys

# Prevent macOS MPS memory allocation ceiling crash
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.visdrone import VisDroneDataset
from src.data.collate import collate_fn
from src.models.ucr_mfg import UCRMFG
from src.losses.total_loss import UCRMFGCombinedLoss
from src.utils.checkpoint import (
    save_atomic,
    load_checkpoint,
    create_checkpoint_payload,
    purge_old_checkpoints,
    GracefulInterruptHandler
)
from src.utils.progress import TrainingProgressDisplay
from src.utils.metrics import compute_dece


def parse_args():
    parser = argparse.ArgumentParser(description="UCR-MFG Staged Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/experiments/visdrone_primary.yaml", help="Experiment config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from 'latest', 'best', or custom path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, help="Image size [H, W]")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    ds_cfg = load_yaml(exp_cfg["dataset_config"])
    model_cfg = load_yaml(exp_cfg["model_config"])

    device = torch.device(args.device)
    print(f"[INIT] Launching UCR-MFG Training on Device: {device}")

    # Hyperparameters
    max_epochs = args.epochs or exp_cfg["train"]["max_epochs"]
    batch_size = args.batch_size or exp_cfg["train"]["batch_size"]
    accum_steps = args.accumulate_grad_batches or exp_cfg["train"].get("accumulate_grad_batches", 1)
    lr = float(exp_cfg["train"]["lr"])
    weight_decay = float(exp_cfg["train"]["weight_decay"])
    ckpt_dir = exp_cfg["checkpoint"]["save_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    img_size = tuple(args.img_size) if args.img_size else tuple(ds_cfg["image_size"])

    # 1. Dataset & DataLoader (Primary: VisDrone)
    root_dir = ds_cfg["root_dir"]
    train_dataset = VisDroneDataset(root_dir=root_dir, split="train", img_size=img_size, is_train=True)
    val_dataset = VisDroneDataset(root_dir=root_dir, split="val", img_size=img_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if sys.platform != "win32" else 0,
        persistent_workers=True if sys.platform != "win32" else False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True
    )

    # 2. Model & Combined Loss
    num_classes = ds_cfg["classes"]["num_classes"]
    model = UCRMFG(
        num_classes=num_classes,
        fpn_channels=model_cfg["model"]["fpn_channels"],
        graph_dim=model_cfg["model"]["graph_dim"],
        strides=model_cfg["model"]["strides"],
        dropout_rate=model_cfg["model"]["uncertainty"]["dropout_rate"],
        mc_passes=model_cfg["model"]["uncertainty"]["mc_passes"],
        pretrained=model_cfg["model"]["pretrained"]
    ).to(device)

    loss_fn = UCRMFGCombinedLoss(
        strides=model_cfg["model"]["strides"],
        lambda_u=model_cfg["loss"]["lambda_u"],
        lambda_graph=model_cfg["loss"]["lambda_graph"],
        lambda_inst=model_cfg["loss"]["lambda_inst"],
        lambda_benefit=model_cfg["loss"]["lambda_benefit"],
        lambda_consistency=model_cfg["loss"]["lambda_consistency"]
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    use_amp = device.type in ("cuda", "mps")
    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    amp_dtype = torch.float16
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    elif device.type == "mps":
        scaler = torch.amp.GradScaler("mps")
    else:
        scaler = None

    # 4. Checkpoint Resumption
    start_epoch = 1
    global_step = 0
    best_metric = 0.0

    if args.resume:
        resume_target = args.resume
        if resume_target in ("latest", "best"):
            resume_target = os.path.join(ckpt_dir, f"{resume_target}.pth")
        res_info = load_checkpoint(resume_target, model, optimizer, scheduler, scaler, current_cfg=exp_cfg)
        start_epoch = res_info["start_epoch"]
        global_step = res_info["global_step"]
        best_metric = res_info["best_metric"]

    # 5. Progress Display & Interrupt Handler
    progress = TrainingProgressDisplay(total_epochs=max_epochs, num_batches=len(train_loader))
    interrupt_handler = GracefulInterruptHandler()

    # Pre-generate spatial grid locations
    shapes = [(img_size[0] // s, img_size[1] // s) for s in model_cfg["model"]["strides"]]
    locations = model.det_head.get_grid_locations(shapes, device)

    optimizer.zero_grad()

    # 6. Training Loop (Epochs start_epoch to max_epochs)
    for epoch in range(start_epoch, max_epochs + 1):
        # Determine active curriculum stage
        if epoch <= 30:
            stage = 1
        elif epoch <= 60:
            stage = 2
        elif epoch <= 90:
            stage = 3
        else:
            stage = 4
        model.set_curriculum_stage(stage)

        progress.start_epoch(epoch)
        model.train()
        epoch_losses = {}

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            boxes = [b.to(device) for b in batch["boxes"]]
            labels = [l.to(device) for l in batch["labels"]]
            ignore_masks = batch["ignore_masks"].to(device)

            # Target assignment
            cls_t, reg_t, cen_t, pos_m = model.det_head.assign_targets(locations, boxes, labels, ignore_masks)
            targets = {
                "cls_targets": cls_t,
                "reg_targets": reg_t,
                "cen_targets": cen_t,
                "pos_masks": pos_m,
            }

            # Forward pass & loss computation
            if use_amp:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    outputs = model(images, base_seed=42, epoch=epoch, batch_idx=batch_idx)

                    benefit_grads = None
                    if stage >= 3 and (global_step % model_cfg["loss"]["benefit_step_interval"] == 0):
                        with torch.enable_grad():
                            soft_s = [r["s_ij"].clone().requires_grad_(True) for r in outputs["router_data"]]
                            loss_approx = sum([s.sigmoid().mean() for s in soft_s])
                            grads = torch.autograd.grad(loss_approx, soft_s, retain_graph=False)
                            benefit_grads = [-g for g in grads]

                    loss_dict = loss_fn(outputs, targets, locations, boxes, labels, benefit_grads)
            else:
                outputs = model(images, base_seed=42, epoch=epoch, batch_idx=batch_idx)

                benefit_grads = None
                if stage >= 3 and (global_step % model_cfg["loss"]["benefit_step_interval"] == 0):
                    with torch.enable_grad():
                        soft_s = [r["s_ij"].clone().requires_grad_(True) for r in outputs["router_data"]]
                        loss_approx = sum([s.sigmoid().mean() for s in soft_s])
                        grads = torch.autograd.grad(loss_approx, soft_s, retain_graph=False)
                        benefit_grads = [-g for g in grads]

                loss_dict = loss_fn(outputs, targets, locations, boxes, labels, benefit_grads)

            loss_total = loss_dict["loss_total"] / float(accum_steps)

            if scaler is not None:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            # Step optimizer upon reaching accumulation threshold
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            # Accumulate metrics
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else float(v))

            current_lr = optimizer.param_groups[0]["lr"]
            progress.update_batch(epoch, batch_idx, {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in loss_dict.items()}, current_lr)

            # Explicit memory cleanup to prevent MPS memory pooling
            del outputs, targets, loss_dict, loss_total, images, boxes, labels, ignore_masks
            if device.type == "mps" and (batch_idx % 10 == 0):
                torch.mps.empty_cache()

            # Check graceful interrupt
            if interrupt_handler.interrupted:
                emergency_payload = create_checkpoint_payload(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, stage, best_metric, exp_cfg
                )
                save_atomic(emergency_payload, os.path.join(ckpt_dir, "interrupted.pth"))
                save_atomic(emergency_payload, os.path.join(ckpt_dir, "latest.pth"))
                print(f"\n[INTERRUPT] Saved emergency checkpoints to {ckpt_dir}. Exiting safely.")
                sys.exit(0)

        scheduler.step()

        # End of epoch summary
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        progress.end_epoch(epoch, avg_losses)

        # Atomic checkpoint persistence
        payload = create_checkpoint_payload(
            model, optimizer, scheduler, scaler,
            epoch, global_step, stage, best_metric, exp_cfg
        )
        save_atomic(payload, os.path.join(ckpt_dir, "latest.pth"))
        save_atomic(payload, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        purge_old_checkpoints(ckpt_dir, keep_last_n=exp_cfg["checkpoint"]["keep_last_n"])

    print(f"\n🎉 [COMPLETE] Finished training run of {max_epochs} epochs!")


if __name__ == "__main__":
    main()
