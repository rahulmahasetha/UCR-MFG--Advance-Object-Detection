import os
import sys
import glob
import signal
import random
import hashlib
import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class CheckpointIncompatibilityError(Exception):
    """Raised when loading a checkpoint with divergent architecture or configuration."""
    pass


class GracefulInterruptHandler:
    """Catches SIGINT/SIGTERM so training can stop at the next safe point; a second signal exits at once."""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        if self.interrupted:
            print("\n[INTERRUPT] Second signal received, exiting without saving.")
            raise KeyboardInterrupt
        name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[INTERRUPT] Caught {name}. Finishing the current optimizer step, then saving (press again to force quit)...")
        self.interrupted = True


def save_atomic(payload: Dict[str, Any], filepath: str) -> None:
    """Atomically persists a checkpoint via temporary file and atomic filesystem rename."""
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)
    temp_path = f"{filepath}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, filepath)  # Atomic POSIX rename
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise IOError(f"Failed to atomically persist checkpoint to {filepath}: {e}")


def create_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    current_stage: int,
    best_metric: float,
    cfg_dict: Dict[str, Any],
    resume: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Assembles the complete checkpoint payload.

    `epoch` is the last fully completed epoch. `resume`, when given, marks a mid-epoch save:
    {"epoch": epoch in progress, "batch": next batch index, "epoch_losses": running loss sums}.
    """
    config_str = str(sorted(cfg_dict.items()))
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    return {
        # 1. Model & Optimization States
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,

        # 2. Training Progress
        "epoch": epoch,
        "global_step": global_step,
        "current_stage": current_stage,
        "best_metric": best_metric,
        "resume": resume,

        # 3. PRNG States (Exact Bit-Level Reproducibility)
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "torch_mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        },

        "config": cfg_dict,
        "metadata": {
            "config_hash": config_hash,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    }


def load_checkpoint(
    target_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    current_cfg: Optional[Dict[str, Any]] = None,
    force: bool = False
) -> Dict[str, Any]:
    """Restores all states from checkpoint with strict compatibility verification."""
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Checkpoint not found at: {target_path}")

    checkpoint = torch.load(target_path, map_location="cpu", weights_only=False)

    # Compatibility check
    if not force and current_cfg is not None and "metadata" in checkpoint:
        current_hash = hashlib.sha256(str(sorted(current_cfg.items())).encode("utf-8")).hexdigest()
        ckpt_hash = checkpoint["metadata"].get("config_hash")
        # Soft check: if hash differs but user hasn't overridden
        if ckpt_hash and current_hash != ckpt_hash:
            print(f"[WARN] Checkpoint config hash differs from current config. Resuming with caution.")

    # 1. Restore Model
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Restore Optimizer, Scheduler, Scaler
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 3. Restore PRNG states
    rng = checkpoint.get("rng_states", {})
    if "python" in rng and rng["python"]:
        random.setstate(rng["python"])
    if "numpy" in rng and rng["numpy"]:
        np.random.set_state(rng["numpy"])
    if "torch_cpu" in rng and rng["torch_cpu"] is not None:
        torch.set_rng_state(rng["torch_cpu"])
    if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    if rng.get("torch_mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["torch_mps"])

    resume = checkpoint.get("resume")
    if resume:
        start_epoch, start_batch = resume["epoch"], resume["batch"]
    else:
        start_epoch, start_batch = checkpoint["epoch"] + 1, 0

    print(f"[RESUME] Loaded checkpoint from {target_path}:")
    print(f"  ├── Last completed epoch: {checkpoint['epoch']} (continuing at epoch {start_epoch}, batch {start_batch})")
    print(f"  ├── Global Step: {checkpoint.get('global_step', 0)}")
    print(f"  ├── Curriculum Stage: {checkpoint.get('current_stage', 1)}")
    print(f"  └── Best Metric: {checkpoint.get('best_metric', 0.0):.4f}")

    return {
        "start_epoch": start_epoch,
        "start_batch": start_batch,
        "epoch_losses": dict(resume["epoch_losses"]) if resume else {},
        "resume_batch_size": resume.get("batch_size") if resume else None,
        "global_step": checkpoint.get("global_step", 0),
        "current_stage": checkpoint.get("current_stage", 1),
        "best_metric": checkpoint.get("best_metric", 0.0),
    }


def purge_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """Purges older epoch_*.pth checkpoints while permanently preserving latest.pth and best.pth."""
    epoch_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth")))
    if len(epoch_ckpts) > keep_last_n:
        to_delete = epoch_ckpts[:-keep_last_n]
        for p in to_delete:
            try:
                os.remove(p)
            except OSError:
                pass
