import sys
import time
from typing import Dict, Any, Optional
import torch


class TrainingProgressDisplay:
    """Informative live terminal animated progress display.
    
    Shows:
        - Current epoch / Total epochs
        - Batch progress bar & percentage
        - Elapsed time & estimated time of arrival (ETA)
        - Multi-loss breakdown (total, det, unc, etc.)
        - Current learning rate
        - GPU / VRAM usage (or RAM on CPU)
        - Overall training progress
    """

    def __init__(self, total_epochs: int, num_batches: int, width: int = 25):
        self.total_epochs = total_epochs
        self.num_batches = num_batches
        self.bar_width = width
        self.epoch_start_time = 0.0
        self.training_start_time = time.time()
        self.last_step_time = 0.0
        self.step_ema = None

    def start_epoch(self, epoch: int) -> None:
        """Call at the start of each training epoch."""
        self.epoch_start_time = time.time()
        self.last_step_time = 0.0
        self.step_ema = None
        print(f"\n{'='*78}")
        print(f"🚀 [EPOCH {epoch:03d} / {self.total_epochs:03d}] Active Training Phase")
        print(f"{'='*78}")

    def update_batch(
        self,
        epoch: int,
        batch_idx: int,
        losses: Dict[str, float],
        lr: float
    ) -> None:
        """Updates the animated in-place status line."""
        current_time = time.time()
        elapsed_epoch = current_time - self.epoch_start_time
        batch_num = batch_idx + 1

        # Smooth EMA for realistic ETA calculation (avoids first-step startup / download distortion)
        if self.last_step_time > 0:
            dt = current_time - self.last_step_time
            if self.step_ema is None:
                self.step_ema = dt
            else:
                self.step_ema = 0.9 * self.step_ema + 0.1 * dt
        self.last_step_time = current_time

        # Percentage & Progress bar
        fraction = batch_num / float(self.num_batches)
        filled = int(round(self.bar_width * fraction))
        bar = "█" * filled + "░" * (self.bar_width - filled)

        # Time estimation using rolling EMA
        if self.step_ema is not None and batch_num > 1:
            eta_epoch = self.step_ema * (self.num_batches - batch_num)
        else:
            rate = elapsed_epoch / max(batch_num, 1)
            eta_epoch = rate * (self.num_batches - batch_num)

        elapsed_str = self._format_seconds(elapsed_epoch)
        eta_str = self._format_seconds(eta_epoch)

        # Memory usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            mem_str = f"VRAM:{vram_mb:.0f}MB"
        elif torch.backends.mps.is_available():
            mem_str = "MPS:Active"
        else:
            mem_str = "CPU"

        # Format loss strings
        l_tot = losses.get("loss_total", 0.0)
        l_det = losses.get("loss_det", 0.0)
        l_u = losses.get("loss_u", 0.0)

        # Single animated line (using \r)
        status_line = (
            f"\rEp {epoch:03d}/{self.total_epochs:03d} |{bar}| {fraction*100:5.1f}% "
            f"[{batch_num}/{self.num_batches}] "
            f"[{elapsed_str}<{eta_str}] "
            f"L_tot:{l_tot:.3f} L_det:{l_det:.3f} L_u:{l_u:.3f} "
            f"LR:{lr:.2e} {mem_str}"
        )
        sys.stdout.write(status_line)
        sys.stdout.flush()

    def end_epoch(self, epoch: int, avg_losses: Dict[str, float], val_metric: Optional[float] = None) -> None:
        """Prints a clean epoch summary block upon completion."""
        sys.stdout.write("\n")
        duration = time.time() - self.epoch_start_time
        dur_str = self._format_seconds(duration)

        overall_elapsed = time.time() - self.training_start_time
        est_total_time = (overall_elapsed / max(epoch, 1)) * self.total_epochs
        est_remaining = max(0.0, est_total_time - overall_elapsed)

        metric_str = f" | Val mAP@0.5: {val_metric:.4f}" if val_metric is not None else ""
        print(f"✅ Finished Epoch {epoch:03d} in {dur_str} (Overall ETA: {self._format_seconds(est_remaining)})")
        loss_summary = " ".join([f"{k}:{v:.4f}" for k, v in avg_losses.items() if k.startswith("loss_")])
        print(f"   Summary -> {loss_summary}{metric_str}")

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        s = int(round(seconds))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

