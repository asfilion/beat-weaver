"""Training loop with mixed-precision, checkpointing, and TensorBoard logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset, build_weighted_sampler, collate_fn
from beat_weaver.model.tokenizer import PAD
from beat_weaver.model.transformer import BeatWeaverModel

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: ModelConfig, steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine LR schedule with linear warmup."""
    total_steps = config.max_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Wraps training state and provides train/validate methods."""

    def __init__(
        self,
        model: BeatWeaverModel,
        config: ModelConfig,
        output_dir: Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or _get_device()

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD,
            label_smoothing=config.label_smoothing,
        )
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad()

        for batch_idx, (mel, mel_mask, tokens, token_mask) in enumerate(dataloader):
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]

            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(mel, input_tokens, mel_mask, input_mask)
                # logits: (batch, seq_len-1, vocab_size)
                # target: (batch, seq_len-1)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
                loss = loss / accum_steps  # Scale for accumulation

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            total_loss += loss.item() * accum_steps  # Unscale for logging
            n_batches += 1
            self.global_step += 1

            # Log every 50 steps
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss_step", loss.item() * accum_steps, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        for mel, mel_mask, tokens, token_mask in dataloader:
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]
            target_mask = token_mask[:, 1:]

            logits = self.model(mel, input_tokens, mel_mask, input_mask)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

            # Token accuracy (ignoring padding)
            preds = logits.argmax(dim=-1)
            mask = target_mask & (target_tokens != PAD)
            total_correct += (preds == target_tokens)[mask].sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, n_batches)
        accuracy = total_correct / max(1, total_tokens)
        return {"val_loss": avg_loss, "val_token_accuracy": accuracy}

    def save_checkpoint(self, name: str) -> Path:
        """Save model + optimizer + training state."""
        ckpt_dir = self.output_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        self.config.save(ckpt_dir / "config.json")

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        (ckpt_dir / "training_state.json").write_text(json.dumps(state, indent=2))
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: Path) -> None:
        """Resume from a checkpoint."""
        ckpt_dir = Path(ckpt_dir)
        self.model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=self.device, weights_only=True),
        )
        self.optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=self.device, weights_only=True),
        )
        state = json.loads((ckpt_dir / "training_state.json").read_text())
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]


def train(
    config: ModelConfig,
    train_dataset: BeatSaberDataset,
    val_dataset: BeatSaberDataset,
    output_dir: Path,
    resume_from: Path | None = None,
) -> Path:
    """Main training entry point.

    Returns path to the best checkpoint directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BeatWeaverModel(config)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    trainer = Trainer(model, config, output_dir)

    if resume_from:
        trainer.load_checkpoint(resume_from)
        logger.info("Resumed from %s (epoch %d)", resume_from, trainer.epoch)

    sampler = build_weighted_sampler(train_dataset, config.official_ratio)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,  # shuffle only when no weighted sampler
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # audio loading not picklable
        pin_memory=trainer.device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=trainer.device.type == "cuda",
    )

    # Build scheduler after knowing steps_per_epoch
    trainer.scheduler = _build_lr_scheduler(
        trainer.optimizer, config, len(train_loader),
    )

    config.save(output_dir / "config.json")

    training_start = time.time()
    epoch_times: list[float] = []

    logger.info(
        "Training: %d train samples, %d val samples, %d batches/epoch, device=%s",
        len(train_dataset), len(val_dataset), len(train_loader), trainer.device,
    )

    for epoch in range(trainer.epoch, config.max_epochs):
        trainer.epoch = epoch
        t0 = time.time()

        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        total_elapsed = time.time() - training_start

        logger.info(
            "Epoch %d/%d (%.1fs, total %.0fs): train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, config.max_epochs, elapsed, total_elapsed,
            train_loss, val_metrics["val_loss"], val_metrics["val_token_accuracy"],
        )

        # TensorBoard
        trainer.writer.add_scalar("train/loss_epoch", train_loss, epoch)
        trainer.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
        trainer.writer.add_scalar("val/token_accuracy", val_metrics["val_token_accuracy"], epoch)
        trainer.writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
        trainer.writer.add_scalar("timing/total_seconds", total_elapsed, epoch)

        # Checkpoint every epoch
        trainer.save_checkpoint(f"epoch_{epoch + 1:03d}")

        # Best model
        if val_metrics["val_loss"] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics["val_loss"]
            trainer.patience_counter = 0
            trainer.save_checkpoint("best")
            logger.info("New best model (val_loss=%.4f)", trainer.best_val_loss)
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    trainer.writer.close()

    # Write training summary
    total_time = time.time() - training_start
    epochs_completed = len(epoch_times)
    avg_epoch = sum(epoch_times) / max(1, epochs_completed)
    summary = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batches_per_epoch": len(train_loader),
        "batch_size": config.batch_size,
        "epochs_completed": epochs_completed,
        "total_time_seconds": round(total_time, 1),
        "avg_epoch_seconds": round(avg_epoch, 1),
        "samples_per_second": round(len(train_dataset) / avg_epoch, 1),
        "best_val_loss": round(trainer.best_val_loss, 4),
        "device": str(trainer.device),
        "model_parameters": model.count_parameters(),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Training complete: %d epochs in %.0fs (avg %.1fs/epoch, %.1f samples/s)",
        epochs_completed, total_time, avg_epoch,
        len(train_dataset) / avg_epoch,
    )

    return output_dir / "checkpoints" / "best"
