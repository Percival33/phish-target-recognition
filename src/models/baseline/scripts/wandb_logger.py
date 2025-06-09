#!/usr/bin/env python3
"""Optional wandb logger for baseline evaluation (only if needed)."""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class WandbLogger:
    """Simple optional wandb logger."""

    def __init__(self, enabled: bool = False, project: str = "baseline-evaluation"):
        self.enabled = enabled
        self.run = None

        if enabled:
            try:
                import wandb

                self.wandb = wandb
                self.run = wandb.init(
                    project=project,
                    name=f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                )
                logger.info("Wandb logging enabled")
            except ImportError:
                logger.warning("Wandb not available, logging disabled")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.enabled = False

    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log metrics if enabled."""
        if not self.enabled or not self.run:
            return

        try:
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            self.run.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def save_artifact(self, file_path: Path, name: str):
        """Save file as artifact if enabled."""
        if not self.enabled or not self.run:
            return

        try:
            artifact = self.wandb.Artifact(name, type="results")
            artifact.add_file(str(file_path))
            self.run.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to save artifact: {e}")

    def finish(self):
        """Finish wandb run if enabled."""
        if self.enabled and self.run:
            try:
                self.wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb: {e}")


# Usage example in main script:
# wandb_logger = WandbLogger(enabled=args.log)
# wandb_logger.log_metrics({"accuracy": 0.85}, prefix="split_0")
# wandb_logger.save_artifact(results_path, "results")
# wandb_logger.finish()
