"""Shared utilities for all main scripts."""

import os
import pickle
import random
import logging
import wandb
from typing import Any
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import multiprocessing as mp

from MambaVLA.utils.scaler import Scaler, ActionScaler, MinMaxScaler
from MambaVLA.utils.ema import ExponentialMovingAverage
from MambaVLA.utils import visuals

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

log = logging.getLogger(__name__)


class Trainer:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            training_dataset: Any,
            validation_dataset: Any,
            training_batch_size: int = 512,
            validation_batch_size: int = 512,
            dataloader_workers: int = 8,
            device: str = 'cpu',
            total_epochs: int = 100,
            enable_data_scaling: bool = True,
            data_scaler_type: str = "minmax",
            evaluation_frequency: int = 50,
            observation_sequence_length: int = 1,
            ema_decay_rate: float = 0.999,
            enable_ema: bool = False,
            checkpoint_frequency: int = 10,
            visuals_config: Any | None = None
    ):
        """Initialize."""
        
        # Dataset and data loading configuration
        self.trainset = training_dataset
        self.valset = validation_dataset
        self.train_batch_size = training_batch_size
        self.val_batch_size = validation_batch_size
        self.num_workers = dataloader_workers
        
        # Training configuration
        self.epoch = total_epochs
        self.perception_seq_len = observation_sequence_length
        self.eval_every_n_epochs = evaluation_frequency
        self.save_every_n_epochs = checkpoint_frequency
        
        # Device and environment configuration
        self.device = device
        self.working_dir = os.getcwd()
        
        # Data scaling configuration
        self.scale_data = enable_data_scaling
        self.scaling_type = data_scaler_type
        
        # EMA configuration
        self.decay_ema = ema_decay_rate
        self.if_use_ema = enable_ema

        # Visualization configuration
        self.visuals_config = visuals_config
        self.visuals_training_dir: str | None = None
        self.global_step = 0
        self._time_loss_pairs: list[tuple[float, float]] = []
        self._trend_steps: list[int] = []
        self._trend_loss_fm: list[float] = []
        self._trend_loss_u: list[float] = []
        self._trend_pred_uncertainty: list[float] = []
        self._trend_residual_magnitude: list[float] = []
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize scaler
        self._setup_scaler()

        log.info("Number of training samples: {}".format(len(self.trainset)))

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

        # self.test_dataloader = DataLoader(
        #     self.valset,
        #     batch_size=self.val_batch_size,
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True,
        #     drop_last=False
        # )

    def _setup_scaler(self):
        """Setup data scaler based on configuration."""
        if self.scaling_type == 'minmax':
            self.scaler = MinMaxScaler(self.trainset.get_all_actions(), self.scale_data, self.device)
        else:
            self.scaler = ActionScaler(self.trainset.get_all_actions(), self.scale_data, self.device)

    def main(self, model):
        """Run main training/testing pipeline."""
        self._setup_training_components(model)
        self._run_training_loop(model)
        self._finalize_training(model)

    def _setup_training_components(self, model):
        """Setup scaler, EMA, and optimizer for training."""
        # assign scaler to model calss
        model.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(model.parameters(), self.decay_ema, self.device)

        # define optimizer
        if model.use_lr_scheduler:
            self.optimizer, self.scheduler = model.configure_optimizers()
        else:
            self.optimizer = model.configure_optimizers()

    def _run_training_loop(self, model):
        """Execute the main training loop over all epochs."""
        for num_epoch in tqdm(range(self.epoch), desc="Epochs", dynamic_ncols=True):
            epoch_loss, epoch_metrics = self._train_single_epoch(model, num_epoch)
            self._log_epoch_results(num_epoch, epoch_loss, epoch_metrics)
            self._save_checkpoint_if_needed(model, num_epoch)

    def _train_single_epoch(self, model, num_epoch):
        """Train for a single epoch and return the average loss."""
        epoch_loss = torch.tensor(0.0).to(self.device)
        epoch_metrics = {
            "loss_fm": 0.0,
            "loss_u": 0.0,
            "effective_lambda_u": 0.0,
            "s_hat_mean": 0.0,
            "s_hat_min": 0.0,
            "s_hat_max": 0.0,
            "exp_s_hat_mean": 0.0,
            "loss_count": 0,
            "s_hat_count": 0,
        }

        for data in tqdm(self.train_dataloader, desc="Batches", leave=False, dynamic_ncols=True):
            obs_dict, action, mask = data
            obs_dict, action = self._prepare_batch_data(obs_dict, action)
            batch_loss, diagnostics = self.train_one_step(model, obs_dict, action, num_epoch)
            epoch_loss += batch_loss
            self.global_step += 1
            if diagnostics is not None:
                epoch_metrics["loss_fm"] += float(diagnostics["loss_fm"])
                loss_u = diagnostics.get("loss_u")
                if loss_u is not None:
                    epoch_metrics["loss_u"] += float(loss_u)
                epoch_metrics["effective_lambda_u"] += float(diagnostics["effective_lambda_u"])
                epoch_metrics["loss_count"] += 1
                s_hat_mean = diagnostics.get("s_hat_mean")
                s_hat_min = diagnostics.get("s_hat_min")
                s_hat_max = diagnostics.get("s_hat_max")
                exp_s_hat_mean = diagnostics.get("exp_s_hat_mean")
                if s_hat_mean is not None:
                    epoch_metrics["s_hat_mean"] += float(s_hat_mean)
                    epoch_metrics["s_hat_min"] += float(s_hat_min)
                    epoch_metrics["s_hat_max"] += float(s_hat_max)
                    epoch_metrics["exp_s_hat_mean"] += float(exp_s_hat_mean)
                    epoch_metrics["s_hat_count"] += 1
            self._maybe_update_visuals(diagnostics, model)

        if epoch_metrics["loss_count"] > 0:
            for key in ["loss_fm", "loss_u", "effective_lambda_u"]:
                epoch_metrics[key] /= epoch_metrics["loss_count"]
            if epoch_metrics["s_hat_count"] > 0:
                for key in ["s_hat_mean", "s_hat_min", "s_hat_max", "exp_s_hat_mean"]:
                    epoch_metrics[key] /= epoch_metrics["s_hat_count"]
            else:
                epoch_metrics["s_hat_mean"] = None
                epoch_metrics["s_hat_min"] = None
                epoch_metrics["s_hat_max"] = None
                epoch_metrics["exp_s_hat_mean"] = None
        else:
            epoch_metrics = None

        return epoch_loss / len(self.train_dataloader), epoch_metrics

    def _prepare_batch_data(self, obs_dict, action):
        """Prepare observation and action data for training."""
        # put data on cuda
        for camera in obs_dict.keys():
            if camera == 'lang':
                continue
            
            obs_dict[camera] = obs_dict[camera].to(self.device)

            if 'rgb' not in camera and 'image' not in camera:
                continue
            obs_dict[camera] = obs_dict[camera][:, :self.perception_seq_len].contiguous()

        action = self.scaler.scale_output(action)
        action = action[:, self.perception_seq_len - 1:, :].contiguous()

        return obs_dict, action

    def _log_epoch_results(self, num_epoch, epoch_loss, epoch_metrics):
        """Log epoch results to wandb and console."""
        log_data = {"train_loss": epoch_loss.item()}
        if epoch_metrics is not None:
            log_data.update(
                {
                    "loss_fm": epoch_metrics["loss_fm"],
                    "loss_u": epoch_metrics["loss_u"],
                    "effective_lambda_u": epoch_metrics["effective_lambda_u"],
                }
            )
            if epoch_metrics.get("s_hat_count", 0) > 0:
                log_data.update(
                    {
                        "s_hat_mean": epoch_metrics["s_hat_mean"],
                        "s_hat_min": epoch_metrics["s_hat_min"],
                        "s_hat_max": epoch_metrics["s_hat_max"],
                        "exp_s_hat_mean": epoch_metrics["exp_s_hat_mean"],
                    }
                )
        wandb.log(log_data)
        log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))

    def _save_checkpoint_if_needed(self, model, num_epoch):
        """Save model checkpoint if it's time to do so."""
        if (num_epoch + 1) % self.save_every_n_epochs == 0:
            try:
                model.store_model_weights(self.working_dir, sv_name=f"epoch_{num_epoch + 1:05d}")
            except Exception as e:
                log.warning(f"Failed to save checkpoint at epoch {num_epoch + 1}: {e}")

    def _finalize_training(self, model):
        """Finalize training by applying EMA and saving final model."""
        log.info("training done")

        if self.if_use_ema:
            self.ema_helper.store(model.parameters())
            self.ema_helper.copy_to(model.parameters())

        model.store_model_weights(model.working_dir, sv_name='final_model')
        # or send weight out of the class

    def train_one_step(self, model, obs_dict, action, num_epoch: int):
        """Run a single training step."""
        model.train()

        output = model(obs_dict, action)
        diagnostics = None
        if isinstance(output, dict):
            v_hat = output["v_hat"]
            s_hat = output["s_hat"]
            v_target = output["v_target"]
            residual = v_hat - v_target
            batchwise_mse = (residual ** 2).mean(dim=list(range(1, residual.dim())))
            loss_fm = batchwise_mse.mean()
            if getattr(model, "use_uncertainty", False):
                s_hat_clamped = torch.clamp(s_hat, min=-8.0, max=8.0)
                denom = torch.exp(s_hat_clamped) + 1e-6
                loss_u = (residual.pow(2) / denom + s_hat_clamped).mean()
                effective_lambda_u = self._compute_effective_lambda_u(num_epoch, model)
                loss = loss_fm + effective_lambda_u * loss_u
                s_hat_mean = s_hat_clamped.mean().detach()
                s_hat_min = s_hat_clamped.min().detach()
                s_hat_max = s_hat_clamped.max().detach()
                exp_s_hat_mean = torch.exp(s_hat_clamped).mean().detach()
            else:
                loss = loss_fm
                loss_u = None
                effective_lambda_u = 0.0
                s_hat_clamped = None
                s_hat_mean = None
                s_hat_min = None
                s_hat_max = None
                exp_s_hat_mean = None

            diagnostics = {
                "loss_fm": loss_fm.detach(),
                "loss_u": loss_u.detach() if loss_u is not None else None,
                "effective_lambda_u": effective_lambda_u,
                "s_hat_mean": s_hat_mean,
                "s_hat_min": s_hat_min,
                "s_hat_max": s_hat_max,
                "exp_s_hat_mean": exp_s_hat_mean,
                "mean_residual_magnitude": residual.abs().mean().detach(),
            }

            if self._visuals_enabled():
                diagnostics.update(
                    {
                        "time_loss_pairs": output.get("time_loss_pairs", []),
                        "mean_pred_uncertainty": torch.exp(s_hat_clamped).mean().detach()
                        if s_hat_clamped is not None
                        else None,
                        "s_hat": s_hat_clamped.detach() if s_hat_clamped is not None else None,
                        "v_hat": v_hat.detach(),
                        "v_target": v_target.detach(),
                    }
                )
        else:
            loss = output

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if model.use_lr_scheduler:
            self.scheduler.step()

        if self.if_use_ema:
            self.ema_helper.update(model.parameters())

        return loss, diagnostics

    @staticmethod
    def _compute_effective_lambda_u(num_epoch: int, model) -> float:
        warmup_epochs = int(getattr(model, "lambda_u_warmup_epochs", 0))
        lambda_u_target = float(getattr(model, "lambda_u_target", getattr(model, "lambda_u", 0.0)))
        if warmup_epochs <= 0:
            return lambda_u_target
        if num_epoch < warmup_epochs:
            return 0.0
        if num_epoch < warmup_epochs * 2:
            progress = (num_epoch - warmup_epochs) / float(warmup_epochs)
            return lambda_u_target * progress
        return lambda_u_target

    def configure_visuals_dir(self, training_dir: str) -> None:
        self.visuals_training_dir = training_dir
        if training_dir:
            os.makedirs(training_dir, exist_ok=True)

    def _visuals_enabled(self) -> bool:
        return bool(getattr(self.visuals_config, "enabled", False)) and self.visuals_training_dir is not None

    def _maybe_update_visuals(self, diagnostics: dict | None, model) -> None:
        if not self._visuals_enabled() or diagnostics is None:
            return

        time_loss_pairs = diagnostics.get("time_loss_pairs", [])
        if time_loss_pairs:
            self._time_loss_pairs.extend(time_loss_pairs)

        self._trend_steps.append(self.global_step)
        self._trend_loss_fm.append(float(diagnostics["loss_fm"]))
        loss_u = diagnostics.get("loss_u")
        if loss_u is not None:
            self._trend_loss_u.append(float(loss_u))
        else:
            self._trend_loss_u.append(float("nan"))
        pred_uncertainty = diagnostics.get("mean_pred_uncertainty")
        if pred_uncertainty is not None:
            self._trend_pred_uncertainty.append(float(pred_uncertainty))
        else:
            self._trend_pred_uncertainty.append(float("nan"))
        self._trend_residual_magnitude.append(float(diagnostics["mean_residual_magnitude"]))

        if self.global_step % int(getattr(self.visuals_config, "train_every_steps", 1000)) != 0:
            return

        step = self.global_step
        if self._time_loss_pairs:
            visuals.plot_flow_time_loss(
                self._time_loss_pairs,
                os.path.join(self.visuals_training_dir, f"flow_time_loss_step_{step}.png"),
            )
            self._time_loss_pairs = []

        if getattr(self.visuals_config, "save_calibration", True) and getattr(model, "use_uncertainty", False):
            residual = diagnostics["v_hat"] - diagnostics["v_target"]
            empirical = (residual ** 2).mean(dim=list(range(1, residual.dim())))
            predicted = torch.exp(diagnostics["s_hat"]).mean(dim=list(range(1, diagnostics["s_hat"].dim())))
            visuals.plot_uncertainty_calibration(
                predicted,
                empirical,
                os.path.join(self.visuals_training_dir, f"uncertainty_calibration_step_{step}.png"),
            )

        if getattr(self.visuals_config, "save_trends", True):
            visuals.plot_training_trends(
                self._trend_steps,
                self._trend_loss_fm,
                self._trend_loss_u,
                self._trend_pred_uncertainty,
                self._trend_residual_magnitude,
                os.path.join(self.visuals_training_dir, f"training_trends_up_to_step_{step}.png"),
            )

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None
