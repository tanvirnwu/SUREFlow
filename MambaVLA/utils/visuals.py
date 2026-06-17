"""Utilities for saving training/testing visualizations."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _to_numpy(values: Iterable) -> np.ndarray:
    if values is None:
        return np.array([])
    if isinstance(values, np.ndarray):
        return values
    try:
        import torch

        if torch.is_tensor(values):
            return values.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(values)


def plot_flow_time_loss(
    time_loss_pairs: Iterable[tuple[float, float]],
    save_path: str,
    *,
    num_bins: int = 10,
) -> None:
    pairs = list(time_loss_pairs)
    if not pairs:
        return
    times = _to_numpy([pair[0] for pair in pairs]).astype(float)
    losses = _to_numpy([pair[1] for pair in pairs]).astype(float)

    plt.figure(figsize=(6, 4))
    plt.scatter(times, losses, alpha=0.3, s=12, label="samples")

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(times, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    for idx in range(num_bins):
        mask = bin_indices == idx
        if np.any(mask):
            bin_means.append(np.mean(losses[mask]))
        else:
            bin_means.append(np.nan)
    plt.plot(bin_centers, bin_means, color="red", linewidth=2, label="binned mean")

    plt.xlim(0.0, 1.0)
    plt.xlabel("t")
    plt.ylabel("loss")
    plt.title("Flow time loss vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_uncertainty_calibration(
    predicted: Iterable[float],
    empirical: Iterable[float],
    save_path: str,
    *,
    num_bins: int = 10,
) -> None:
    predicted_arr = _to_numpy(predicted).astype(float)
    empirical_arr = _to_numpy(empirical).astype(float)
    if predicted_arr.size == 0 or empirical_arr.size == 0:
        return

    min_val = np.min(predicted_arr)
    max_val = np.max(predicted_arr)
    if np.isclose(min_val, max_val):
        max_val = min_val + 1e-6

    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_indices = np.digitize(predicted_arr, bins) - 1
    bin_means_pred = []
    bin_means_emp = []
    for idx in range(num_bins):
        mask = bin_indices == idx
        if np.any(mask):
            bin_means_pred.append(np.mean(predicted_arr[mask]))
            bin_means_emp.append(np.mean(empirical_arr[mask]))

    plt.figure(figsize=(5, 5))
    plt.plot(bin_means_pred, bin_means_emp, marker="o", label="calibration")
    line_min = min(np.min(predicted_arr), np.min(empirical_arr))
    line_max = max(np.max(predicted_arr), np.max(empirical_arr))
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="gray", label="y=x")
    plt.xlabel("Mean predicted uncertainty")
    plt.ylabel("Mean empirical error")
    plt.title("Uncertainty calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_trends(
    steps: Iterable[int],
    loss_fm: Iterable[float],
    loss_u: Optional[Iterable[float]],
    mean_pred_uncertainty: Iterable[float],
    mean_residual_magnitude: Iterable[float],
    save_path: str,
) -> None:
    steps_arr = _to_numpy(steps)
    plt.figure(figsize=(7, 4))
    plt.plot(steps_arr, _to_numpy(loss_fm), label="loss_fm")
    if loss_u is not None:
        loss_u_arr = _to_numpy(loss_u)
        if loss_u_arr.size:
            plt.plot(steps_arr, loss_u_arr, label="loss_u")
    plt.plot(steps_arr, _to_numpy(mean_pred_uncertainty), label="mean predicted uncertainty")
    plt.plot(steps_arr, _to_numpy(mean_residual_magnitude), label="mean residual magnitude")
    plt.xlabel("global step")
    plt.ylabel("value")
    plt.title("Training trends")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_uncertainty_heatmap(
    uncertainty: Iterable,
    save_path: str,
    *,
    cmap: str = "viridis",
) -> None:
    uncertainty_arr = _to_numpy(uncertainty).astype(float)
    if uncertainty_arr.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.imshow(uncertainty_arr, aspect="auto", cmap=cmap)
    plt.colorbar(label="s_hat")
    plt.xlabel("Action dimension")
    plt.ylabel("Timestep")
    plt.title("Uncertainty heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_refinement_effect(
    mean_s_hat: Iterable[float],
    mean_residual: Optional[Iterable[float]],
    save_path: str,
) -> None:
    mean_s_hat_arr = _to_numpy(mean_s_hat)
    if mean_s_hat_arr.size == 0:
        return
    iterations = np.arange(1, len(mean_s_hat_arr) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, mean_s_hat_arr, marker="o", label="mean s_hat")
    if mean_residual is not None:
        mean_residual_arr = _to_numpy(mean_residual)
        if mean_residual_arr.size:
            plt.plot(iterations, mean_residual_arr, marker="x", label="mean residual magnitude")
    plt.xlabel("Refinement iteration")
    plt.ylabel("value")
    plt.title("Refinement effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_task_summary(
    task_labels: Iterable[str],
    success_rate: Iterable[float],
    mean_uncertainty: Iterable[float],
    mean_failure_uncertainty: Iterable[float],
    save_path: str,
) -> None:
    labels = list(task_labels)
    success_arr = _to_numpy(success_rate).astype(float)
    mean_uncertainty_arr = _to_numpy(mean_uncertainty).astype(float)
    mean_failure_arr = _to_numpy(mean_failure_uncertainty).astype(float)

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(10, 4))
    plt.bar(x - width, success_arr, width=width, label="success rate")
    plt.bar(x, mean_uncertainty_arr, width=width, label="mean uncertainty")
    plt.bar(x + width, mean_failure_arr, width=width, label="mean uncertainty (failure)")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("value")
    plt.title("Per-task summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
