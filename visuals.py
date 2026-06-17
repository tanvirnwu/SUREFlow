"""Standalone evaluation + plotting script for LIBERO/LIBERO-PRO checkpoints.

Example:
    python visuals.py --checkpoint_path PATH --train_suite libero_goal --eval_suite libero_goal_temp --output_dir PATH
    python visuals.py --checkpoint_path PATH --train_suite libero_goal --eval_suite temp --output_dir PATH
    python visuals.py --checkpoint_path PATH --train_suite libero_goal --output_dir PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LIBERO_PRO_PYTHON_ROOT = os.path.join(REPO_ROOT, "LIBERO-PRO")
if os.path.isdir(LIBERO_PRO_PYTHON_ROOT) and LIBERO_PRO_PYTHON_ROOT not in sys.path:
    sys.path.insert(1, LIBERO_PRO_PYTHON_ROOT)

from configs.config import create_libero_pro_eval_config, create_libero_train_config
from configs.factory import create_model, create_simulation, create_trainer
from MambaVLA.benchmark.libero.libero_sim import OffScreenRenderEnv, benchmark

log = logging.getLogger("visuals")
logging.basicConfig(level=logging.INFO)


def _plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visuals.py plotting. Install it (e.g., `pip install matplotlib`) and rerun."
        ) from exc
    return plt


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _configure_libero_pro_paths(repo_root: str) -> None:
    libero_root = os.path.join(repo_root, "LIBERO-PRO", "libero", "libero")
    if not os.path.isdir(libero_root):
        log.warning("LIBERO-PRO root not found at %s; using existing LIBERO path configuration.", libero_root)
        return

    config_root = os.path.join(repo_root, ".libero")
    os.makedirs(config_root, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = config_root

    path_config = {
        "benchmark_root": libero_root,
        "bddl_files": os.path.join(libero_root, "bddl_files"),
        "init_states": os.path.join(libero_root, "init_files"),
        "datasets": os.path.join(libero_root, "..", "datasets"),
        "assets": os.path.join(libero_root, "assets"),
    }

    config_file = os.path.join(config_root, "config.yaml")
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(path_config, f)


def _clear_local_libero_pro_override(repo_root: str) -> None:
    config_root = os.path.join(repo_root, ".libero")
    configured_path = os.environ.get("LIBERO_CONFIG_PATH")
    if configured_path and os.path.abspath(configured_path) == os.path.abspath(config_root):
        os.environ.pop("LIBERO_CONFIG_PATH", None)


def _resolve_eval_cfg(train_suite: str, eval_suite: str | None):
    if not eval_suite:
        _clear_local_libero_pro_override(REPO_ROOT)
        return create_libero_train_config(train_suite)

    suffixes = {"swap", "object", "lan", "task", "temp"}
    if eval_suite in suffixes:
        cfg = create_libero_pro_eval_config(train_suite, eval_suite)
    elif eval_suite.startswith(f"{train_suite}_"):
        suffix = eval_suite[len(train_suite) + 1 :]
        if suffix not in suffixes:
            raise ValueError(f"Unsupported eval suite suffix '{suffix}'.")
        cfg = create_libero_pro_eval_config(train_suite, suffix)
    else:
        raise ValueError(
            "--eval_suite must be one of {swap,object,lan,task,temp} or train_suite-suffixed like "
            f"'{train_suite}_temp'. Got: {eval_suite}"
        )
    _configure_libero_pro_paths(REPO_ROOT)
    return cfg


def _resolve_checkpoint_file(checkpoint_path: str) -> tuple[str, str]:
    cp = os.path.abspath(checkpoint_path)
    if os.path.isfile(cp):
        return cp, os.path.dirname(cp)

    if not os.path.isdir(cp):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    candidates = [
        os.path.join(cp, "final_model.pth"),
        os.path.join(cp, "model_state_dict.pth"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand, cp
    raise FileNotFoundError(
        f"No checkpoint file found in {checkpoint_path} (looked for final_model.pth, model_state_dict.pth)"
    )




def _set_model_scaler(cfg, model, checkpoint_dir: str, scaler_path: str | None = None) -> None:
    candidates: list[str] = []
    if scaler_path:
        candidates.append(os.path.abspath(scaler_path))

    candidates.extend([
        os.path.join(checkpoint_dir, "model_scaler.pkl"),
        os.path.join(os.path.dirname(checkpoint_dir), "model_scaler.pkl"),
    ])

    for cand in candidates:
        if os.path.isfile(cand):
            model.load_model_scaler(os.path.dirname(cand), os.path.basename(cand))
            log.info("Loaded scaler from %s", cand)
            return

    log.warning("No scaler file found near checkpoint. Falling back to trainer-derived scaler from dataset config.")
    try:
        trainer = create_trainer(cfg)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            "Could not find model_scaler.pkl and failed to build fallback scaler from dataset. "
            "Provide --scaler_path explicitly or ensure dataset path is valid. "
            f"Original error: {exc}"
        ) from exc

    model.set_scaler(trainer.scaler)
    log.info("Using fallback scaler from trainer dataset statistics.")

def _load_model_from_checkpoint(cfg, checkpoint_path: str, scaler_path: str | None = None):
    checkpoint_file, checkpoint_dir = _resolve_checkpoint_file(checkpoint_path)
    model = create_model(cfg)

    try:
        state_dict = torch.load(checkpoint_file, map_location=cfg.device, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load checkpoint file '{checkpoint_file}': {exc}") from exc

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        log.warning("Checkpoint key mismatch. missing=%s unexpected=%s", missing_keys, unexpected_keys)

    _set_model_scaler(cfg, model, checkpoint_dir, scaler_path=scaler_path)
    model = model.to(cfg.device)
    model.eval()
    return model


def _task_embedding_for_context(sim, context: int, idx_within_task: int):
    benchmark_type = benchmark.get_benchmark_dict()[sim.benchmark_type]()
    task_bddl_file = benchmark_type.get_task_bddl_file_path(context)
    file_name = os.path.basename(task_bddl_file).split(".")[0]

    if isinstance(sim.task_embs, dict):
        task_emb = sim.task_embs[file_name].to(sim.device).unsqueeze(0)
    else:
        emb_index = context if context < len(sim.task_embs) else idx_within_task
        task_emb = sim.task_embs[emb_index].to(sim.device).unsqueeze(0)
    return task_emb, file_name, task_bddl_file, benchmark_type


def _rollout_episode(
    sim,
    model,
    context: int,
    context_ind: int,
    max_steps: int,
    tau_for_activation: float | None,
) -> dict[str, Any]:
    task_emb, file_name, task_bddl_file, benchmark_type = _task_embedding_for_context(sim, context, context_ind)

    init_states = benchmark_type.get_task_init_states(context)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)

    model.reset()
    env.seed(sim.seed)
    env.reset()
    obs = env.set_init_state(init_state=init_states[context_ind])

    dummy = np.zeros(7)
    dummy[-1] = -1.0
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)

    actions: list[list[float]] = []
    uncertainty_seq: list[float] = []
    uncertainty_raw_seq: list[float] = []
    activation_any_seq: list[float] = []
    activation_dim_frac_seq: list[float] = []

    success = 0
    success_step = None

    for step in range(max_steps):
        agentview_rgb = (
            torch.from_numpy(obs["agentview_image"]).to(sim.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
        )
        eye_in_hand_rgb = (
            torch.from_numpy(obs["robot0_eye_in_hand_image"]).to(sim.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
        )
        robot_states = torch.from_numpy(
            np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]], axis=-1)
        ).to(sim.device).float().unsqueeze(0).unsqueeze(0)

        obs_dict = {
            "agentview_image": agentview_rgb,
            "eye_in_hand_image": eye_in_hand_rgb,
            "lang_emb": task_emb,
            "robot_states": robot_states,
        }

        action_out = model.predict(obs_dict, return_diagnostics=True, collect_refinement_stats=False)
        action, diagnostics = action_out

        action_np = action.detach().cpu().numpy()
        actions.append(action_np.tolist())

        if diagnostics and diagnostics.get("s_hat") is not None:
            s_hat = diagnostics["s_hat"].detach().cpu().numpy()
            uncertainty_raw_seq.append(float(np.mean(s_hat)))
            uncertainty_seq.append(float(np.mean(np.exp(np.clip(s_hat, -8.0, 8.0)))))
            if tau_for_activation is not None:
                mask = s_hat > tau_for_activation
                activation_any_seq.append(float(np.any(mask)))
                activation_dim_frac_seq.append(float(np.mean(mask.astype(np.float32))))
        else:
            uncertainty_raw_seq.append(float("nan"))
            uncertainty_seq.append(float("nan"))

        obs, r, _, _ = env.step(action_np)
        if r == 1:
            success = 1
            success_step = step + 1
            break

    env.close()

    executed_steps = success_step if success_step is not None else max_steps
    return {
        "task_id": int(context),
        "task_name": file_name,
        "episode_id": int(context_ind),
        "success": int(success),
        "episode_length": int(executed_steps),
        "success_step": int(success_step) if success_step is not None else None,
        "actions": actions,
        "uncertainty_exp_s_hat": uncertainty_seq,
        "uncertainty_s_hat": uncertainty_raw_seq,
        "activation_any": activation_any_seq,
        "activation_dim_frac": activation_dim_frac_seq,
    }


def run_condition(cfg, model, horizon: int, urflow: bool, tau: float | None = None) -> dict[str, Any]:
    sim = create_simulation(cfg)
    sim.use_multiprocessing = False
    sim.max_step_per_episode = horizon

    model.use_uncertainty = urflow
    if tau is not None:
        model.uncertainty_threshold = tau

    sim._collect_bddl_paths()
    if sim._is_pro_benchmark():
        sim.load_task_embeddings_runtime()
    else:
        sim.load_task_embeddings_for_benchmark()

    if sim.benchmark_type == "libero_90":
        num_tasks = 50
    else:
        num_tasks = 10

    episodes: list[dict[str, Any]] = []
    for task_id in range(num_tasks):
        for ep in range(sim.rollouts):
            result = _rollout_episode(sim, model, task_id, ep, horizon, tau if urflow else None)
            result["horizon"] = horizon
            result["urflow"] = bool(urflow)
            result["tau"] = tau
            episodes.append(result)

    success_rate = float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0
    return {
        "horizon": horizon,
        "urflow": urflow,
        "tau": tau,
        "success_rate": success_rate,
        "episodes": episodes,
    }


def _save_episode_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["horizon", "urflow", "tau", "task_id", "task_name", "episode_id", "success", "episode_length", "success_step"],
        )
        writer.writeheader()
        for ep in episodes:
            writer.writerow({k: ep.get(k) for k in writer.fieldnames})


def _plot_success_vs_horizon(path: Path, on_results: list[dict[str, Any]], off_results: list[dict[str, Any]]) -> None:
    plt = _plt()
    x_on = [r["horizon"] for r in on_results]
    y_on = [r["success_rate"] for r in on_results]
    x_off = [r["horizon"] for r in off_results]
    y_off = [r["success_rate"] for r in off_results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_on, y_on, marker="o", label="URFlow on")
    plt.plot(x_off, y_off, marker="o", label="URFlow off")
    plt.xlabel("Episode horizon")
    plt.ylabel("Success rate")
    plt.title("Figure A: Success vs episode horizon")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _cum_success_curve(episodes: list[dict[str, Any]], horizon: int) -> np.ndarray:
    steps = np.arange(1, horizon + 1)
    success_steps = [ep["success_step"] if ep["success_step"] is not None else np.inf for ep in episodes]
    arr = np.array(success_steps, dtype=float)
    return np.array([(arr <= s).mean() for s in steps])


def _plot_success_over_time(path: Path, on_eps: list[dict[str, Any]], off_eps: list[dict[str, Any]], horizon: int) -> None:
    plt = _plt()
    steps = np.arange(1, horizon + 1)
    y_on = _cum_success_curve(on_eps, horizon)
    y_off = _cum_success_curve(off_eps, horizon)

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, y_on, label="URFlow on")
    plt.plot(steps, y_off, label="URFlow off")
    plt.xlabel("Step")
    plt.ylabel("Cumulative success by step")
    plt.title(f"Figure B: Success over time (H={horizon})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _mean_step_series(episodes: list[dict[str, Any]], key: str, horizon: int) -> np.ndarray:
    vals = np.full((len(episodes), horizon), np.nan, dtype=float)
    for i, ep in enumerate(episodes):
        seq = ep.get(key, [])
        upto = min(len(seq), horizon)
        if upto:
            vals[i, :upto] = np.asarray(seq[:upto], dtype=float)
    return np.nanmean(vals, axis=0)


def _plot_compounding_proxy(path: Path, on_eps: list[dict[str, Any]], off_eps: list[dict[str, Any]], horizon: int) -> str:
    plt = _plt()
    on_unc = _mean_step_series(on_eps, "uncertainty_exp_s_hat", horizon)
    off_unc = _mean_step_series(off_eps, "uncertainty_exp_s_hat", horizon)

    proxy = "uncertainty exp(s_hat)"
    if np.all(np.isnan(on_unc)) and np.all(np.isnan(off_unc)):
        proxy = "delta action L2"

        def delta_action(episodes):
            vals = np.full((len(episodes), horizon), np.nan, dtype=float)
            for i, ep in enumerate(episodes):
                acts = np.asarray(ep.get("actions", []), dtype=float)
                if len(acts) > 1:
                    d = np.linalg.norm(np.diff(acts, axis=0), axis=1)
                    vals[i, 1 : 1 + len(d)] = d[: horizon - 1]
            return np.nanmean(vals, axis=0)

        on_series = delta_action(on_eps)
        off_series = delta_action(off_eps)
    else:
        on_series = on_unc
        off_series = off_unc

    steps = np.arange(1, horizon + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, on_series, label="URFlow on")
    plt.plot(steps, off_series, label="URFlow off")
    plt.xlabel("Step")
    plt.ylabel(f"Mean {proxy}")
    plt.title(f"Figure C: Compounding error proxy over time (H={horizon})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return proxy


def _plot_refinement_activation(path: Path, rows: list[dict[str, Any]], horizons: list[int], taus: list[float]) -> None:
    plt = _plt()
    plt.figure(figsize=(7.5, 4.5))
    for tau in taus:
        y = []
        for h in horizons:
            ep = next((r for r in rows if r["horizon"] == h and r["tau"] == tau), None)
            if ep is None:
                y.append(np.nan)
                continue
            vals = [v for e in ep["episodes"] for v in e.get("activation_any", [])]
            y.append(float(np.mean(vals)) if vals else np.nan)
        plt.plot(horizons, y, marker="o", label=f"tau={tau}")

    plt.xlabel("Episode horizon")
    plt.ylabel("Refinement activation rate")
    plt.title("Figure D: Refinement activation vs horizon")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_heatmap(path: Path, heat: np.ndarray, rows: list[float], cols: list[int]) -> None:
    plt = _plt()
    plt.figure(figsize=(6.5, 4.8))
    im = plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Success rate")
    plt.xticks(np.arange(len(cols)), [str(c) for c in cols])
    plt.yticks(np.arange(len(rows)), [str(r) for r in rows])
    plt.xlabel("Horizon")
    plt.ylabel("Tau")
    plt.title("Figure E: Long-horizon sensitivity heatmap (URFlow on)")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_calibration(path: Path, episodes: list[dict[str, Any]]) -> str:
    plt = _plt()
    pred = []
    emp = []
    for ep in episodes:
        unc = np.asarray(ep.get("uncertainty_exp_s_hat", []), dtype=float)
        acts = np.asarray(ep.get("actions", []), dtype=float)
        if len(unc) < 2 or len(acts) < 2:
            continue
        err = np.linalg.norm(np.diff(acts, axis=0), axis=1)
        unc = unc[1 : 1 + len(err)]
        m = (~np.isnan(unc)) & (~np.isnan(err))
        if np.any(m):
            pred.extend(unc[m].tolist())
            emp.extend(err[m].tolist())

    pred_arr = np.asarray(pred, dtype=float)
    emp_arr = np.asarray(emp, dtype=float)
    if len(pred_arr) < 10:
        raise RuntimeError("Insufficient uncertainty/error samples for calibration plot.")

    bins = np.quantile(pred_arr, np.linspace(0.0, 1.0, 11))
    bins = np.unique(bins)
    if len(bins) < 3:
        raise RuntimeError("Predicted uncertainty has near-zero variance; cannot form calibration bins.")

    x_means = []
    y_means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred_arr >= lo) & (pred_arr <= hi)
        if np.any(m):
            x_means.append(float(np.mean(pred_arr[m])))
            y_means.append(float(np.mean(emp_arr[m])))

    plt.figure(figsize=(6.2, 4.5))
    plt.plot(x_means, y_means, marker="o")
    plt.xlabel("Mean predicted uncertainty")
    plt.ylabel("Mean empirical error proxy (delta action L2)")
    plt.title("Figure F: Uncertainty calibration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return "delta action L2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on LIBERO/LIBERO-PRO and generate paper-ready figures.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--train_suite", type=str, required=True, choices=["libero_object", "libero_spatial", "libero_goal", "libero_90", "libero_10"])
    parser.add_argument("--eval_suite", type=str, default=None, help="Optional LIBERO-PRO suite suffix (temp/task/object/lan/swap) or full suite name train_suffix.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scaler_path", type=str, default=None, help="Optional explicit path to model_scaler.pkl")
    parser.add_argument("--horizons", type=int, nargs="+", default=[50, 100, 200, 400, 600])
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--tau_list", type=float, nargs="+", default=[0.6, 0.8, 0.9])
    parser.add_argument("--refinement_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_eval_cfg(args.train_suite, args.eval_suite)
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed_everywhere(cfg.seed)

    if args.rollouts is not None:
        cfg.simulation.rollouts = args.rollouts
    cfg.simulation.use_multiprocessing = False
    cfg.simulation.render_image = False
    cfg.simulation.save_video = False

    model = _load_model_from_checkpoint(cfg, args.checkpoint_path, scaler_path=args.scaler_path)
    if args.refinement_steps is not None:
        model.refinement_steps = args.refinement_steps

    all_results: dict[str, Any] = {
        "args": vars(args),
        "config": asdict(cfg),
        "conditions": [],
        "notes": [],
    }

    ur_on_rows = []
    ur_off_rows = []

    for h in args.horizons:
        cfg.simulation.max_step_per_episode = h
        ur_on_rows.append(run_condition(cfg, model, h, urflow=True, tau=model.uncertainty_threshold))
        ur_off_rows.append(run_condition(cfg, model, h, urflow=False, tau=None))

    all_results["conditions"].extend(ur_on_rows)
    all_results["conditions"].extend(ur_off_rows)

    tau_rows = []
    for tau in args.tau_list:
        for h in args.horizons:
            cfg.simulation.max_step_per_episode = h
            tau_rows.append(run_condition(cfg, model, h, urflow=True, tau=tau))
    all_results["conditions"].extend(tau_rows)

    heatmap_taus = [0.4, 0.6, 0.8, 0.9]
    heatmap_horizons = [50, 100, 200, 600]
    heat = np.zeros((len(heatmap_taus), len(heatmap_horizons)), dtype=float)
    heat_rows = []
    for i, tau in enumerate(heatmap_taus):
        for j, h in enumerate(heatmap_horizons):
            cfg.simulation.max_step_per_episode = h
            out = run_condition(cfg, model, h, urflow=True, tau=tau)
            heat[i, j] = out["success_rate"]
            heat_rows.append(out)
    all_results["conditions"].extend(heat_rows)

    # Flatten episodes for reproducibility outputs.
    all_episodes = [ep for cond in all_results["conditions"] for ep in cond["episodes"]]

    json_path = output_dir / "evaluation_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    _save_episode_csv(output_dir / "episode_metrics.csv", all_episodes)

    _plot_success_vs_horizon(output_dir / "figure_A_success_vs_horizon.png", ur_on_rows, ur_off_rows)

    rep_h = max(args.horizons)
    rep_on = next(r for r in ur_on_rows if r["horizon"] == rep_h)
    rep_off = next(r for r in ur_off_rows if r["horizon"] == rep_h)
    _plot_success_over_time(output_dir / f"figure_B_success_over_time_H{rep_h}.png", rep_on["episodes"], rep_off["episodes"], rep_h)

    proxy_used = _plot_compounding_proxy(
        output_dir / f"figure_C_compounding_metric_H{rep_h}.png",
        rep_on["episodes"],
        rep_off["episodes"],
        rep_h,
    )
    if proxy_used != "residual norm":
        note = f"Figure C used proxy '{proxy_used}' because residual norms are not exposed by evaluation diagnostics."
        all_results["notes"].append(note)
        log.warning(note)

    _plot_refinement_activation(output_dir / "figure_D_refinement_activation_vs_horizon.png", tau_rows, args.horizons, args.tau_list)
    _plot_heatmap(output_dir / "figure_E_long_horizon_heatmap.png", heat, heatmap_taus, heatmap_horizons)

    with (output_dir / "figure_E_long_horizon_heatmap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau\\horizon", *heatmap_horizons])
        for tau, row in zip(heatmap_taus, heat):
            writer.writerow([tau, *row.tolist()])

    try:
        cal_proxy = _plot_calibration(output_dir / "figure_F_calibration.png", rep_on["episodes"])
        all_results["notes"].append(f"Figure F empirical error proxy: {cal_proxy}.")
    except RuntimeError as exc:
        warn_msg = f"Figure F skipped: {exc}"
        log.warning(warn_msg)
        all_results["notes"].append(warn_msg)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSummary table (success rate):")
    print("condition\thorizon\ttau\tsuccess_rate")
    for row in ur_on_rows:
        print(f"URFlow_ON\t{row['horizon']}\t{row['tau']}\t{row['success_rate']:.4f}")
    for row in ur_off_rows:
        print(f"URFlow_OFF\t{row['horizon']}\tNone\t{row['success_rate']:.4f}")
    for row in tau_rows:
        print(f"URFlow_tau\t{row['horizon']}\t{row['tau']}\t{row['success_rate']:.4f}")


if __name__ == "__main__":
    main()
