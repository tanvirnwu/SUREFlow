"""
Main training script using dataclass configuration instead of Hydra.
"""


import os
import sys
import logging
import random
import shutil
import datetime
import yaml
import numpy as np
import torch
import wandb
from wandb.errors import CommError, UsageError

import multiprocessing as mp

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


os.environ['NUMEXPR_MAX_THREADS'] = '64'
# Add the current directory to Python path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# Prefer the bundled LIBERO-PRO package over any globally installed libero package.
# We want imports like `from libero.libero import ...` to resolve to:
# <repo>/LIBERO-PRO/libero/libero/...
LIBERO_PRO_PYTHON_ROOT = os.path.join(REPO_ROOT, "LIBERO-PRO")
if os.path.isdir(LIBERO_PRO_PYTHON_ROOT) and LIBERO_PRO_PYTHON_ROOT not in sys.path:
    sys.path.insert(1, LIBERO_PRO_PYTHON_ROOT)

from configs.config import (
    create_libero_train_config,
    create_libero_pro_eval_config,
)
from configs.factory import create_model, create_trainer, create_simulation

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _safe_dirname(value: str | None, fallback: str) -> str:
    """Return a filesystem-safe directory name."""
    name = (value or "").strip()
    if not name:
        name = fallback
    return name.replace(os.sep, "_").replace(" ", "_")


def _ensure_unique_run_dir(base_dir: str) -> str:
    """Ensure the run directory is unique by appending a counter if needed."""
    if not os.path.exists(base_dir):
        return base_dir
    counter = 1
    while True:
        candidate = f"{base_dir}_{counter:02d}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _build_training_run_dir(cfg) -> tuple[str, str, str]:
    project_name = _safe_dirname(cfg.wandb.project, "default_project")
    mode_name = _safe_dirname(cfg.wandb.mode, "default_mode")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    run_dir = os.path.join(output_root, project_name, mode_name, run_id)
    run_dir = _ensure_unique_run_dir(run_dir)
    run_id = os.path.basename(run_dir)
    return run_dir, project_name, mode_name


def _resolve_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = checkpoint_path
    if os.path.basename(os.path.normpath(checkpoint_dir)) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    return checkpoint_dir


def _extract_project_mode(run_dir: str) -> tuple[str | None, str | None]:
    parts = os.path.normpath(run_dir).split(os.sep)
    if "logs" in parts:
        idx = parts.index("logs")
        if len(parts) >= idx + 3:
            return parts[idx + 1], parts[idx + 2]
    return None, None


def _ensure_run_subdirs(run_dir: str) -> dict[str, str]:
    subdirs = {
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "wandb": os.path.join(run_dir, "wandb"),
        "logs": os.path.join(run_dir, "logs"),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    return subdirs


def _copy_config_file(run_dir: str) -> None:
    config_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config.py")
    config_dest = os.path.join(run_dir, "config.py")
    shutil.copyfile(config_src, config_dest)


def _resolve_wandb_mode(wandb_cfg) -> str | None:
    valid_modes = {"dryrun", "online", "run", "offline", "disabled"}
    env_mode = (os.getenv("WANDB_MODE") or "").strip()
    if env_mode:
        if env_mode in valid_modes:
            return env_mode
        log.warning(
            "Ignoring unsupported WANDB_MODE '%s'. Expected one of %s.",
            env_mode,
            sorted(valid_modes),
        )
        return None
    cfg_mode = (wandb_cfg.mode or "").strip()
    if not cfg_mode:
        return None
    if cfg_mode not in valid_modes:
        log.warning(
            "Ignoring unsupported W&B mode '%s'. Expected one of %s.",
            cfg_mode,
            sorted(valid_modes),
        )
        return None
    return cfg_mode


def _merge_wandb_tags(existing_tags, required_tags: list[str]) -> list[str]:
    merged: list[str] = []
    if existing_tags:
        if isinstance(existing_tags, (list, tuple, set)):
            merged.extend([str(tag) for tag in existing_tags])
        else:
            merged.append(str(existing_tags))
    for tag in required_tags:
        if tag not in merged:
            merged.append(tag)
    return merged


def init_wandb_logging(
    cfg,
    wandb_config,
    *,
    run_name: str,
    wandb_dir: str,
    group: str | None,
    job_type: str,
    tags: list[str],
):
    """Initialise Weights & Biases logging if enabled in the config."""

    wandb_cfg = cfg.wandb

    if not getattr(wandb_cfg, "enabled", True):
        log.info("W&B logging disabled via configuration; skipping initialisation.")
        return None

    project = wandb_cfg.project
    entity = os.getenv("WANDB_ENTITY", wandb_cfg.entity)
    mode = _resolve_wandb_mode(wandb_cfg) or ""

    if mode:
        os.environ["WANDB_MODE"] = mode

    if project is None:
        log.info("No W&B project specified; skipping remote logging.")
        return None

    os.makedirs(wandb_dir, exist_ok=True)
    init_kwargs = {
        "project": project,
        "config": wandb_config,
        "name": run_name,
        "dir": wandb_dir,
        "job_type": job_type,
    }

    if group:
        init_kwargs["group"] = group

    if entity:
        init_kwargs["entity"] = entity

    merged_tags = _merge_wandb_tags(getattr(wandb_cfg, "tags", None), tags)
    if merged_tags:
        init_kwargs["tags"] = merged_tags

    try:
        return wandb.init(**init_kwargs)
    except (CommError, UsageError) as err:
        log.warning("W&B initialisation failed (%s); continuing without remote logging.", err)
    except Exception as err:  # noqa: BLE001 - broad catch to keep training running
        log.warning("Unexpected error during W&B initialisation (%s). Continuing without remote logging.", err)

    return None


def _configure_libero_pro_paths(repo_root: str) -> None:
    """Point LIBERO path resolution to the bundled LIBERO-PRO assets."""
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

    log.info(
        "Configured LIBERO paths for LIBERO-PRO eval (bddl=%s, init=%s).",
        path_config["bddl_files"],
        path_config["init_states"],
    )


def _clear_local_libero_pro_override(repo_root: str) -> None:
    """Clear repo-local LIBERO-PRO path override so vanilla LIBERO config is used."""
    config_root = os.path.join(repo_root, ".libero")
    configured_path = os.environ.get("LIBERO_CONFIG_PATH")
    if configured_path and os.path.abspath(configured_path) == os.path.abspath(config_root):
        os.environ.pop("LIBERO_CONFIG_PATH", None)
        log.info("Cleared repo-local LIBERO_CONFIG_PATH override to use vanilla LIBERO paths.")


def main(train_suite: str = "libero_object", eval_suite: str | None = None, checkpoint_path: str | None = None) -> None:
    """
    Main training function.

    Args:
        train_suite: The train task suite to use ('libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10')
        eval_suite: Optional LIBERO PRO eval suffix ('swap', 'object', 'lan', 'task', 'temp')
    """

    repo_root = os.path.dirname(os.path.abspath(__file__))

    if eval_suite is None:
        _clear_local_libero_pro_override(repo_root)
        cfg = create_libero_train_config(train_suite)
    else:
        cfg = create_libero_pro_eval_config(train_suite, eval_suite)
        _configure_libero_pro_paths(repo_root)

    set_seed_everywhere(cfg.seed)

    # Initialize wandb logger
    wandb_config = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "group": cfg.wandb.mode or cfg.group,
        "seed": cfg.seed,
        "benchmark_type": cfg.dataset.benchmark_type,
        "demos_per_task": cfg.dataset.demos_per_task,
        "chunck_size": cfg.chunck_size,
        "perception_seq_len": cfg.perception_seq_len,
        "action_seq_len": cfg.action_seq_len,
        "train_batch_size": cfg.train_batch_size,
        "epoch": cfg.epoch,
        "device": cfg.device,
        "len_embd": cfg.len_embd,
        "latent_dim": cfg.latent_dim,
        "action_dim": cfg.action_dim,
        "state_dim": cfg.state_dim,
    }

    is_evaluation = checkpoint_path is not None
    if is_evaluation:
        run_dir = _resolve_run_dir_from_checkpoint(checkpoint_path)
        run_id = os.path.basename(os.path.normpath(run_dir))
        extracted_project, extracted_mode = _extract_project_mode(run_dir)
        project_name = extracted_project or cfg.wandb.project
        mode_name = cfg.wandb.mode or extracted_mode
        evaluation_dir = os.path.join(run_dir, "evaluation")
        os.makedirs(evaluation_dir, exist_ok=True)
        wandb_dir = os.path.join(evaluation_dir, "wandb")
        logs_dir = os.path.join(evaluation_dir, "logs")
        os.makedirs(wandb_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
    else:
        run_dir, project_name, mode_name = _build_training_run_dir(cfg)
        subdirs = _ensure_run_subdirs(run_dir)
        wandb_dir = subdirs["wandb"]
        _copy_config_file(run_dir)
        run_id = os.path.basename(run_dir)

    if is_evaluation:
        wandb_run_name = "evaluation"
        wandb_group = mode_name
        wandb_job_type = "eval"
        wandb_tags = ["evaluation", run_id]
    else:
        wandb_run_name = cfg.wandb.mode
        wandb_group = cfg.wandb.mode
        wandb_job_type = "train"
        wandb_tags = [cfg.wandb.mode, run_id]

    visuals_dir = os.path.join(run_dir, "visuals")
    visuals_training_dir = os.path.join(visuals_dir, "training")
    visuals_testing_dir = os.path.join(visuals_dir, "testing")
    os.makedirs(visuals_training_dir, exist_ok=True)
    os.makedirs(visuals_testing_dir, exist_ok=True)

    run = init_wandb_logging(
        cfg,
        wandb_config,
        run_name=wandb_run_name,
        wandb_dir=wandb_dir,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=wandb_tags,
    )

    if is_evaluation:
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
    else:
        checkpoints_dir = os.path.join(run_dir, "checkpoints")

    # Create model and set its working_dir to the run-specific directory
    model = create_model(cfg)
    model.working_dir = checkpoints_dir

    # Create trainer and set its working_dir as well
    trainer = create_trainer(cfg)
    trainer.working_dir = checkpoints_dir
    trainer.configure_visuals_dir(visuals_training_dir)

    # Get model parameters for logging
    model.get_params()

    # If a checkpoint is provided, load it and skip training
    if checkpoint_path is not None:
        # Set scaler from trainer to ensure inference works
        model.set_scaler(trainer.scaler)

        # Resolve checkpoint: file path or directory
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, weights_only=True)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                log.warning(
                    "Checkpoint load had mismatched keys (missing=%s, unexpected=%s).",
                    missing_keys,
                    unexpected_keys,
                )
            log.info(f"Loaded checkpoint from file: {checkpoint_path}")
        elif os.path.isdir(checkpoint_path):
            candidates = [
                os.path.join(checkpoint_path, "final_model.pth"),
                os.path.join(checkpoint_path, "model_state_dict.pth"),
            ]
            loaded = False
            for cand in candidates:
                if os.path.isfile(cand):
                    state_dict = torch.load(cand, weights_only=True)
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    if missing_keys or unexpected_keys:
                        log.warning(
                            "Checkpoint load had mismatched keys (missing=%s, unexpected=%s).",
                            missing_keys,
                            unexpected_keys,
                        )
                    log.info(f"Loaded checkpoint from directory: {cand}")
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path} (looked for final_model.pth, model_state_dict.pth)")
    else:
        # Train the model if no checkpoint provided
        trainer.main(model)

    # Create simulation environment
    env_sim = create_simulation(cfg)
    env_sim.configure_visuals(cfg.visuals, visuals_testing_dir)

    # Test the model
    env_sim.test_model(model, cfg.model_cfg, epoch=cfg.epoch)

    log.info("Training done")
    log.info("state_dict saved in {}".format(model.working_dir))

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MambaVLA model")
    parser.add_argument(
        "--train_suite",
        type=str,
        default="libero_object",
        choices=["libero_object", "libero_spatial", "libero_goal", "libero_90", "libero_10"],
        help="Task suite to use for training dataset and training-language embeddings"
    )
    parser.add_argument(
        "--eval_suite",
        type=str,
        default=None,
        choices=["swap", "object", "lan", "task", "temp"],
        help="Optional LIBERO PRO evaluation suffix; when set, simulation uses <train_suite>_<eval_suite>."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint (.pth file or directory). If provided, skips training and evaluates with this checkpoint."
    )

    args = parser.parse_args()
    main(args.train_suite, args.eval_suite, args.checkpoint_path)
