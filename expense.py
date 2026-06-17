import argparse
import itertools
import random
from typing import Dict, List, Optional, Tuple

import torch

from configs.config import ALLOWED_TRAIN_SUITES, MainConfig, create_libero_train_config
from configs.factory import create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bool_str(value: bool) -> str:
    return "T" if value else "F"


def _make_dummy_obs(cfg: MainConfig, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    obs: Dict[str, torch.Tensor] = {}

    for camera_name in cfg.camera_names:
        key = f"{camera_name}_image"
        shape_meta = cfg.shape_meta.obs.get(key)
        if shape_meta is None or "shape" not in shape_meta:
            raise KeyError(f"Missing shape metadata for observation key '{key}'.")
        channels, height, width = shape_meta["shape"]
        obs[key] = torch.randn(batch_size, seq_len, channels, height, width, device=device)

    obs["lang_emb"] = torch.randn(batch_size, cfg.lang_emb_dim, device=device)

    if cfg.consider_robot_states:
        obs["robot_states"] = torch.randn(batch_size, seq_len, cfg.state_dim, device=device)

    return obs


def _load_checkpoint_if_possible(model: torch.nn.Module, checkpoint_path: Optional[str], device: torch.device) -> None:
    if not checkpoint_path:
        return

    try:
        checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        print(f"Warning: could not load checkpoint from '{checkpoint_path}': {exc}")
        return

    state_dict = None
    if isinstance(checkpoint_obj, dict):
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            state_dict = checkpoint_obj["state_dict"]
        elif "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
            state_dict = checkpoint_obj["model_state_dict"]
        else:
            state_dict = checkpoint_obj

    if not isinstance(state_dict, dict):
        print("Warning: checkpoint format is not recognized as a state_dict; continuing with random weights.")
        return

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                "Warning: checkpoint loaded with key mismatches "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
            )
    except Exception as exc:
        print(f"Warning: failed to apply checkpoint weights: {exc}")


def _get_flops_backend():
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        return "fvcore", FlopCountAnalysis
    except Exception:
        try:
            from thop import profile  # type: ignore

            return "thop", profile
        except Exception:
            return None, None


def _compute_flops(
    model: torch.nn.Module,
    obs: Dict[str, torch.Tensor],
    backend_name: Optional[str],
    backend_obj,
) -> Tuple[float, Optional[float]]:
    if backend_name is None:
        raise RuntimeError("No FLOPs backend available. Please install fvcore or thop.")

    with torch.no_grad():
        if backend_name == "fvcore":
            flops = backend_obj(model, (obs,)).total()
            return float(flops) / 1e9, None

        macs, _ = backend_obj(model, inputs=(obs,), verbose=False)
        flops = float(macs) * 2.0
        return flops / 1e9, float(macs) / 1e9


def _build_model_for_combo(
    cfg: MainConfig,
    use_sigma_film: bool,
    use_action_decoder: bool,
    decoder_use_action_tokens_as_queries: bool,
    use_uncertainty: bool,
) -> torch.nn.Module:
    cfg.model_cfg.model.backbones.use_sigma_film = use_sigma_film
    cfg.model_cfg.model.backbones.use_action_decoder = use_action_decoder
    cfg.model_cfg.model.backbones.decoder_use_action_tokens_as_queries = decoder_use_action_tokens_as_queries
    cfg.model_cfg.use_uncertainty = use_uncertainty

    return create_model(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute parameters and FLOPs for model feature-flag combinations.")
    parser.add_argument("--suite", type=str, required=True, help="LIBERO suite, e.g. libero_object, libero_spatial.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on, auto picks cuda if available.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy input.")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Observation sequence length for dummy input. Defaults to config.perception_seq_len.",
    )
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cpu":
        raise RuntimeError(
            "This model uses Mamba selective scan that requires CUDA tensors. "
            "Please run with device cuda, for example: python expense.py --device cuda ..."
        )
    if args.suite not in ALLOWED_TRAIN_SUITES:
        raise ValueError(f"Unsupported suite '{args.suite}'. Supported suites: {sorted(ALLOWED_TRAIN_SUITES)}")

    cfg = create_libero_train_config(args.suite)
    cfg.device = args.device
    cfg.model_cfg.device = args.device
    cfg.model_cfg.model.device = args.device

    seq_len = args.seq_len if args.seq_len is not None else cfg.perception_seq_len

    set_seed(cfg.seed)
    device = torch.device(args.device)
    backend_name, backend_obj = _get_flops_backend()

    skipped_invalid = 0
    evaluated = 0

    combinations: List[Tuple[bool, bool, bool, bool]] = list(
        itertools.product([False, True], [False, True], [False, True], [False, True])
    )

    for use_sigma_film, use_action_decoder, decoder_queries, use_uncertainty in combinations:
        if not use_action_decoder and decoder_queries:
            skipped_invalid += 1
            continue

        effective_decoder_queries = decoder_queries if use_action_decoder else False

        try:
            model = _build_model_for_combo(
                cfg=cfg,
                use_sigma_film=use_sigma_film,
                use_action_decoder=use_action_decoder,
                decoder_use_action_tokens_as_queries=effective_decoder_queries,
                use_uncertainty=use_uncertainty,
            )
        except Exception as exc:
            raise RuntimeError(
                "Model construction failed for combination "
                f"(use_sigma_film={use_sigma_film}, use_action_decoder={use_action_decoder}, "
                f"decoder_use_action_tokens_as_queries={effective_decoder_queries}, use_uncertainty={use_uncertainty})."
            ) from exc

        model = model.to(device)
        model.eval()

        _load_checkpoint_if_possible(model, args.checkpoint, device)

        obs = _make_dummy_obs(cfg, args.batch_size, seq_len, device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        flops_g, macs_g = _compute_flops(model, obs, backend_name, backend_obj)

        line = (
            f"use_sigma_film={_bool_str(use_sigma_film)} "
            f"use_action_decoder={_bool_str(use_action_decoder)} "
            f"decoder_use_action_tokens_as_queries={_bool_str(effective_decoder_queries)} "
            f"use_uncertainty={_bool_str(use_uncertainty)} | "
            f"total_params={total_params} trainable_params={trainable_params} | "
            f"flops_g={flops_g:.6f}"
        )
        if macs_g is not None:
            line += f" macs_g={macs_g:.6f}"
        print(line)

        evaluated += 1

    print("Summary:")
    print(f"combinations_evaluated={evaluated}")
    print(f"flops_backend={backend_name or 'none'}")
    print(f"combinations_skipped_invalid={skipped_invalid}")


if __name__ == "__main__":
    main()
