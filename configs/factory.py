"""
Factory functions to instantiate objects from dataclass configurations.
This replaces Hydra's instantiate functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Type, Union
from configs.config import (
    MainConfig, ModelConfig, TrainerConfig, SimulationConfig, 
    ModelConfig, ObsEncoderConfig, LanguageEncoderConfig,
    OptimizerConfig, LRSchedulerConfig
)


def instantiate_from_config(config: Any, **kwargs) -> Any:
    """
    Instantiate an object from a configuration dataclass.
    This replaces hydra.utils.instantiate functionality.
    """
    if hasattr(config, '_target_'):
        target_class = _get_class_from_target(config._target_)
        
        # Extract parameters from config, excluding _target_ and _recursive_
        params = {}
        for key, value in config.__dict__.items():
            if key not in ['_target_', '_recursive_']:
                params[key] = value
        
        # Add any additional kwargs (these override config values)
        params.update(kwargs)
        
        return target_class(**params)
    else:
        # If no _target_, return the config as is
        return config


def _get_class_from_target(target: str) -> Type:
    """
    Get a class from its target string (e.g., 'torch.optim.AdamW').
    """
    if target == "torch.optim.AdamW":
        return torch.optim.AdamW
    elif target == "torch.optim.Adam":
        return torch.optim.Adam
    elif target == "torch.optim.SGD":
        return torch.optim.SGD
    elif target == "MambaVLA.MambaVLA" or target == "MambaVLA.MambaVLA":
        # Import via package __init__ to keep a stable public API
        from MambaVLA import MambaVLA
        return MambaVLA
    elif target == "MambaVLA.Trainer":
        from MambaVLA.main import Trainer
        return Trainer
    elif target == "MambaVLA.benchmark.libero.libero_sim.LiberoSim":
        from MambaVLA.benchmark.libero.libero_sim import MultiTaskSim
        return MultiTaskSim
    elif target == "MambaVLA.ActionFLowMatching":
        from MambaVLA import ActionFLowMatching
        return ActionFLowMatching
    elif target == "MambaVLA.MambaVLAPolicy":
        from MambaVLA import MambaVLAPolicy
        return MambaVLAPolicy
    elif target == "MambaVLA.MambaModel":
        from MambaVLA import MambaModel
        return MambaModel
    elif target == "MambaVLA.CLIPImgEncoder":
        from MambaVLA import CLIPImgEncoder
        return CLIPImgEncoder
    elif target == "MambaVLA.LangClip":
        from MambaVLA import LangClip
        return LangClip
    elif target == "MambaVLA.MultiImageObsEncoder":
        from MambaVLA import MultiImageObsEncoder
        return MultiImageObsEncoder
    elif target == "MambaVLA.ResNetEncoder":
        from MambaVLA import ResNetEncoder
        return ResNetEncoder
    elif target == "MambaVLA.benchmark.libero.libero_dataset.LiberoDataset":
        from MambaVLA.benchmark.libero.libero_dataset import LiberoDataset
        return LiberoDataset
    else:
        raise ValueError(f"Unknown target class: {target}")


def create_model(config: MainConfig) -> Any:
    """Create an model from the main configuration."""
    model_config = config.model_cfg
    
    # Create the encoder first
    encoder = instantiate_from_config(model_config.model.backbones.encoder)
    
    # Create the backbone with the encoder
    backbone = instantiate_from_config(
        model_config.model.backbones,
        encoder=encoder
    )
    
    # Create the model with the backbone
    model = instantiate_from_config(
        model_config.model,
        backbones=backbone
    )
    
    # Create encoders
    obs_encoder = instantiate_from_config(model_config.obs_encoders)
    lang_encoder = instantiate_from_config(model_config.language_encoders)
    
    # Create the model
    model = instantiate_from_config(
        model_config,
        model=model,
        obs_encoders=obs_encoder,
        language_encoders=lang_encoder,
        action_dim=config.action_dim,
        perception_seq_len=config.perception_seq_len,
        action_seq_len=config.action_seq_len,
        cam_names=config.camera_names,
        device=config.device,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        sampling_steps=config.num_sampling_steps
    )
    
    return model


def create_trainer(config: MainConfig) -> Any:
    """Create a trainer from the main configuration."""
    trainer_config = config.trainer
    
    # Create dataset directly with only the required parameters
    from MambaVLA.benchmark.libero.libero_dataset import LiberoDataset
    from pathlib import Path
    dataset = LiberoDataset(
        data_directory=Path(config.dataset.dataset_path),
        device=config.device,
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        state_dim=config.state_dim,
        max_len_data=config.max_len_data,
        chunck_size=config.chunck_size,
        demos_per_task=config.dataset.demos_per_task
    )
    
    # Create the trainer directly with individual parameters from sub-configurations
    from MambaVLA.main import Trainer
    trainer = Trainer(
        training_dataset=dataset,
        validation_dataset=dataset,  # Using same dataset for validation
        training_batch_size=trainer_config.train_batch_size,
        validation_batch_size=trainer_config.val_batch_size,
        dataloader_workers=trainer_config.num_workers,
        device=trainer_config.device,
        total_epochs=trainer_config.epoch,
        enable_data_scaling=trainer_config.scale_data,
        data_scaler_type=trainer_config.scaling_type,
        evaluation_frequency=trainer_config.eval_every_n_epochs,
        observation_sequence_length=trainer_config.perception_seq_len,
        ema_decay_rate=trainer_config.decay_ema,
        enable_ema=trainer_config.if_use_ema,
        checkpoint_frequency=trainer_config.save_every_n_epochs,
        visuals_config=config.visuals
    )
    
    return trainer


def create_simulation(config: MainConfig) -> Any:
    """Create a simulation from the main configuration."""
    sim_config = config.simulation
    
    # Create the simulation with all required parameters
    simulation = instantiate_from_config(
        sim_config,
        rollouts=sim_config.rollouts,
        max_step_per_episode=sim_config.max_step_per_episode,
        benchmark_type=(config.eval_benchmark_type or config.dataset.benchmark_type),
        use_eye_in_hand=sim_config.use_eye_in_hand,
        seed=config.seed,
        device=config.device,
        render_image=sim_config.render_image,
        n_cores=sim_config.n_cores,
        use_multiprocessing=sim_config.use_multiprocessing
    )
    simulation.cfg = config

    return simulation


def create_optimizer(model: Any, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Create an optimizer for the model."""
    if config._target_ == "torch.optim.AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.betas[0], config.betas[1]),
            weight_decay=config.transformer_weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config._target_}")


def create_lr_scheduler(optimizer: torch.optim.Optimizer, config: LRSchedulerConfig) -> Any:
    """Create a learning rate scheduler."""
    from MambaVLA.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
    from omegaconf import DictConfig
    
    # Create a DictConfig-like structure for the scheduler
    scheduler_config = DictConfig({
        'lr_scheduler': {
            'init_lr': config.init_lr,
            'init_lr_scale': config.init_lr_scale,
            'final_lr_scale': config.final_lr_scale,
            'total_steps': config.total_steps,
            'phase_ratio': config.phase_ratio,
            'lr': config.lr
        }
    })
    
    return TriStageLRScheduler(
        optimizer=optimizer,
        configs=scheduler_config
    )
