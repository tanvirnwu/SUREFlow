"""
Dataclass-based configuration system to replace Hydra configuration.d
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
import torch


# Common configuration values to avoid duplication
DEVICE = "cuda"
ACTION_DIM = 7
STATE_DIM = 9
LATENT_DIM = 256
LANG_EMB_DIM = 512
len_embd = 256
perception_seq_len = 1
action_seq_len = 10
consider_robot_states = False
CAMERA_NAMES = ["agentview", "eye_in_hand"]


@dataclass
class WandbConfig:
    """Wandb configuration."""
    enabled: bool = True
    entity: Optional[str] = "tanvirnwu"
    project: Optional[str] = "SUREFlow"
    mode: Optional[str] = "LG_AQD_lambda_u:0.001_bs:256,demo:70,sam:70"
    tags: List[str] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    _target_: str = "torch.optim.AdamW"
    transformer_weight_decay: float = 0.05
    obs_encoder_weight_decay: float = 0.05
    learning_rate: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.9])


@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""
    init_lr: float = 1e-4
    init_lr_scale: float = 0.1
    final_lr_scale: float = 1e-6
    total_steps: int = 50000
    phase_ratio: str = "(0.02, 0.08, 0.9)"
    lr: float = 1e-4


@dataclass
class ShapeMetaConfig:
    """Shape metadata configuration for observations."""
    obs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "agentview_image": {
            "shape": [3, 128, 128],
            "type": "rgb"
        },
        "eye_in_hand_image": {
            "shape": [3, 128, 128],
            "type": "rgb"
        }
    })


@dataclass
class MambaEncoderConfig:
    """Mamba encoder configuration."""
    _target_: str = "SUREFlow.MambaModel"
    d_model: int = 256
    n_layer: int = 5
    d_intermediate: int = 256
    ssm_cfg: Dict[str, Any] = field(default_factory=lambda: {
        "layer": "Mamba1",
        "d_state": 64,
        "d_conv": 4,
        "expand": 2
    })


@dataclass
class BackboneConfig:
    """Backbone configuration."""
    _target_: str = "SUREFlow.SUREFlowPolicy"
    latent_dim: int = LATENT_DIM
    action_dim: int = ACTION_DIM
    lang_emb_dim: int = LANG_EMB_DIM
    goal_conditioned: bool = True
    lang_tok_len: int = 1
    obs_tok_len: int = 2  
    action_seq_len: int = action_seq_len
    embed_pdrob: int = 0
    embed_dim: int = LATENT_DIM
    device: str = DEVICE
    linear_output: bool = True
    use_ada_conditioning: bool = False
    use_sigma_film: bool = False
    use_action_decoder: bool = True
    action_decoder_heads: int = 4
    action_decoder_mlp_ratio: float = 2.0
    action_decoder_dropout: float = 0.0
    decoder_use_action_tokens_as_queries: bool = True
    encoder: MambaEncoderConfig = field(default_factory=MambaEncoderConfig)


@dataclass
class ActionFlowMatchingConfig:
    """Model configuration."""
    _target_: str = "SUREFlow.ActionFLowMatching"
    ln: bool = False
    device: str = DEVICE
    backbones: BackboneConfig = field(default_factory=BackboneConfig)


@dataclass
class ObsEncoderConfig:
    """Observation encoder configuration."""
    _target_: str = "SUREFlow.MultiImageObsEncoder"
    shape_meta: Dict = field(default_factory=lambda: {
        'obs': {
            'agentview_image': {'shape': [3, 128, 128], 'type': 'rgb'},
            'eye_in_hand_image': {'shape': [3, 128, 128], 'type': 'rgb'}
        }
    })
    rgb_model: Dict = field(default_factory=lambda: {
        '_target_': 'SUREFlow.ResNetEncoder',
        'latent_dim': 256,
        'pretrained': False,
        'freeze_backbone': False,
        'use_mlp': True
    })
    resize_shape: Optional[Tuple[int, int]] = None
    random_crop: bool = False
    use_group_norm: bool = True
    share_rgb_model: bool = False
    imagenet_norm: bool = True


@dataclass
class LanguageEncoderConfig:
    """Language encoder configuration."""
    _target_: str = "SUREFlow.LangClip"
    # model_name: str = "ViT-B/32"


@dataclass
class ModelConfig:
    """Model configuration."""
    _target_: str = "SUREFlow.SUREFlow"
    if_film_condition: bool = False
    consider_robot_states: bool = consider_robot_states
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    use_lr_scheduler: bool = False
    perception_seq_len: int = perception_seq_len
    action_seq_len: int = action_seq_len
    cam_names: List[str] = field(default_factory=lambda: CAMERA_NAMES)
    device: str = DEVICE
    state_dim: int = STATE_DIM
    latent_dim: int = LATENT_DIM
    action_dim: int = ACTION_DIM
    sampling_steps: int = 70
    use_uncertainty: bool = True
    lambda_u: float = 0.001
    lambda_u_warmup_epochs: int = 5
    lambda_u_target: float = 0.001
    refinement_steps: int = 3
    uncertainty_threshold: float = 0.9
    model: ActionFlowMatchingConfig = field(default_factory=ActionFlowMatchingConfig)
    obs_encoders: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    language_encoders: LanguageEncoderConfig = field(default_factory=LanguageEncoderConfig)


@dataclass
class DataLoadingConfig:
    """Data loading configuration."""
    train_batch_size: int = 256
    val_batch_size: int = 256
    num_workers: int = 8


@dataclass
class TrainingConfig:
    """Training process configuration."""
    epoch: int = 300
    perception_seq_len: int = perception_seq_len
    eval_every_n_epochs: int = 5
    save_every_n_epochs: int = 50


@dataclass
class DataScalingConfig:
    """Data scaling configuration."""
    scale_data: bool = True
    scaling_type: str = "minmax"


@dataclass
class EMAConfig:
    """Exponential Moving Average configuration."""
    decay_ema: float = 0.995
    if_use_ema: bool = True


@dataclass
class VisualsConfig:
    """Visualization configuration."""
    enabled: bool = False
    train_every_steps: int = 1000
    test_num_episodes_per_task: int = 3
    save_heatmaps: bool = True
    save_calibration: bool = True
    save_trends: bool = True
    save_refinement_effect: bool = True


@dataclass
class TrainerConfig:
    """Trainer configuration."""
    _target_: str = "SUREFlow.Trainer"
    device: str = DEVICE
    
    # Sub-configurations
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_scaling: DataScalingConfig = field(default_factory=DataScalingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    
    @property
    def train_batch_size(self) -> int:
        return self.data_loading.train_batch_size
    
    @property
    def val_batch_size(self) -> int:
        return self.data_loading.val_batch_size
    
    @property
    def num_workers(self) -> int:
        return self.data_loading.num_workers
    
    @property
    def epoch(self) -> int:
        return self.training.epoch
    
    @property
    def perception_seq_len(self) -> int:
        return self.training.perception_seq_len
    
    @property
    def eval_every_n_epochs(self) -> int:
        return self.training.eval_every_n_epochs
    
    @property
    def save_every_n_epochs(self) -> int:
        return self.training.save_every_n_epochs
    
    @property
    def scale_data(self) -> bool:
        return self.data_scaling.scale_data
    
    @property
    def scaling_type(self) -> str:
        return self.data_scaling.scaling_type
    
    @property
    def decay_ema(self) -> float:
        return self.ema.decay_ema
    
    @property
    def if_use_ema(self) -> bool:
        return self.ema.if_use_ema




@dataclass
class DatasetConfig:
    """Dataset configuration."""
    _target_: str = "dataloader.libero_dataset.LiberoDataset"
    # Note: benchmark_type is not passed to LiberoDataset, it's extracted from data_directory
    benchmark_type: str = "libero_object"  # Used for path construction
    demos_per_task: int = 70
    dataset_path: str = "/home/HDD/tanvir_HDD/datasets/robot/"
    perception_seq_len: int = perception_seq_len
    action_seq_len: int = action_seq_len
    multistep: int = 10

    goal_conditioned: bool = True
    use_pos_emb: bool = True
    num_sampling_steps: int = 70
    if_use_ema: bool = True
    obs_tokens: int = 2
    obs_dim: int = 9
    action_dim: int = ACTION_DIM
    state_dim: int = STATE_DIM
    max_len_data: int = 347
    consider_robot_states: bool = consider_robot_states
    camera_names: List[str] = field(default_factory=lambda: CAMERA_NAMES)
    shape_meta: ShapeMetaConfig = field(default_factory=ShapeMetaConfig)
    len_embd: int = len_embd
    lang_emb_dim: int = LANG_EMB_DIM
    latent_dim: int = LATENT_DIM
    n_heads: int = 4
    mamba_encoder_cfg: Dict[str, Any] = field(default_factory=dict)
    mamba_n_layer_encoder: int = 4


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    _target_: str = "dataloader.libero_sim.LiberoSim"
    rollouts: int = 1
    max_step_per_episode: int = 600
    benchmark_type: str = DatasetConfig.benchmark_type
    use_eye_in_hand: bool = False
    seed: int = 11
    device: str = DEVICE
    render_image: bool = False
    n_cores: int = 2
    use_multiprocessing: bool = False
    save_video: bool = True
    save_video_dir: str = "/home/HDD/tanvir_HDD/robot/SUREFlow/Evl_Video"

@dataclass
class MainConfig:
    """Main configuration class that contains all other configurations."""
    # Basic settings
    model_name: str = "mamba"
    group: str = "SUREFlow"
    seed: int = 0
    project: Optional[str] = "SUREFlow"
    mode: Optional[str] = None
    eval_benchmark_type: Optional[str] = None
    
    # Wandb configuration
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Model configuration
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    
    # Trainer configuration
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Visualization configuration
    visuals: VisualsConfig = field(default_factory=VisualsConfig)
    
    # Simulation configuration
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Training parameters
    train_batch_size: int = 256
    val_batch_size: int = 256
    num_workers: int = 4
    device: str = DEVICE
    epoch: int = 300
    eval_every_n_epochs: int = 5
    scale_data: bool = True
    scaling_type: str = "minmax"
    
    # Environment parameters
    obs_dim: int = 9
    action_dim: int = ACTION_DIM
    state_dim: int = STATE_DIM
    max_len_data: int = 260
    
    # Observations
    consider_robot_states: bool = consider_robot_states
    camera_names: List[str] = field(default_factory=lambda: CAMERA_NAMES)
    shape_meta: ShapeMetaConfig = field(default_factory=ShapeMetaConfig)
    
    # Model parameters
    chunck_size: int = 10
    perception_seq_len: int = perception_seq_len
    action_seq_len: int = action_seq_len
    multistep: int = 10

    goal_conditioned: bool = True
    use_pos_emb: bool = True
    num_sampling_steps: int = 70
    if_use_ema: bool = True
    obs_tokens: int = 2
    
    # Architecture parameters
    len_embd: int = len_embd
    lang_emb_dim: int = LANG_EMB_DIM
    latent_dim: int = LATENT_DIM

    n_heads: int = 4
    mamba_encoder_cfg: Dict[str, Any] = field(default_factory=lambda: {
        "layer": "Mamba1",
        "d_state": 64,
        "d_conv": 5,
        "expand": 2
    })
    mamba_n_layer_encoder: int = 5


def create_config() -> MainConfig:
    """Create and return the main configuration."""
    return MainConfig()


def create_libero_object_config() -> MainConfig:
    """Create configuration for libero_object task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_object"
    config.dataset.dataset_path = "/home/tanvir/projects/RoboMani/data/libero_object/"
    return config


def create_libero_spatial_config() -> MainConfig:
    """Create configuration for libero_spatial task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_spatial"
    config.dataset.dataset_path = "/home/tanvir/projects/RoboMani/data/libero_spatial/"
    return config


def create_libero_goal_config() -> MainConfig:
    """Create configuration for libero_goal task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_goal"
    config.dataset.dataset_path = "/home/HDD/tanvir_HDD/datasets/robot/libero_goal/"

    return config

def create_libero_90_config() -> MainConfig:
    """Create configuration for libero_90 task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_90"
    config.dataset.dataset_path = "/home/tanvir/projects/RoboMani/data/libero_90/" 
    config.max_len_data = config.dataset.max_len_data
    config.trainer.data_loading.num_workers = 0

    return config


def create_libero_10_config() -> MainConfig:
    """Create configuration for libero_10 task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_10"
    config.dataset.dataset_path = "/home/tanvir/projects/RoboMani/data/libero_10/"
    config.max_len_data = config.dataset.max_len_data
    config.trainer.data_loading.num_workers = 0

    return config



ALLOWED_TRAIN_SUITES = {
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_90",
    "libero_10",
}

ALLOWED_PRO_SUFFIXES = {"swap", "object", "lan", "task", "temp"}


def create_libero_train_config(train_suite: str) -> MainConfig:
    """Create configuration for a LIBERO train suite."""
    if train_suite not in ALLOWED_TRAIN_SUITES:
        raise ValueError(f"Unsupported train_suite '{train_suite}'. Expected one of: {sorted(ALLOWED_TRAIN_SUITES)}")

    config = MainConfig()
    config.dataset.benchmark_type = train_suite
    #    config.dataset.dataset_path = f"/home/tanvir/projects/RoboMani/data/{train_suite}/"
    config.dataset.dataset_path = f"/home/HDD/tanvir_HDD/datasets/robot/{train_suite}/"

    if train_suite in {"libero_90", "libero_10"}:
        config.max_len_data = config.dataset.max_len_data
        # Large suites keep full trajectories/images in memory. With spawn workers,
        # each worker receives a pickled copy of the full dataset and can OOM.
        config.trainer.data_loading.num_workers = 0

    return config


def create_libero_pro_eval_config(train_suite: str, pro_suffix: str) -> MainConfig:
    """Create configuration for LIBERO PRO evaluation from a train suite."""
    if pro_suffix not in ALLOWED_PRO_SUFFIXES:
        raise ValueError(f"Unsupported pro_suffix '{pro_suffix}'. Expected one of: {sorted(ALLOWED_PRO_SUFFIXES)}")

    config = create_libero_train_config(train_suite)
    config.eval_benchmark_type = f"{train_suite}_{pro_suffix}"
    return config
