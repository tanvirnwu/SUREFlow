"""
Dataclass-based configuration system to replace Hydra configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Shared constants
#
# Single source of truth for values referenced by more than one config. Edit
# here to change a value everywhere it is used.
# ---------------------------------------------------------------------------

# Hardware / dimensions
DEVICE = "cuda"
ACTION_DIM = 7
STATE_DIM = 9
LATENT_DIM = 256
LANG_EMB_DIM = 512
LEN_EMBD = 256

# Sequence / observation settings
PERCEPTION_SEQ_LEN = 1
ACTION_SEQ_LEN = 10
CONSIDER_ROBOT_STATES = False
CAMERA_NAMES = ["agentview", "eye_in_hand"]

# Training defaults (shared between MainConfig and TrainerConfig sub-configs)
EPOCHS = 400
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
NUM_WORKERS = 0

# Sampling
NUM_SAMPLING_STEPS = 50

# Dataset paths
DATA_ROOT = "/home/HDD/tanvir_HDD/datasets/robot/"

# Suites whose full trajectories/images are kept in memory. With spawn workers
# each worker receives a pickled copy of the full dataset and can OOM, so these
# run single-process (num_workers = 0).
LARGE_SUITES = {"libero_90", "libero_10"}
LARGE_SUITE_NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

@dataclass
class WandbConfig:
    """Wandb configuration."""
    enabled: bool = True
    entity: Optional[str] = "tanvirnwu"
    project: Optional[str] = "SUREFlow_demo70_FiLM"
    mode: Optional[str] = "LO_E400_B256_TS100k"
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

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
    total_steps: int = 100000
    phase_ratio: str = "(0.02, 0.08, 0.9)"
    lr: float = 1e-4


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

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
    action_seq_len: int = ACTION_SEQ_LEN
    embed_pdrob: int = 0
    embed_dim: int = LATENT_DIM
    device: str = DEVICE
    linear_output: bool = True
    use_ada_conditioning: bool = False
    use_sigma_film: bool = True
    use_action_decoder: bool = True
    action_decoder_heads: int = 4
    action_decoder_mlp_ratio: float = 2.0
    action_decoder_dropout: float = 0.0
    decoder_use_action_tokens_as_queries: bool = True
    encoder: MambaEncoderConfig = field(default_factory=MambaEncoderConfig)


@dataclass
class ActionFlowMatchingConfig:
    """Flow-matching model configuration."""
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


@dataclass
class ModelConfig:
    """Model configuration."""
    _target_: str = "SUREFlow.SUREFlow"
    if_film_condition: bool = False # not used in the code
    consider_robot_states: bool = CONSIDER_ROBOT_STATES
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    use_lr_scheduler: bool = True
    perception_seq_len: int = PERCEPTION_SEQ_LEN
    action_seq_len: int = ACTION_SEQ_LEN
    cam_names: List[str] = field(default_factory=lambda: CAMERA_NAMES)
    device: str = DEVICE
    state_dim: int = STATE_DIM
    latent_dim: int = LATENT_DIM
    action_dim: int = ACTION_DIM
    sampling_steps: int = NUM_SAMPLING_STEPS
    use_uncertainty: bool = True
    lambda_u: float = 0.001
    lambda_u_warmup_epochs: int = 5
    lambda_u_target: float = 0.001
    refinement_steps: int = 3
    uncertainty_threshold: float = 0.9
    model: ActionFlowMatchingConfig = field(default_factory=ActionFlowMatchingConfig)
    obs_encoders: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    language_encoders: LanguageEncoderConfig = field(default_factory=LanguageEncoderConfig)


# ---------------------------------------------------------------------------
# Trainer (data loading, training loop, scaling, EMA)
# ---------------------------------------------------------------------------

@dataclass
class DataLoadingConfig:
    """Data loading configuration."""
    train_batch_size: int = TRAIN_BATCH_SIZE
    val_batch_size: int = VAL_BATCH_SIZE
    num_workers: int = NUM_WORKERS


@dataclass
class TrainingConfig:
    """Training process configuration."""
    epoch: int = EPOCHS
    perception_seq_len: int = PERCEPTION_SEQ_LEN
    eval_every_n_epochs: int = 5
    save_every_n_epochs: int = 100


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
class TrainerConfig:
    """Trainer configuration."""
    _target_: str = "SUREFlow.Trainer"
    device: str = DEVICE

    # Sub-configurations
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_scaling: DataScalingConfig = field(default_factory=DataScalingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)

    # Flat accessors so the factory can read trainer.<field> directly
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    _target_: str = "dataloader.libero_dataset.LiberoDataset"
    # Note: benchmark_type is not passed to LiberoDataset, it's extracted from data_directory
    benchmark_type: str = "libero_object"  # Used for path construction
    demos_per_task: int = 70
    dataset_path: str = DATA_ROOT
    max_len_data: int = 347


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Simulation configuration."""
    _target_: str = "dataloader.libero_sim.LiberoSim"
    rollouts: int = 50
    max_step_per_episode: int = 350
    benchmark_type: str = DatasetConfig.benchmark_type
    use_eye_in_hand: bool = True
    seed: int = 11
    device: str = DEVICE
    render_image: bool = False
    n_cores: int = 2
    use_multiprocessing: bool = False
    save_video: bool = False
    save_video_dir: str = "/home/HDD/tanvir_HDD/robot/SUREFlow/Evl_Video"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class MainConfig:
    """Main configuration class that contains all other configurations."""
    # Basic settings
    group: str = "SUREFlow"
    seed: int = 0
    eval_benchmark_type: Optional[str] = None

    # Sub-configurations
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    visuals: VisualsConfig = field(default_factory=VisualsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Training parameters (read directly off the top-level cfg, e.g. by run.py)
    train_batch_size: int = TRAIN_BATCH_SIZE
    device: str = DEVICE
    epoch: int = EPOCHS

    # Environment parameters
    obs_dim: int = 9
    action_dim: int = ACTION_DIM
    state_dim: int = STATE_DIM
    max_len_data: int = 260

    # Observations
    camera_names: List[str] = field(default_factory=lambda: CAMERA_NAMES)

    # Model parameters
    chunck_size: int = 10
    perception_seq_len: int = PERCEPTION_SEQ_LEN
    action_seq_len: int = ACTION_SEQ_LEN
    num_sampling_steps: int = NUM_SAMPLING_STEPS

    # Architecture parameters
    len_embd: int = LEN_EMBD
    latent_dim: int = LATENT_DIM


def create_config() -> MainConfig:
    """Create and return the main configuration."""
    return MainConfig()


def create_libero_object_config() -> MainConfig:
    """Create configuration for libero_object task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_object"
    config.dataset.dataset_path = f"{DATA_ROOT}libero_object/"
    return config


def create_libero_spatial_config() -> MainConfig:
    """Create configuration for libero_spatial task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_spatial"
    config.dataset.dataset_path = f"{DATA_ROOT}libero_spatial/"
    return config


def create_libero_goal_config() -> MainConfig:
    """Create configuration for libero_goal task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_goal"
    config.dataset.dataset_path = f"{DATA_ROOT}libero_goal/"
    return config


def create_libero_90_config() -> MainConfig:
    """Create configuration for libero_90 task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_90"
    config.dataset.dataset_path = f"{DATA_ROOT}libero_90/"
    config.max_len_data = config.dataset.max_len_data
    config.trainer.data_loading.num_workers = LARGE_SUITE_NUM_WORKERS
    return config


def create_libero_10_config() -> MainConfig:
    """Create configuration for libero_10 task suite."""
    config = MainConfig()
    config.dataset.benchmark_type = "libero_10"
    config.dataset.dataset_path = f"{DATA_ROOT}libero_10/"
    config.max_len_data = config.dataset.max_len_data
    config.trainer.data_loading.num_workers = LARGE_SUITE_NUM_WORKERS
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
    config.dataset.dataset_path = f"{DATA_ROOT}{train_suite}/"

    if train_suite in LARGE_SUITES:
        config.max_len_data = config.dataset.max_len_data
        config.trainer.data_loading.num_workers = LARGE_SUITE_NUM_WORKERS

    return config


def create_libero_pro_eval_config(train_suite: str, pro_suffix: str) -> MainConfig:
    """Create configuration for LIBERO PRO evaluation from a train suite."""
    if pro_suffix not in ALLOWED_PRO_SUFFIXES:
        raise ValueError(f"Unsupported pro_suffix '{pro_suffix}'. Expected one of: {sorted(ALLOWED_PRO_SUFFIXES)}")

    config = create_libero_train_config(train_suite)
    config.eval_benchmark_type = f"{train_suite}_{pro_suffix}"
    return config
