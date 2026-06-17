import torch
import torch.nn as nn
from transformers import AutoConfig, SiglipImageProcessor, SiglipVisionModel


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower: str = "google/siglip-so400m-patch14-384", args=None, delay_load: bool = False, **kwargs):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # allow hydra to pass extra, unused kwargs like latent_dim, pretrained, etc.
        # configure select feature from args if provided
        self.select_feature = 'patch'
        if args is not None:
            self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # output projection settings (allow hydra kwargs like latent_dim, use_mlp)
        self.output_dim = kwargs.get('latent_dim', None)
        self.use_mlp = kwargs.get('use_mlp', False)
        self.project = None  # initialized after loading when hidden size known

        if not delay_load:
            self.load_model()
        elif (args is not None) and getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.eval()

        # build projection if requested
        hidden = self.vision_tower.config.hidden_size
        if self.output_dim is None:
            self.output_dim = hidden
        if self.use_mlp and self.output_dim != hidden:
            self.project = nn.Sequential(
                nn.Linear(hidden, self.output_dim),
                nn.GELU(),
                nn.Linear(self.output_dim, self.output_dim),
            )
        elif self.output_dim != hidden:
            self.project = nn.Linear(hidden, self.output_dim)
        else:
            self.project = nn.Identity()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            image_features = image_forward_outs.last_hidden_state  # (B, 729, 1536)
            # pool patches to a single vector per image
            image_features = image_features.mean(dim=1)  # (B, 1536)
        elif self.select_feature == 'cls_patch':
            image_features = image_forward_outs.pooler_output  # (B, 1536) or (B, 1, 1536)
            if image_features.dim() == 3 and image_features.size(1) == 1:
                image_features = image_features[:, 0, :]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_feature = self.project(image_feature)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = self.project(image_features)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2