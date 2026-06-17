import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Callable
import torchvision.models as models
import einops


def set_parameter_requires_grad(model, requires_grad):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad
            

def freeze_params(model):
    set_parameter_requires_grad(model, requires_grad=False)
    

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


class BesoResNetEncoder(nn.Module):
    """BesoResNetEncoder that matches the checkpoint structure"""

    def __init__(
        self,
        latent_dim: int = 256,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        use_mlp: bool = True,
        device: str = 'cuda:0'
    ):
        super(BesoResNetEncoder, self).__init__()
        self.latent_dim = latent_dim
        backbone = models.resnet18(pretrained=pretrained)
        n_inputs = backbone.fc.in_features
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            freeze_params(self.backbone)
        
        # substitute norm for ema diffusion stuff
        replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.fc_layers = nn.Sequential(nn.Linear(n_inputs, latent_dim))

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        batch_size = len(x)
        t_steps = 1
        time_series = False
        
        if len(x.shape) == 5:
            t_steps = x.shape[1]
            x = einops.rearrange(x, 'b t n x_dim y_dim -> (b t) n x_dim y_dim')
            time_series = True
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.conv_forward(x)
        if self.use_mlp:
            x = self.fc_layers(x)
        
        if time_series:
            x = einops.rearrange(x, '(b t) d -> b t d', b=batch_size, t=t_steps, d=self.latent_dim)        
        return x


class MultiImageResNetEncoder(nn.Module):
    """Multi-image ResNet encoder that matches the checkpoint structure"""
    
    def __init__(self, camera_names: List[str], latent_dim: int = 256, input_channels: int = 3):
        super(MultiImageResNetEncoder, self).__init__()
        
        self.camera_names = camera_names
        self.latent_dim = latent_dim
        
        # Create a model map for each camera using BesoResNetEncoder
        self.key_model_map = nn.ModuleDict()
        for camera_name in camera_names:
            self.key_model_map[f"{camera_name}_image"] = BesoResNetEncoder(
                latent_dim=latent_dim,
                pretrained=False,
                freeze_backbone=False,
                use_mlp=True
            )
        
        # Dummy variable for compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the multi-image encoder
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
                    Keys should be like "agentview_image", "eye_in_hand_image"
        
        Returns:
            torch.Tensor: Concatenated features from all cameras [batch_size, num_cameras, latent_dim]
        """
        features = []
        
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            if image_key in obs_dict:
                # Get the image tensor [batch_size, channels, height, width]
                image = obs_dict[image_key]
                # Encode the image
                encoded = self.key_model_map[image_key](image)
                features.append(encoded)
            else:
                raise KeyError(f"Expected key '{image_key}' not found in obs_dict")
        
        # Stack features from all cameras [batch_size, num_cameras, latent_dim]
        features = torch.stack(features, dim=1)
        
        return features


if __name__ == "__main__":
    # Test the encoder
    camera_names = ["agentview", "eye_in_hand"]
    encoder = MultiImageResNetEncoder(camera_names, latent_dim=256)
    
    # Create dummy observation dictionary
    batch_size = 2
    obs_dict = {
        "agentview_image": torch.randn(batch_size, 3, 128, 128),
        "eye_in_hand_image": torch.randn(batch_size, 3, 128, 128)
    }
    
    # Forward pass
    output = encoder(obs_dict)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, num_cameras, latent_dim]
