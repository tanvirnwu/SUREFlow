"""
Eagle-based observation encoder for replacing ResNet in MambaVLA.
This encoder processes multi-camera observations using the Eagle backbone.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers.feature_extraction_utils import BatchFeature
from .eagle_backbone import EagleBackbone, DEFAULT_EAGLE_MODEL_NAME
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues, build_transform
import torchvision.transforms.functional as TF
from PIL import Image


class EagleObsEncoder(nn.Module):
    """
    Eagle-based observation encoder that processes multi-camera images.
    Replaces ResNet-based encoders with Eagle backbone for better vision-language understanding.
    """
    
    def __init__(
        self,
        camera_names: List[str],
        latent_dim: int = 256,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = False,
        tune_visual: bool = True,  # Allow tuning visual features for observation encoding
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: Optional[dict] = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        use_local_eagle_hg_model: bool = True,
        input_size: int = 224,
        norm_type: str = "siglip",
    ):
        super().__init__()
        
        self.camera_names = camera_names
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.norm_type = norm_type
        
        # Initialize Eagle backbone
        self.eagle_backbone = EagleBackbone(
            select_layer=12,
            model_name=model_name,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            reproject_vision=reproject_vision,
            scale_image_resolution=scale_image_resolution,
            processor_cfg=processor_cfg,
            projector_dim=projector_dim,
            allow_reshape_visual=allow_reshape_visual,
            use_local_eagle_hg_model=use_local_eagle_hg_model,
        )
        
        # Initialize processor for image preprocessing
        if processor_cfg is None:
            processor_cfg = {
                "model_path": model_name,
                "max_input_tiles": 1,
                "model_spec": {
                    "template": "qwen2-chat",
                    "num_image_token": 64
                }
            }
        
        self.processor = EagleProcessor(
            model_path=processor_cfg["model_path"],
            max_input_tiles=processor_cfg["max_input_tiles"],
            model_spec=ModelSpecificValues(**processor_cfg["model_spec"]),
            use_local_eagle_hg_model=use_local_eagle_hg_model,
        )
        
        # Get image context token ID
        self.img_context_token_id = self.processor.get_img_context_token()
        
        # Build transform for image preprocessing
        self.transform = build_transform(input_size=input_size, norm_type=norm_type)
        
        # Projection layer to match expected latent dimension
        eagle_output_dim = 1536  # Eagle's default output dimension
        if projector_dim != -1:
            eagle_output_dim = projector_dim
            
        self.projection = nn.Linear(eagle_output_dim, latent_dim)
        
        # Dummy variable for device compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
    
    def _preprocess_images(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess images from multiple cameras into the format expected by Eagle.
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
            
        Returns:
            torch.Tensor: Preprocessed images [batch_size, channels, height, width]
        """
        batch_size = None
        processed_images = []
        
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            if image_key not in obs_dict:
                raise KeyError(f"Expected key '{image_key}' not found in obs_dict")
            
            images = obs_dict[image_key]  # [batch_size, channels, height, width]
            
            if batch_size is None:
                batch_size = images.shape[0]
            else:
                assert batch_size == images.shape[0], "Batch size mismatch across cameras"
            
            # Process each image in the batch
            camera_images = []
            for i in range(batch_size):
                img_tensor = images[i].cpu()
                
                # Ensure values are in [0, 1] range
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # Convert to PIL image
                img_pil = TF.to_pil_image(img_tensor)
                
                # Apply transform (resize to input_size and normalize)
                img_transformed = self.transform(img_pil)
                camera_images.append(img_transformed)
            
            # Stack processed images for this camera
            camera_images = torch.stack(camera_images).to(images.device)
            processed_images.append(camera_images)
        
        # For now, we'll use the first camera's images
        # TODO: Consider how to handle multiple cameras with Eagle
        return processed_images[0]
    
    def _create_eagle_inputs(self, images: torch.Tensor, batch_size: int) -> BatchFeature:
        """
        Create inputs for Eagle backbone.
        
        Args:
            images: Preprocessed images [batch_size, channels, height, width]
            batch_size: Batch size
            
        Returns:
            BatchFeature: Inputs for Eagle backbone
        """
        # Create input IDs with image context tokens
        seq_length = 64  # Number of image tokens
        input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long, device=images.device)
        
        # Fill with image context tokens
        input_ids[:] = self.img_context_token_id
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_length, device=images.device)
        
        return BatchFeature(data={
            "pixel_values": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    
    def forward(self, obs_dict: Dict[str, torch.Tensor], lang_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Eagle observation encoder.
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
            lang_cond: Optional language conditioning (not used in this implementation)
            
        Returns:
            torch.Tensor: Encoded features [batch_size, num_cameras, latent_dim]
        """
        batch_size = None
        features = []
        
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            if image_key not in obs_dict:
                raise KeyError(f"Expected key '{image_key}' not found in obs_dict")
            
            images = obs_dict[image_key]  # [batch_size, channels, height, width]
            
            if batch_size is None:
                batch_size = images.shape[0]
            else:
                assert batch_size == images.shape[0], "Batch size mismatch across cameras"
            
            # Process images for this camera
            processed_images = self._preprocess_images({image_key: images})
            
            # Create Eagle inputs
            eagle_inputs = self._create_eagle_inputs(processed_images, batch_size)
            
            # Forward through Eagle backbone
            with torch.no_grad() if not self.training else torch.enable_grad():
                eagle_outputs = self.eagle_backbone(eagle_inputs)
            
            # Extract features and project to latent dimension
            backbone_features = eagle_outputs["backbone_features"]  # [batch_size, seq_length, hidden_dim]
            
            # Take the first token (image token) and project to latent dimension
            image_features = backbone_features[:, 0, :]  # [batch_size, hidden_dim]
            projected_features = self.projection(image_features)  # [batch_size, latent_dim]
            
            features.append(projected_features)
        
        # Stack features from all cameras [batch_size, num_cameras, latent_dim]
        features = torch.stack(features, dim=1)
        
        return features
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class MultiImageEagleObsEncoder(nn.Module):
    """
    Multi-image Eagle observation encoder that maintains compatibility with existing code.
    This is a wrapper around EagleObsEncoder to match the expected interface.
    """
    
    def __init__(
        self,
        shape_meta: dict,
        latent_dim: int = 256,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = False,
        tune_visual: bool = True,
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: Optional[dict] = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        use_local_eagle_hg_model: bool = True,
        input_size: int = 224,
        norm_type: str = "siglip",
    ):
        super().__init__()
        
        # Extract camera names from shape_meta
        obs_shape_meta = shape_meta['obs']
        camera_names = []
        for key, attr in obs_shape_meta.items():
            if attr.get('type') == 'rgb':
                # Extract camera name from key (e.g., "agentview_image" -> "agentview")
                camera_name = key.replace('_image', '')
                camera_names.append(camera_name)
        
        self.camera_names = camera_names
        self.shape_meta = shape_meta
        
        # Initialize Eagle encoder
        self.eagle_encoder = EagleObsEncoder(
            camera_names=camera_names,
            latent_dim=latent_dim,
            model_name=model_name,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            reproject_vision=reproject_vision,
            scale_image_resolution=scale_image_resolution,
            processor_cfg=processor_cfg,
            projector_dim=projector_dim,
            allow_reshape_visual=allow_reshape_visual,
            use_local_eagle_hg_model=use_local_eagle_hg_model,
            input_size=input_size,
            norm_type=norm_type,
        )
        
        # Dummy variable for device compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
    
    def forward(self, obs_dict: Dict[str, torch.Tensor], lang_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the multi-image Eagle encoder.
        
        Args:
            obs_dict: Dictionary containing image tensors for each camera
            lang_cond: Optional language conditioning
            
        Returns:
            torch.Tensor: Encoded features [batch_size, num_cameras, latent_dim]
        """
        return self.eagle_encoder(obs_dict, lang_cond)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    @torch.no_grad()
    def output_shape(self):
        """Get the output shape of the encoder."""
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=self.dtype,
                device=self.device
            )
            example_obs_dict[key] = this_obs
        
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

