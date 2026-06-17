"""
Eagle-based language encoder for replacing CLIP in MambaVLA.
This encoder processes language instructions using the Eagle backbone.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers.feature_extraction_utils import BatchFeature
from .eagle_backbone import EagleBackbone, DEFAULT_EAGLE_MODEL_NAME
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues


class EagleLangEncoder(nn.Module):
    """
    Eagle-based language encoder that processes text instructions.
    Replaces CLIP-based language encoder with Eagle backbone for better language understanding.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = True,  # Allow tuning language model for language encoding
        tune_visual: bool = False,  # Don't need visual features for language only
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: Optional[dict] = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        use_local_eagle_hg_model: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.freeze_backbone = freeze_backbone
        
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
        
        # Initialize processor for text preprocessing
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
        
        # Get image context token ID (needed for Eagle input format)
        self.img_context_token_id = self.processor.get_img_context_token()
        
        # Projection layer to match expected language embedding dimension
        eagle_output_dim = 1536  # Eagle's default output dimension
        if projector_dim != -1:
            eagle_output_dim = projector_dim
            
        self.projection = nn.Linear(eagle_output_dim, 512)  # Match CLIP's output dimension
        
        # Dummy variable for device compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.eagle_backbone.parameters():
                param.requires_grad = False
    
    def _preprocess_text(self, text_list: List[str]) -> tuple:
        """
        Preprocess text inputs for Eagle backbone.
        
        Args:
            text_list: List of text strings to process
            
        Returns:
            tuple: (input_ids, attention_mask) for Eagle backbone
        """
        # Process text using Eagle processor
        processed = self.processor.process_text(text_list)
        
        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]
        
        return input_ids, attention_mask
    
    def _create_eagle_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BatchFeature:
        """
        Create inputs for Eagle backbone with dummy images.
        
        Args:
            input_ids: Tokenized text input [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            BatchFeature: Inputs for Eagle backbone
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create dummy images (Eagle requires both text and images)
        # We'll use a small dummy image since we only care about text processing
        dummy_images = torch.zeros(batch_size, 3, 224, 224, device=device, dtype=torch.float16)
        
        return BatchFeature(data={
            "pixel_values": dummy_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    
    def forward(self, text_list: List[str]) -> torch.Tensor:
        """
        Forward pass through the Eagle language encoder.
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            torch.Tensor: Encoded language features [batch_size, 1, lang_emb_dim]
        """
        # Preprocess text
        input_ids, attention_mask = self._preprocess_text(text_list)
        
        # Create Eagle inputs
        eagle_inputs = self._create_eagle_inputs(input_ids, attention_mask)
        
        # Forward through Eagle backbone
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            eagle_outputs = self.eagle_backbone(eagle_inputs)
        
        # Extract features and project to language embedding dimension
        backbone_features = eagle_outputs["backbone_features"]  # [batch_size, seq_length, hidden_dim]
        
        # Use the last token (typically contains the most relevant information)
        # or pool over all tokens
        if backbone_features.shape[1] > 1:
            # Pool over sequence length (mean pooling)
            pooled_features = torch.mean(backbone_features, dim=1)  # [batch_size, hidden_dim]
        else:
            pooled_features = backbone_features[:, 0, :]  # [batch_size, hidden_dim]
        
        # Project to language embedding dimension
        lang_features = self.projection(pooled_features)  # [batch_size, lang_emb_dim]
        
        # Add sequence dimension to match expected output format
        lang_features = torch.unsqueeze(lang_features, 1)  # [batch_size, 1, lang_emb_dim]
        
        return lang_features
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class LangEagle(nn.Module):
    """
    Eagle-based language encoder that maintains compatibility with existing code.
    This is a wrapper around EagleLangEncoder to match the expected interface.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",  # Keep same interface as CLIP
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Map CLIP model names to Eagle configurations if needed
        eagle_model_name = DEFAULT_EAGLE_MODEL_NAME
        
        # Initialize Eagle language encoder
        self.eagle_encoder = EagleLangEncoder(
            model_name=eagle_model_name,
            tune_llm=not freeze_backbone,
            tune_visual=False,
            freeze_backbone=freeze_backbone,
            **kwargs
        )
        
        # Dummy variable for device compatibility
        self._dummy_variable = nn.Parameter(torch.zeros(0))
    
    def forward(self, text_list: List[str]) -> torch.Tensor:
        """
        Forward pass through the Eagle language encoder.
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            torch.Tensor: Encoded language features [batch_size, 1, lang_emb_dim]
        """
        return self.eagle_encoder(text_list)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
