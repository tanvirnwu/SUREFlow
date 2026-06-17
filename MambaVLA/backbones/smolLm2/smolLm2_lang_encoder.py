"""
smolLm2-based language encoder for generating text embeddings.
This encoder processes language instructions using the smolLm2 model.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer


class SmolLm2LangEncoder(nn.Module):
    """
    smolLm2-based language encoder that processes text instructions.
    Uses a small language model for efficient text embedding generation.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",  # Default small model, can be changed
        freeze_backbone: bool = True,
        max_length: int = 512,
        output_dim: int = 512,
        use_pooling: bool = True,
        pooling_strategy: str = "mean",  # "mean", "cls", "last"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        self.output_dim = output_dim
        self.use_pooling = use_pooling
        self.pooling_strategy = pooling_strategy
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model's hidden dimension
        self.hidden_dim = self.model.config.hidden_size
        
        # Projection layer to match expected output dimension
        if self.hidden_dim != output_dim:
            self.projection = nn.Linear(self.hidden_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _tokenize_texts(self, text_list: List[str]) -> dict:
        """
        Tokenize a list of texts.
        
        Args:
            text_list: List of text strings to tokenize
            
        Returns:
            dict: Tokenized inputs with input_ids and attention_mask
        """
        return self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def _pool_embeddings(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool the hidden states to get sentence-level embeddings.
        
        Args:
            last_hidden_states: [batch_size, seq_length, hidden_dim]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return last_hidden_states[:, 0, :]
        elif self.pooling_strategy == "last":
            # Use last non-padded token
            sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 because of 0-indexing
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size), sequence_lengths]
        else:  # mean pooling
            # Mean pooling over non-padded tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
    
    def forward(self, text_list: List[str]) -> torch.Tensor:
        """
        Forward pass through the smolLm2 language encoder.
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            torch.Tensor: Encoded language features [batch_size, 1, output_dim]
        """
        # Tokenize texts
        tokenized = self._tokenize_texts(text_list)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Forward through model
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Get hidden states
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, hidden_dim]
        
        # Pool embeddings if requested
        if self.use_pooling:
            pooled_embeddings = self._pool_embeddings(last_hidden_states, attention_mask)
        else:
            # Use [CLS] token by default
            pooled_embeddings = last_hidden_states[:, 0, :]
        
        # Project to output dimension
        projected_embeddings = self.projection(pooled_embeddings)  # [batch_size, output_dim]
        
        # Add sequence dimension to match expected output format
        projected_embeddings = torch.unsqueeze(projected_embeddings, 1)  # [batch_size, 1, output_dim]
        
        return projected_embeddings
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class LangSmolLm2(nn.Module):
    """
    smolLm2-based language encoder that maintains compatibility with existing code.
    This is a wrapper around SmolLm2LangEncoder to match the expected interface.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",  # Keep same interface as CLIP for compatibility
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Map CLIP model names to smolLm2 configurations if needed
        # You can customize this mapping based on your needs
        smolLm2_model_name = "microsoft/DialoGPT-small"  # Default small model
        
        # Override with custom model if provided
        if "smolLm2_model_name" in kwargs:
            smolLm2_model_name = kwargs.pop("smolLm2_model_name")
        
        # Initialize smolLm2 language encoder
        self.smolLm2_encoder = SmolLm2LangEncoder(
            model_name=smolLm2_model_name,
            freeze_backbone=freeze_backbone,
            **kwargs
        )
    
    def forward(self, text_list: List[str]) -> torch.Tensor:
        """
        Forward pass through the smolLm2 language encoder.
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            torch.Tensor: Encoded language features [batch_size, 1, output_dim]
        """
        return self.smolLm2_encoder(text_list)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Example usage and testing
if __name__ == "__main__":
    # Test the encoder
    encoder = LangSmolLm2(
        smolLm2_model_name="microsoft/DialoGPT-small",
        freeze_backbone=True,
        output_dim=512
    )
    
    # Test with sample texts
    sample_texts = [
        "Pick up the red block",
        "Place the object in the container",
        "Move the robot arm to the target position"
    ]
    
    # Generate embeddings
    embeddings = encoder(sample_texts)
    
    print(f"Input texts: {sample_texts}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Device: {encoder.device}")
    print(f"Model name: {encoder.smolLm2_encoder.model_name}")
