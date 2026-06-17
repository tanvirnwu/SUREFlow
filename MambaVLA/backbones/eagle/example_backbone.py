import torch
from transformers.feature_extraction_utils import BatchFeature
from MambaVLA.backbones.eagle.eagle_backbone import EagleBackbone, DEFAULT_EAGLE_MODEL_NAME
from MambaVLA.backbones.eagle.eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues, build_transform

def create_random_inputs(batch_size=256, seq_length=64, hidden_size=1536, device="cuda"):
    """
    Create random inputs for the EagleBackbone.

    Args:
        batch_size (int): Number of samples in the batch
        seq_length (int): Length of the sequence (must be >= number of image tokens)
        hidden_size (int): Size of the hidden dimension
        device (str): Device to create tensors on

    Returns:
        BatchFeature: Random inputs for the backbone
    """
    # Create random pixel values (simulating 128x128 images)
    # Shape: [batch_size, channels, height, width]
    pixel_values_128 = torch.randn(batch_size, 3, 128, 128, device=device, dtype=torch.float16)
    
    # Resize from 128x128 to 224x224 using the built-in transform
    transform = build_transform(input_size=224, norm_type="siglip")
    
    # Convert tensor to PIL images, apply transform, then back to tensor
    pixel_values_224 = []
    for i in range(batch_size):
        # Convert tensor to PIL image (assuming values are in [0, 1] range)
        img_tensor = pixel_values_128[i].cpu()
        if img_tensor.min() < 0:  # If values are in [-1, 1], convert to [0, 1]
            img_tensor = (img_tensor + 1) / 2
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PIL image
        from PIL import Image
        import torchvision.transforms.functional as TF
        img_pil = TF.to_pil_image(img_tensor)
        
        # Apply transform (resize to 224x224 and normalize)
        img_transformed = transform(img_pil)
        pixel_values_224.append(img_transformed)
    
    # Stack all transformed images
    pixel_values = torch.stack(pixel_values_224).to(device)

    # Initialize processor to get the image context token ID
    processor = EagleProcessor(
        model_path=DEFAULT_EAGLE_MODEL_NAME,
        max_input_tiles=1,
        model_spec=ModelSpecificValues(
            template="qwen2-chat",
            num_image_token=64
        )
    )
    img_context_token_id = processor.get_img_context_token()

    # Create input IDs with image context tokens
    # Shape: [batch_size, seq_length]
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)

    # Number of image tokens
    num_img_tokens = 64
    if seq_length < num_img_tokens:
        raise ValueError("seq_length must be at least as large as the number of image tokens (64).")

    # Create a template for each sample that includes the image tokens
    for i in range(batch_size):
        # Create a sequence that starts with image tokens
        sequence = [img_context_token_id] * num_img_tokens
        # Fill the rest with random tokens
        sequence.extend(input_ids[i, num_img_tokens:].tolist())
        # Truncate if sequence is longer than seq_length
        sequence = sequence[:seq_length]
        input_ids[i] = torch.tensor(sequence, device=device)

    # Create attention mask (1 for real tokens, 0 for padding)
    # Shape: [batch_size, seq_length]
    attention_mask = torch.ones(batch_size, seq_length, device=device)

    return BatchFeature(data={
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

def main():
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. Flash Attention requires CUDA.")
        return

    print(f"Using device: {device}")

    # Create the backbone model
    backbone = EagleBackbone(
        select_layer=12,  # Number of transformer layers to use
        model_name=DEFAULT_EAGLE_MODEL_NAME,  # Use the default model path
        tune_llm=False,
        tune_visual=False,
        reproject_vision=False,
        processor_cfg={
            "model_path": DEFAULT_EAGLE_MODEL_NAME,  # Use the same path for processor
            "max_input_tiles": 1,
            "model_spec": {
                "template": "qwen2-chat",
                "num_image_token": 64
            }
        }
    )

    # Move model to GPU
    backbone = backbone.to(device)

    # Create random inputs
    inputs = create_random_inputs(device=device)

    # Forward pass
    with torch.autocast(device_type=device, dtype=torch.float16):
        outputs = backbone(inputs)

    # Print shapes
    print("Input shapes:")
    print(f"pixel_values: {inputs['pixel_values'].shape}")
    print(f"input_ids: {inputs['input_ids'].shape}")
    print(f"attention_mask: {inputs['attention_mask'].shape}")

    print("\nOutput shapes:")
    print(f"backbone_features: {outputs['backbone_features'].shape}")
    print(f"backbone_attention_mask: {outputs['backbone_attention_mask'].shape}")

if __name__ == "__main__":
    main()
