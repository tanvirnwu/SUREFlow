"""
Test script for smolLm2 language encoder.
"""

import torch
from smolLm2_lang_encoder import LangSmolLm2


def test_smolLm2_encoder():
    """Test the smolLm2 language encoder."""
    
    print("Testing smolLm2 Language Encoder...")
    
    # Initialize encoder
    encoder = LangSmolLm2(
        smolLm2_model_name="microsoft/DialoGPT-small",
        freeze_backbone=True,
        output_dim=512,
        max_length=256
    )
    
    print(f"Model loaded: {encoder.smolLm2_encoder.model_name}")
    print(f"Device: {encoder.device}")
    print(f"Output dimension: {encoder.smolLm2_encoder.output_dim}")
    
    # Test with different types of robot instructions
    test_texts = [
        "Pick up the red block from the table",
        "Place the object in the blue container",
        "Move the robot arm to the target position",
        "Open the gripper and grasp the object",
        "Navigate to the kitchen and find the cup"
    ]
    
    print(f"\nTesting with {len(test_texts)} sample texts:")
    for i, text in enumerate(test_texts):
        print(f"{i+1}. {text}")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    with torch.no_grad():
        embeddings = encoder(test_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected shape: [batch_size, 1, output_dim] = [{len(test_texts)}, 1, 512]")
    
    # Test individual text
    print("\nTesting single text embedding...")
    single_text = ["Move the robot forward"]
    single_embedding = encoder(single_text)
    print(f"Single embedding shape: {single_embedding.shape}")
    
    # Test similarity between similar texts
    print("\nTesting text similarity...")
    similar_texts = [
        "Pick up the red block",
        "Grasp the red block",
        "Lift the red block"
    ]
    
    similar_embeddings = encoder(similar_texts)
    
    # Compute cosine similarity
    from torch.nn.functional import cosine_similarity
    
    # Compare first two similar texts
    sim_1_2 = cosine_similarity(
        similar_embeddings[0].flatten(), 
        similar_embeddings[1].flatten(), 
        dim=0
    )
    print(f"Similarity between 'Pick up the red block' and 'Grasp the red block': {sim_1_2.item():.4f}")
    
    # Compare with different text
    different_text = ["Navigate to the kitchen"]
    different_embedding = encoder(different_text)
    
    sim_1_diff = cosine_similarity(
        similar_embeddings[0].flatten(),
        different_embedding[0].flatten(),
        dim=0
    )
    print(f"Similarity between 'Pick up the red block' and 'Navigate to the kitchen': {sim_1_diff.item():.4f}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_smolLm2_encoder()
