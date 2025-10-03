# two_tower/frozen_embeddings.py (NEW FILE)

import torch
import pandas as pd
import numpy as np
from typing import Dict

def load_frozen_item_embeddings(
    model_path: str,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Load frozen item embeddings from previous HH-Item model.
    
    Args:
        model_path: Path to previous two-tower model (.pt file)
        device: Device to load model on
        
    Returns:
        Dict mapping item_id → embedding (numpy array)
    """
    print(f"Loading frozen item embeddings from {model_path}...")
    
    # Load previous model state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract item embeddings
    # Adjust this based on your previous model structure
    if 'item_embeddings' in checkpoint:
        item_embeddings = checkpoint['item_embeddings']
    elif 'model_state_dict' in checkpoint:
        # Extract from model state dict
        item_tower_weights = {
            k: v for k, v in checkpoint['model_state_dict'].items() 
            if 'item_tower' in k or 'item_embedding' in k
        }
        # You may need to extract the actual embedding table here
        item_embeddings = checkpoint['model_state_dict']['item_embeddings.weight']
    else:
        raise KeyError("Cannot find item embeddings in checkpoint")
    
    # Convert to numpy
    if isinstance(item_embeddings, torch.Tensor):
        item_embeddings = item_embeddings.cpu().numpy()
    
    # Map to item IDs
    # You need item_id → index mapping from previous model
    item_id_map = checkpoint.get('item_id_map', None)
    
    if item_id_map is None:
        print("⚠️  No item_id_map found, using indices as keys")
        item_embedding_dict = {
            f"item_{i}": item_embeddings[i]
            for i in range(len(item_embeddings))
        }
    else:
        item_embedding_dict = {
            item_id: item_embeddings[idx]
            for item_id, idx in item_id_map.items()
        }
    
    print(f"✅ Loaded {len(item_embedding_dict)} frozen item embeddings")
    return item_embedding_dict