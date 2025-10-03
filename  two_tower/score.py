# two_tower/score.py (COMPLETELY REWRITTEN)

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from .model import ItemAttributeAlignmentModel  # CHANGED
from .frozen_embeddings import load_frozen_item_embeddings  # NEW

def score_item_attribute_affinity(  # RENAMED function
    model_path: str,
    frozen_item_embeddings_path: str,  # CHANGED from hh_attr_history_path
    attr_embeddings_path: str,
    provider: str,
    model_version: str,
    output_path: str,
    top_k: int = 50,
    batch_size: int = 1024
):
    """
    Generate item-attribute affinity scores for all items.
    
    Args:
        model_path: Path to trained alignment model (.pt file)
        frozen_item_embeddings_path: Path to frozen item embeddings from Phase 1
        attr_embeddings_path: Path to attribute embeddings
        provider: Provider name
        model_version: Model version
        output_path: Where to save results
        top_k: Keep top-K attributes per item
        batch_size: Batch size for inference
    """
    
    print("="*60)
    print("Generating Item-Attribute Affinity Scores")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    
    # Load frozen item embeddings (CHANGED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frozen_item_embeddings = load_frozen_item_embeddings(
        frozen_item_embeddings_path, 
        device=device
    )
    
    # Load attribute embeddings (SAME)
    attr_emb_df = pd.read_parquet(attr_embeddings_path)
    attr_emb_df = attr_emb_df[
        (attr_emb_df['provider'] == provider) & 
        (attr_emb_df['model_version'] == model_version)
    ]
    
    attr_embeddings = {
        row['attr_id']: np.array(row['vector'])
        for _, row in attr_emb_df.iterrows()
    }
    
    attr_ids = list(attr_embeddings.keys())
    attr_embed_matrix = np.stack([attr_embeddings[aid] for aid in attr_ids])
    
    print(f"   {len(frozen_item_embeddings)} items")
    print(f"   {len(attr_embeddings)} attributes")
    
    # Load model
    print("\n🏗️  Loading model...")
    
    attr_embed_dim = attr_embed_matrix.shape[1]
    item_embed_dim = len(next(iter(frozen_item_embeddings.values())))
    
    model = ItemAttributeAlignmentModel(  # CHANGED model class
        item_embed_dim=item_embed_dim,
        attr_embed_dim=attr_embed_dim,
        output_dim=64
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded from {model_path}")
    print(f"   Device: {device}")
    
    # NO LONGER BUILD HOUSEHOLD PROFILES ❌
    # Items already have embeddings - just use them directly
    
    # Compute affinity scores
    print("\n🎯 Computing affinity scores...")
    
    all_item_ids = list(frozen_item_embeddings.keys())
    attr_embed_tensor = torch.FloatTensor(attr_embed_matrix).to(device)
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(all_item_ids), batch_size)):
        batch_item_ids = all_item_ids[i:i+batch_size]
        
        # Get frozen item embeddings for this batch (CHANGED)
        batch_item_embeddings = np.stack([
            frozen_item_embeddings[item_id] for item_id in batch_item_ids
        ])
        batch_item_embeddings = torch.FloatTensor(batch_item_embeddings).to(device)
        
        with torch.no_grad():
            # Compute affinity: [batch_items, n_attrs]
            affinity_scores = model.compute_affinity(
                batch_item_embeddings,  # CHANGED: items instead of households
                attr_embed_tensor
            )
            affinity_scores = affinity_scores.cpu().numpy()
        
        # For each item, keep top-K attributes (CHANGED)
        for j, item_id in enumerate(batch_item_ids):
            scores = affinity_scores[j]
            
            # Get top-K indices
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            
            for rank, idx in enumerate(top_k_indices):
                results.append({
                    'item_id': item_id,  # CHANGED from household_id
                    'attr_id': attr_ids[idx],
                    'affinity_score': scores[idx],
                    'rank': rank + 1
                })
    
    # Save results
    print(f"\n💾 Saving results to {output_path}...")
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_path, index=False)
    
    print(f"✅ Saved {len(results_df)} affinity scores")
    print(f"   Average score: {results_df['affinity_score'].mean():.4f}")
    print(f"   Score range: [{results_df['affinity_score'].min():.4f}, {results_df['affinity_score'].max():.4f}]")
    
    # Summary stats
    print("\n📊 Summary Statistics:")
    print(f"   Items scored: {results_df['item_id'].nunique()}")  # CHANGED
    print(f"   Unique attributes: {results_df['attr_id'].nunique()}")
    print(f"   Avg attributes per item: {len(results_df) / results_df['item_id'].nunique():.1f}")  # CHANGED
    
    return results_df