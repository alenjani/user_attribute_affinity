# two_tower/score.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from .model import TwoTowerModel
from .dataset import HouseholdAttributeDataset

def score_household_attribute_affinity(
    model_path: str,
    hh_attr_history_path: str,
    attr_embeddings_path: str,
    provider: str,
    model_version: str,
    output_path: str,
    top_k: int = 50,
    batch_size: int = 1024
):
    """
    Generate household-attribute affinity scores for all households.
    
    Args:
        model_path: Path to trained two-tower model (.pt file)
        hh_attr_history_path: Path to household history
        attr_embeddings_path: Path to attribute embeddings
        provider: Provider name
        model_version: Model version
        output_path: Where to save results
        top_k: Keep top-K attributes per household
        batch_size: Batch size for inference
    """
    
    print("="*60)
    print("Generating Household-Attribute Affinity Scores")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    hh_attr_history = pd.read_parquet(hh_attr_history_path)
    
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
    
    print(f"   {len(hh_attr_history['household_id'].unique())} households")
    print(f"   {len(attr_embeddings)} attributes")
    
    # Load model
    print("\n🏗️  Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    attr_embed_dim = attr_embed_matrix.shape[1]
    model = TwoTowerModel(attr_embed_dim=attr_embed_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded from {model_path}")
    print(f"   Device: {device}")
    
    # Build household profiles
    print("\n🔨 Building household profiles...")
    household_profiles = {}
    
    for hh_id, group in hh_attr_history.groupby('household_id'):
        attrs = []
        weights = []
        
        for _, row in group.iterrows():
            attr_id = row['attr_id']
            if attr_id in attr_embeddings:
                attrs.append(attr_id)
                weights.append(row['hist_score'])
        
        if len(attrs) > 0:
            # Weighted average of attribute embeddings
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            embeddings = np.stack([attr_embeddings[attr] for attr in attrs])
            hh_profile = np.average(embeddings, weights=weights, axis=0)
            
            household_profiles[hh_id] = hh_profile
    
    print(f"   Built {len(household_profiles)} household profiles")
    
    # Compute affinity scores
    print("\n🎯 Computing affinity scores...")
    
    all_hh_ids = list(household_profiles.keys())
    attr_embed_tensor = torch.FloatTensor(attr_embed_matrix).to(device)
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(all_hh_ids), batch_size)):
        batch_hh_ids = all_hh_ids[i:i+batch_size]
        batch_profiles = np.stack([household_profiles[hh_id] for hh_id in batch_hh_ids])
        batch_profiles = torch.FloatTensor(batch_profiles).to(device)
        
        with torch.no_grad():
            # Compute affinity: [batch_hh, n_attrs]
            affinity_scores = model.compute_affinity(batch_profiles, attr_embed_tensor)
            affinity_scores = affinity_scores.cpu().numpy()
        
        # For each household, keep top-K attributes
        for j, hh_id in enumerate(batch_hh_ids):
            scores = affinity_scores[j]
            
            # Get top-K indices
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            
            for rank, idx in enumerate(top_k_indices):
                results.append({
                    'household_id': hh_id,
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
    print(f"   Households scored: {results_df['household_id'].nunique()}")
    print(f"   Unique attributes: {results_df['attr_id'].nunique()}")
    print(f"   Avg attributes per household: {len(results_df) / results_df['household_id'].nunique():.1f}")
    
    return results_df