# test_two_tower_local.py

import numpy as np
import pandas as pd
import torch

from two_tower.model import TwoTowerModel
from two_tower.dataset import HouseholdAttributeDataset

# Create tiny test data
attr_embeddings = {
    'brand_a': np.random.randn(32),
    'brand_b': np.random.randn(32),
    'brand_c': np.random.randn(32),
}

hh_attr_history = pd.DataFrame({
    'household_id': ['hh1', 'hh1', 'hh2', 'hh2', 'hh3'],
    'attr_id': ['brand_a', 'brand_b', 'brand_b', 'brand_c', 'brand_a'],
    'hist_score': [10.0, 5.0, 8.0, 3.0, 12.0],
    'last_seen_days': [7, 14, 5, 30, 3]
})

# Create dataset
dataset = HouseholdAttributeDataset(
    hh_attr_history, 
    attr_embeddings, 
    negatives_per_pos=2
)

print(f"✅ Dataset created with {len(dataset)} samples")

# Test model
model = TwoTowerModel(attr_embed_dim=32, output_dim=16)

# Get one batch
hh_profile, pos_attr, neg_attrs = dataset[0]
print(f"✅ Sample shapes:")
print(f"   HH profile: {hh_profile.shape}")
print(f"   Positive attr: {pos_attr.shape}")
print(f"   Negative attrs: {neg_attrs.shape}")

# Forward pass
hh_profile = hh_profile.unsqueeze(0)
pos_attr = pos_attr.unsqueeze(0)
neg_attrs = neg_attrs.unsqueeze(0)

logits = model(hh_profile, pos_attr, neg_attrs)
print(f"✅ Logits shape: {logits.shape}")
print(f"✅ Logits: {logits}")

print("\n✅ Two-Tower model working correctly!")