# two_tower/dataset.py (MODIFIED)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
from collections import defaultdict

class ItemAttributeDataset(Dataset):
    """
    Dataset for Item-Attribute alignment training.
    
    Each sample:
    - Item embedding (frozen from previous model)
    - Positive attribute (item has this)
    - Negative attributes (item doesn't have these)
    """
    
    def __init__(
        self,
        map_item_attr: pd.DataFrame,           # Item→attribute mappings
        frozen_item_embeddings: Dict[str, np.ndarray],  # Frozen from Phase 1
        attr_embeddings: Dict[str, np.ndarray],         # From CoocAE/GCL
        negatives_per_pos: int = 5,
        negative_sampling: str = 'random'
    ):
        """
        Args:
            map_item_attr: DataFrame with (item_id, attr_id, p_attr)
            frozen_item_embeddings: {item_id: frozen_embedding}
            attr_embeddings: {attr_id: embedding from provider}
            negatives_per_pos: Number of negative samples per positive
            negative_sampling: 'random' or 'popularity'
        """
        self.map_item_attr = map_item_attr
        self.frozen_item_embeddings = frozen_item_embeddings
        self.attr_embeddings = attr_embeddings
        self.negatives_per_pos = negatives_per_pos
        self.negative_sampling = negative_sampling
        
        # Build item profiles
        self._build_item_profiles()
        
        # Build positive samples
        self._build_positive_samples()
        
        # Compute attribute popularity
        self._compute_attribute_popularity()
        
        print(f"✅ Dataset created:")
        print(f"   {len(self.item_profiles)} items")
        print(f"   {len(self.positive_samples)} positive samples")
        print(f"   {len(self.attr_embeddings)} attributes")
    
    def _build_item_profiles(self):
        """Build dict of item → list of attr_ids."""
        self.item_profiles = defaultdict(list)
        
        for _, row in self.map_item_attr.iterrows():
            item_id = row['item_id']  # Adjust column name if needed
            attr_id = row['attr_id']
            
            # Only include if we have embeddings for both
            if (item_id in self.frozen_item_embeddings and 
                attr_id in self.attr_embeddings):
                self.item_profiles[item_id].append(attr_id)
        
        # Convert to list for indexing
        self.items = list(self.item_profiles.keys())
    
    def _build_positive_samples(self):
        """Create positive samples: (item_id, attr_id)."""
        self.positive_samples = []
        
        for item_id, attr_list in self.item_profiles.items():
            for attr_id in attr_list:
                self.positive_samples.append((item_id, attr_id))
    
    def _compute_attribute_popularity(self):
        """Compute popularity distribution for negative sampling."""
        attr_counts = self.map_item_attr['attr_id'].value_counts()
        
        self.all_attrs = list(self.attr_embeddings.keys())
        
        self.attr_popularity = {}
        for attr in self.all_attrs:
            self.attr_popularity[attr] = attr_counts.get(attr, 1)
        
        # Convert to probability distribution
        total = sum(self.attr_popularity.values())
        self.attr_probs = np.array([
            self.attr_popularity[attr] / total 
            for attr in self.all_attrs
        ])
    
    def _sample_negatives(self, item_id: str, positive_attr: str, k: int) -> list:
        """Sample k negative attributes for an item."""
        # Get attributes this item has
        item_positive_attrs = set(self.item_profiles[item_id])
        
        negatives = []
        attempts = 0
        max_attempts = k * 10
        
        while len(negatives) < k and attempts < max_attempts:
            if self.negative_sampling == 'random':
                neg_attr = np.random.choice(self.all_attrs)
            elif self.negative_sampling == 'popularity':
                neg_attr = np.random.choice(self.all_attrs, p=self.attr_probs)
            else:
                neg_attr = np.random.choice(self.all_attrs)
            
            # Ensure it's not a positive
            if neg_attr not in item_positive_attrs:
                negatives.append(neg_attr)
            
            attempts += 1
        
        # Pad if needed
        while len(negatives) < k:
            negatives.append(np.random.choice(self.all_attrs))
        
        return negatives
    
    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            item_embedding: [64] - frozen from Phase 1
            positive_attr: [64] - attribute embedding
            negative_attrs: [negatives_per_pos, 64] - negative attribute embeddings
        """
        item_id, pos_attr = self.positive_samples[idx]
        
        # 1. Get frozen item embedding
        item_embedding = self.frozen_item_embeddings[item_id]
        
        # 2. Get positive attribute embedding
        pos_embedding = self.attr_embeddings[pos_attr]
        
        # 3. Sample negative attributes
        neg_attrs = self._sample_negatives(item_id, pos_attr, self.negatives_per_pos)
        neg_embeddings = np.stack([self.attr_embeddings[attr] for attr in neg_attrs])
        
        return (
            torch.FloatTensor(item_embedding),
            torch.FloatTensor(pos_embedding),
            torch.FloatTensor(neg_embeddings)
        )


def create_dataloaders(
    map_item_attr: pd.DataFrame,
    frozen_item_embeddings: Dict[str, np.ndarray],
    attr_embeddings: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    batch_size: int = 256,
    negatives_per_pos: int = 5
):
    """
    Create train and validation dataloaders.
    
    Split by items (not households this time).
    """
    # Split by items
    unique_items = list(frozen_item_embeddings.keys())
    np.random.shuffle(unique_items)
    
    split_idx = int(len(unique_items) * train_ratio)
    train_items = set(unique_items[:split_idx])
    val_items = set(unique_items[split_idx:])
    
    # Filter map_item_attr by split
    train_data = map_item_attr[map_item_attr['item_id'].isin(train_items)]
    val_data = map_item_attr[map_item_attr['item_id'].isin(val_items)]
    
    print(f"Train: {len(train_items)} items, {len(train_data)} mappings")
    print(f"Val:   {len(val_items)} items, {len(val_data)} mappings")
    
    # Create datasets
    train_dataset = ItemAttributeDataset(
        train_data, frozen_item_embeddings, attr_embeddings, negatives_per_pos
    )
    val_dataset = ItemAttributeDataset(
        val_data, frozen_item_embeddings, attr_embeddings, negatives_per_pos
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader