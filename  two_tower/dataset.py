# two_tower/dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
from collections import defaultdict

class HouseholdAttributeDataset(Dataset):
    """
    Dataset for Two-Tower training.
    
    Each sample:
    - Household purchase history (list of attributes they bought)
    - Positive attribute (one they bought)
    - Negative attributes (ones they didn't buy)
    """
    
    def __init__(
        self,
        hh_attr_history: pd.DataFrame,
        attr_embeddings: Dict[str, np.ndarray],
        negatives_per_pos: int = 5,
        negative_sampling: str = 'random'  # 'random', 'popularity', 'hard'
    ):
        """
        Args:
            hh_attr_history: DataFrame with (household_id, attr_id, hist_score)
            attr_embeddings: Dict of {attr_id: embedding vector}
            negatives_per_pos: Number of negative samples per positive
            negative_sampling: Strategy for negative sampling
        """
        self.hh_attr_history = hh_attr_history
        self.attr_embeddings = attr_embeddings
        self.negatives_per_pos = negatives_per_pos
        self.negative_sampling = negative_sampling
        
        # Build household purchase profiles
        self._build_household_profiles()
        
        # Build positive samples
        self._build_positive_samples()
        
        # Compute attribute popularity for sampling
        self._compute_attribute_popularity()
        
        print(f"✅ Dataset created:")
        print(f"   {len(self.households)} households")
        print(f"   {len(self.positive_samples)} positive samples")
        print(f"   {len(self.attr_embeddings)} attributes")
    
    def _build_household_profiles(self):
        """Build dict of household → list of (attr_id, weight)."""
        self.household_profiles = defaultdict(list)
        
        for _, row in self.hh_attr_history.iterrows():
            hh_id = row['household_id']
            attr_id = row['attr_id']
            score = row['hist_score']
            
            # Only include attributes we have embeddings for
            if attr_id in self.attr_embeddings:
                self.household_profiles[hh_id].append((attr_id, score))
        
        # Normalize weights per household
        for hh_id in self.household_profiles:
            attrs, scores = zip(*self.household_profiles[hh_id])
            scores = np.array(scores)
            scores = scores / scores.sum()  # Normalize to sum to 1
            self.household_profiles[hh_id] = list(zip(attrs, scores))
        
        self.households = list(self.household_profiles.keys())
    
    def _build_positive_samples(self):
        """Create positive samples: (household_id, positive_attr_id)."""
        self.positive_samples = []
        
        for hh_id, attr_list in self.household_profiles.items():
            # Each attribute this household bought is a positive sample
            for attr_id, weight in attr_list:
                self.positive_samples.append((hh_id, attr_id, weight))
    
    def _compute_attribute_popularity(self):
        """Compute popularity distribution for negative sampling."""
        attr_counts = self.hh_attr_history['attr_id'].value_counts()
        
        # Only include attributes with embeddings
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
    
    def _sample_negatives(self, household_id: str, positive_attr: str, k: int) -> list:
        """Sample k negative attributes for a household."""
        # Get attributes this household has interacted with
        hh_positive_attrs = set([attr for attr, _ in self.household_profiles[household_id]])
        
        negatives = []
        attempts = 0
        max_attempts = k * 10
        
        while len(negatives) < k and attempts < max_attempts:
            if self.negative_sampling == 'random':
                # Uniform random
                neg_attr = np.random.choice(self.all_attrs)
            elif self.negative_sampling == 'popularity':
                # Sample by popularity (more popular = more likely)
                neg_attr = np.random.choice(self.all_attrs, p=self.attr_probs)
            else:
                # TODO: Hard negative sampling (similar attributes)
                neg_attr = np.random.choice(self.all_attrs)
            
            # Ensure it's not a positive
            if neg_attr not in hh_positive_attrs:
                negatives.append(neg_attr)
            
            attempts += 1
        
        # If couldn't find enough, pad with random
        while len(negatives) < k:
            negatives.append(np.random.choice(self.all_attrs))
        
        return negatives
    
    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hh_profile: Household embedding (aggregated from purchase history)
            positive_attr: Positive attribute embedding
            negative_attrs: Negative attribute embeddings [negatives_per_pos, dim]
        """
        hh_id, pos_attr, weight = self.positive_samples[idx]
        
        # 1. Build household embedding (weighted average of purchased attributes)
        hh_attrs, hh_weights = zip(*self.household_profiles[hh_id])
        hh_embeddings = np.stack([self.attr_embeddings[attr] for attr in hh_attrs])
        hh_weights = np.array(hh_weights)
        
        hh_profile = np.average(hh_embeddings, weights=hh_weights, axis=0)
        
        # 2. Get positive attribute embedding
        pos_embedding = self.attr_embeddings[pos_attr]
        
        # 3. Sample negative attributes
        neg_attrs = self._sample_negatives(hh_id, pos_attr, self.negatives_per_pos)
        neg_embeddings = np.stack([self.attr_embeddings[attr] for attr in neg_attrs])
        
        return (
            torch.FloatTensor(hh_profile),
            torch.FloatTensor(pos_embedding),
            torch.FloatTensor(neg_embeddings)
        )


def create_dataloaders(
    hh_attr_history: pd.DataFrame,
    attr_embeddings: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    batch_size: int = 256,
    negatives_per_pos: int = 5
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        hh_attr_history: Full household-attribute history
        attr_embeddings: Attribute embeddings from provider
        train_ratio: Fraction for training
        batch_size: Batch size
        negatives_per_pos: Number of negatives per positive
    
    Returns:
        train_loader, val_loader
    """
    # Time-based split (more realistic)
    # Sort by last_seen_days (most recent first)
    hh_attr_history = hh_attr_history.sort_values('last_seen_days')
    
    # Split by households (avoid data leakage)
    unique_hhs = hh_attr_history['household_id'].unique()
    np.random.shuffle(unique_hhs)
    
    split_idx = int(len(unique_hhs) * train_ratio)
    train_hhs = set(unique_hhs[:split_idx])
    val_hhs = set(unique_hhs[split_idx:])
    
    train_data = hh_attr_history[hh_attr_history['household_id'].isin(train_hhs)]
    val_data = hh_attr_history[hh_attr_history['household_id'].isin(val_hhs)]
    
    print(f"Train: {len(train_hhs)} households, {len(train_data)} interactions")
    print(f"Val:   {len(val_hhs)} households, {len(val_data)} interactions")
    
    # Create datasets
    train_dataset = HouseholdAttributeDataset(
        train_data, attr_embeddings, negatives_per_pos
    )
    val_dataset = HouseholdAttributeDataset(
        val_data, attr_embeddings, negatives_per_pos
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader