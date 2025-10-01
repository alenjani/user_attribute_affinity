# attr_embed/cooc_ae.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import AttributeEmbeddingProvider

class CoocAEProvider(AttributeEmbeddingProvider):
    """
    Co-occurrence Autoencoder for attribute embeddings.
    
    Approach:
    1. Build attribute co-occurrence matrix from item-level or household-level data
    2. Apply SPPMI (Shifted Positive PMI) transformation
    3. Train denoising autoencoder
    4. Extract bottleneck embeddings
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.source = config.get('source', 'item')  # 'item' or 'household'
        self.latent_dim = config.get('latent_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.epochs = config.get('epochs', 40)
        self.noise_prob = config.get('noise_prob', 0.2)
        self.pmi_shift = config.get('pmi_shift', 1.0)
        self.lr = config.get('lr', 0.001)
        self.batch_size = config.get('batch_size', 256)
        
        self.attr_to_idx = {}
        self.idx_to_attr = {}
        self.cooccur_matrix = None
        self.model = None
        
    def fit(self, inputs: Dict[str, pd.DataFrame]) -> None:
        """Train CoocAE on co-occurrence data."""
        
        print(f"🔨 Building co-occurrence matrix (source={self.source})...")
        
        if self.source == 'item':
            self.cooccur_matrix = self._build_item_cooccurrence(
                inputs['map_item_attribute']
            )
        elif self.source == 'household':
            self.cooccur_matrix = self._build_household_cooccurrence(
                inputs['hh_attr_history']
            )
        else:
            raise ValueError(f"Unknown source: {self.source}")
        
        print(f"   Matrix shape: {self.cooccur_matrix.shape}")
        print(f"   Sparsity: {100 * (1 - self.cooccur_matrix.nnz / np.prod(self.cooccur_matrix.shape)):.1f}%")
        
        print("🔨 Applying SPPMI transformation...")
        sppmi_matrix = self._compute_sppmi(self.cooccur_matrix)
        
        print("🔨 Training autoencoder...")
        self.model = self._train_autoencoder(sppmi_matrix)
        
        print("🔨 Extracting embeddings...")
        self._extract_embeddings(sppmi_matrix)
        
        print("✅ CoocAE training complete!")
    
    def _build_item_cooccurrence(self, map_item_attr: pd.DataFrame) -> csr_matrix:
        """
        Build attribute co-occurrence from item-level data.
        Two attributes co-occur if they appear in the same item.
        """
        # Get unique attributes
        unique_attrs = map_item_attr['attr_id'].unique()
        self.attr_to_idx = {attr: idx for idx, attr in enumerate(unique_attrs)}
        self.idx_to_attr = {idx: attr for attr, idx in self.attr_to_idx.items()}
        n_attrs = len(unique_attrs)
        
        # Build co-occurrence matrix
        from collections import defaultdict
        cooccur = defaultdict(int)
        
        # Group by item
        for upc, group in map_item_attr.groupby('upc_nbr'):
            attrs = group['attr_id'].values
            # Every pair of attributes in this item co-occurs
            for i, attr1 in enumerate(attrs):
                for attr2 in attrs[i+1:]:
                    idx1 = self.attr_to_idx[attr1]
                    idx2 = self.attr_to_idx[attr2]
                    if idx1 < idx2:
                        cooccur[(idx1, idx2)] += 1
                    else:
                        cooccur[(idx2, idx1)] += 1
        
        # Convert to sparse matrix
        row, col, data = [], [], []
        for (i, j), count in cooccur.items():
            row.extend([i, j])
            col.extend([j, i])
            data.extend([count, count])
        
        matrix = csr_matrix((data, (row, col)), shape=(n_attrs, n_attrs))
        return matrix
    
    def _build_household_cooccurrence(self, hh_attr_hist: pd.DataFrame) -> csr_matrix:
        """
        Build attribute co-occurrence from household purchase history.
        Two attributes co-occur if bought by the same household.
        """
        # Get unique attributes
        unique_attrs = hh_attr_hist['attr_id'].unique()
        self.attr_to_idx = {attr: idx for idx, attr in enumerate(unique_attrs)}
        self.idx_to_attr = {idx: attr for attr, idx in self.attr_to_idx.items()}
        n_attrs = len(unique_attrs)
        
        # Build co-occurrence matrix
        from collections import defaultdict
        cooccur = defaultdict(float)
        
        # Group by household
        for hh_id, group in hh_attr_hist.groupby('household_id'):
            attrs = group['attr_id'].values
            weights = group['hist_score'].values
            
            # Weighted co-occurrence
            for i, (attr1, w1) in enumerate(zip(attrs, weights)):
                for attr2, w2 in zip(attrs[i+1:], weights[i+1:]):
                    idx1 = self.attr_to_idx[attr1]
                    idx2 = self.attr_to_idx[attr2]
                    weight = np.sqrt(w1 * w2)  # Geometric mean
                    
                    if idx1 < idx2:
                        cooccur[(idx1, idx2)] += weight
                    else:
                        cooccur[(idx2, idx1)] += weight
        
        # Convert to sparse matrix
        row, col, data = [], [], []
        for (i, j), count in cooccur.items():
            row.extend([i, j])
            col.extend([j, i])
            data.extend([count, count])
        
        matrix = csr_matrix((data, (row, col)), shape=(n_attrs, n_attrs))
        return matrix
    
    def _compute_sppmi(self, cooccur_matrix: csr_matrix) -> np.ndarray:
        """
        Compute Shifted Positive PMI.
        PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
        SPPMI = max(PMI - shift, 0)
        """
        # Convert to dense for easier computation
        C = cooccur_matrix.toarray()
        
        # Compute probabilities
        total = C.sum()
        P_ij = C / total
        P_i = C.sum(axis=1, keepdims=True) / total
        P_j = C.sum(axis=0, keepdims=True) / total
        
        # PMI
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(P_ij / (P_i @ P_j))
            pmi[np.isnan(pmi)] = 0
            pmi[np.isinf(pmi)] = 0
        
        # Shifted Positive PMI
        sppmi = np.maximum(pmi - self.pmi_shift, 0)
        
        return sppmi
    
    def _train_autoencoder(self, sppmi_matrix: np.ndarray) -> nn.Module:
        """Train denoising autoencoder."""
        
        n_attrs = sppmi_matrix.shape[0]
        
        # Define model
        class DenoisingAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x):
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon, z
        
        model = DenoisingAutoencoder(n_attrs, self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Convert to tensor
        X = torch.FloatTensor(sppmi_matrix)
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            # Add noise (dropout mask)
            noise_mask = torch.bernoulli(torch.ones_like(X) * (1 - self.noise_prob))
            X_noisy = X * noise_mask
            
            # Forward pass
            X_recon, _ = model(X_noisy)
            
            # Loss (MSE)
            loss = F.mse_loss(X_recon, X)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return model
    
    def _extract_embeddings(self, sppmi_matrix: np.ndarray) -> None:
        """Extract embeddings from trained encoder."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(sppmi_matrix)
            _, Z = self.model(X)
            embeddings = Z.numpy()
        
        # L2 normalize
        embeddings = normalize(embeddings, norm='l2', axis=1)
        
        # Store
        self.attr_vectors = {
            self.idx_to_attr[idx]: embeddings[idx]
            for idx in range(len(embeddings))
        }
    
    def get_vectors(self) -> Dict[str, np.ndarray]:
        """Return computed embeddings."""
        return self.attr_vectors