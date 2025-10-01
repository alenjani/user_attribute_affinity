# attr_embed/gcl.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from typing import Dict, Any, Tuple

from .base import AttributeEmbeddingProvider
from .utils import build_attribute_graph, build_node_features

class GCLProvider(AttributeEmbeddingProvider):
    """
    Graph Contrastive Learning for attribute embeddings.
    
    Approach:
    1. Build attribute co-occurrence graph
    2. Create node features from metadata and item statistics
    3. Apply Graph Neural Network with message passing
    4. Self-supervised contrastive learning with graph augmentation
    5. Extract final node embeddings
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.dim = config.get('dim', 64)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.n_layers = config.get('n_layers', 3)
        self.drop_edge_prob = config.get('drop_edge_prob', 0.2)
        self.drop_feat_prob = config.get('drop_feat_prob', 0.1)
        self.temperature = config.get('temperature', 0.5)
        self.epochs = config.get('epochs', 100)
        self.lr = config.get('lr', 0.001)
        self.batch_size = config.get('batch_size', 256)
        self.use_household_edges = config.get('use_household_edges', True)
        
        self.attr_to_idx = {}
        self.idx_to_attr = {}
        self.adjacency_matrix = None
        self.node_features = None
        self.model = None
    
    def fit(self, inputs: Dict[str, pd.DataFrame]) -> None:
        """Train GCL on attribute graph."""
        
        print(f"🔨 Building attribute graph...")
        
        # Build graph
        self.adjacency_matrix, self.attr_to_idx, self.idx_to_attr = build_attribute_graph(
            map_item_attr=inputs['map_item_attribute'],
            hh_attr_history=inputs.get('hh_attr_history'),
            use_household_edges=self.use_household_edges,
            min_cooccur=3
        )
        
        # Build node features
        print(f"🔨 Building node features...")
        self.node_features = build_node_features(
            attr_meta=inputs['attr_meta'],
            map_item_attr=inputs['map_item_attribute'],
            item_nutrition=inputs.get('item_nutrition'),
            attr_to_idx=self.attr_to_idx
        )
        
        print(f"   Graph: {self.adjacency_matrix.shape[0]} nodes, {self.adjacency_matrix.nnz // 2} edges")
        print(f"   Node features: {self.node_features.shape[1]} dimensions")
        
        # Convert to PyTorch
        edge_index, edge_weight = self._sparse_to_edge_index(self.adjacency_matrix)
        node_features = torch.FloatTensor(self.node_features)
        
        # Train model
        print(f"🔨 Training GCL model...")
        self.model = self._train_gcl(node_features, edge_index, edge_weight)
        
        # Extract embeddings
        print(f"🔨 Extracting embeddings...")
        self._extract_embeddings(node_features, edge_index, edge_weight)
        
        print("✅ GCL training complete!")
    
    def _sparse_to_edge_index(self, adj_matrix: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sparse adjacency matrix to edge_index format."""
        coo = adj_matrix.tocoo()
        edge_index = torch.LongTensor(np.vstack([coo.row, coo.col]))
        edge_weight = torch.FloatTensor(coo.data)
        return edge_index, edge_weight
    
    def _augment_graph(
        self, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor,
        node_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply graph augmentation for contrastive learning.
        
        Augmentations:
        1. Edge dropout (randomly drop edges)
        2. Feature masking (randomly mask node features)
        """
        
        # 1. Edge dropout
        n_edges = edge_index.size(1)
        edge_mask = torch.rand(n_edges) > self.drop_edge_prob
        edge_index_aug = edge_index[:, edge_mask]
        edge_weight_aug = edge_weight[edge_mask]
        
        # 2. Feature masking
        feat_mask = torch.rand_like(node_features) > self.drop_feat_prob
        node_features_aug = node_features * feat_mask
        
        return edge_index_aug, edge_weight_aug, node_features_aug
    
    def _train_gcl(
        self, 
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> nn.Module:
        """Train Graph Contrastive Learning model."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Initialize model
        model = GraphEncoder(
            in_dim=node_features.size(1),
            hidden_dim=self.hidden_dim,
            out_dim=self.dim,
            n_layers=self.n_layers
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        
        n_nodes = node_features.size(0)
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            # Create two augmented views
            edge_index_1, edge_weight_1, node_feat_1 = self._augment_graph(
                edge_index, edge_weight, node_features
            )
            edge_index_2, edge_weight_2, node_feat_2 = self._augment_graph(
                edge_index, edge_weight, node_features
            )
            
            edge_index_1 = edge_index_1.to(device)
            edge_weight_1 = edge_weight_1.to(device)
            node_feat_1 = node_feat_1.to(device)
            
            edge_index_2 = edge_index_2.to(device)
            edge_weight_2 = edge_weight_2.to(device)
            node_feat_2 = node_feat_2.to(device)
            
            # Forward pass on both views
            z1 = model(node_feat_1, edge_index_1, edge_weight_1)
            z2 = model(node_feat_2, edge_index_2, edge_weight_2)
            
            # Contrastive loss (InfoNCE)
            loss = self._contrastive_loss(z1, z2, self.temperature)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return model
    
    def _contrastive_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        temperature: float
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Positive pairs: (z1[i], z2[i]) - same node in two views
        Negative pairs: (z1[i], z2[j]) where i != j
        """
        
        # L2 normalize
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        batch_size = z1.size(0)
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2*batch, dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / temperature  # [2*batch, 2*batch]
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_indices = torch.arange(batch_size, device=z.device)
        pos_indices = torch.cat([pos_indices + batch_size, pos_indices])
        
        # Loss: log-softmax
        log_prob = F.log_softmax(sim_matrix, dim=1)
        loss = -log_prob[torch.arange(2 * batch_size, device=z.device), pos_indices].mean()
        
        return loss
    
    def _extract_embeddings(
        self, 
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> None:
        """Extract final embeddings from trained model."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        
        with torch.no_grad():
            embeddings = self.model(node_features, edge_index, edge_weight)
            embeddings = embeddings.cpu().numpy()
        
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


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder.
    
    Uses GraphSAGE-style message passing with mean aggregation.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3):
        super().__init__()
        
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers - 1:
                # Last layer outputs final dimension
                self.convs.append(GraphConvLayer(hidden_dim, out_dim))
            else:
                self.convs.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < n_layers - 1 else out_dim)
            for i in range(n_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [n_nodes, in_dim]
            edge_index: Edge indices [2, n_edges]
            edge_weight: Edge weights [n_edges]
        
        Returns:
            embeddings: [n_nodes, out_dim]
        """
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # Message passing
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index, edge_weight)
            x_new = bn(x_new)
            
            if i < self.n_layers - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=0.2, training=self.training)
            
            # Residual connection (if dimensions match)
            if x.size(1) == x_new.size(1):
                x = x + x_new
            else:
                x = x_new
        
        return x


class GraphConvLayer(nn.Module):
    """
    Single graph convolution layer (GraphSAGE-style with mean aggregation).
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Transform for self features
        self.linear_self = nn.Linear(in_dim, out_dim)
        
        # Transform for neighbor features
        self.linear_neigh = nn.Linear(in_dim, out_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Message passing.
        
        Args:
            x: Node features [n_nodes, in_dim]
            edge_index: [2, n_edges] - (source, target) pairs
            edge_weight: [n_edges]
        
        Returns:
            x_new: [n_nodes, out_dim]
        """
        
        n_nodes = x.size(0)
        
        # Self features
        x_self = self.linear_self(x)
        
        # Aggregate neighbor features (weighted mean)
        row, col = edge_index[0], edge_index[1]
        
        # Weighted sum of neighbor features
        x_neigh = torch.zeros(n_nodes, self.in_dim, device=x.device)
        degree = torch.zeros(n_nodes, device=x.device)
        
        x_neigh.index_add_(0, col, x[row] * edge_weight.unsqueeze(1))
        degree.index_add_(0, col, edge_weight)
        
        # Normalize by degree (mean aggregation)
        degree = degree.clamp(min=1.0)
        x_neigh = x_neigh / degree.unsqueeze(1)
        
        # Transform neighbor features
        x_neigh = self.linear_neigh(x_neigh)
        
        # Combine self and neighbor
        x_new = x_self + x_neigh
        
        return x_new