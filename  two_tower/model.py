# two_tower/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """
    Two-Tower model for household-attribute affinity.
    
    Architecture:
    - Household Tower: MLP that processes household purchase profile
    - Attribute Tower: Optional projection of attribute embeddings
    - Dot product similarity between towers
    """
    
    def __init__(
        self,
        attr_embed_dim: int,
        hh_hidden_dims: list = [128, 64],
        output_dim: int = 64,
        dropout: float = 0.2,
        freeze_attr_embeddings: bool = True
    ):
        """
        Args:
            attr_embed_dim: Dimension of attribute embeddings (from provider)
            hh_hidden_dims: Hidden layer dimensions for household tower
            output_dim: Final embedding dimension (both towers)
            dropout: Dropout probability
            freeze_attr_embeddings: If True, don't fine-tune attribute embeddings
        """
        super().__init__()
        
        self.attr_embed_dim = attr_embed_dim
        self.output_dim = output_dim
        self.freeze_attr_embeddings = freeze_attr_embeddings
        
        # Household Tower
        hh_layers = []
        prev_dim = attr_embed_dim
        
        for hidden_dim in hh_hidden_dims:
            hh_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection to output_dim
        hh_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.household_tower = nn.Sequential(*hh_layers)
        
        # Attribute Tower (optional projection)
        if attr_embed_dim != output_dim:
            self.attribute_projection = nn.Linear(attr_embed_dim, output_dim)
        else:
            self.attribute_projection = nn.Identity()
        
        # Temperature parameter for scaling logits (learnable)
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward_household(self, hh_profile: torch.Tensor) -> torch.Tensor:
        """
        Encode household profile.
        
        Args:
            hh_profile: [batch, attr_embed_dim] - aggregated purchase history
        
        Returns:
            hh_embedding: [batch, output_dim]
        """
        hh_embedding = self.household_tower(hh_profile)
        # L2 normalize
        hh_embedding = F.normalize(hh_embedding, p=2, dim=-1)
        return hh_embedding
    
    def forward_attribute(self, attr_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode attribute.
        
        Args:
            attr_embedding: [batch, attr_embed_dim] or [batch, k, attr_embed_dim]
        
        Returns:
            attr_embedding_proj: [batch, output_dim] or [batch, k, output_dim]
        """
        if self.freeze_attr_embeddings:
            attr_embedding = attr_embedding.detach()
        
        attr_embedding_proj = self.attribute_projection(attr_embedding)
        # L2 normalize
        attr_embedding_proj = F.normalize(attr_embedding_proj, p=2, dim=-1)
        return attr_embedding_proj
    
    def forward(
        self, 
        hh_profile: torch.Tensor, 
        pos_attr: torch.Tensor, 
        neg_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for contrastive learning.
        
        Args:
            hh_profile: [batch, attr_embed_dim]
            pos_attr: [batch, attr_embed_dim]
            neg_attrs: [batch, k_neg, attr_embed_dim]
        
        Returns:
            logits: [batch, 1 + k_neg] - similarity scores
        """
        # Encode household
        hh_emb = self.forward_household(hh_profile)  # [batch, output_dim]
        
        # Encode positive attribute
        pos_emb = self.forward_attribute(pos_attr)   # [batch, output_dim]
        
        # Encode negative attributes
        neg_embs = self.forward_attribute(neg_attrs) # [batch, k_neg, output_dim]
        
        # Compute similarities (dot product)
        pos_logit = (hh_emb * pos_emb).sum(dim=-1, keepdim=True)  # [batch, 1]
        neg_logits = torch.bmm(neg_embs, hh_emb.unsqueeze(-1)).squeeze(-1)  # [batch, k_neg]
        
        # Concatenate: [positive, negatives]
        logits = torch.cat([pos_logit, neg_logits], dim=-1)  # [batch, 1 + k_neg]
        
        # Scale by temperature
        logits = logits / self.temperature
        
        return logits
    
    def compute_affinity(
        self, 
        hh_profile: torch.Tensor, 
        attr_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute affinity scores between households and attributes (inference).
        
        Args:
            hh_profile: [batch_hh, attr_embed_dim]
            attr_embeddings: [n_attrs, attr_embed_dim]
        
        Returns:
            affinity_scores: [batch_hh, n_attrs]
        """
        hh_emb = self.forward_household(hh_profile)          # [batch_hh, output_dim]
        attr_embs = self.forward_attribute(attr_embeddings)  # [n_attrs, output_dim]
        
        # Dot product: [batch_hh, n_attrs]
        affinity_scores = torch.mm(hh_emb, attr_embs.T)
        
        return affinity_scores