# two_tower/model.py (MODIFIED - ADD NEW CLASS)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep existing TwoTowerModel class for reference
# Add new class:

class ItemAttributeAlignmentModel(nn.Module):
    """
    Aligns attribute embeddings to frozen item embedding space.
    
    Architecture:
    - Item Tower: FROZEN (just pass through)
    - Attribute Tower: Trainable MLP for alignment
    """
    
    def __init__(
        self,
        item_embed_dim: int = 64,
        attr_embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Args:
            item_embed_dim: Dimension of frozen item embeddings
            attr_embed_dim: Dimension of attribute embeddings (from provider)
            hidden_dim: Hidden layer size for alignment network
            output_dim: Final aligned dimension (should match item_embed_dim)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.item_embed_dim = item_embed_dim
        self.attr_embed_dim = attr_embed_dim
        self.output_dim = output_dim
        
        # Item tower: Just pass through (frozen embeddings)
        # No parameters needed
        
        # Attribute alignment network (trainable)
        self.attr_alignment = nn.Sequential(
            nn.Linear(attr_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Temperature parameter for scaling logits
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward_item(self, item_embedding: torch.Tensor) -> torch.Tensor:
        """
        Item tower (frozen - just normalize).
        
        Args:
            item_embedding: [batch, item_embed_dim] - frozen embeddings
        
        Returns:
            item_embedding_normalized: [batch, output_dim]
        """
        # Detach to ensure no gradients flow to frozen embeddings
        item_emb = item_embedding.detach()
        
        # L2 normalize
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        
        return item_emb
    
    def forward_attribute(self, attr_embedding: torch.Tensor) -> torch.Tensor:
        """
        Attribute alignment tower (trainable).
        
        Args:
            attr_embedding: [batch, attr_embed_dim] or [batch, k, attr_embed_dim]
        
        Returns:
            attr_embedding_aligned: [batch, output_dim] or [batch, k, output_dim]
        """
        # Align to item space
        attr_aligned = self.attr_alignment(attr_embedding)
        
        # L2 normalize
        attr_aligned = F.normalize(attr_aligned, p=2, dim=-1)
        
        return attr_aligned
    
    def forward(
        self,
        item_embedding: torch.Tensor,
        pos_attr: torch.Tensor,
        neg_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for contrastive learning.
        
        Args:
            item_embedding: [batch, item_embed_dim] - frozen
            pos_attr: [batch, attr_embed_dim]
            neg_attrs: [batch, k_neg, attr_embed_dim]
        
        Returns:
            logits: [batch, 1 + k_neg] - similarity scores
        """
        # Encode item (frozen)
        item_emb = self.forward_item(item_embedding)  # [batch, output_dim]
        
        # Encode positive attribute
        pos_emb = self.forward_attribute(pos_attr)   # [batch, output_dim]
        
        # Encode negative attributes
        neg_embs = self.forward_attribute(neg_attrs) # [batch, k_neg, output_dim]
        
        # Compute similarities (dot product)
        pos_logit = (item_emb * pos_emb).sum(dim=-1, keepdim=True)  # [batch, 1]
        neg_logits = torch.bmm(neg_embs, item_emb.unsqueeze(-1)).squeeze(-1)  # [batch, k_neg]
        
        # Concatenate: [positive, negatives]
        logits = torch.cat([pos_logit, neg_logits], dim=-1)  # [batch, 1 + k_neg]
        
        # Scale by temperature
        logits = logits / self.temperature
        
        return logits
    
    def compute_affinity(
        self,
        item_embeddings: torch.Tensor,
        attr_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute affinity scores between items and attributes (inference).
        
        Args:
            item_embeddings: [batch_item, item_embed_dim] - frozen
            attr_embeddings: [n_attrs, attr_embed_dim] - from provider
        
        Returns:
            affinity_scores: [batch_item, n_attrs]
        """
        item_emb = self.forward_item(item_embeddings)        # [batch_item, output_dim]
        attr_embs = self.forward_attribute(attr_embeddings)  # [n_attrs, output_dim]
        
        # Dot product: [batch_item, n_attrs]
        affinity_scores = torch.mm(item_emb, attr_embs.T)
        
        return affinity_scores