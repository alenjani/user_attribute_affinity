# attr_embed/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import pandas as pd

class AttributeEmbeddingProvider(ABC):
    """
    Base interface for all attribute embedding providers.
    All providers (CoocAE, GCL, SVD, SBERT) implement this.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Provider-specific configuration dict
        """
        self.config = config
        self.attr_vectors = {}  # Will store {attr_id: np.ndarray}
        
    @abstractmethod
    def fit(self, inputs: Dict[str, pd.DataFrame]) -> None:
        """
        Train/compute embeddings from input data.
        
        Args:
            inputs: Dict containing DataFrames:
                - 'attr_meta': Attribute metadata
                - 'map_item_attribute': Item→attribute mappings
                - 'hh_attr_history': Household→attribute history
                - 'hh_item': Transaction data (optional)
                - 'item_nutrition': Nutrition data (optional)
        """
        pass
    
    @abstractmethod
    def get_vectors(self) -> Dict[str, np.ndarray]:
        """
        Return computed embeddings.
        
        Returns:
            Dict mapping attr_id → L2-normalized vector
        """
        pass
    
    def save(self, output_path: str, metadata: Dict[str, Any]) -> None:
        """
        Save embeddings to vector store (Parquet).
        
        Args:
            output_path: Path to save (e.g., 'gs://bucket/models/attr_embeddings.parquet')
            metadata: Metadata to store (provider, version, etc.)
        """
        vectors = self.get_vectors()
        
        # Convert to DataFrame
        rows = []
        for attr_id, vector in vectors.items():
            rows.append({
                'attr_id': attr_id,
                'provider': metadata['provider'],
                'model_version': metadata['model_version'],
                'vocab_version': metadata.get('vocab_version', 'v1'),
                'built_from': metadata.get('built_from', ''),
                'dim': len(vector),
                'vector': vector.tolist(),
                'created_at': metadata.get('created_at', pd.Timestamp.now())
            })
        
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
        print(f"✅ Saved {len(rows)} embeddings to {output_path}")
    
    def load(self, path: str, provider: str, model_version: str) -> None:
        """
        Load previously saved embeddings.
        
        Args:
            path: Path to vector store
            provider: Provider name to filter
            model_version: Model version to filter
        """
        df = pd.read_parquet(path)
        df = df[
            (df['provider'] == provider) & 
            (df['model_version'] == model_version)
        ]
        
        self.attr_vectors = {
            row['attr_id']: np.array(row['vector'])
            for _, row in df.iterrows()
        }
        print(f"✅ Loaded {len(self.attr_vectors)} embeddings")