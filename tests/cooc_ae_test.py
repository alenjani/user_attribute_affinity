# test_local.py

import pandas as pd
from attr_embed.cooc_ae import CoocAEProvider

# Create small test data
attr_meta = pd.DataFrame({
    'attr_id': ['brand_a', 'brand_b', 'brand_c'],
    'attr_family': ['brand', 'brand', 'brand'],
    'display_name': ['Brand A', 'Brand B', 'Brand C']
})

map_item_attr = pd.DataFrame({
    'upc_nbr': ['item1', 'item1', 'item2', 'item2', 'item3'],
    'attr_id': ['brand_a', 'brand_b', 'brand_b', 'brand_c', 'brand_a'],
    'p_attr': [1.0, 1.0, 1.0, 1.0, 1.0],
    'evidence_sources': ['test'] * 5
})

hh_attr_history = pd.DataFrame({
    'household_id': ['hh1', 'hh1', 'hh2', 'hh2'],
    'attr_id': ['brand_a', 'brand_b', 'brand_b', 'brand_c'],
    'hist_score': [10.0, 5.0, 8.0, 3.0],
    'last_seen_days': [7, 14, 5, 30]
})

# Test CoocAE
config = {
    'source': 'item',
    'latent_dim': 8,
    'hidden_dim': 16,
    'epochs': 10
}

provider = CoocAEProvider(config)
provider.fit({
    'attr_meta': attr_meta,
    'map_item_attribute': map_item_attr,
    'hh_attr_history': hh_attr_history
})

vectors = provider.get_vectors()
print(f"Generated {len(vectors)} embeddings")
print(f"Embedding dimension: {len(next(iter(vectors.values())))}")
print(f"Sample: {list(vectors.keys())[:3]}")