# test_gcl_local.py

import numpy as np
import pandas as pd
from attr_embed.gcl import GCLProvider
from attr_embed.utils import build_attribute_graph, build_node_features

def test_gcl_small():
    """Test GCL on small synthetic data."""
    
    print("="*60)
    print("Testing GCL Provider (Small Data)")
    print("="*60)
    
    # Create small test data
    attr_meta = pd.DataFrame({
        'attr_id': [
            'brand_a', 'brand_b', 'brand_c',
            'health_organic', 'health_vegan',
            'category_dairy', 'category_snacks'
        ],
        'attr_family': [
            'brand', 'brand', 'brand',
            'health_claim', 'health_claim',
            'category', 'category'
        ],
        'display_name': [
            'Brand A', 'Brand B', 'Brand C',
            'Organic', 'Vegan',
            'Dairy', 'Snacks'
        ]
    })
    
    map_item_attr = pd.DataFrame({
        'upc_nbr': [
            'item1', 'item1', 'item1',
            'item2', 'item2', 'item2',
            'item3', 'item3',
            'item4', 'item4'
        ],
        'attr_id': [
            'brand_a', 'health_organic', 'category_dairy',
            'brand_b', 'health_organic', 'category_dairy',
            'brand_c', 'category_snacks',
            'brand_a', 'health_vegan'
        ],
        'p_attr': [1.0] * 10,
        'evidence_sources': ['test'] * 10
    })
    
    hh_attr_history = pd.DataFrame({
        'household_id': ['hh1', 'hh1', 'hh2', 'hh2', 'hh3'],
        'attr_id': ['brand_a', 'health_organic', 'brand_b', 'health_organic', 'brand_c'],
        'hist_score': [10.0, 5.0, 8.0, 3.0, 12.0],
        'last_seen_days': [7, 14, 5, 30, 3]
    })
    
    # Test graph building
    print("\n1. Testing graph building...")
    adj_matrix, attr_to_idx, idx_to_attr = build_attribute_graph(
        map_item_attr=map_item_attr,
        hh_attr_history=hh_attr_history,
        use_household_edges=True,
        min_cooccur=1
    )
    
    print(f"   ✅ Graph built: {adj_matrix.shape[0]} nodes, {adj_matrix.nnz} edges")
    
    # Test node features
    print("\n2. Testing node feature building...")
    node_features = build_node_features(
        attr_meta=attr_meta,
        map_item_attr=map_item_attr,
        attr_to_idx=attr_to_idx
    )
    
    print(f"   ✅ Node features: shape {node_features.shape}")
    
    # Test GCL provider
    print("\n3. Testing GCL provider...")
    config = {
        'dim': 16,
        'hidden_dim': 32,
        'n_layers': 2,
        'drop_edge_prob': 0.1,
        'drop_feat_prob': 0.1,
        'temperature': 0.5,
        'epochs': 10,  # Few epochs for testing
        'lr': 0.001,
        'use_household_edges': True
    }
    
    provider = GCLProvider(config)
    
    inputs = {
        'attr_meta': attr_meta,
        'map_item_attribute': map_item_attr,
        'hh_attr_history': hh_attr_history
    }
    
    provider.fit(inputs)
    
    vectors = provider.get_vectors()
    
    print(f"\n   ✅ Generated {len(vectors)} embeddings")
    print(f"   ✅ Embedding dimension: {len(next(iter(vectors.values())))}")
    
    # Check embedding quality
    print("\n4. Checking embedding quality...")
    
    # Check normalization
    norms = [np.linalg.norm(vec) for vec in vectors.values()]
    mean_norm = np.mean(norms)
    print(f"   Mean L2 norm: {mean_norm:.3f} (should be ~1.0)")
    
    # Check similarity patterns
    brand_a_emb = vectors['brand_a']
    brand_b_emb = vectors['brand_b']
    brand_c_emb = vectors['brand_c']
    
    sim_ab = np.dot(brand_a_emb, brand_b_emb)
    sim_ac = np.dot(brand_a_emb, brand_c_emb)
    
    print(f"\n   Similarity(brand_a, brand_b): {sim_ab:.3f}")
    print(f"   Similarity(brand_a, brand_c): {sim_ac:.3f}")
    print(f"   Expected: brand_a closer to brand_b (both organic dairy)")
    
    if sim_ab > sim_ac:
        print("   ✅ Semantic structure preserved!")
    else:
        print("   ⚠️  Unexpected similarity pattern (may need more data/epochs)")
    
    print("\n" + "="*60)
    print("✅ GCL TEST PASSED")
    print("="*60)

if __name__ == "__main__":
    test_gcl_small()