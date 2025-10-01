# attr_embed/utils.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, List
from collections import defaultdict

def build_attribute_graph(
    map_item_attr: pd.DataFrame,
    hh_attr_history: pd.DataFrame = None,
    use_household_edges: bool = True,
    min_cooccur: int = 3
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
    """
    Build attribute co-occurrence graph.
    
    Args:
        map_item_attr: Item→attribute mappings
        hh_attr_history: Household→attribute history (optional)
        use_household_edges: If True, add household-level co-occurrence edges
        min_cooccur: Minimum co-occurrence count to create edge
    
    Returns:
        adjacency_matrix: Sparse adjacency matrix [n_attrs, n_attrs]
        attr_to_idx: Dict mapping attr_id → index
        idx_to_attr: Dict mapping index → attr_id
    """
    
    # Get unique attributes
    unique_attrs = set(map_item_attr['attr_id'].unique())
    if hh_attr_history is not None and use_household_edges:
        unique_attrs.update(hh_attr_history['attr_id'].unique())
    
    unique_attrs = sorted(list(unique_attrs))
    attr_to_idx = {attr: idx for idx, attr in enumerate(unique_attrs)}
    idx_to_attr = {idx: attr for attr, idx in attr_to_idx.items()}
    n_attrs = len(unique_attrs)
    
    print(f"   Building graph with {n_attrs} attribute nodes...")
    
    # Build co-occurrence from item-level
    print("   Adding item-level co-occurrence edges...")
    item_cooccur = defaultdict(float)
    
    for upc, group in map_item_attr.groupby('upc_nbr'):
        attrs = group['attr_id'].values
        weights = group['p_attr'].values
        
        # Pairwise co-occurrence
        for i in range(len(attrs)):
            for j in range(i + 1, len(attrs)):
                attr1, attr2 = attrs[i], attrs[j]
                weight = weights[i] * weights[j]  # Confidence product
                
                idx1 = attr_to_idx[attr1]
                idx2 = attr_to_idx[attr2]
                
                if idx1 < idx2:
                    item_cooccur[(idx1, idx2)] += weight
                else:
                    item_cooccur[(idx2, idx1)] += weight
    
    print(f"      {len(item_cooccur)} unique item-level edges")
    
    # Build co-occurrence from household-level
    hh_cooccur = defaultdict(float)
    
    if hh_attr_history is not None and use_household_edges:
        print("   Adding household-level co-occurrence edges...")
        
        for hh_id, group in hh_attr_history.groupby('household_id'):
            attrs = group['attr_id'].values
            scores = group['hist_score'].values
            
            # Weighted co-occurrence
            for i in range(len(attrs)):
                for j in range(i + 1, len(attrs)):
                    attr1, attr2 = attrs[i], attrs[j]
                    
                    if attr1 in attr_to_idx and attr2 in attr_to_idx:
                        weight = np.sqrt(scores[i] * scores[j])
                        
                        idx1 = attr_to_idx[attr1]
                        idx2 = attr_to_idx[attr2]
                        
                        if idx1 < idx2:
                            hh_cooccur[(idx1, idx2)] += weight
                        else:
                            hh_cooccur[(idx2, idx1)] += weight
        
        print(f"      {len(hh_cooccur)} unique household-level edges")
    
    # Combine edges
    all_edges = defaultdict(float)
    
    for edge, weight in item_cooccur.items():
        all_edges[edge] += weight
    
    for edge, weight in hh_cooccur.items():
        all_edges[edge] += weight * 0.5  # Downweight household edges slightly
    
    # Filter by minimum co-occurrence
    all_edges = {k: v for k, v in all_edges.items() if v >= min_cooccur}
    
    print(f"   Total edges after filtering: {len(all_edges)}")
    
    # Convert to sparse adjacency matrix
    row, col, data = [], [], []
    
    for (i, j), weight in all_edges.items():
        # Symmetric (undirected graph)
        row.extend([i, j])
        col.extend([j, i])
        data.extend([weight, weight])
    
    adjacency_matrix = csr_matrix((data, (row, col)), shape=(n_attrs, n_attrs))
    
    # Add self-loops with weight 1.0
    adjacency_matrix += csr_matrix(np.eye(n_attrs))
    
    print(f"   Graph density: {100 * adjacency_matrix.nnz / (n_attrs ** 2):.2f}%")
    
    return adjacency_matrix, attr_to_idx, idx_to_attr


def build_node_features(
    attr_meta: pd.DataFrame,
    map_item_attr: pd.DataFrame,
    item_nutrition: pd.DataFrame = None,
    attr_to_idx: Dict[str, int] = None
) -> np.ndarray:
    """
    Build node feature matrix for attributes.
    
    Features include:
    - Attribute family one-hot encoding
    - Popularity (how many items/households have this attribute)
    - Nutrition statistics (if available)
    
    Args:
        attr_meta: Attribute metadata
        map_item_attr: Item→attribute mappings
        item_nutrition: Item nutrition data (optional)
        attr_to_idx: Mapping from attr_id to index
    
    Returns:
        node_features: [n_attrs, n_features]
    """
    
    n_attrs = len(attr_to_idx)
    features_list = []
    
    print(f"   Building node features for {n_attrs} attributes...")
    
    # 1. Attribute family one-hot encoding
    print("      Adding attribute family features...")
    attr_families = attr_meta.set_index('attr_id')['attr_family'].to_dict()
    unique_families = sorted(attr_meta['attr_family'].unique())
    family_to_idx = {fam: idx for idx, fam in enumerate(unique_families)}
    
    family_features = np.zeros((n_attrs, len(unique_families)))
    for attr_id, idx in attr_to_idx.items():
        if attr_id in attr_families:
            fam_idx = family_to_idx[attr_families[attr_id]]
            family_features[idx, fam_idx] = 1.0
    
    features_list.append(family_features)
    print(f"         {family_features.shape[1]} family features")
    
    # 2. Popularity features
    print("      Adding popularity features...")
    attr_item_counts = map_item_attr['attr_id'].value_counts().to_dict()
    attr_item_diversity = map_item_attr.groupby('attr_id')['upc_nbr'].nunique().to_dict()
    
    popularity_features = np.zeros((n_attrs, 2))
    for attr_id, idx in attr_to_idx.items():
        popularity_features[idx, 0] = np.log1p(attr_item_counts.get(attr_id, 0))
        popularity_features[idx, 1] = np.log1p(attr_item_diversity.get(attr_id, 0))
    
    features_list.append(popularity_features)
    print(f"         {popularity_features.shape[1]} popularity features")
    
    # 3. Nutrition features (if available)
    if item_nutrition is not None:
        print("      Adding nutrition features...")
        
        # Merge items with attributes
        item_attr_nutrition = map_item_attr.merge(
            item_nutrition,
            left_on='upc_nbr',
            right_on='upc_id',
            how='inner'
        )
        
        # Get nutrition columns (exclude upc_id)
        nutrition_cols = [col for col in item_nutrition.columns if col not in ['upc_id', 'upc_nbr']]
        
        # Aggregate nutrition by attribute (mean)
        attr_nutrition = item_attr_nutrition.groupby('attr_id')[nutrition_cols].mean()
        
        # Fill missing with 0
        nutrition_features = np.zeros((n_attrs, len(nutrition_cols)))
        for attr_id, idx in attr_to_idx.items():
            if attr_id in attr_nutrition.index:
                nutrition_features[idx, :] = attr_nutrition.loc[attr_id].values
        
        # Normalize (z-score)
        nutrition_features = (nutrition_features - nutrition_features.mean(axis=0)) / (nutrition_features.std(axis=0) + 1e-6)
        nutrition_features = np.nan_to_num(nutrition_features, 0.0)
        
        features_list.append(nutrition_features)
        print(f"         {nutrition_features.shape[1]} nutrition features")
    
    # Concatenate all features
    node_features = np.concatenate(features_list, axis=1)
    
    print(f"   Total node features: {node_features.shape[1]} dimensions")
    
    return node_features.astype(np.float32)