# etl/build_hh_attr_history.py

import numpy as np
from datetime import datetime

def build_hh_attr_history(
    txn_df: pd.DataFrame,
    map_item_attr: pd.DataFrame,
    decay_days: float = 90.0,
    ref_date: str = None
) -> pd.DataFrame:
    """
    Compute time-decayed household→attribute scores.
    
    Args:
        txn_df: Transaction data
        map_item_attr: Item→attribute mapping
        decay_days: Half-life for exponential decay
        ref_date: Reference date (default: max date in txn_df)
    """
    
    # Join transactions with attributes
    txn_attr = txn_df.merge(
        map_item_attr[['upc_nbr', 'attr_id', 'p_attr']],
        left_on='upc_nbr',
        right_on='upc_nbr',
        how='inner'
    )
    
    # Compute days ago from reference date
    if ref_date is None:
        ref_date = txn_attr['txn_date'].max()
    else:
        ref_date = pd.to_datetime(ref_date)
    
    txn_attr['days_ago'] = (ref_date - pd.to_datetime(txn_attr['txn_date'])).dt.days
    
    # Time-decayed weight
    txn_attr['weight'] = (
        txn_attr['item_qount'] * 
        txn_attr['p_attr'] * 
        np.exp(-txn_attr['days_ago'] / decay_days)
    )
    
    # Aggregate by household × attribute
    hh_attr_hist = txn_attr.groupby(['household_id', 'attr_id']).agg({
        'weight': 'sum',
        'days_ago': 'min'  # Most recent purchase
    }).reset_index()
    
    hh_attr_hist.columns = ['household_id', 'attr_id', 'hist_score', 'last_seen_days']
    
    # Filter out very low scores (noise reduction)
    hh_attr_hist = hh_attr_hist[hh_attr_hist['hist_score'] > 0.01]
    
    return hh_attr_hist


# Usage
hh_attr_history = build_hh_attr_history(txn_df, map_item_attr)
hh_attr_history.to_parquet('gs://your-bucket/data/hh_attr_history.parquet', index=False)