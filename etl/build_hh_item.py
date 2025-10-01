# etl/build_hh_item.py

def build_hh_item(txn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to match design contract.
    """
    hh_item = txn_df.rename(columns={
        'upc_nbr': 'upc_id',  # Standardize to upc_id
        'txn_date': 'ts'
    }).copy()
    
    # Ensure types
    hh_item['ts'] = pd.to_datetime(hh_item['ts'])
    
    # Optional: Add basket_id if txn_id represents basket
    # If txn_id is transaction (not basket), you may need to group by (household_id, date)
    hh_item['basket_id'] = hh_item['txn_id']  # Or create composite: household_id + date
    
    # Select final columns
    hh_item = hh_item[['household_id', 'upc_id', 'basket_id', 'ts', 'item_qount']]
    hh_item = hh_item.rename(columns={'item_qount': 'quantity'})
    
    return hh_item


# Usage
txn_df = bq.read_from_gcp_table("SELECT ... FROM gcp_txn_table")
hh_item = build_hh_item(txn_df)
hh_item.to_parquet('gs://your-bucket/data/hh_item.parquet', index=False)