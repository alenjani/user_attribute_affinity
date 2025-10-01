# etl/run_etl.py

from google.cloud import bigquery
import pandas as pd

from etl.build_attr_meta import build_attr_meta
from etl.build_map_item_attribute import build_map_item_attribute
from etl.build_hh_item import build_hh_item
from etl.build_hh_attr_history import build_hh_attr_history
from etl.build_item_text import build_item_text
from etl.build_item_nutrition import build_item_nutrition

def run_full_etl(output_bucket: str):
    """Full ETL pipeline."""
    
    bq_client = bigquery.Client()
    
    print("=" * 60)
    print("ETL PIPELINE")
    print("=" * 60)
    
    # Load from BigQuery
    print("\n📂 Loading from BigQuery...")
    
    item_df = bq_client.query("""
        SELECT 
            upc_nbr, item_dsc, smic_class_cd, smic_category_cd,
            department_nm, shelf_nm, health_clain_type_dsc, 
            nutrition_nm, iri_brands_nm
        FROM `your_project.your_dataset.gcp_item_table`
    """).to_dataframe()
    
    txn_df = bq_client.query("""
        SELECT 
            household_id, upc_nbr, txn_id, txn_date, item_qount
        FROM `your_project.your_dataset.gcp_txn_table`
        WHERE txn_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
    """).to_dataframe()
    
    household_df = bq_client.query("""
        SELECT household_id, age_group, segment, income_group
        FROM `your_project.your_dataset.gcp_hhs_table`
    """).to_dataframe()
    
    print(f"   Items: {len(item_df):,}")
    print(f"   Transactions: {len(txn_df):,}")
    print(f"   Households: {len(household_df):,}")
    
    # Build tables
    print("\n🔨 Building tables...")
    
    attr_meta = build_attr_meta(item_df)
    attr_meta.to_parquet(f'gs://{output_bucket}/data/attr_meta.parquet')
    print(f"   ✅ attr_meta: {len(attr_meta):,} attributes")
    
    map_item_attr = build_map_item_attribute(item_df)
    map_item_attr.to_parquet(f'gs://{output_bucket}/data/map_item_attribute.parquet')
    print(f"   ✅ map_item_attribute: {len(map_item_attr):,} mappings")
    
    hh_item = build_hh_item(txn_df)
    hh_item.to_parquet(f'gs://{output_bucket}/data/hh_item.parquet')
    print(f"   ✅ hh_item: {len(hh_item):,} transactions")
    
    hh_attr_history = build_hh_attr_history(txn_df, map_item_attr)
    hh_attr_history.to_parquet(f'gs://{output_bucket}/data/hh_attr_history.parquet')
    print(f"   ✅ hh_attr_history: {len(hh_attr_history):,} interactions")
    
    item_nutrition = build_item_nutrition(item_df)
    item_nutrition.to_parquet(f'gs://{output_bucket}/data/item_nutrition.parquet')
    print(f"   ✅ item_nutrition: {len(item_nutrition):,} items")
    
    household_df.to_parquet(f'gs://{output_bucket}/data/hh_meta.parquet')
    print(f"   ✅ hh_meta: {len(household_df):,} households")
    
    print("\n✅ ETL COMPLETE")
    print(f"   Output: gs://{output_bucket}/data/")

if __name__ == "__main__":
    run_full_etl(output_bucket='your-recsys-bucket')