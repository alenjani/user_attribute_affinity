# run_inference.py

from two_tower.score import score_household_attribute_affinity

def main():
    # Paths
    DATA_PATH = "gs://your-recsys-bucket/data"
    MODELS_PATH = "gs://your-recsys-bucket/models"
    
    PROVIDER = "cooc_ae_item"
    MODEL_VERSION = "cooc_ae_item_2025-09-30"
    
    MODEL_PATH = f"{MODELS_PATH}/two_tower_{PROVIDER}/best_model.pt"
    HH_ATTR_HISTORY_PATH = f"{DATA_PATH}/hh_attr_history.parquet"
    ATTR_EMBEDDINGS_PATH = f"{MODELS_PATH}/attr_embeddings.parquet"
    OUTPUT_PATH = f"{MODELS_PATH}/hh_attribute_affinity_{PROVIDER}.parquet"
    
    # Run scoring
    results_df = score_household_attribute_affinity(
        model_path=MODEL_PATH,
        hh_attr_history_path=HH_ATTR_HISTORY_PATH,
        attr_embeddings_path=ATTR_EMBEDDINGS_PATH,
        provider=PROVIDER,
        model_version=MODEL_VERSION,
        output_path=OUTPUT_PATH,
        top_k=50,
        batch_size=1024
    )
    
    # Sample results
    print("\n📋 Sample Results:")
    print(results_df.head(20))

if __name__ == "__main__":
    main()