# run_provider.py

# run_provider.py (UPDATED)

import pandas as pd
import yaml
import time
from pathlib import Path

from attr_embed.cooc_ae import CoocAEProvider
from attr_embed.gcl import GCLProvider  # NEW

# Provider registry
PROVIDERS = {
    "cooc_ae_item": CoocAEProvider,
    "cooc_ae_hh": CoocAEProvider,
    "gcl": GCLProvider,  # NEW
}

def run_one_provider(name: str, model_version: str, config: dict, data_path: str, output_path: str):
    """Run a single provider."""
    
    print(f"\n{'='*60}")
    print(f"Running Provider: {name}")
    print(f"Version: {model_version}")
    print(f"{'='*60}\n")
    
    # Load inputs
    print("📂 Loading data...")
    inputs = {
        "attr_meta": pd.read_parquet(f"{data_path}/attr_meta.parquet"),
        "map_item_attribute": pd.read_parquet(f"{data_path}/map_item_attribute.parquet"),
        "hh_attr_history": pd.read_parquet(f"{data_path}/hh_attr_history.parquet"),
    }
    
    # For GCL, also load optional data
    if name.startswith("gcl"):
        try:
            inputs["item_nutrition"] = pd.read_parquet(f"{data_path}/item_nutrition.parquet")
            print("   ✅ Loaded item_nutrition for GCL")
        except:
            print("   ⚠️  item_nutrition not found (GCL will use basic features)")
    
    print(f"   attr_meta: {len(inputs['attr_meta'])} attributes")
    print(f"   map_item_attribute: {len(inputs['map_item_attribute'])} mappings")
    print(f"   hh_attr_history: {len(inputs['hh_attr_history'])} hh-attr pairs")
    
    # Initialize provider
    provider_class = PROVIDERS[name]
    provider = provider_class(config)
    
    # Train
    start_time = time.time()
    provider.fit(inputs)
    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")
    
    # Save
    metadata = {
        'provider': name,
        'model_version': model_version,
        'vocab_version': 'v1',
        'built_from': data_path,
        'created_at': pd.Timestamp.now()
    }
    
    provider.save(output_path, metadata)
    
    print(f"\n✅ {name} complete!\n")

def run_all_providers(config_path: str, data_path: str, output_path: str):
    """Run all providers defined in config."""
    
    # Load config
    with open(config_path) as f:
        spec = yaml.safe_load(f)
    
    # Run each provider
    for provider_spec in spec['providers']:
        run_one_provider(
            name=provider_spec['name'],
            model_version=provider_spec['model_version'],
            config=provider_spec['config'],
            data_path=data_path,
            output_path=output_path
        )

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "gs://your-recsys-bucket/data"  # Or local path for testing
    OUTPUT_PATH = "gs://your-recsys-bucket/models/attr_embeddings.parquet"
    CONFIG_PATH = "configs/providers.yaml"
    
    run_all_providers(CONFIG_PATH, DATA_PATH, OUTPUT_PATH)