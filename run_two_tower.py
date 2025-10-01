# run_two_tower.py (UPDATED)

import yaml
import argparse
from pathlib import Path
from two_tower.train import train_two_tower

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', type=str, default='cooc_ae_item',
                       choices=['cooc_ae_item', 'cooc_ae_hh', 'gcl'])
    parser.add_argument('--data_path', type=str, default='gs://your-recsys-bucket/data')
    parser.add_argument('--models_path', type=str, default='gs://your-recsys-bucket/models')
    args = parser.parse_args()
    
    # Paths
    HH_ATTR_HISTORY_PATH = f"{args.data_path}/hh_attr_history.parquet"
    ATTR_EMBEDDINGS_PATH = f"{args.models_path}/attr_embeddings.parquet"
    
    # Get model version from providers.yaml
    with open("configs/providers.yaml") as f:
        providers_config = yaml.safe_load(f)
    
    model_version = None
    for p in providers_config['providers']:
        if p['name'] == args.provider:
            model_version = p['model_version']
            break
    
    if model_version is None:
        raise ValueError(f"Provider {args.provider} not found in providers.yaml")
    
    OUTPUT_DIR = f"{args.models_path}/two_tower_{args.provider}"
    
    # Load config
    with open("configs/two_tower.yaml") as f:
        config = yaml.safe_load(f)['training']
    
    # Train
    print(f"\n🚀 Training Two-Tower with provider: {args.provider}")
    print(f"   Model version: {model_version}")
    
    metrics = train_two_tower(
        hh_attr_history_path=HH_ATTR_HISTORY_PATH,
        attr_embeddings_path=ATTR_EMBEDDINGS_PATH,
        provider=args.provider,
        model_version=model_version,
        output_dir=OUTPUT_DIR,
        config=config
    )
    
    print("\n" + "="*60)
    print("FINAL METRICS:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()