# compare_providers.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compare_embeddings(
    attr_embeddings_path: str,
    providers: list = ["cooc_ae_item", "cooc_ae_hh", "gcl"]
):
    """
    Compare embeddings from different providers.
    
    Metrics:
    1. Embedding similarity (how different are the embeddings?)
    2. Neighborhood coherence (are neighbors in same attribute family?)
    3. Embedding distribution (normalized? clustered?)
    """
    
    print("="*60)
    print("Comparing Attribute Embedding Providers")
    print("="*60)
    
    # Load all embeddings
    attr_emb_df = pd.read_parquet(attr_embeddings_path)
    
    provider_embeddings = {}
    for provider in providers:
        provider_df = attr_emb_df[attr_emb_df['provider'] == provider]
        
        if len(provider_df) == 0:
            print(f"⚠️  No embeddings found for provider: {provider}")
            continue
        
        # Get latest version
        latest_version = provider_df['model_version'].max()
        provider_df = provider_df[provider_df['model_version'] == latest_version]
        
        embeddings = {
            row['attr_id']: np.array(row['vector'])
            for _, row in provider_df.iterrows()
        }
        
        provider_embeddings[provider] = embeddings
        print(f"✅ Loaded {len(embeddings)} embeddings from {provider}")
    
    if len(provider_embeddings) < 2:
        print("❌ Need at least 2 providers to compare")
        return
    
    # Find common attributes across all providers
    common_attrs = set.intersection(*[set(embs.keys()) for embs in provider_embeddings.values()])
    print(f"\n📊 Common attributes across providers: {len(common_attrs)}")
    
    # 1. Embedding Similarity Analysis
    print("\n" + "="*60)
    print("1. EMBEDDING SIMILARITY (How different are the providers?)")
    print("="*60)
    
    similarity_matrix = pd.DataFrame(
        index=providers,
        columns=providers,
        dtype=float
    )
    
    for i, prov1 in enumerate(providers):
        if prov1 not in provider_embeddings:
            continue
        for j, prov2 in enumerate(providers):
            if prov2 not in provider_embeddings:
                continue
            
            if i <= j:
                # Compute average cosine similarity between embeddings
                similarities = []
                for attr in common_attrs:
                    emb1 = provider_embeddings[prov1][attr]
                    emb2 = provider_embeddings[prov2][attr]
                    
                    # Cosine similarity
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                similarity_matrix.loc[prov1, prov2] = avg_sim
                similarity_matrix.loc[prov2, prov1] = avg_sim
    
    print("\nAverage Cosine Similarity Between Providers:")
    print(similarity_matrix)
    print("\nInterpretation:")
    print("  1.0 = Identical embeddings")
    print("  0.0 = Orthogonal embeddings")
    print("  Typical range: 0.3-0.7 (providers capture different but related patterns)")
    
    # 2. Neighborhood Coherence Analysis
    print("\n" + "="*60)
    print("2. NEIGHBORHOOD COHERENCE (Are similar attributes close?)")
    print("="*60)
    
    # Load attribute metadata for families
    attr_meta_path = Path(attr_embeddings_path).parent.parent / "data" / "attr_meta.parquet"
    
    if attr_meta_path.exists():
        attr_meta = pd.read_parquet(attr_meta_path)
        attr_families = attr_meta.set_index('attr_id')['attr_family'].to_dict()
        
        coherence_scores = {}
        
        for provider, embeddings in provider_embeddings.items():
            # For each attribute, find k nearest neighbors
            k = 10
            same_family_count = 0
            total_count = 0
            
            attr_list = list(common_attrs)
            emb_matrix = np.stack([embeddings[attr] for attr in attr_list])
            
            # Compute pairwise similarities
            similarities = np.dot(emb_matrix, emb_matrix.T)
            
            for i, attr in enumerate(attr_list):
                if attr not in attr_families:
                    continue
                
                attr_family = attr_families[attr]
                
                # Get top-k neighbors (excluding self)
                neighbor_indices = np.argsort(similarities[i])[::-1][1:k+1]
                neighbors = [attr_list[idx] for idx in neighbor_indices]
                
                # Count how many are same family
                for neighbor in neighbors:
                    if neighbor in attr_families:
                        if attr_families[neighbor] == attr_family:
                            same_family_count += 1
                        total_count += 1
            
            coherence = same_family_count / total_count if total_count > 0 else 0
            coherence_scores[provider] = coherence
        
        print("\nNeighborhood Coherence (% same family in top-10 neighbors):")
        for provider, score in coherence_scores.items():
            print(f"  {provider:20s}: {score:.2%}")
        
        print("\nInterpretation:")
        print("  Higher = better (similar attributes are closer)")
        print("  Baseline (random): ~10-20% (depends on family distribution)")
        print("  Good: >40%")
    else:
        print("⚠️  attr_meta.parquet not found, skipping coherence analysis")
    
    # 3. Embedding Distribution Analysis
    print("\n" + "="*60)
    print("3. EMBEDDING DISTRIBUTION (Are embeddings well-distributed?)")
    print("="*60)
    
    for provider, embeddings in provider_embeddings.items():
        emb_matrix = np.stack([embeddings[attr] for attr in common_attrs])
        
        # Compute statistics
        norms = np.linalg.norm(emb_matrix, axis=1)
        mean_norm = norms.mean()
        std_norm = norms.std()
        
        # Pairwise distances
        distances = []
        sample_size = min(1000, len(common_attrs))
        sample_attrs = np.random.choice(list(common_attrs), sample_size, replace=False)
        
        for attr in sample_attrs:
            emb = embeddings[attr]
            other_attrs = [a for a in sample_attrs if a != attr]
            other_embs = np.stack([embeddings[a] for a in other_attrs[:100]])
            
            dists = np.linalg.norm(other_embs - emb, axis=1)
            distances.extend(dists)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        print(f"\n{provider}:")
        print(f"  Embedding norms: {mean_norm:.3f} ± {std_norm:.3f}")
        print(f"  Pairwise distances: {mean_dist:.3f} ± {std_dist:.3f}")
        print(f"  Interpretation:")
        if mean_norm < 0.95 or mean_norm > 1.05:
            print(f"    ⚠️  Norms not normalized (should be ~1.0)")
        else:
            print(f"    ✅ Well-normalized embeddings")
        
        if std_dist / mean_dist < 0.2:
            print(f"    ⚠️  Low variance (embeddings too clustered)")
        else:
            print(f"    ✅ Good distribution spread")
    
    # 4. Visualization
    print("\n" + "="*60)
    print("4. GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix.astype(float),
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        square=True
    )
    plt.title('Provider Embedding Similarity')
    plt.tight_layout()
    plt.savefig(output_dir / 'provider_similarity.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'provider_similarity.png'}")
    
    # Plot 2: Neighborhood coherence bar chart
    if 'coherence_scores' in locals():
        plt.figure(figsize=(10, 6))
        providers_sorted = sorted(coherence_scores.keys())
        scores = [coherence_scores[p] for p in providers_sorted]
        
        plt.bar(providers_sorted, scores)
        plt.ylabel('Coherence Score')
        plt.title('Neighborhood Coherence by Provider')
        plt.ylim(0, 1)
        plt.axhline(y=0.4, color='r', linestyle='--', label='Good threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'coherence_comparison.png', dpi=150)
        print(f"✅ Saved: {output_dir / 'coherence_comparison.png'}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    
    # Return summary
    return {
        'similarity_matrix': similarity_matrix,
        'coherence_scores': coherence_scores if 'coherence_scores' in locals() else None
    }


if __name__ == "__main__":
    MODELS_PATH = "gs://your-recsys-bucket/models"
    ATTR_EMBEDDINGS_PATH = f"{MODELS_PATH}/attr_embeddings.parquet"
    
    results = compare_embeddings(
        attr_embeddings_path=ATTR_EMBEDDINGS_PATH,
        providers=["cooc_ae_item", "cooc_ae_hh", "gcl"]
    )