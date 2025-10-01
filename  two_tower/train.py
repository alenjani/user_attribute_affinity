# two_tower/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
from tqdm import tqdm

from .model import TwoTowerModel
from .dataset import create_dataloaders

def train_two_tower(
    hh_attr_history_path: str,
    attr_embeddings_path: str,
    provider: str,
    model_version: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Train two-tower model.
    
    Args:
        hh_attr_history_path: Path to household-attribute history
        attr_embeddings_path: Path to vector store (attr_embeddings.parquet)
        provider: Provider name to load (e.g., 'cooc_ae_item')
        model_version: Model version to load
        output_dir: Directory to save model and metrics
        config: Training configuration
    
    Returns:
        metrics: Dict of evaluation metrics
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"Training Two-Tower Model")
    print(f"Provider: {provider}")
    print(f"Model Version: {model_version}")
    print("="*60)
    
    # 1. Load data
    print("\n📂 Loading data...")
    hh_attr_history = pd.read_parquet(hh_attr_history_path)
    
    # Load attribute embeddings
    attr_emb_df = pd.read_parquet(attr_embeddings_path)
    attr_emb_df = attr_emb_df[
        (attr_emb_df['provider'] == provider) & 
        (attr_emb_df['model_version'] == model_version)
    ]
    
    attr_embeddings = {
        row['attr_id']: np.array(row['vector'])
        for _, row in attr_emb_df.iterrows()
    }
    
    attr_embed_dim = len(next(iter(attr_embeddings.values())))
    
    print(f"   Loaded {len(hh_attr_history)} hh-attr interactions")
    print(f"   Loaded {len(attr_embeddings)} attribute embeddings")
    print(f"   Attribute embedding dim: {attr_embed_dim}")
    
    # 2. Create dataloaders
    print("\n📊 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        hh_attr_history,
        attr_embeddings,
        train_ratio=config.get('train_ratio', 0.8),
        batch_size=config.get('batch_size', 256),
        negatives_per_pos=config.get('negatives_per_pos', 5)
    )
    
    # 3. Create model
    print("\n🏗️  Creating model...")
    model = TwoTowerModel(
        attr_embed_dim=attr_embed_dim,
        hh_hidden_dims=config.get('hh_hidden_dims', [128, 64]),
        output_dim=config.get('output_dim', 64),
        dropout=config.get('dropout', 0.2),
        freeze_attr_embeddings=config.get('freeze_attr_embeddings', True)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 0.001)
    )
    
    # 5. Training loop
    print("\n🚀 Training...")
    epochs = config.get('epochs', 10)
    best_val_loss = float('inf')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            hh_profile, pos_attr, neg_attrs = [x.to(device) for x in batch]
            
            # Forward pass
            logits = model(hh_profile, pos_attr, neg_attrs)
            
            # Loss: positive should be ranked first
            # Labels: first position (index 0) is positive
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hh_profile, pos_attr, neg_attrs = [x.to(device) for x in batch]
                
                logits = model(hh_profile, pos_attr, neg_attrs)
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()
                
                # Accuracy: is positive ranked first?
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
    
    # 6. Save final model and metrics
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    metrics = {
        'best_val_loss': best_val_loss,
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final_train_loss': history['train_loss'][-1]
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training history
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    
    print("\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final val acc: {val_acc:.4f}")
    print(f"   Saved to: {output_dir}")
    
    return metrics