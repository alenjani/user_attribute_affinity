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

def train_item_attribute_alignment(
    map_item_attr_path: str,
    frozen_item_model_path: str,        # NEW: path to frozen item embeddings
    attr_embeddings_path: str,
    provider: str,
    model_version: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Train item-attribute alignment model.
    
    Args:
        map_item_attr_path: Path to item-attribute mappings
        frozen_item_model_path: Path to previous HH-Item model (for frozen item embeddings)
        attr_embeddings_path: Path to vector store (attr_embeddings.parquet)
        provider: Provider name to load (e.g., 'gcl')
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
    map_item_attr = pd.read_parquet(map_item_attr_path)
    
    # Load frozen item embeddings (NEW)
    frozen_item_embeddings = load_frozen_item_embeddings(frozen_item_model_path, device=device)
    
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
    
    print(f"   Loaded {len(map_item_attr)} item-attr mappings")
    print(f"   Loaded {len(frozen_item_embeddings)} frozen item embeddings")
    print(f"   Loaded {len(attr_embeddings)} attribute embeddings")
    
    # 2. Create dataloaders
    print("\n📊 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        map_item_attr,
        frozen_item_embeddings,
        attr_embeddings,
        train_ratio=config.get('train_ratio', 0.8),
        batch_size=config.get('batch_size', 256),
        negatives_per_pos=config.get('negatives_per_pos', 5)
    )
    
    # 3. Create model (use new ItemAttributeAlignmentModel)
    print("\n🏗️  Creating model...")
    model = ItemAttributeAlignmentModel(
        item_embed_dim=64,  # Frozen embeddings dimension
        attr_embed_dim=64,  # Attribute embeddings dimension
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=64,
        dropout=config.get('dropout', 0.2)
    )
    
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