"""
Train flood detection model on Bangladesh multi-location data
Simplified and robust training pipeline
"""
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from datetime import datetime
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir

# Create output directories
ensure_dir("outputs")
ensure_dir("outputs/models")
ensure_dir("outputs/predictions")
ensure_dir("outputs/metrics")

class BangladeshFloodDataset(Dataset):
    """Dataset for Bangladesh flood tiles"""
    
    def __init__(self, catalog_path='data/tiles/tile_catalog.json', mode='train'):
        # Load tile catalog
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
        
        self.tiles = catalog['tiles']
        self.locations = catalog['locations']
        
        # Filter SAR tiles (S1)
        self.tiles = [t for t in self.tiles if 'S1' in t['sensor']]
        
        # Split train/val (80/20)
        np.random.seed(42)
        np.random.shuffle(self.tiles)
        split_idx = int(len(self.tiles) * 0.8)
        
        if mode == 'train':
            self.tiles = self.tiles[:split_idx]
        else:
            self.tiles = self.tiles[split_idx:]
        
        print(f"{mode.upper()} dataset: {len(self.tiles)} tiles")
        locations = set(t['location'] for t in self.tiles)
        print(f"  Locations: {locations}")
        
        self.mode = mode
        
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        
        # Load tile
        tile = np.load(tile_info['path'])
        
        # Prepare input (ensure 3 channels)
        if len(tile.shape) == 2:
            # Single channel - replicate
            x = np.stack([tile, tile, tile], axis=0)
        elif len(tile.shape) == 3:
            if tile.shape[2] == 3:
                # Already 3 channels, transpose to CHW
                x = tile.transpose(2, 0, 1)
            elif tile.shape[2] == 2:
                # 2 channels (VV, VH) - add ratio as third
                vv, vh = tile[:,:,0], tile[:,:,1]
                ratio = np.clip(vv / (vh + 1e-6), 0, 2) / 2
                x = np.stack([vv, vh, ratio], axis=0)
            else:
                # Other number of channels - take first 3 or pad
                if tile.shape[2] >= 3:
                    x = tile[:,:,:3].transpose(2, 0, 1)
                else:
                    x = np.zeros((3, tile.shape[0], tile.shape[1]))
                    x[:tile.shape[2]] = tile.transpose(2, 0, 1)
        else:
            # Fallback
            x = np.stack([tile, tile, tile], axis=0)
        
        # Ensure float32 and proper range
        x = x.astype(np.float32)
        x = np.clip(x, 0, 1)
        
        # Create synthetic flood mask based on period and intensity
        h, w = x.shape[1:]
        
        if tile_info['period'] == 'flood':
            # High flood probability
            # Use SAR intensity to detect water (low backscatter)
            water_mask = x[0] < 0.3  # VV channel
            
            # Add some spatial coherence
            from scipy import ndimage
            water_mask = ndimage.binary_dilation(water_mask, iterations=2)
            water_mask = ndimage.binary_erosion(water_mask, iterations=1)
            flood_mask = water_mask.astype(np.float32)
            
        elif tile_info['period'] == 'pre':
            # Low flood probability
            water_mask = x[0] < 0.15  # Only very dark areas
            flood_mask = water_mask.astype(np.float32) * 0.5
            
        else:  # post
            # Medium flood probability
            water_mask = x[0] < 0.2
            flood_mask = water_mask.astype(np.float32) * 0.7
        
        # Convert to binary classification
        flood_mask = (flood_mask > 0.5).astype(np.long)
        
        # Data augmentation for training
        if self.mode == 'train' and np.random.rand() > 0.5:
            # Horizontal flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
                flood_mask = np.flip(flood_mask, axis=1).copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                flood_mask = np.flip(flood_mask, axis=0).copy()
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(flood_mask, dtype=torch.long)

def create_model(in_channels=3, num_classes=2):
    """Create segmentation model"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=(256, 128, 64, 32, 16)
    )
    return model

def train():
    """Main training function"""
    print("="*60)
    print("BANGLADESH FLOOD DETECTION TRAINING")
    print("="*60)
    
    # Check for scipy (needed for morphological operations)
    try:
        from scipy import ndimage
    except ImportError:
        print("Warning: scipy not found. Installing basic flood masks only.")
    
    # Create datasets
    train_dataset = BangladeshFloodDataset(mode='train')
    val_dataset = BangladeshFloodDataset(mode='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    print(f"\n🚀 Model initialized on {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Training loop
    print("\n📊 Starting training...")
    epochs = 20  # Quick training
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Record metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < metrics['best_val_loss']:
            metrics['best_val_loss'] = avg_val_loss
            metrics['best_epoch'] = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'metrics': metrics
            }, 'outputs/models/best_flood_model.pt')
            
            print(f"   ✓ Saved best model (loss: {avg_val_loss:.4f})")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, 'outputs/models/final_flood_model.pt')
    
    # Save metrics
    with open('outputs/metrics/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Best model: Epoch {metrics['best_epoch']} with loss {metrics['best_val_loss']:.4f}")
    print(f"✓ Models saved to: outputs/models/")
    print(f"✓ Metrics saved to: outputs/metrics/")
    
    # Generate sample predictions
    print("\n🎨 Generating sample predictions...")
    generate_predictions(model, val_dataset, device)

def generate_predictions(model, dataset, device):
    """Generate sample predictions"""
    model.eval()
    
    # Get a few samples
    samples = min(6, len(dataset))
    
    for i in range(samples):
        image, mask = dataset[i]
        
        # Add batch dimension
        x = image.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(x)
            pred = output.argmax(1)[0].cpu().numpy()
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        
        # Prepare visualization
        img = image.permute(1, 2, 0).numpy()
        
        # Create overlay
        overlay = img.copy()
        overlay[:,:,2] = np.where(pred > 0, 1.0, overlay[:,:,2])  # Blue for water
        overlay[:,:,0] = np.where(pred > 0, 0.3, overlay[:,:,0])  # Reduce red
        
        # Save
        tile_info = dataset.tiles[i]
        name = f"{tile_info['location']}_{tile_info['period']}_{i}"
        save_img(overlay, f"outputs/predictions/{name}.png")
    
    print(f"   ✓ Saved {samples} predictions to outputs/predictions/")

if __name__ == "__main__":
    # Check if catalog exists
    if not os.path.exists('data/tiles/tile_catalog.json'):
        print("❌ Error: tile_catalog.json not found!")
        print("   Please run preprocessing first: python 02_preprocess_bangladesh.py")
        sys.exit(1)
    
    # Run training
    train()