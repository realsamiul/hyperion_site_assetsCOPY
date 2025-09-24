"""
Preprocess Bangladesh flood imagery
Creates tiles from downloaded NPY files
"""
import numpy as np
import glob
import os
import json
from pathlib import Path
import sys
sys.path.append('..')
from common.utils import ensure_dir, save_img

# Paths
RAW = "../data/raw"
OUT = "data/tiles"
PREVIEW = "data/previews"

# Create directories
ensure_dir(OUT)
ensure_dir(PREVIEW)

class BangladeshPreprocessor:
    def __init__(self):
        self.tile_size = 256
        self.overlap = 0.5
        self.all_tiles = []
        self.location_stats = {}
        
    def parse_filename(self, filename):
        """Extract info from filename like S1_Gaibandha_2020_pre.npy"""
        parts = Path(filename).stem.split('_')
        
        # Handle different filename formats
        if len(parts) >= 4:
            sensor = parts[0]
            location = f"{parts[1]}_{parts[2]}"
            period = parts[3]
        else:
            sensor = parts[0] if len(parts) > 0 else 'unknown'
            location = parts[1] if len(parts) > 1 else 'unknown'
            period = parts[2] if len(parts) > 2 else 'unknown'
            
        return {
            'sensor': sensor,
            'location': location,
            'period': period
        }
    
    def preprocess_sar(self, data):
        """Process SAR data"""
        # Handle structured array from Earth Engine
        if data.dtype.names:
            # It's a structured array
            vv = data['VV'].astype(np.float32) if 'VV' in data.dtype.names else None
            vh = data['VH'].astype(np.float32) if 'VH' in data.dtype.names else None
            
            if vv is not None:
                # Data is already normalized 0-1 from Earth Engine
                vv = np.clip(vv, 0, 1)
                
                if vh is not None:
                    vh = np.clip(vh, 0, 1)
                    # Create 3-channel image: VV, VH, VV/VH ratio
                    ratio = np.clip(vv / (vh + 1e-6), 0, 2) / 2
                    return np.stack([vv, vh, ratio], axis=-1)
                else:
                    # Single channel - replicate to 3 channels
                    return np.stack([vv, vv, vv], axis=-1)
        else:
            # Regular numpy array
            if len(data.shape) == 3 and data.shape[2] >= 2:
                # Has VV, VH channels
                vv = data[:,:,0].astype(np.float32)
                vh = data[:,:,1].astype(np.float32)
                ratio = np.clip(vv / (vh + 1e-6), 0, 2) / 2
                return np.stack([vv, vh, ratio], axis=-1)
            elif len(data.shape) == 2:
                # Single channel - replicate
                d = data.astype(np.float32)
                return np.stack([d, d, d], axis=-1)
            else:
                # Use first channel
                d = data[:,:,0].astype(np.float32) if len(data.shape) == 3 else data.astype(np.float32)
                return np.stack([d, d, d], axis=-1)
    
    def preprocess_optical(self, data):
        """Process optical data"""
        # Handle structured array
        if data.dtype.names:
            # Extract RGB bands
            if 'B4' in data.dtype.names:  # Red
                r = data['B4'].astype(np.float32)
                g = data['B3'].astype(np.float32) if 'B3' in data.dtype.names else r
                b = data['B2'].astype(np.float32) if 'B2' in data.dtype.names else r
                rgb = np.stack([r, g, b], axis=-1)
            else:
                # Fallback to first field
                d = data[data.dtype.names[0]].astype(np.float32)
                rgb = np.stack([d, d, d], axis=-1)
        else:
            # Regular array
            if len(data.shape) == 3 and data.shape[2] >= 3:
                # Take first 3 channels as RGB
                rgb = data[:,:,:3].astype(np.float32)
            elif len(data.shape) == 3 and data.shape[2] < 3:
                # Pad to 3 channels
                rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)
                rgb[:,:,:data.shape[2]] = data
            else:
                # Single channel - replicate to RGB
                d = data.astype(np.float32)
                rgb = np.stack([d, d, d], axis=-1)
        
        # Data from Earth Engine is already 0-1, but enhance contrast
        rgb = np.clip(rgb, 0, 1)
        
        # Enhance contrast
        for i in range(3):
            channel = rgb[:,:,i]
            # Only enhance if there's actual variation
            if channel.std() > 0.01:
                p2, p98 = np.percentile(channel, (2, 98))
                if p98 > p2:
                    rgb[:,:,i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
        
        return rgb
    
    def create_tiles(self, image, file_info):
        """Create tiles from image"""
        h, w = image.shape[:2]
        tiles = []
        
        step = int(self.tile_size * (1 - self.overlap))
        tile_count = 0
        
        for i in range(0, h - self.tile_size + 1, step):
            for j in range(0, w - self.tile_size + 1, step):
                tile = image[i:i+self.tile_size, j:j+self.tile_size]
                
                # Calculate quality metrics
                variance = tile.std()
                mean_val = tile.mean()
                
                # Skip low quality tiles
                if variance < 0.01:  # Too uniform
                    continue
                if mean_val < 0.05 or mean_val > 0.95:  # Too dark/bright
                    continue
                
                # Save tile
                tile_name = f"{file_info['sensor']}_{file_info['location']}_{file_info['period']}_{tile_count:03d}"
                tile_path = os.path.join(OUT, f"{tile_name}.npy")
                np.save(tile_path, tile)
                
                # Save preview for first few tiles
                if tile_count < 3:
                    preview_path = os.path.join(PREVIEW, f"{tile_name}.png")
                    # Ensure 3 channels for preview
                    if tile.shape[2] < 3:
                        preview_tile = np.zeros((tile.shape[0], tile.shape[1], 3))
                        preview_tile[:,:,:tile.shape[2]] = tile
                    else:
                        preview_tile = tile[:,:,:3]
                    save_img(preview_tile, preview_path)
                
                # Record tile info
                tile_info = {
                    'path': tile_path,
                    'name': tile_name,
                    'location': file_info['location'],
                    'period': file_info['period'],
                    'sensor': file_info['sensor'],
                    'variance': float(variance),
                    'mean': float(mean_val),
                    'position': [i, j],
                    'size': self.tile_size,
                    'channels': image.shape[2] if len(image.shape) > 2 else 1
                }
                
                tiles.append(tile_info)
                tile_count += 1
        
        return tiles
    
    def process_file(self, filepath):
        """Process a single NPY file"""
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        
        # Parse filename
        file_info = self.parse_filename(filename)
        
        try:
            # Load data
            data = np.load(filepath, allow_pickle=True)
            print(f"  Shape: {data.shape}, dtype: {data.dtype}")
            
            # Preprocess based on sensor
            if file_info['sensor'] == 'S1':
                processed = self.preprocess_sar(data)
            else:  # S2
                processed = self.preprocess_optical(data)
            
            print(f"  Processed shape: {processed.shape}")
            
            # Create tiles
            tiles = self.create_tiles(processed, file_info)
            print(f"  ✓ Generated {len(tiles)} quality tiles")
            
            # Update statistics
            location = file_info['location']
            if location not in self.location_stats:
                self.location_stats[location] = {
                    'total_tiles': 0,
                    'periods': set(),
                    'sensors': set(),
                    'files': []
                }
            
            self.location_stats[location]['total_tiles'] += len(tiles)
            self.location_stats[location]['periods'].add(file_info['period'])
            self.location_stats[location]['sensors'].add(file_info['sensor'])
            self.location_stats[location]['files'].append(filename)
            
            self.all_tiles.extend(tiles)
            
            return len(tiles)
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def create_summary_preview(self):
        """Create a summary preview image"""
        print("\n🎨 Creating summary previews...")
        
        for location, stats in self.location_stats.items():
            print(f"  {location}: {stats['total_tiles']} tiles")
            
            # Get sample tiles from this location
            location_tiles = [t for t in self.all_tiles if t['location'] == location]
            
            if location_tiles:
                # Get samples from different periods
                sample_tiles = []
                for period in ['pre', 'flood', 'post']:
                    period_tiles = [t for t in location_tiles if t['period'] == period]
                    if period_tiles:
                        sample_tiles.append(period_tiles[0])
                
                if sample_tiles:
                    # Create simple preview grid
                    n = len(sample_tiles)
                    grid_w = min(n, 3) * 256
                    grid_h = ((n + 2) // 3) * 256
                    grid = np.ones((grid_h, grid_w, 3), dtype=np.float32)
                    
                    for idx, tile_info in enumerate(sample_tiles):
                        row = idx // 3
                        col = idx % 3
                        
                        # Load tile
                        tile = np.load(tile_info['path'])
                        
                        # Ensure 3 channels for grid
                        if len(tile.shape) == 2:
                            tile_rgb = np.stack([tile, tile, tile], axis=-1)
                        elif tile.shape[2] < 3:
                            tile_rgb = np.zeros((tile.shape[0], tile.shape[1], 3), dtype=np.float32)
                            tile_rgb[:,:,:tile.shape[2]] = tile
                            # Fill remaining channels with mean
                            for c in range(tile.shape[2], 3):
                                tile_rgb[:,:,c] = tile.mean(axis=2) if len(tile.shape) > 2 else tile
                        else:
                            tile_rgb = tile[:,:,:3]
                        
                        # Ensure float and proper range
                        tile_rgb = np.clip(tile_rgb.astype(np.float32), 0, 1)
                        
                        # Place in grid
                        y1 = row * 256
                        y2 = y1 + 256
                        x1 = col * 256
                        x2 = x1 + 256
                        
                        if y2 <= grid_h and x2 <= grid_w:
                            grid[y1:y2, x1:x2] = tile_rgb
                    
                    # Save preview
                    preview_path = os.path.join(PREVIEW, f"{location}_summary.png")
                    save_img(grid, preview_path)
                    print(f"    ✓ Saved summary preview")
    
    def save_catalog(self):
        """Save tile catalog"""
        # Convert sets to lists for JSON
        for stats in self.location_stats.values():
            stats['periods'] = list(stats['periods'])
            stats['sensors'] = list(stats['sensors'])
        
        catalog = {
            'total_tiles': len(self.all_tiles),
            'locations': self.location_stats,
            'tiles': self.all_tiles,
            'tile_size': self.tile_size,
            'overlap': self.overlap
        }
        
        catalog_path = os.path.join(OUT, 'tile_catalog.json')
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f"\n✓ Saved tile catalog: {catalog_path}")
    
    def run(self):
        """Main processing pipeline"""
        print("="*60)
        print("BANGLADESH FLOOD PREPROCESSING")
        print("="*60)
        
        # Find all NPY files
        npy_files = glob.glob(os.path.join(RAW, "*.npy"))
        print(f"Found {len(npy_files)} files to process")
        
        # Process each file
        total_tiles = 0
        for filepath in npy_files:
            n_tiles = self.process_file(filepath)
            total_tiles += n_tiles
        
        # Create previews
        try:
            self.create_summary_preview()
        except Exception as e:
            print(f"Warning: Could not create all previews: {e}")
        
        # Save catalog
        self.save_catalog()
        
        # Summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"✓ Total tiles created: {total_tiles}")
        print(f"✓ Locations processed: {len(self.location_stats)}")
        
        for location, stats in self.location_stats.items():
            print(f"\n  {location}:")
            print(f"    Tiles: {stats['total_tiles']}")
            print(f"    Periods: {', '.join(sorted(stats['periods']))}")
            print(f"    Sensors: {', '.join(sorted(stats['sensors']))}")
        
        if total_tiles > 0:
            print(f"\n✅ Success! Tiles saved to: {OUT}")
            print(f"   Preview images in: {PREVIEW}")
            print(f"   Catalog: {OUT}/tile_catalog.json")
            print("\n🎯 Ready for training!")
            print("   Next: python 03_train_flood_model.py")
        else:
            print("\n⚠️ No tiles were created. Check the input data.")

if __name__ == "__main__":
    preprocessor = BangladeshPreprocessor()
    preprocessor.run()