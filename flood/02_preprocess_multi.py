"""
Advanced Preprocessing for Multiple Flood Locations
Handles multiple areas with intelligent tiling and quality filtering
"""
import numpy as np, glob, os, json
from pathlib import Path
import sys
sys.path.append('..')
from common.utils import ensure_dir, save_img
import cv2
from typing import Dict, List

RAW = "../data/raw"
OUT = "data/tiles"
METADATA = "../data/acquisition.json"

ensure_dir(OUT)
ensure_dir("data/previews")

class MultiLocationPreprocessor:
    def __init__(self):
        self.metadata = self.load_metadata()
        self.location_stats = {}
        self.all_tiles = []
        
    def load_metadata(self) -> Dict:
        """Load acquisition metadata to understand data structure"""
        if os.path.exists(METADATA):
            with open(METADATA, 'r') as f:
                return json.load(f)
        return {}
    
    def preprocess_sar(self, data: np.ndarray) -> np.ndarray:
        """Advanced SAR preprocessing"""
        if len(data.shape) == 3:
            # If we have VV, VH, water_prob
            vv = data[:,:,0] if data.shape[2] > 0 else data
            vh = data[:,:,1] if data.shape[2] > 1 else None
        else:
            vv = data
            vh = None
        
        # Convert to dB and normalize (your proven method)
        vv_db = np.clip((10*np.log10(vv+1e-6)+25)/30, 0, 1)
        
        if vh is not None:
            vh_db = np.clip((10*np.log10(vh+1e-6)+30)/35, 0, 1)
            # Create false color composite
            ratio = np.clip(vv_db / (vh_db + 1e-6), 0, 2) / 2
            return np.stack([vv_db, vh_db, ratio], axis=-1)
        
        return vv_db
    
    def preprocess_optical(self, data: np.ndarray) -> np.ndarray:
        """Process S2 optical with water indices"""
        if len(data.shape) == 3:
            # Expecting B4,B3,B2,B8,B11,B12,NDWI,MNDWI
            rgb = data[:,:,:3] if data.shape[2] >= 3 else data
            
            # Enhance contrast for visualization
            for i in range(min(3, data.shape[2])):
                p2, p98 = np.percentile(rgb[:,:,i], (2, 98))
                rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2), 0, 1)
            
            # If we have water indices, create enhanced visualization
            if data.shape[2] >= 7:  # Has NDWI
                ndwi = data[:,:,6]
                water_mask = ndwi > 0.3
                # Highlight water in blue
                rgb[:,:,2] = np.where(water_mask, 1.0, rgb[:,:,2])
                rgb[:,:,0] = np.where(water_mask, 0.3, rgb[:,:,0])
            
            return rgb
        return data
    
    def create_smart_tiles(self, image: np.ndarray, location: str, 
                          period: str, sensor: str) -> List[Dict]:
        """Create tiles with intelligent selection"""
        h, w = image.shape[:2]
        tiles = []
        
        # Adaptive tile size based on image dimensions
        if h < 512 or w < 512:
            tile_size = 256
            step = 128  # 50% overlap
        else:
            tile_size = 512
            step = 256
        
        print(f"   Image {h}x{w} → tiles {tile_size}x{tile_size}")
        
        tile_id = 0
        for i in range(0, h - tile_size + 1, step):
            for j in range(0, w - tile_size + 1, step):
                tile = image[i:i+tile_size, j:j+tile_size]
                
                # Quality metrics
                if len(tile.shape) == 2:
                    variance = tile.std()
                    mean_val = tile.mean()
                    # Water detection heuristic for SAR
                    water_score = np.sum(tile < 0.3) / tile.size
                else:
                    variance = tile.mean(axis=2).std()
                    mean_val = tile.mean()
                    water_score = 0
                
                # Skip low-quality tiles
                if variance < 0.01:  # Too uniform
                    continue
                if mean_val < 0.05 or mean_val > 0.95:  # Too dark/bright
                    continue
                
                # Save tile
                tile_name = f"{sensor}_{location}_{period}_{tile_id:03d}"
                tile_path = f"{OUT}/{tile_name}.npy"
                np.save(tile_path, tile)
                
                # Save preview for first few tiles
                if tile_id < 3:
                    if len(tile.shape) == 2:
                        save_img(tile, f"data/previews/{tile_name}.png", cmap="gray")
                    else:
                        save_img(tile, f"data/previews/{tile_name}.png")
                
                tiles.append({
                    'path': tile_path,
                    'location': location,
                    'period': period,
                    'sensor': sensor,
                    'variance': float(variance),
                    'water_score': float(water_score),
                    'position': [i, j],
                    'size': tile_size
                })
                
                tile_id += 1
        
        return tiles
    
    def process_location_files(self):
        """Process all downloaded files from multiple locations"""
        
        # Get all NPY files
        all_files = glob.glob(f"{RAW}/*.npy")
        print(f"Found {len(all_files)} raw files to process")
        
        # Group by location from filename
        location_files = {}
        for fp in all_files:
            fn = Path(fp).stem
            
            # Parse filename (e.g., "S1_Sylhet_2022_Extreme_pre")
            parts = fn.split('_')
            sensor = parts[0]  # S1, S2, or L89
            
            # Extract location (might be multi-word)
            if 'pre' in fn:
                period = 'pre'
            elif 'flood' in fn:
                period = 'flood'
            elif 'post' in fn:
                period = 'post'
            else:
                period = 'unknown'
            
            # Location is everything between sensor and period
            location_parts = []
            for part in parts[1:]:
                if part in ['pre', 'flood', 'post']:
                    break
                location_parts.append(part)
            location = '_'.join(location_parts) if location_parts else 'default'
            
            if location not in location_files:
                location_files[location] = []
            
            location_files[location].append({
                'path': fp,
                'sensor': sensor,
                'period': period,
                'filename': fn
            })
        
        # Process each location
        for location, files in location_files.items():
            print(f"\n📍 Processing location: {location}")
            print(f"   Files: {len(files)}")
            
            location_tiles = []
            
            for file_info in files:
                print(f"\n   Processing {file_info['filename']}...")
                
                # Load data
                data = np.load(file_info['path'], allow_pickle=True)
                print(f"     Shape: {data.shape}, dtype: {data.dtype}")
                
                # Preprocess based on sensor type
                if file_info['sensor'] == 'S1':
                    processed = self.preprocess_sar(data)
                elif file_info['sensor'] in ['S2', 'L89']:
                    processed = self.preprocess_optical(data)
                else:
                    processed = data
                
                # Create tiles
                tiles = self.create_smart_tiles(
                    processed, 
                    location,
                    file_info['period'],
                    file_info['sensor']
                )
                
                location_tiles.extend(tiles)
                print(f"     ✓ Generated {len(tiles)} quality tiles")
            
            # Store location statistics
            self.location_stats[location] = {
                'total_tiles': len(location_tiles),
                'periods': list(set(t['period'] for t in location_tiles)),
                'sensors': list(set(t['sensor'] for t in location_tiles)),
                'avg_variance': np.mean([t['variance'] for t in location_tiles]),
                'water_tiles': sum(1 for t in location_tiles if t['water_score'] > 0.3)
            }
            
            self.all_tiles.extend(location_tiles)
        
        return self.all_tiles
    
    def create_composite_previews(self):
        """Create location-based composite previews"""
        print("\n🎨 Creating composite previews...")
        
        for location, stats in self.location_stats.items():
            print(f"   {location}: {stats['total_tiles']} tiles")
            
            # Create a grid preview for each location
            location_tiles = [t for t in self.all_tiles if t['location'] == location]
            
            if not location_tiles:
                continue
            
            # Sample tiles for preview (max 9 for 3x3 grid)
            sample_tiles = location_tiles[:9]
            grid_size = int(np.ceil(np.sqrt(len(sample_tiles))))
            
            # Create grid
            tile_size = 256
            grid = np.zeros((grid_size * tile_size, grid_size * tile_size, 3))
            
            for idx, tile_info in enumerate(sample_tiles):
                tile = np.load(tile_info['path'])
                
                # Ensure 3 channels for grid
                if len(tile.shape) == 2:
                    tile = np.stack([tile]*3, axis=-1)
                elif tile.shape[2] > 3:
                    tile = tile[:,:,:3]
                
                # Resize if needed
                if tile.shape[0] != tile_size:
                    tile = cv2.resize(tile, (tile_size, tile_size))
                
                # Place in grid
                row = idx // grid_size
                col = idx % grid_size
                grid[row*tile_size:(row+1)*tile_size, 
                     col*tile_size:(col+1)*tile_size] = tile
            
            # Save grid
            save_img(grid, f"data/previews/{location}_grid.png")
    
    def save_metadata(self):
        """Save comprehensive preprocessing metadata"""
        metadata = {
            'timestamp': os.path.getmtime(__file__),
            'total_tiles': len(self.all_tiles),
            'locations': self.location_stats,
            'tile_distribution': {
                'by_period': {},
                'by_sensor': {},
                'by_location': {}
            }
        }
        
        # Calculate distributions
        for tile in self.all_tiles:
            period = tile['period']
            sensor = tile['sensor']
            location = tile['location']
            
            metadata['tile_distribution']['by_period'][period] = \
                metadata['tile_distribution']['by_period'].get(period, 0) + 1
            metadata['tile_distribution']['by_sensor'][sensor] = \
                metadata['tile_distribution']['by_sensor'].get(sensor, 0) + 1
            metadata['tile_distribution']['by_location'][location] = \
                metadata['tile_distribution']['by_location'].get(location, 0) + 1
        
        # Save tile catalog
        with open('data/tile_catalog.json', 'w') as f:
            json.dump({
                'metadata': metadata,
                'tiles': self.all_tiles
            }, f, indent=2)
        
        print(f"\n📊 Preprocessing Summary:")
        print(f"   Total tiles: {len(self.all_tiles)}")
        print(f"   Locations: {len(self.location_stats)}")
        for loc, stats in self.location_stats.items():
            print(f"     {loc}: {stats['total_tiles']} tiles, "
                  f"water detected in {stats['water_tiles']} tiles")

# Run preprocessing
if __name__ == "__main__":
    preprocessor = MultiLocationPreprocessor()
    tiles = preprocessor.process_location_files()
    preprocessor.create_composite_previews()
    preprocessor.save_metadata()
    print("\n✅ Multi-location preprocessing complete!")