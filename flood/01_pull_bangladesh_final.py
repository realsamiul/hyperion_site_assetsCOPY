"""
Pull Bangladesh flood imagery - CORRECTED GEOMETRY VERSION
Fixed the geometry type error for proper downloads
"""
import os, json, logging, requests, numpy as np, ee
from datetime import datetime
import sys
sys.path.append('..')

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Your project
GCP_PROJECT = "hyperion-472805"

# Initialize Earth Engine
try:
    ee.Initialize(project=GCP_PROJECT)
    log.info(f"✓ Earth Engine initialized with project: {GCP_PROJECT}")
except Exception as e:
    log.error(f"Failed to initialize: {e}")
    sys.exit(1)

# Create directories
DATA_DIR = "../data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Working locations
LOCATIONS = [
    {
        "name": "Gaibandha_2020",
        "center": [89.543, 25.3297],
        "size": 0.3,
        "periods": {
            "pre": ["2020-06-01", "2020-06-30"],
            "flood": ["2020-07-15", "2020-08-15"],
            "post": ["2020-08-16", "2020-09-15"]
        }
    },
    {
        "name": "Sylhet_2024",
        "center": [91.8687, 24.8949],
        "size": 0.3,
        "periods": {
            "pre": ["2024-05-01", "2024-05-31"],
            "flood": ["2024-06-01", "2024-06-30"],
            "post": ["2024-07-01", "2024-07-31"]
        }
    }
]

class CorrectedBangladeshPull:
    def __init__(self):
        self.downloaded = []
        self.failed = []
        
    def create_aoi(self, center, size):
        """Create AOI - returns ee.Geometry object"""
        lon, lat = center
        half = size / 2
        
        # Create bounds
        bounds = [
            lon - half,  # west
            lat - half,  # south
            lon + half,  # east
            lat + half   # north
        ]
        
        # Return as ee.Geometry.Rectangle
        return ee.Geometry.Rectangle(bounds)
    
    def download_sar_data(self, location, period_name, dates):
        """Download SAR with corrected geometry handling"""
        try:
            log.info(f"  SAR {period_name}: {dates[0]} to {dates[1]}")
            
            # Create AOI as ee.Geometry
            aoi = self.create_aoi(location['center'], location['size'])
            
            # Build collection
            collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                         .filterBounds(aoi)
                         .filterDate(dates[0], dates[1])
                         .filter(ee.Filter.eq('instrumentMode', 'IW'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                         .select(['VV', 'VH']))
            
            count = collection.size().getInfo()
            log.info(f"    Found {count} images")
            
            if count == 0:
                log.warning(f"    No SAR images available")
                return False
            
            # Create composite and clip with ee.Geometry
            composite = collection.median().clip(aoi)
            
            # Simple preprocessing
            vv = composite.select('VV').unitScale(-25, 0)
            vh = composite.select('VH').unitScale(-30, -5)
            combined = ee.Image.cat([vv, vh]).rename(['VV', 'VH'])
            
            # Method 1: Try NPY download
            try:
                log.info(f"    Downloading as NPY...")
                
                # Get bounds for download - FIXED
                bounds = aoi.bounds()
                
                url = combined.getDownloadURL({
                    'scale': 30,
                    'region': bounds,  # Pass ee.Geometry directly
                    'format': 'NPY'
                })
                
                response = requests.get(url, timeout=180)
                
                if response.status_code == 200:
                    filename = f"S1_{location['name']}_{period_name}.npy"
                    filepath = os.path.join(RAW_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify
                    data = np.load(filepath, allow_pickle=True)
                    size_mb = os.path.getsize(filepath) / (1024*1024)
                    log.info(f"    ✓ Downloaded: {filename} ({size_mb:.1f} MB, shape: {data.shape})")
                    
                    self.downloaded.append(filepath)
                    return True
                else:
                    log.warning(f"    NPY download failed: HTTP {response.status_code}")
                    
            except Exception as e:
                log.warning(f"    NPY error: {str(e)[:100]}")
            
            # Method 2: Try GeoTIFF
            try:
                log.info(f"    Trying GeoTIFF format...")
                
                url = combined.getDownloadURL({
                    'scale': 30,
                    'region': aoi,
                    'format': 'GEO_TIFF'
                })
                
                response = requests.get(url, timeout=180)
                
                if response.status_code == 200:
                    filename = f"S1_{location['name']}_{period_name}.tif"
                    filepath = os.path.join(RAW_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    size_mb = os.path.getsize(filepath) / (1024*1024)
                    log.info(f"    ✓ Downloaded: {filename} ({size_mb:.1f} MB)")
                    
                    self.downloaded.append(filepath)
                    return True
                    
            except Exception as e:
                log.warning(f"    GeoTIFF error: {str(e)[:100]}")
            
            # Method 3: Get as PNG (always works)
            try:
                log.info(f"    Trying PNG visualization...")
                
                # Visualization parameters
                vis_params = {
                    'min': 0,
                    'max': 1,
                    'bands': ['VV']
                }
                
                # Get thumbnail
                url = combined.select('VV').getThumbURL({
                    'dimensions': 1024,
                    'region': aoi,
                    'format': 'png',
                    'min': 0,
                    'max': 1
                })
                
                response = requests.get(url, timeout=60)
                
                if response.status_code == 200:
                    filename = f"S1_{location['name']}_{period_name}.png"
                    filepath = os.path.join(RAW_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    log.info(f"    ✓ Downloaded: {filename} (visualization)")
                    self.downloaded.append(filepath)
                    return True
                    
            except Exception as e:
                log.warning(f"    PNG error: {str(e)[:100]}")
            
            self.failed.append(f"S1_{location['name']}_{period_name}")
            return False
            
        except Exception as e:
            log.error(f"    SAR error: {e}")
            self.failed.append(f"S1_{location['name']}_{period_name}")
            return False
    
    def download_optical_data(self, location, period_name, dates):
        """Download optical with corrected geometry"""
        try:
            log.info(f"  Optical {period_name}: {dates[0]} to {dates[1]}")
            
            # Create AOI
            aoi = self.create_aoi(location['center'], location['size'])
            
            # Build collection
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(dates[0], dates[1])
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                         .select(['B4', 'B3', 'B2', 'B8']))
            
            count = collection.size().getInfo()
            log.info(f"    Found {count} images")
            
            if count == 0:
                log.warning(f"    No optical images available")
                return False
            
            # Get best image
            if count == 1:
                image = collection.first()
            else:
                image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            # Process and clip
            processed = image.divide(10000).clip(aoi)
            
            # Add NDWI
            ndwi = processed.normalizedDifference(['B3', 'B8']).rename('NDWI')
            final = processed.addBands(ndwi)
            
            # Try download methods
            # Method 1: NPY
            try:
                log.info(f"    Downloading as NPY...")
                
                url = final.getDownloadURL({
                    'scale': 30,
                    'region': aoi,  # Pass ee.Geometry directly
                    'format': 'NPY'
                })
                
                response = requests.get(url, timeout=180)
                
                if response.status_code == 200:
                    filename = f"S2_{location['name']}_{period_name}.npy"
                    filepath = os.path.join(RAW_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    data = np.load(filepath, allow_pickle=True)
                    size_mb = os.path.getsize(filepath) / (1024*1024)
                    log.info(f"    ✓ Downloaded: {filename} ({size_mb:.1f} MB, shape: {data.shape})")
                    
                    self.downloaded.append(filepath)
                    return True
                    
            except Exception as e:
                log.warning(f"    NPY error: {str(e)[:100]}")
            
            # Method 2: RGB visualization
            try:
                log.info(f"    Trying RGB visualization...")
                
                rgb = processed.select(['B4', 'B3', 'B2'])
                
                url = rgb.getThumbURL({
                    'dimensions': 1024,
                    'region': aoi,
                    'format': 'png',
                    'min': 0,
                    'max': 0.3
                })
                
                response = requests.get(url, timeout=60)
                
                if response.status_code == 200:
                    filename = f"S2_{location['name']}_{period_name}_rgb.png"
                    filepath = os.path.join(RAW_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    log.info(f"    ✓ Downloaded: {filename} (RGB visualization)")
                    self.downloaded.append(filepath)
                    return True
                    
            except Exception as e:
                log.warning(f"    RGB error: {str(e)[:100]}")
            
            self.failed.append(f"S2_{location['name']}_{period_name}")
            return False
            
        except Exception as e:
            log.error(f"    Optical error: {e}")
            self.failed.append(f"S2_{location['name']}_{period_name}")
            return False
    
    def run(self):
        """Main execution"""
        log.info("="*60)
        log.info("BANGLADESH FLOOD IMAGERY ACQUISITION")
        log.info("="*60)
        
        # Process each location
        for location in LOCATIONS:
            log.info(f"\n📍 {location['name']}")
            log.info(f"   Center: {location['center']}")
            log.info(f"   Size: {location['size']}° (~{location['size']*111:.0f} km)")
            
            for period_name, dates in location['periods'].items():
                log.info(f"\n {period_name.upper()} Period:")
                
                # Download SAR
                self.download_sar_data(location, period_name, dates)
                
                # Download Optical
                self.download_optical_data(location, period_name, dates)
        
        # Summary
        log.info("\n" + "="*60)
        log.info("DOWNLOAD SUMMARY")
        log.info("="*60)
        
        if self.downloaded:
            log.info(f"✓ Successfully downloaded {len(self.downloaded)} files:")
            for fp in self.downloaded:
                log.info(f"  - {os.path.basename(fp)}")
        
        if self.failed:
            log.warning(f"\n⚠ Failed downloads: {len(self.failed)}")
            for name in self.failed:
                log.warning(f"  - {name}")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "project": GCP_PROJECT,
            "locations": LOCATIONS,
            "downloaded": self.downloaded,
            "failed": self.failed,
            "summary": {
                "total_attempts": len(self.downloaded) + len(self.failed),
                "successful": len(self.downloaded),
                "failed": len(self.failed)
            }
        }
        
        metadata_path = os.path.join(DATA_DIR, "acquisition.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"\n✓ Metadata saved: {metadata_path}")
        
        if self.downloaded:
            log.info("\n🎉 Success! Data ready for processing.")
            log.info("\nNext steps:")
            log.info("1. Check the ../data/raw/ folder for downloaded files")
            log.info("2. Run preprocessing script to prepare tiles")
        else:
            log.error("\n❌ No image files downloaded successfully.")

if __name__ == "__main__":
    puller = CorrectedBangladeshPull()
    puller.run()