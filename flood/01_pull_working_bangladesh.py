"""
Pull Bangladesh flood imagery using discovered available dates
"""
import os, json, logging, requests, numpy as np, ee
from typing import Dict
import sys
sys.path.append('..')
from common.config import GCP_PROJECT, ASSET_DIR, SCALE, BANGLADESH_FLOODS
from common.utils import ensure_dir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Initialize Earth Engine
try:
    ee.Initialize(project=GCP_PROJECT)
    log.info(f"Earth Engine initialised ({GCP_PROJECT})")
except Exception:
    log.error("Run earthengine authenticate then re-run")
    raise

# Use the updated config with discovered imagery
VERIFIED_FLOODS = BANGLADESH_FLOODS

# Data paths
DATA_BASE = os.path.join(ASSET_DIR, "../data")
ensure_dir(os.path.join(DATA_BASE, "raw"))

class WorkingBangladeshPull:
    def __init__(self):
        self.dl = []
        self.scale = SCALE or 10
        
    def pull_imagery(self, location):
        """Pull imagery for a verified location"""
        log.info(f"\n📍 Processing: {location['name']}")
        
        aoi = ee.Geometry.Rectangle(location['aoi'])
        results = {}
        
        for period_name, period in location['periods'].items():
            log.info(f"\n--- {period_name.upper()} PERIOD ---")
            log.info(f"   Dates: {period['start']} to {period['end']}")
            
            # Try to get Sentinel-1
            try:
                s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filterBounds(aoi) \
                    .filterDate(period['start'], period['end']) \
                    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                    .select(['VV', 'VH'])
                
                s1_count = s1.size().getInfo()
                log.info(f"   Found {s1_count} Sentinel-1 images")
                
                if s1_count > 0:
                    # Create composite
                    composite = s1.median().clip(aoi)
                    
                    # Normalize
                    normalized = composite.unitScale(-25, 5)
                    
                    # Download
                    url = normalized.getDownloadURL({
                        "scale": self.scale,
                        "crs": "EPSG:4326",
                        "region": aoi,
                        "format": "NPY"
                    })
                    
                    filename = f"S1_{location['name']}_{period_name}.npy"
                    filepath = os.path.join(DATA_BASE, "raw", filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(requests.get(url).content)
                    
                    # Verify
                    data = np.load(filepath, allow_pickle=True)
                    log.info(f"   ✓ SAR saved: {filename} (shape: {data.shape})")
                    self.dl.append(filepath)
                    
            except Exception as e:
                log.warning(f"   SAR failed: {e}")
            
            # Try to get Sentinel-2
            try:
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(aoi) \
                    .filterDate(period['start'], period['end']) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                
                s2_count = s2.size().getInfo()
                log.info(f"   Found {s2_count} Sentinel-2 images")
                
                if s2_count > 0:
                    # Get best image (least clouds)
                    best = s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
                    
                    # Select bands and normalize
                    rgbn = best.select(['B4','B3','B2','B8']).divide(10000).clip(aoi)
                    
                    # Add water index
                    ndwi = rgbn.normalizedDifference(['B3', 'B8']).rename('NDWI')
                    combined = rgbn.addBands(ndwi)
                    
                    # Download
                    url = combined.getDownloadURL({
                        "scale": self.scale,
                        "crs": "EPSG:4326",
                        "region": aoi,
                        "format": "NPY"
                    })
                    
                    filename = f"S2_{location['name']}_{period_name}.npy"
                    filepath = os.path.join(DATA_BASE, "raw", filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(requests.get(url).content)
                    
                    log.info(f"   ✓ Optical saved: {filename}")
                    self.dl.append(filepath)
                    
            except Exception as e:
                log.warning(f"   Optical failed: {e}")
            
            results[period_name] = {
                "s1_downloaded": f"S1_{location['name']}_{period_name}.npy" in [os.path.basename(f) for f in self.dl],
                "s2_downloaded": f"S2_{location['name']}_{period_name}.npy" in [os.path.basename(f) for f in self.dl]
            }
        
        return results
    
    def run(self):
        """Pull all available imagery"""
        log.info("="*60)
        log.info("BANGLADESH FLOOD IMAGERY ACQUISITION")
        log.info("="*60)
        
        all_results = {}
        
        # Try each location
        for location in VERIFIED_FLOODS:
            try:
                results = self.pull_imagery(location)
                all_results[location['name']] = results
            except Exception as e:
                log.error(f"Failed for {location['name']}: {e}")
        
        # Save metadata
        metadata = {
            "locations_processed": len(all_results),
            "files_downloaded": len(self.dl),
            "results": all_results,
            "files": self.dl
        }
        
        metadata_path = os.path.join(DATA_BASE, "acquisition.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"\n✓ Downloaded {len(self.dl)} files")
        log.info(f"✓ Metadata saved: {metadata_path}")
        
        if len(self.dl) == 0:
            log.warning("\n⚠️ No files downloaded. Run 00_discover_bangladesh_floods.py first to find available imagery!")
        
        return metadata

if __name__ == "__main__":
    puller = WorkingBangladeshPull()
    puller.run()