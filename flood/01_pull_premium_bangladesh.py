"""
Advanced Satellite Acquisition for Bangladesh Floods
Maintains proven connection method while finding BEST flood imagery
"""
import os, json, logging, requests, numpy as np, ee
from typing import Dict, List, Tuple
from datetime import datetime
import sys
sys.path.append('..')
from common.config import GCP_PROJECT, ASSET_DIR, SCALE
from common.utils import ts, ensure, save_json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── EE login (YOUR PROVEN METHOD) ───────────────────────────────
try:
    ee.Initialize(project=GCP_PROJECT)
    log.info(f"Earth Engine initialised  ({GCP_PROJECT})")
except Exception:
    log.error("Run  earthengine authenticate  then re-run the script")
    raise
# ─────────────────────────────────────────────────────────────────

# Premium Bangladesh flood events with verified excellent imagery
BANGLADESH_FLOODS = [
    {
        "name": "Sylhet_2022_Extreme",
        "aoi": [24.7, 91.5, 25.0, 91.9],  # Sylhet region
        "periods": {
            "pre": {"start": "2022-05-01", "end": "2022-05-20"},
            "flood": {"start": "2022-06-15", "end": "2022-06-22"},
            "post": {"start": "2022-07-10", "end": "2022-07-25"}
        },
        "description": "Worst flooding in 122 years, entire city underwater"
    },
    {
        "name": "Kurigram_2024_Monsoon", 
        "aoi": [25.7, 89.5, 26.0, 89.8],  # Kurigram district
        "periods": {
            "pre": {"start": "2024-06-01", "end": "2024-06-15"},
            "flood": {"start": "2024-07-01", "end": "2024-07-10"},
            "post": {"start": "2024-07-20", "end": "2024-08-05"}
        },
        "description": "Recent monsoon flooding with clear imagery"
    },
    {
        "name": "Dhaka_2020_Urban",
        "aoi": [23.5, 90.2, 23.9, 90.5],  # Greater Dhaka
        "periods": {
            "pre": {"start": "2020-06-01", "end": "2020-06-20"},
            "flood": {"start": "2020-07-15", "end": "2020-07-25"},
            "post": {"start": "2020-08-10", "end": "2020-08-25"}
        },
        "description": "Urban flooding showing infrastructure impact"
    }
]

# Data directories
DATA_BASE = os.path.join(ASSET_DIR, "../data")
OUTPUT_BASE = os.path.join(ASSET_DIR, "../outputs")
ensure(os.path.join(DATA_BASE, "raw"))
ensure(os.path.join(OUTPUT_BASE, "metrics"))

class PremiumSatelliteAcquisition:
    """Enhanced acquisition with quality scoring and best imagery selection"""
    
    def __init__(self):
        self.scale = SCALE or 10  # Default 10m resolution
        self.dl = []  # Downloaded file paths
        self.quality_scores = {}
        self.best_location = None
        
    def evaluate_location(self, location: Dict) -> Tuple[float, Dict]:
        """Evaluate imagery quality for a location"""
        log.info(f"\n🔍 Evaluating: {location['name']}")
        log.info(f"   {location['description']}")
        
        aoi = ee.Geometry.Rectangle(location['aoi'])
        scores = {"location": location['name'], "details": {}}
        total_score = 0
        
        for period_name, period in location['periods'].items():
            # Check SAR availability and quality
            s1_coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filterBounds(aoi)
                      .filterDate(period['start'], period['end'])
                      .filter(ee.Filter.eq('instrumentMode', 'IW'))
                      .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')))
            
            # Check optical availability and cloud cover
            s2_coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(aoi)
                      .filterDate(period['start'], period['end'])
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
            
            s1_count = s1_coll.size().getInfo()
            s2_count = s2_coll.size().getInfo()
            
            # Get cloud statistics for S2
            cloud_score = 100
            if s2_count > 0:
                best_s2 = s2_coll.sort('CLOUDY_PIXEL_PERCENTAGE').first()
                cloud_cover = best_s2.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                cloud_score = 100 - cloud_cover
            
            # Calculate period score
            period_score = (
                min(100, s1_count * 20) * 0.4 +  # SAR availability (40%)
                min(100, s2_count * 20) * 0.3 +  # Optical availability (30%)
                cloud_score * 0.3                 # Cloud-free quality (30%)
            )
            
            scores["details"][period_name] = {
                "s1_images": s1_count,
                "s2_images": s2_count,
                "cloud_score": cloud_score,
                "quality_score": period_score
            }
            total_score += period_score
            
        scores["total_score"] = total_score / 3  # Average across periods
        
        log.info(f"   Quality Score: {scores['total_score']:.1f}/100")
        log.info(f"   SAR: {scores['details']['flood']['s1_images']} images (flood)")
        log.info(f"   Optical: {scores['details']['flood']['s2_images']} images (flood)")
        
        return scores["total_score"], scores
    
    def download_sar_advanced(self, start: str, end: str, tag: str, aoi) -> bool:
        """Enhanced SAR download with preprocessing"""
        try:
            coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                   .filterBounds(aoi)
                   .filterDate(start, end)
                   .filter(ee.Filter.eq('instrumentMode', 'IW'))
                   .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                   .select(['VV', 'VH', 'angle']))
            
            if coll.size().getInfo() == 0:
                log.warning(f"   no SAR images for {tag}")
                return False
            
            # Advanced preprocessing
            def preprocess_s1(image):
                """Apply refined preprocessing to S1"""
                vv = image.select('VV')
                vh = image.select('VH')
                angle = image.select('angle')
                
                # Incidence angle correction
                vv_corrected = vv.subtract(angle.multiply(0.05))
                vh_corrected = vh.subtract(angle.multiply(0.05))
                
                # Calculate VV/VH ratio (good for water detection)
                ratio = vv_corrected.divide(vh_corrected).rename('ratio')
                
                return image.addBands([vv_corrected.rename('VV_corr'),
                                      vh_corrected.rename('VH_corr'),
                                      ratio])
            
            # Apply preprocessing and create composite
            processed = coll.map(preprocess_s1).map(lambda i: i.clip(aoi))
            
            # Use median for speckle reduction
            composite = processed.median()
            
            # Normalize to 0-1 range (your proven method)
            normalized = composite.select(['VV', 'VH']).unitScale(-25, 5)
            
            # Add water probability layer
            water_prob = normalized.select('VV').lt(0.2).rename('water_prob')
            final = normalized.addBands(water_prob)
            
            # Download using your proven method
            url = final.getDownloadURL({
                "scale": self.scale,
                "crs": "EPSG:4326",
                "region": aoi,
                "format": "NPY"
            })
            
            fp = os.path.join(DATA_BASE, "raw", f"S1_{tag}.npy")
            with open(fp, "wb") as f:
                f.write(requests.get(url).content)
            
            # Verify download
            data = np.load(fp, allow_pickle=True)
            log.info(f"   ✓ SAR saved → {fp} (shape: {data.shape})")
            self.dl.append(fp)
            return True
            
        except Exception as e:
            log.error(f"   ✗ SAR {tag}: {e}")
            return False
    
    def download_optical_advanced(self, start: str, end: str, tag: str, aoi) -> bool:
        """Enhanced optical download with water indices"""
        try:
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(aoi)
                   .filterDate(start, end)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
            
            if coll.size().getInfo() == 0:
                log.warning(f"   no optical images for {tag}")
                return False
            
            # Cloud masking function
            def mask_clouds(image):
                qa = image.select('QA60')
                clouds = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
                return image.updateMask(clouds)
            
            # Get best image and apply cloud mask
            masked = coll.map(mask_clouds).map(lambda i: i.clip(aoi))
            best = masked.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            # Select RGB + NIR + SWIR bands
            bands = best.select(['B4','B3','B2','B8','B11','B12']).divide(10000)
            
            # Calculate water indices
            ndwi = bands.normalizedDifference(['B3', 'B8']).rename('NDWI')
            mndwi = bands.normalizedDifference(['B3', 'B11']).rename('MNDWI')
            
            # Combine all bands
            final = bands.addBands([ndwi, mndwi])
            
            # Download
            url = final.getDownloadURL({
                "scale": self.scale,
                "crs": "EPSG:4326",
                "region": aoi,
                "format": "NPY"
            })
            
            fp = os.path.join(DATA_BASE, "raw", f"S2_{tag}.npy")
            with open(fp, "wb") as f:
                f.write(requests.get(url).content)
            
            log.info(f"   ✓ Optical saved → {fp}")
            self.dl.append(fp)
            return True
            
        except Exception as e:
            log.error(f"   ✗ Optical {tag}: {e}")
            return False
    
    def download_landsat_bonus(self, start: str, end: str, tag: str, aoi) -> bool:
        """Bonus: Download Landsat for additional coverage"""
        try:
            # Merge Landsat 8 and 9
            l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi)
            l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(aoi)
            
            coll = l8.merge(l9).filterDate(start, end) \
                    .filter(ee.Filter.lt('CLOUD_COVER', 20))
            
            if coll.size().getInfo() == 0:
                return False
            
            # Process Landsat
            def process_landsat(image):
                optical = image.select(['SR_B.*']).multiply(0.0000275).add(-0.2)
                thermal = image.select(['ST_B10']).multiply(0.00341802).add(149.0)
                return optical.addBands(thermal).clip(aoi)
            
            composite = coll.map(process_landsat).median()
            
            url = composite.getDownloadURL({
                "scale": 30,  # Landsat resolution
                "crs": "EPSG:4326",
                "region": aoi,
                "format": "NPY"
            })
            
            fp = os.path.join(DATA_BASE, "raw", f"L89_{tag}.npy")
            with open(fp, "wb") as f:
                f.write(requests.get(url).content)
            
            log.info(f"   ✓ Landsat bonus saved → {fp}")
            self.dl.append(fp)
            return True
            
        except:
            return False
    
    def run(self):
        """Find and download the best Bangladesh flood imagery"""
        log.info("="*60)
        log.info("PREMIUM BANGLADESH FLOOD IMAGERY ACQUISITION")
        log.info("="*60)
        
        # Evaluate all locations
        all_scores = []
        for location in BANGLADESH_FLOODS:
            score, details = self.evaluate_location(location)
            all_scores.append({
                "location": location,
                "score": score,
                "details": details
            })
        
        # Select best location
        best = max(all_scores, key=lambda x: x["score"])
        self.best_location = best["location"]
        
        log.info("\n" + "="*60)
        log.info(f"🏆 SELECTED: {self.best_location['name']}")
        log.info(f"   Quality Score: {best['score']:.1f}/100")
        log.info(f"   {self.best_location['description']}")
        log.info("="*60)
        
        # Download imagery from best location
        aoi = ee.Geometry.Rectangle(self.best_location['aoi'])
        results = {}
        
        for period_name, period in self.best_location['periods'].items():
            log.info(f"\n--- {period_name.upper()} PERIOD ---")
            
            # Check availability
            avail = self.check_availability(period['start'], period['end'], aoi)
            log.info(f"   Available: {avail['s1']} SAR, {avail['s2']} optical")
            
            # Download with advanced processing
            sar_ok = self.download_sar_advanced(
                period['start'], period['end'], period_name, aoi)
            opt_ok = self.download_optical_advanced(
                period['start'], period['end'], period_name, aoi)
            landsat_ok = self.download_landsat_bonus(
                period['start'], period['end'], period_name, aoi)
            
            results[period_name] = {
                "availability": avail,
                "sar_downloaded": sar_ok,
                "optical_downloaded": opt_ok,
                "landsat_bonus": landsat_ok
            }
        
        # Save comprehensive metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "selected_location": self.best_location['name'],
            "aoi": self.best_location['aoi'],
            "description": self.best_location['description'],
            "quality_score": best['score'],
            "quality_details": best['details'],
            "download_results": results,
            "files_downloaded": self.dl,
            "total_files": len(self.dl)
        }
        
        metadata_fp = os.path.join(DATA_BASE, "acquisition.json")
        with open(metadata_fp, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"\n✓ Acquisition complete. Metadata: {metadata_fp}")
        log.info(f"✓ Downloaded {len(self.dl)} files")
        
        return metadata
    
    def check_availability(self, start: str, end: str, aoi) -> Dict:
        """Check imagery availability (your proven method)"""
        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(aoi)
              .filterDate(start, end)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')))
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(aoi)
              .filterDate(start, end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)))
        return {"s1": s1.size().getInfo(), "s2": s2.size().getInfo()}


if __name__ == "__main__":
    log.info("="*60)
    log.info("START SATELLITE PULL")
    log.info("="*60)
    acquisition = PremiumSatelliteAcquisition()
    acquisition.run()