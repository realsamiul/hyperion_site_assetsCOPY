"""
Global configuration for Hyperion Flood Detection
Bangladesh locations with VERIFIED imagery availability
"""
import os

# Google Earth Engine
GCP_PROJECT = "hyperion-472805"

# VERIFIED Bangladesh flood locations with excellent imagery
# Using coordinates from discovery script with proper AOI rectangles
BANGLADESH_FLOODS = [
    {
        "name": "Gaibandha_2020",
        "center": [25.3297, 89.543],  # From discovery script
        "aoi": [24.8297, 89.043, 25.8297, 90.043],  # 50km buffer around center
        "periods": {
            "pre": {"start": "2020-06-01", "end": "2020-06-30"},
            "flood": {"start": "2020-07-01", "end": "2020-08-31"},  # Wider range
            "post": {"start": "2020-08-01", "end": "2020-09-15"}
        },
        "description": "Major 2020 monsoon floods - 65 SAR, 20 optical images"
    },
    {
        "name": "Sylhet_2024",
        "center": [24.8949, 91.8687],  # From discovery script
        "aoi": [24.3949, 91.3687, 25.3949, 92.3687],  # 50km buffer around center
        "periods": {
            "pre": {"start": "2024-05-01", "end": "2024-05-31"},
            "flood": {"start": "2024-06-01", "end": "2024-07-31"},  # Wider range
            "post": {"start": "2024-07-01", "end": "2024-08-31"}
        },
        "description": "2024 monsoon floods - 50 SAR, 19 optical images"
    },
    {
        "name": "Cox_Bazar_2023",
        "center": [21.4272, 92.0058],  # From discovery script
        "aoi": [20.9272, 91.5058, 21.9272, 92.5058],  # 50km buffer around center
        "periods": {
            "pre": {"start": "2023-05-01", "end": "2023-05-31"},
            "flood": {"start": "2023-06-01", "end": "2023-07-31"},  # Wider range
            "post": {"start": "2023-08-01", "end": "2023-08-31"}
        },
        "description": "2023 coastal floods - 48 SAR, 18 optical images"
    }
]

# Default location (best imagery)
DEFAULT_FLOOD = BANGLADESH_FLOODS[0]

# Backward compatibility
FLOOD_AOI = BANGLADESH_FLOODS[0]["aoi"]
SEARCH_PERIODS = BANGLADESH_FLOODS[0]["periods"]

# Paths
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ASSET_DIR, "../data")
OUTPUT_DIR = os.path.join(ASSET_DIR, "../outputs")

# Processing
SCALE = 10  # Resolution in meters
TILE_SIZE = 256

def initialize_ee():
    """Initialize Earth Engine with error handling"""
    import ee
    try:
        ee.Initialize(project=GCP_PROJECT)
        return True
    except:
        ee.Authenticate()
        ee.Initialize(project=GCP_PROJECT)
        return True
