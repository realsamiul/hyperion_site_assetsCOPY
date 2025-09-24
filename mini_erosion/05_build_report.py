import json, datetime
import sys
sys.path.append('..')
from common.utils import load_schema, save_json

# Load template schema
schema = load_schema()

# Update metadata
schema["meta"].update({
    "model_id": "hyperion-erosion",
    "model_name": "Hyperion Coastal Erosion Demo",
    "generated_at": datetime.datetime.utcnow().isoformat()+"Z",
    "aoi_bbox": [91.80, 20.75, 91.95, 21.95],
    "aoi_name": "Kutubdia Island",
    "periods": ["1990", "2023"],
    "tags": ["erosion", "Landsat", "U-Net"]
})

# Update story
schema["story"]["one_liner"] = "Coastal erosion monitoring through satellite imagery."
schema["story"]["elevator"] = "U-Net-based shoreline detection for long-term coastal erosion analysis."

# Update dataset info
schema["dataset"]["sensors"] = {
    "landsat8": {"total": 30}
}

# Update results
schema["results"]["qualitative"] = {
    "retreat_curve": "assets/charts/retreat_curve.png"
}

# Update artifacts
schema["artifacts"]["images"]["shoreline"] = {
    "1990": "assets/hero/shore_1990.jpg",
    "2023": "assets/hero/shore_2023.jpg"
}

# Save report
save_json(schema, "../../model_report_erosion.json")
print("Erosion model_report.json written.")
