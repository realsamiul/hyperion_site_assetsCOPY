import json, datetime
import sys
sys.path.append('..')
from common.utils import load_schema, save_json

# Load template schema
schema = load_schema()

# Update metadata
schema["meta"].update({
    "model_id": "hyperion-crop",
    "model_name": "Hyperion Crop Stress Demo",
    "generated_at": datetime.datetime.utcnow().isoformat()+"Z",
    "aoi_bbox": [89.10, 23.05, 89.30, 23.25],
    "aoi_name": "Agricultural Region",
    "periods": ["2023"],
    "tags": ["crop", "Sentinel-2", "SimSiam"]
})

# Update story
schema["story"]["one_liner"] = "Crop stress monitoring through satellite imagery."
schema["story"]["elevator"] = "SimSiam embeddings and KMeans clustering for crop stress analysis."

# Update dataset info
schema["dataset"]["sensors"] = {
    "sentinel2": {"total": 12}
}

# Update results
schema["results"]["qualitative"] = {
    "stress_heatmap": "assets/hero/stress_heatmap.jpg",
    "classification_map": "assets/charts/stress_classification.png"
}

# Update artifacts
schema["artifacts"]["images"]["crop"] = {
    "stress_heatmap": "assets/hero/stress_heatmap.jpg"
}

# Save report
save_json(schema, "../../model_report_crop.json")
print("Crop model_report.json written.")
