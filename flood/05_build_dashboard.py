"""
Build Comprehensive Dashboard for Multi-Location Flood Detection
Combines all visualization assets into a complete dashboard
"""
import json, datetime, numpy as np, os
import sys
sys.path.append('..')
from common.utils import load_schema, save_json, ensure_dir

def build_dashboard():
    """Build comprehensive dashboard with all assets"""
    print("🏗️ Building comprehensive flood detection dashboard...")
    
    # Load base schema
    schema = load_schema()
    
    # Update metadata for multi-location flood detection
    schema["meta"].update({
        "model_id": "hyperion-flood-multi-location",
        "model_name": "Hyperion Multi-Location Flood Detection",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "aoi_bbox": [89.5, 23.5, 91.9, 26.0],  # Bangladesh coverage
        "aoi_name": "Bangladesh Flood Events",
        "periods": ["pre", "flood", "post"],
        "tags": ["flood", "Sentinel-1", "Sentinel-2", "multi-location", "Bangladesh"]
    })
    
    # Update story
    schema["story"]["one_liner"] = "Advanced multi-location flood detection across Bangladesh using premium satellite imagery"
    schema["story"]["description"] = "Comprehensive flood monitoring system covering multiple Bangladesh flood events with advanced ML models and stunning visualizations"
    
    # Dataset information
    schema["dataset"]["sensors"] = {
        "sentinel1": {"total": 9, "description": "SAR imagery for water detection"},
        "sentinel2": {"total": 9, "description": "Optical imagery with water indices"},
        "landsat": {"total": 3, "description": "Additional coverage for validation"}
    }
    
    # Results - qualitative visualizations
    schema["results"]["qualitative"] = {
        "hero_image": "assets/hero/main_hero_flood.jpg",
        "location_comparison": "assets/comparisons/multi_location_analysis.html",
        "timeline_animation": "assets/hero/location_timeline.mp4",
        "3d_visualization": "assets/3d/flood_terrain_3d.html",
        "analytics_dashboard": "assets/dashboard/analytics.html"
    }
    
    # Artifacts - all generated assets
    schema["artifacts"]["images"]["hero"] = {
        "main": "assets/hero/main_hero_flood.jpg",
        "pre_flood": "assets/hero/pre_advanced.jpg",
        "during_flood": "assets/hero/flood_advanced.jpg", 
        "post_flood": "assets/hero/post_advanced.jpg"
    }
    
    schema["artifacts"]["videos"] = {
        "timeline_hd": "assets/hero/flood_timeline_hd.mp4",
        "location_timeline": "assets/hero/location_timeline.mp4",
        "transition_gif": "assets/hero/flood_transition.gif"
    }
    
    schema["artifacts"]["interactive"] = {
        "3d_terrain": "assets/3d/flood_terrain_3d.html",
        "analytics_dashboard": "assets/dashboard/analytics.html",
        "location_comparison": "assets/comparisons/multi_location_analysis.html"
    }
    
    # Technical details
    schema["technical"] = {
        "model_architecture": "LocationAwareFloodModel with EfficientNet-B2 encoder",
        "training_data": "Multi-location Bangladesh flood events",
        "validation_metrics": "Dice Loss + Focal Loss combination",
        "inference_time": "< 1 second per tile",
        "coverage_area": "Multiple Bangladesh regions"
    }
    
    # Save dashboard data
    ensure_dir("assets/dashboard")
    with open("assets/dashboard/dashboard_data.json", 'w') as f:
        json.dump(schema, f, indent=2)
    
    # Save model report
    save_json(schema, "../../model_report_flood_multi.json")
    
    print("✅ Dashboard built successfully!")
    print("   📊 Dashboard data: assets/dashboard/dashboard_data.json")
    print("   📋 Model report: ../../model_report_flood_multi.json")
    
    return schema

if __name__ == "__main__":
    build_dashboard()
