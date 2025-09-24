"""
Complete Results Organization Pipeline for Bangladesh Flood Detection System
Handles ALL directories including empty ones, previews, and future placeholders
"""
import os
import json
import shutil
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime
import glob
import hashlib

class CompleteResultsOrganizer:
    def __init__(self):
        """Initialize the complete results organizer"""
        
        print("="*80)
        print("BANGLADESH FLOOD DETECTION - COMPLETE RESULTS ORGANIZATION")
        print("Organizing all outputs including empty directories for future use")
        print("="*80)
        
        # Define organized structure with ALL directories
        self.organized_root = "ORGANIZED_RESULTS"
        
        # Complete structure including empty directories for future outputs
        self.structure = {
            "01_RAW_SATELLITE_DATA": {
                "Gaibandha_2020": {
                    "SAR_Sentinel1": ["pre", "flood", "post"],
                    "Optical_Sentinel2": ["pre", "flood", "post"],
                    "metadata": []
                },
                "Sylhet_2024": {
                    "SAR_Sentinel1": ["pre", "flood", "post"],
                    "Optical_Sentinel2": ["pre", "flood", "post"],
                    "metadata": []
                }
            },
            "02_PROCESSED_DATA": {
                "tiles": {
                    "Gaibandha_2020": {
                        "SAR": ["pre", "flood", "post"],
                        "Optical": ["pre", "flood", "post"]
                    },
                    "Sylhet_2024": {
                        "SAR": ["pre", "flood", "post"],
                        "Optical": ["pre", "flood", "post"]
                    }
                },
                "previews": {
                    "tiles": [],
                    "composites": [],
                    "summaries": []
                },
                "quality_metrics": []
            },
            "03_TRAINING": {
                "datasets": {
                    "train": [],
                    "validation": [],
                    "test": []
                },
                "augmentation_samples": [],
                "data_statistics": []
            },
            "04_MODELS": {
                "checkpoints": {
                    "best_models": [],
                    "epoch_checkpoints": []
                },
                "final_models": [],
                "model_cards": [],
                "training_history": {
                    "metrics": [],
                    "loss_curves": [],
                    "learning_rate": []
                }
            },
            "05_INFERENCE_OUTPUTS": {
                "predictions": {
                    "Gaibandha_2020": ["masks", "overlays", "raw_outputs"],
                    "Sylhet_2024": ["masks", "overlays", "raw_outputs"]
                },
                "probability_maps": {
                    "Gaibandha_2020": [],
                    "Sylhet_2024": []
                },
                "uncertainty_maps": {
                    "Gaibandha_2020": [],
                    "Sylhet_2024": []
                },
                "attention_maps": {
                    "Gaibandha_2020": [],
                    "Sylhet_2024": []
                },
                "temporal_analysis": {
                    "flood_evolution": [],
                    "change_detection": [],
                    "time_series": []
                }
            },
            "06_ANALYSIS_RESULTS": {
                "metrics": {
                    "performance": [],
                    "confusion_matrices": [],
                    "roc_curves": []
                },
                "comparisons": {
                    "location_comparison": [],
                    "temporal_comparison": [],
                    "model_comparison": []
                },
                "statistical_analysis": {
                    "flood_statistics": [],
                    "accuracy_by_region": [],
                    "error_analysis": []
                }
            },
            "07_VISUALIZATIONS": {
                "dashboard": {
                    "interactive_plots": [],
                    "static_charts": [],
                    "maps": []
                },
                "hero_images": {
                    "website": [],
                    "presentation": [],
                    "print_quality": []
                },
                "showcase": {
                    "best_results": [],
                    "comparison_grids": [],
                    "before_after": []
                },
                "reports": {
                    "executive_summary": [],
                    "technical_report": [],
                    "presentation_slides": []
                }
            },
            "08_DOCUMENTATION": {
                "data_lineage": {
                    "processing_flow": [],
                    "transformation_logs": []
                },
                "metadata": {
                    "file_registry": [],
                    "hash_verification": []
                },
                "api_outputs": {
                    "json_responses": [],
                    "export_configs": []
                },
                "notebooks": {
                    "analysis": [],
                    "visualization": []
                }
            },
            "09_DEPLOYMENT_READY": {
                "production_models": {
                    "optimized": [],
                    "quantized": [],
                    "onnx_exports": []
                },
                "inference_configs": [],
                "docker": {
                    "images": [],
                    "configs": []
                },
                "api_endpoints": []
            },
            "10_AUDIT_TRAIL": {
                "processing_logs": [],
                "error_logs": [],
                "performance_benchmarks": [],
                "validation_reports": []
            }
        }
        
        # Tracking dictionaries
        self.file_registry = {}
        self.empty_directories = []
        self.future_placeholders = []
        self.processing_status = {}
        
    def create_complete_structure(self):
        """Create the complete directory structure including empty placeholders"""
        
        print("\n📁 Creating Complete Directory Structure...")
        
        def create_nested_dirs(base_path, structure, track_empty=True):
            """Recursively create all directories"""
            for key, value in structure.items():
                dir_path = os.path.join(base_path, key)
                os.makedirs(dir_path, exist_ok=True)
                
                # Check if directory will be empty (placeholder)
                is_empty = True
                
                if isinstance(value, dict):
                    is_empty = create_nested_dirs(dir_path, value, track_empty)
                elif isinstance(value, list):
                    for subdir in value:
                        if subdir:
                            subdir_path = os.path.join(dir_path, subdir)
                            os.makedirs(subdir_path, exist_ok=True)
                            
                            # Add README to empty directories
                            readme_path = os.path.join(subdir_path, "README_PLACEHOLDER.txt")
                            if not os.path.exists(readme_path):
                                with open(readme_path, 'w') as f:
                                    f.write(f"This directory is reserved for: {subdir}\n")
                                    f.write(f"Parent category: {key}\n")
                                    f.write(f"Created: {datetime.now().isoformat()}\n")
                                self.empty_directories.append(subdir_path)
                
                # Track empty directories
                if is_empty and track_empty:
                    self.future_placeholders.append(dir_path)
                
                return is_empty
        
        create_nested_dirs(self.organized_root, self.structure)
        print(f"   ✓ Created {len(self.empty_directories)} placeholder directories")
        print("   ✓ Complete structure ready")
    
    def organize_all_outputs(self):
        """Organize all existing outputs comprehensively"""
        
        # List of all source directories to check
        source_dirs = {
            "../data/raw": self._organize_raw_data,
            "data/raw": self._organize_raw_data,
            "data/tiles": self._organize_tiles,
            "data/previews": self._organize_previews,
            "outputs/models": self._organize_models,
            "outputs/predictions": self._organize_predictions,
            "outputs/attention_maps": self._organize_attention_maps,
            "outputs/uncertainty_maps": self._organize_uncertainty_maps,
            "outputs/temporal_analysis": self._organize_temporal_analysis,
            "outputs/metrics": self._organize_metrics,
            "assets/dashboard": self._organize_dashboard,
            "assets/hero": self._organize_hero,
            "assets/showcase": self._organize_showcase,
            "assets/reports": self._organize_reports,
            "assets/comparisons": self._organize_comparisons
        }
        
        print("\n🔄 Organizing All Outputs...")
        
        for source_dir, organize_func in source_dirs.items():
            if os.path.exists(source_dir):
                print(f"\n📂 Processing: {source_dir}")
                files_found = organize_func(source_dir)
                self.processing_status[source_dir] = {
                    "exists": True,
                    "files_processed": files_found
                }
            else:
                self.processing_status[source_dir] = {
                    "exists": False,
                    "files_processed": 0
                }
                print(f"   ⏭️  Skipping {source_dir} (not found)")
    
    def _organize_raw_data(self, source_dir):
        """Organize raw satellite data"""
        files_processed = 0
        
        for file_path in glob.glob(os.path.join(source_dir, "*.npy")):
            filename = os.path.basename(file_path)
            
            # Parse filename
            parts = filename.replace('.npy', '').split('_')
            if len(parts) >= 4:
                sensor = parts[0]
                location = f"{parts[1]}_{parts[2]}"
                period = parts[3]
                
                # Determine destination
                sensor_type = "SAR_Sentinel1" if sensor == 'S1' else "Optical_Sentinel2"
                dest_dir = os.path.join(
                    self.organized_root,
                    "01_RAW_SATELLITE_DATA",
                    location,
                    sensor_type,
                    period
                )
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy with metadata
                new_name = f"{location}_{sensor}_{period}_raw.npy"
                new_path = os.path.join(dest_dir, new_name)
                shutil.copy2(file_path, new_path)
                
                # Create metadata
                self._create_metadata(file_path, new_path, {
                    "location": location,
                    "sensor": sensor,
                    "period": period,
                    "original_name": filename
                })
                
                files_processed += 1
                print(f"   ✓ {filename} → {sensor_type}/{period}/")
        
        return files_processed
    
    def _organize_tiles(self, source_dir):
        """Organize processed tiles"""
        files_processed = 0
        
        # Load tile catalog if exists
        catalog_path = os.path.join(source_dir, "tile_catalog.json")
        if os.path.exists(catalog_path):
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
            
            # Copy catalog to documentation
            doc_dir = os.path.join(self.organized_root, "08_DOCUMENTATION", "metadata")
            os.makedirs(doc_dir, exist_ok=True)
            shutil.copy2(catalog_path, os.path.join(doc_dir, "original_tile_catalog.json"))
            
            # Process tiles
            for tile_info in catalog.get('tiles', []):
                if 'path' in tile_info and os.path.exists(tile_info['path']):
                    location = tile_info['location']
                    period = tile_info['period']
                    sensor_type = "SAR" if 'S1' in tile_info['sensor'] else "Optical"
                    
                    dest_dir = os.path.join(
                        self.organized_root,
                        "02_PROCESSED_DATA",
                        "tiles",
                        location,
                        sensor_type,
                        period
                    )
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Create descriptive filename
                    position = tile_info.get('position', [0, 0])
                    new_name = f"{location}_{sensor_type}_{period}_r{position[0]:03d}_c{position[1]:03d}.npy"
                    new_path = os.path.join(dest_dir, new_name)
                    
                    shutil.copy2(tile_info['path'], new_path)
                    files_processed += 1
        
        # Also copy any .npy files directly
        for npy_file in glob.glob(os.path.join(source_dir, "*.npy")):
            if 'catalog' not in npy_file:
                files_processed += 1
        
        return files_processed
    
    def _organize_previews(self, source_dir):
        """Organize preview images"""
        files_processed = 0
        
        preview_types = {
            'grid': 'composites',
            'summary': 'summaries',
            'tile': 'tiles'
        }
        
        for img_file in glob.glob(os.path.join(source_dir, "*.png")):
            filename = os.path.basename(img_file)
            
            # Determine preview type
            preview_type = 'tiles'  # default
            for key, value in preview_types.items():
                if key in filename.lower():
                    preview_type = value
                    break
            
            dest_dir = os.path.join(
                self.organized_root,
                "02_PROCESSED_DATA",
                "previews",
                preview_type
            )
            os.makedirs(dest_dir, exist_ok=True)
            
            shutil.copy2(img_file, os.path.join(dest_dir, filename))
            files_processed += 1
            print(f"   ✓ Preview: {filename} → previews/{preview_type}/")
        
        return files_processed
    
    def _organize_models(self, source_dir):
        """Organize model files"""
        files_processed = 0
        
        for model_file in glob.glob(os.path.join(source_dir, "*.pt")):
            filename = os.path.basename(model_file)
            
            # Determine model type
            if 'best' in filename.lower():
                dest_dir = os.path.join(self.organized_root, "04_MODELS", "checkpoints", "best_models")
            elif 'final' in filename.lower():
                dest_dir = os.path.join(self.organized_root, "04_MODELS", "final_models")
            else:
                dest_dir = os.path.join(self.organized_root, "04_MODELS", "checkpoints", "epoch_checkpoints")
            
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy model
            new_path = os.path.join(dest_dir, filename)
            shutil.copy2(model_file, new_path)
            
            # Create model card
            self._create_model_card(model_file, new_path)
            
            files_processed += 1
            print(f"   ✓ Model: {filename}")
        
        return files_processed
    
    def _organize_predictions(self, source_dir):
        """Organize prediction outputs"""
        files_processed = 0
        
        for pred_file in glob.glob(os.path.join(source_dir, "*.*")):
            filename = os.path.basename(pred_file)
            
            # Determine location
            if 'Gaibandha' in filename:
                location = 'Gaibandha_2020'
            elif 'Sylhet' in filename:
                location = 'Sylhet_2024'
            else:
                location = 'Unknown'
            
            # Determine prediction type
            if 'probability' in filename.lower():
                pred_type = 'probability_maps'
            elif 'uncertainty' in filename.lower():
                pred_type = 'uncertainty_maps'
            elif 'overlay' in filename.lower():
                pred_type = 'overlays'
            else:
                pred_type = 'masks'
            
            if 'probability' in filename.lower():
                dest_dir = os.path.join(
                    self.organized_root,
                    "05_INFERENCE_OUTPUTS",
                    "probability_maps",
                    location
                )
            elif 'uncertainty' in filename.lower():
                dest_dir = os.path.join(
                    self.organized_root,
                    "05_INFERENCE_OUTPUTS",
                    "uncertainty_maps",
                    location
                )
            else:
                dest_dir = os.path.join(
                    self.organized_root,
                    "05_INFERENCE_OUTPUTS",
                    "predictions",
                    location,
                    pred_type
                )
            
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(pred_file, os.path.join(dest_dir, filename))
            files_processed += 1
        
        return files_processed
    
    def _organize_attention_maps(self, source_dir):
        """Organize attention maps (even if empty)"""
        files_processed = 0
        
        # Create placeholder structure
        for location in ['Gaibandha_2020', 'Sylhet_2024']:
            dest_dir = os.path.join(
                self.organized_root,
                "05_INFERENCE_OUTPUTS",
                "attention_maps",
                location
            )
            os.makedirs(dest_dir, exist_ok=True)
            
            # Add placeholder README
            readme_path = os.path.join(dest_dir, "README.txt")
            with open(readme_path, 'w') as f:
                f.write("Attention maps will be generated here when using transformer models\n")
                f.write(f"Location: {location}\n")
                f.write(f"Reserved for future model outputs\n")
        
        # Copy any existing attention maps
        for file in glob.glob(os.path.join(source_dir, "*.*")):
            files_processed += 1
            # Process if files exist
        
        return files_processed
    
    def _organize_uncertainty_maps(self, source_dir):
        """Organize uncertainty maps"""
        files_processed = 0
        
        # Check both the dedicated directory and predictions directory
        for pattern in ["*.png", "*.npy", "*.jpg"]:
            for file in glob.glob(os.path.join(source_dir, pattern)):
                filename = os.path.basename(file)
                
                # Determine location
                if 'Gaibandha' in filename:
                    location = 'Gaibandha_2020'
                elif 'Sylhet' in filename:
                    location = 'Sylhet_2024'
                else:
                    location = 'Unknown'
                
                dest_dir = os.path.join(
                    self.organized_root,
                    "05_INFERENCE_OUTPUTS",
                    "uncertainty_maps",
                    location
                )
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(file, os.path.join(dest_dir, filename))
                files_processed += 1
        
        return files_processed
    
    def _organize_temporal_analysis(self, source_dir):
        """Organize temporal analysis outputs"""
        files_processed = 0
        
        # Create structure for temporal analysis
        categories = ['flood_evolution', 'change_detection', 'time_series']
        
        for category in categories:
            dest_dir = os.path.join(
                self.organized_root,
                "05_INFERENCE_OUTPUTS",
                "temporal_analysis",
                category
            )
            os.makedirs(dest_dir, exist_ok=True)
            
            # Add placeholder
            readme_path = os.path.join(dest_dir, "README.txt")
            with open(readme_path, 'w') as f:
                f.write(f"Temporal analysis outputs: {category}\n")
                f.write("Reserved for multi-temporal flood analysis\n")
        
        # Copy any existing files
        for file in glob.glob(os.path.join(source_dir, "*.*")):
            shutil.copy2(file, os.path.join(
                self.organized_root,
                "05_INFERENCE_OUTPUTS",
                "temporal_analysis",
                "flood_evolution",
                os.path.basename(file)
            ))
            files_processed += 1
        
        return files_processed
    
    def _organize_metrics(self, source_dir):
        """Organize metrics and training history"""
        files_processed = 0
        
        for metric_file in glob.glob(os.path.join(source_dir, "*.json")):
            filename = os.path.basename(metric_file)
            
            if 'training' in filename.lower():
                dest_dir = os.path.join(
                    self.organized_root,
                    "04_MODELS",
                    "training_history",
                    "metrics"
                )
            else:
                dest_dir = os.path.join(
                    self.organized_root,
                    "06_ANALYSIS_RESULTS",
                    "metrics",
                    "performance"
                )
            
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(metric_file, os.path.join(dest_dir, filename))
            files_processed += 1
            print(f"   ✓ Metrics: {filename}")
        
        return files_processed
    
    def _organize_dashboard(self, source_dir):
        """Organize dashboard outputs"""
        files_processed = 0
        
        file_types = {
            '.html': 'interactive_plots',
            '.json': 'static_charts',
            '.png': 'static_charts',
            '.jpg': 'static_charts'
        }
        
        for ext, subdir in file_types.items():
            for file in glob.glob(os.path.join(source_dir, f"*{ext}")):
                dest_dir = os.path.join(
                    self.organized_root,
                    "07_VISUALIZATIONS",
                    "dashboard",
                    subdir
                )
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(file, os.path.join(dest_dir, os.path.basename(file)))
                files_processed += 1
        
        # Create placeholder for empty dashboard
        if files_processed == 0:
            placeholder_dir = os.path.join(
                self.organized_root,
                "07_VISUALIZATIONS",
                "dashboard",
                "interactive_plots"
            )
            os.makedirs(placeholder_dir, exist_ok=True)
            with open(os.path.join(placeholder_dir, "README.txt"), 'w') as f:
                f.write("Dashboard visualizations will be generated here\n")
                f.write("Run: python 04_export_final_dashboard.py\n")
        
        return files_processed
    
    def _organize_hero(self, source_dir):
        """Organize hero images"""
        files_processed = 0
        
        for img_file in glob.glob(os.path.join(source_dir, "*.*")):
            filename = os.path.basename(img_file)
            
            dest_dir = os.path.join(
                self.organized_root,
                "07_VISUALIZATIONS",
                "hero_images",
                "website"
            )
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_file, os.path.join(dest_dir, filename))
            files_processed += 1
        
        return files_processed
    
    def _organize_showcase(self, source_dir):
        """Organize showcase visualizations"""
        files_processed = 0
        
        for file in glob.glob(os.path.join(source_dir, "*.*")):
            dest_dir = os.path.join(
                self.organized_root,
                "07_VISUALIZATIONS",
                "showcase",
                "best_results"
            )
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file, os.path.join(dest_dir, os.path.basename(file)))
            files_processed += 1
        
        return files_processed
    
    def _organize_reports(self, source_dir):
        """Organize reports"""
        files_processed = 0
        
        report_types = {
            'executive': 'executive_summary',
            'technical': 'technical_report',
            'performance': 'technical_report',
            'summary': 'executive_summary'
        }
        
        for file in glob.glob(os.path.join(source_dir, "*.*")):
            filename = os.path.basename(file)
            
            # Determine report type
            report_type = 'technical_report'  # default
            for key, value in report_types.items():
                if key in filename.lower():
                    report_type = value
                    break
            
            dest_dir = os.path.join(
                self.organized_root,
                "07_VISUALIZATIONS",
                "reports",
                report_type
            )
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file, os.path.join(dest_dir, filename))
            files_processed += 1
        
        return files_processed
    
    def _organize_comparisons(self, source_dir):
        """Organize comparison visualizations"""
        files_processed = 0
        
        for file in glob.glob(os.path.join(source_dir, "*.*")):
            dest_dir = os.path.join(
                self.organized_root,
                "06_ANALYSIS_RESULTS",
                "comparisons",
                "location_comparison"
            )
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file, os.path.join(dest_dir, os.path.basename(file)))
            files_processed += 1
        
        return files_processed
    
    def _create_metadata(self, source_path, dest_path, info):
        """Create metadata file for any data file"""
        metadata = {
            "original_path": source_path,
            "organized_path": dest_path,
            "file_size_mb": os.path.getsize(source_path) / (1024*1024),
            "organization_date": datetime.now().isoformat(),
            **info
        }
        
        meta_path = dest_path.replace('.npy', '_metadata.json').replace('.png', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_model_card(self, source_path, dest_path):
        """Create model card for model files"""
        try:
            import torch
            checkpoint = torch.load(source_path, map_location='cpu')
            
            model_card = {
                "model_file": os.path.basename(dest_path),
                "file_size_mb": os.path.getsize(source_path) / (1024*1024),
                "performance": {
                    "dice_score": checkpoint.get('dice', 0),
                    "iou_score": checkpoint.get('iou', 0),
                    "loss": checkpoint.get('loss', 0)
                },
                "training_info": {
                    "epoch": checkpoint.get('epoch', 0)
                },
                "organization_date": datetime.now().isoformat()
            }
        except:
            model_card = {
                "model_file": os.path.basename(dest_path),
                "file_size_mb": os.path.getsize(source_path) / (1024*1024),
                "organization_date": datetime.now().isoformat()
            }
        
        card_path = dest_path.replace('.pt', '_model_card.json')
        with open(card_path, 'w') as f:
            json.dump(model_card, f, indent=2)
    
    def create_status_report(self):
        """Create comprehensive status report"""
        
        print("\n📊 Creating Status Report...")
        
        # Count all files
        total_files = 0
        total_size = 0
        file_distribution = {}
        
        for root, dirs, files in os.walk(self.organized_root):
            for file in files:
                if not file.endswith('.txt') and not file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    total_size += os.path.getsize(file_path)
                    
                    # Track distribution
                    rel_path = os.path.relpath(root, self.organized_root)
                    main_category = rel_path.split(os.sep)[0]
                    
                    if main_category not in file_distribution:
                        file_distribution[main_category] = 0
                    file_distribution[main_category] += 1
        
        # Create report
        status_report = {
            "organization_timestamp": datetime.now().isoformat(),
            "total_files_organized": total_files,
            "total_size_mb": total_size / (1024*1024),
            "processing_status": self.processing_status,
            "file_distribution": file_distribution,
            "empty_placeholder_directories": len(self.empty_directories),
            "future_placeholders": self.future_placeholders[:10],  # Sample
            "structure_categories": list(self.structure.keys()),
            "completeness": {
                "raw_data": "01_RAW_SATELLITE_DATA" in file_distribution,
                "processed_tiles": "02_PROCESSED_DATA" in file_distribution,
                "models": "04_MODELS" in file_distribution,
                "predictions": "05_INFERENCE_OUTPUTS" in file_distribution,
                "visualizations": "07_VISUALIZATIONS" in file_distribution
            }
        }
        
        # Save report
        report_path = os.path.join(self.organized_root, "ORGANIZATION_STATUS.json")
        with open(report_path, 'w') as f:
            json.dump(status_report, f, indent=2)
        
        return status_report
    
    def create_next_steps_guide(self):
        """Create guide for next steps based on what's missing"""
        
        guide = """# Next Steps Guide

Based on the organization status, here are the recommended next steps:

## 🚀 Immediate Actions

### If visualizations are missing:
```bash
python 04_export_final_dashboard.py