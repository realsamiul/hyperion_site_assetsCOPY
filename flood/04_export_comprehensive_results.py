"""
Comprehensive Results Export for Bangladesh Flood Detection System
Generates professional visualizations, metrics, and interactive dashboard
"""
import numpy as np
import torch
import cv2
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import imageio
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir

# Create comprehensive output structure
ensure_dir("assets/hero")
ensure_dir("assets/dashboard")
ensure_dir("assets/comparisons")
ensure_dir("assets/metrics")
ensure_dir("assets/reports")

class ComprehensiveResultsExporter:
    def __init__(self):
        """Initialize with all available data"""
        
        # Load model
        self.model = self._load_model()
        
        # Load all metadata
        self.tile_catalog = self._load_json('data/tiles/tile_catalog.json')
        self.training_metrics = self._load_json('outputs/metrics/training_report.json')
        self.acquisition_data = self._load_json('../data/acquisition.json')
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Analysis results
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'locations': {},
            'overall_metrics': {},
            'model_performance': {},
            'visualizations': []
        }
        
    def _load_model(self):
        """Load the trained model"""
        model_path = 'outputs/models/best_flood_model.pt'
        if not os.path.exists(model_path):
            model_path = 'outputs/models/final_flood_model.pt'
        
        if os.path.exists(model_path):
            print(f"✓ Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model architecture
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name="efficientnet-b0",
                encoder_weights=None,
                in_channels=3,
                classes=2
            )
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            model.eval()
            return model
        else:
            print("⚠ Model not found, using dummy predictions")
            return None
    
    def _load_json(self, path):
        """Safely load JSON file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_flood_detection_results(self):
        """Comprehensive analysis of flood detection results"""
        print("\n📊 Analyzing Flood Detection Results...")
        
        # Get unique locations
        locations = self.tile_catalog.get('locations', {})
        
        for location_name, location_info in locations.items():
            print(f"\n  Analyzing {location_name}...")
            
            # Get tiles for this location
            location_tiles = [t for t in self.tile_catalog['tiles'] 
                            if t['location'] == location_name]
            
            # Analyze by period
            period_analysis = {}
            for period in ['pre', 'flood', 'post']:
                period_tiles = [t for t in location_tiles if t['period'] == period]
                
                if period_tiles:
                    # Analyze flood coverage
                    flood_percentages = []
                    water_scores = []
                    
                    for tile_info in period_tiles[:10]:  # Sample tiles
                        tile = np.load(tile_info['path'])
                        
                        # Predict if model available
                        if self.model:
                            pred_mask = self._predict_flood(tile)
                            flood_percentage = (pred_mask > 0).mean() * 100
                        else:
                            # Use simple threshold
                            if len(tile.shape) == 3:
                                flood_percentage = (tile[:,:,0] < 0.3).mean() * 100
                            else:
                                flood_percentage = (tile < 0.3).mean() * 100
                        
                        flood_percentages.append(flood_percentage)
                        water_scores.append(tile_info.get('mean', 0))
                    
                    period_analysis[period] = {
                        'tile_count': len(period_tiles),
                        'avg_flood_coverage': np.mean(flood_percentages),
                        'max_flood_coverage': np.max(flood_percentages),
                        'min_flood_coverage': np.min(flood_percentages),
                        'avg_intensity': np.mean(water_scores)
                    }
            
            # Calculate flood evolution
            flood_increase = 0
            flood_decrease = 0
            
            if 'pre' in period_analysis and 'flood' in period_analysis:
                flood_increase = (period_analysis['flood']['avg_flood_coverage'] - 
                                period_analysis['pre']['avg_flood_coverage'])
            
            if 'flood' in period_analysis and 'post' in period_analysis:
                flood_decrease = (period_analysis['flood']['avg_flood_coverage'] - 
                                period_analysis['post']['avg_flood_coverage'])
            
            self.analysis_results['locations'][location_name] = {
                'total_tiles': location_info['total_tiles'],
                'periods_analyzed': list(period_analysis.keys()),
                'period_analysis': period_analysis,
                'flood_dynamics': {
                    'flood_increase_percentage': flood_increase,
                    'flood_decrease_percentage': flood_decrease,
                    'peak_flood_coverage': max([p['avg_flood_coverage'] 
                                               for p in period_analysis.values()])
                }
            }
        
        # Calculate overall metrics
        total_tiles = sum(loc['total_tiles'] for loc in self.analysis_results['locations'].values())
        avg_peak_flood = np.mean([loc['flood_dynamics']['peak_flood_coverage'] 
                                  for loc in self.analysis_results['locations'].values()])
        
        self.analysis_results['overall_metrics'] = {
            'total_locations': len(locations),
            'total_tiles_processed': total_tiles,
            'total_area_km2': total_tiles * 0.0625,  # Assuming 256x256m tiles
            'average_peak_flood_coverage': avg_peak_flood,
            'locations_analyzed': list(locations.keys())
        }
        
        # Add model performance
        if self.training_metrics:
            self.analysis_results['model_performance'] = {
                'best_dice_score': self.training_metrics.get('best_dice', 0),
                'final_iou': self.training_metrics.get('final_metrics', {}).get('iou', 0),
                'training_time_minutes': self.training_metrics.get('training_time_seconds', 0) / 60,
                'model_parameters': self.training_metrics.get('parameters', 0)
            }
        
        return self.analysis_results
    
    def create_professional_visualizations(self):
        """Create professional-grade visualizations"""
        print("\n🎨 Creating Professional Visualizations...")
        
        # 1. Multi-location comparison dashboard
        self._create_location_comparison_dashboard()
        
        # 2. Temporal evolution visualization
        self._create_temporal_evolution_chart()
        
        # 3. Model performance visualization
        self._create_model_performance_dashboard()
        
        # 4. Hero images
        self._create_hero_images()
        
        # 5. Executive summary infographic
        self._create_executive_summary()
    
    def _create_location_comparison_dashboard(self):
        """Create interactive location comparison"""
        
        locations = list(self.analysis_results['locations'].keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Flood Coverage by Location',
                'Temporal Evolution',
                'Peak Flood Intensity',
                'Recovery Analysis'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Flood coverage by location
        coverage_data = []
        for loc_name, loc_data in self.analysis_results['locations'].items():
            for period, analysis in loc_data['period_analysis'].items():
                coverage_data.append({
                    'Location': loc_name.replace('_', ' '),
                    'Period': period.capitalize(),
                    'Coverage': analysis['avg_flood_coverage']
                })
        
        import pandas as pd
        df = pd.DataFrame(coverage_data)
        
        for period in ['Pre', 'Flood', 'Post']:
            period_df = df[df['Period'] == period]
            fig.add_trace(
                go.Bar(
                    name=period,
                    x=period_df['Location'],
                    y=period_df['Coverage'],
                    text=period_df['Coverage'].round(1),
                    textposition='auto',
                ),
                row=1, col=1
            )
        
        # 2. Temporal evolution
        for loc_name, loc_data in self.analysis_results['locations'].items():
            periods = ['pre', 'flood', 'post']
            coverage = []
            for p in periods:
                if p in loc_data['period_analysis']:
                    coverage.append(loc_data['period_analysis'][p]['avg_flood_coverage'])
                else:
                    coverage.append(0)
            
            fig.add_trace(
                go.Scatter(
                    name=loc_name.replace('_', ' '),
                    x=['Pre-Flood', 'During Flood', 'Post-Flood'],
                    y=coverage,
                    mode='lines+markers',
                    line=dict(width=3),
                    marker=dict(size=10)
                ),
                row=1, col=2
            )
        
        # 3. Peak flood intensity
        peak_data = []
        for loc_name, loc_data in self.analysis_results['locations'].items():
            peak_data.append({
                'Location': loc_name.replace('_', ' '),
                'Peak': loc_data['flood_dynamics']['peak_flood_coverage']
            })
        
        peak_df = pd.DataFrame(peak_data)
        
        fig.add_trace(
            go.Bar(
                x=peak_df['Location'],
                y=peak_df['Peak'],
                text=peak_df['Peak'].round(1),
                textposition='outside',
                marker_color='indianred'
            ),
            row=2, col=1
        )
        
        # 4. Recovery analysis
        for loc_name, loc_data in self.analysis_results['locations'].items():
            increase = loc_data['flood_dynamics']['flood_increase_percentage']
            decrease = loc_data['flood_dynamics']['flood_decrease_percentage']
            
            fig.add_trace(
                go.Scatter(
                    name=loc_name.replace('_', ' '),
                    x=['Flood Onset', 'Peak', 'Recovery'],
                    y=[0, increase, increase - decrease],
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Bangladesh Flood Analysis Dashboard",
            title_font_size=24,
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Location", row=1, col=1)
        fig.update_xaxes(title_text="Time Period", row=1, col=2)
        fig.update_xaxes(title_text="Location", row=2, col=1)
        fig.update_xaxes(title_text="Phase", row=2, col=2)
        
        fig.update_yaxes(title_text="Coverage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Coverage (%)", row=1, col=2)
        fig.update_yaxes(title_text="Peak Coverage (%)", row=2, col=1)
        fig.update_yaxes(title_text="Change (%)", row=2, col=2)
        
        fig.write_html("assets/dashboard/location_comparison.html")
        print("   ✓ Location comparison dashboard created")
    
    def _create_temporal_evolution_chart(self):
        """Create temporal evolution visualization"""
        
        # Create animated chart showing flood evolution
        frames = []
        
        for period in ['pre', 'flood', 'post']:
            data = []
            for loc_name, loc_data in self.analysis_results['locations'].items():
                if period in loc_data['period_analysis']:
                    data.append({
                        'Location': loc_name.replace('_', ' '),
                        'Coverage': loc_data['period_analysis'][period]['avg_flood_coverage'],
                        'Period': period.capitalize()
                    })
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                
                frame = go.Frame(
                    data=[go.Bar(
                        x=df['Location'],
                        y=df['Coverage'],
                        text=df['Coverage'].round(1),
                        textposition='outside',
                        marker_color=df['Coverage'],
                        marker_colorscale='Blues'
                    )],
                    name=period
                )
                frames.append(frame)
        
        # Initial data
        initial_data = []
        for loc_name, loc_data in self.analysis_results['locations'].items():
            if 'pre' in loc_data['period_analysis']:
                initial_data.append({
                    'Location': loc_name.replace('_', ' '),
                    'Coverage': loc_data['period_analysis']['pre']['avg_flood_coverage']
                })
        
        import pandas as pd
        initial_df = pd.DataFrame(initial_data)
        
        fig = go.Figure(
            data=[go.Bar(
                x=initial_df['Location'],
                y=initial_df['Coverage'],
                text=initial_df['Coverage'].round(1),
                textposition='outside',
                marker_color=initial_df['Coverage'],
                marker_colorscale='Blues'
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000},
                            'transition': {'duration': 500}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0},
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'label': period.capitalize(),
                        'method': 'animate',
                        'args': [[period], {
                            'frame': {'duration': 0},
                            'transition': {'duration': 0}
                        }]
                    }
                    for period in ['pre', 'flood', 'post']
                ]
            }],
            title="Flood Evolution Timeline",
            xaxis_title="Location",
            yaxis_title="Flood Coverage (%)",
            height=600,
            template='plotly_white'
        )
        
        fig.write_html("assets/dashboard/temporal_evolution.html")
        print("   ✓ Temporal evolution chart created")
    
    def _create_model_performance_dashboard(self):
        """Create model performance visualization"""
        
        if not self.training_metrics:
            print("   ⚠ No training metrics found")
            return
        
        # Get metrics history
        history = self.training_metrics.get('full_history', {})
        if not history:
            history = self.training_metrics.get('metrics_history', {})
        
        if history:
            # Create performance plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Training Progress',
                    'Dice Score Evolution',
                    'IoU Score Evolution',
                    'Final Performance'
                )
            )
            
            epochs = list(range(1, len(history.get('train_loss', [])) + 1))
            
            # 1. Training progress
            if 'train_loss' in history:
                fig.add_trace(
                    go.Scatter(
                        name='Training Loss',
                        x=epochs,
                        y=history['train_loss'],
                        mode='lines',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(
                        name='Validation Loss',
                        x=epochs,
                        y=history['val_loss'],
                        mode='lines',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
            
            # 2. Dice score
            if 'dice_score' in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['dice_score'],
                        mode='lines+markers',
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=2
                )
            
            # 3. IoU score
            if 'iou_score' in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['iou_score'],
                        mode='lines+markers',
                        line=dict(color='purple', width=2),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
            
            # 4. Final performance metrics
            final_metrics = {
                'Dice Score': self.training_metrics.get('best_dice', 0),
                'IoU Score': self.training_metrics.get('final_metrics', {}).get('iou', 0),
                'Best Loss': self.training_metrics.get('best_val_loss', 0)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(final_metrics.keys()),
                    y=list(final_metrics.values()),
                    text=[f"{v:.3f}" for v in final_metrics.values()],
                    textposition='outside',
                    marker_color=['green', 'purple', 'orange']
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Model Performance Dashboard",
                showlegend=True,
                height=700,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_xaxes(title_text="Metric", row=2, col=2)
            
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Dice Score", row=1, col=2)
            fig.update_yaxes(title_text="IoU Score", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=2)
            
            fig.write_html("assets/dashboard/model_performance.html")
            print("   ✓ Model performance dashboard created")
    
    def _create_hero_images(self):
        """Create hero images for website"""
        print("   Creating hero images...")
        
        # Get best tiles from each period
        best_tiles = {
            'pre': None,
            'flood': None,
            'post': None
        }
        
        for period in ['pre', 'flood', 'post']:
            period_tiles = [t for t in self.tile_catalog['tiles'] 
                          if t['period'] == period and 'S1' in t['sensor']]
            
            if period_tiles:
                # Sort by variance (image quality)
                period_tiles.sort(key=lambda x: x.get('variance', 0), reverse=True)
                best_tiles[period] = period_tiles[0]
        
        # Create visualizations
        for period, tile_info in best_tiles.items():
            if tile_info:
                tile = np.load(tile_info['path'])
                
                # Create visualization
                if len(tile.shape) == 3:
                    # Use first channel for visualization
                    vis = tile[:,:,0]
                else:
                    vis = tile
                
                # Enhance contrast
                vis = np.clip(vis, 0, 1)
                p2, p98 = np.percentile(vis, (2, 98))
                vis = np.clip((vis - p2) / (p98 - p2), 0, 1)
                
                # Apply colormap
                vis_colored = cv2.applyColorMap((vis * 255).astype(np.uint8), 
                                               cv2.COLORMAP_OCEAN)
                
                # Add period label
                cv2.putText(vis_colored, f"{period.upper()} FLOOD", 
                          (20, 50), cv2.FONT_HERSHEY_BOLD, 1.5, (255, 255, 255), 3)
                
                # Add location
                location = tile_info['location'].replace('_', ' ')
                cv2.putText(vis_colored, location, 
                          (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save
                cv2.imwrite(f"assets/hero/{period}_hero.jpg", vis_colored,
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print("   ✓ Hero images created")
    
    def _create_executive_summary(self):
        """Create executive summary infographic"""
        
        # Create summary data
        summary = {
            'title': 'Bangladesh Flood Detection System - Executive Summary',
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'key_findings': {
                'Locations Analyzed': len(self.analysis_results['locations']),
                'Total Area (km²)': round(self.analysis_results['overall_metrics']['total_area_km2'], 1),
                'Peak Flood Coverage': f"{self.analysis_results['overall_metrics']['average_peak_flood_coverage']:.1f}%",
                'Total Tiles Processed': self.analysis_results['overall_metrics']['total_tiles_processed']
            },
            'model_performance': {
                'Dice Score': f"{self.analysis_results['model_performance'].get('best_dice_score', 0):.3f}",
                'IoU Score': f"{self.analysis_results['model_performance'].get('final_iou', 0):.3f}",
                'Training Time': f"{self.analysis_results['model_performance'].get('training_time_minutes', 0):.1f} min"
            },
            'locations': []
        }
        
        # Add location summaries
        for loc_name, loc_data in self.analysis_results['locations'].items():
            summary['locations'].append({
                'name': loc_name.replace('_', ' '),
                'tiles': loc_data['total_tiles'],
                'peak_flood': f"{loc_data['flood_dynamics']['peak_flood_coverage']:.1f}%",
                'flood_increase': f"{loc_data['flood_dynamics']['flood_increase_percentage']:.1f}%"
            })
        
        # Save as JSON
        with open('assets/reports/executive_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("   ✓ Executive summary created")
    
    def _predict_flood(self, tile):
        """Predict flood mask for a tile"""
        if self.model is None:
            # Simple threshold-based prediction
            if len(tile.shape) == 3:
                return (tile[:,:,0] < 0.3).astype(np.uint8)
            return (tile < 0.3).astype(np.uint8)
        
        # Prepare input
        if len(tile.shape) == 3:
            x = torch.tensor(tile.transpose(2,0,1)[None], dtype=torch.float32)
        else:
            x = torch.tensor(np.stack([tile]*3, axis=0)[None], dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x.to(self.device))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred_mask = pred.argmax(1)[0].cpu().numpy()
        
        return pred_mask
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n📝 Generating Final Report...")
        
        report = {
            'project': 'Bangladesh Flood Detection System',
            'version': '1.0',
            'generated': datetime.now().isoformat(),
            'executive_summary': {
                'objective': 'Advanced AI-based flood detection and monitoring for Bangladesh',
                'methodology': 'Deep learning segmentation using Sentinel-1 SAR imagery',
                'key_achievements': [
                    f"Analyzed {len(self.analysis_results['locations'])} flood-prone locations",
                    f"Processed {self.analysis_results['overall_metrics']['total_tiles_processed']} satellite tiles",
                    f"Achieved {self.analysis_results['model_performance'].get('best_dice_score', 0):.1%} detection accuracy",
                    f"Covered {self.analysis_results['overall_metrics']['total_area_km2']:.0f} km² total area"
                ]
            },
            'technical_details': {
                'data_sources': {
                    'satellite': 'Sentinel-1 SAR (VV, VH polarization)',
                    'locations': list(self.analysis_results['locations'].keys()),
                    'time_periods': ['Pre-flood', 'During flood', 'Post-flood']
                },
                'model_architecture': {
                    'type': 'U-Net with EfficientNet-B0 encoder',
                    'parameters': self.analysis_results['model_performance'].get('model_parameters', 0),
                    'training_time': f"{self.analysis_results['model_performance'].get('training_time_minutes', 0):.1f} minutes"
                },
                'performance_metrics': self.analysis_results['model_performance']
            },
            'findings': self.analysis_results,
            'visualizations_generated': [
                'assets/dashboard/location_comparison.html',
                'assets/dashboard/temporal_evolution.html',
                'assets/dashboard/model_performance.html',
                'assets/hero/*.jpg',
                'assets/reports/executive_summary.json'
            ],
            'recommendations': [
                'Deploy system for real-time flood monitoring',
                'Integrate with national disaster management systems',
                'Expand coverage to additional high-risk areas',
                'Implement early warning system based on predictions'
            ]
        }
        
        # Save comprehensive report
        with open('assets/reports/final_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown version
        self._create_markdown_report(report)
        
        print("   ✓ Final report generated")
        
        return report
    
    def _create_markdown_report(self, report):
        """Create markdown version of report"""
        
        md_content = f"""# Bangladesh Flood Detection System - Final Report

Generated: {report['generated']}

## Executive Summary

**Objective:** {report['executive_summary']['objective']}

**Methodology:** {report['executive_summary']['methodology']}

### Key Achievements
"""
        
        for achievement in report['executive_summary']['key_achievements']:
            md_content += f"- {achievement}\n"
        
        md_content += f"""

## Technical Details

### Data Sources
- **Satellite:** {report['technical_details']['data_sources']['satellite']}
- **Locations:** {', '.join(report['technical_details']['data_sources']['locations'])}
- **Time Periods:** {', '.join(report['technical_details']['data_sources']['time_periods'])}

### Model Performance
- **Dice Score:** {report['technical_details']['performance_metrics'].get('best_dice_score', 0):.3f}
- **IoU Score:** {report['technical_details']['performance_metrics'].get('final_iou', 0):.3f}
- **Training Time:** {report['technical_details']['performance_metrics'].get('training_time_minutes', 0):.1f} minutes

## Recommendations
"""
        
        for rec in report['recommendations']:
            md_content += f"1. {rec}\n"
        
        with open('assets/reports/final_report.md', 'w') as f:
            f.write(md_content)
    
    def run(self):
        """Execute complete results export"""
        print("="*80)
        print("COMPREHENSIVE RESULTS EXPORT")
        print("="*80)
        
        # Analyze results
        self.analyze_flood_detection_results()
        
        # Create visualizations
        self.create_professional_visualizations()
        
        # Generate final report
        report = self.generate_final_report()
        
        print("\n" + "="*80)
        print("✅ EXPORT COMPLETE!")
        print("="*80)
        print("\n📊 Generated Assets:")
        print("   - Interactive dashboards: assets/dashboard/")
        print("   - Hero images: assets/hero/")
        print("   - Final report: assets/reports/")
        print("\n🎯 Key Results:")
        print(f"   - Locations analyzed: {len(self.analysis_results['locations'])}")
        print(f"   - Total area covered: {self.analysis_results['overall_metrics']['total_area_km2']:.0f} km²")
        print(f"   - Peak flood coverage: {self.analysis_results['overall_metrics']['average_peak_flood_coverage']:.1f}%")
        print(f"   - Model accuracy (Dice): {self.analysis_results['model_performance'].get('best_dice_score', 0):.1%}")
        
        print("\n🚀 Ready for presentation!")
        print("   Open assets/dashboard/location_comparison.html to view interactive dashboard")
        
        return report


if __name__ == "__main__":
    exporter = ComprehensiveResultsExporter()
    exporter.run()