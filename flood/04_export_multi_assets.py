"""
Generate Stunning Multi-Location Visualizations
Creates SOTD-worthy assets showcasing multiple Bangladesh flood events
"""
import numpy as np, torch, cv2, json, plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from common.utils import save_img, ensure_dir
from datetime import datetime
import imageio

ensure_dir("assets/hero")
ensure_dir("assets/dashboard")
ensure_dir("assets/comparisons")

class MultiLocationVisualizer:
    def __init__(self):
        # Load trained model
        checkpoint = torch.load('outputs/models/best_multi_location_model.pt')
        from train_script import LocationAwareFloodModel
        self.model = LocationAwareFloodModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load metadata
        with open('data/tile_catalog.json', 'r') as f:
            self.catalog = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def create_location_comparison(self):
        """Create stunning comparison across locations"""
        print("🎨 Creating multi-location comparison...")
        
        locations = list(self.catalog['metadata']['locations'].keys())
        
        # Create comparison grid
        fig = make_subplots(
            rows=len(locations), 
            cols=3,
            subplot_titles=['Pre-Flood', 'During Flood', 'Post-Flood'] * len(locations),
            vertical_spacing=0.02,
            horizontal_spacing=0.02
        )
        
        comparison_images = []
        
        for loc_idx, location in enumerate(locations):
            location_tiles = [t for t in self.catalog['tiles'] if t['location'] == location]
            
            for period_idx, period in enumerate(['pre', 'flood', 'post']):
                # Get best tile for this period
                period_tiles = [t for t in location_tiles if t['period'] == period]
                
                if period_tiles:
                    # Process tile
                    tile = np.load(period_tiles[0]['path'])
                    
                    # Predict flood
                    pred_mask = self.predict_flood(tile)
                    
                    # Create visualization
                    if len(tile.shape) == 2:
                        base = cv2.applyColorMap((tile * 255).astype(np.uint8), cv2.COLORMAP_BONE)
                    else:
                        base = (tile[:,:,:3] * 255).astype(np.uint8) if tile.shape[2] >= 3 else \
                               cv2.applyColorMap((tile[:,:,0] * 255).astype(np.uint8), cv2.COLORMAP_BONE)
                    
                    # Overlay flood prediction
                    overlay = base.copy()
                    flood_color = [255, 100, 100] if period == 'flood' else [100, 100, 255]
                    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array(flood_color) * 0.5
                    
                    comparison_images.append({
                        'location': location,
                        'period': period,
                        'image': overlay
                    })
                    
                    # Add to plotly figure
                    fig.add_trace(
                        go.Heatmap(z=pred_mask, colorscale='Blues', showscale=False),
                        row=loc_idx+1, col=period_idx+1
                    )
        
        # Save comparison figure
        fig.update_layout(
            title="Multi-Location Flood Analysis: Bangladesh",
            height=300 * len(locations),
            width=1200
        )
        fig.write_html("assets/comparisons/multi_location_analysis.html")
        
        # Create animated GIF showing all locations
        self.create_location_animation(comparison_images)
        
        print("   ✓ Multi-location comparison created")
    
    def create_location_animation(self, images):
        """Create smooth animation across locations and time"""
        frames = []
        
        for img_data in images:
            # Add location and period text
            img = img_data['image'].copy()
            cv2.putText(img, f"{img_data['location']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"{img_data['period'].upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            frames.append(img)
        
        # Save as GIF and MP4
        imageio.mimsave("assets/hero/location_timeline.gif", frames, fps=2)
        imageio.mimsave("assets/hero/location_timeline.mp4", frames, fps=2)
    
    def create_dashboard_data(self):
        """Generate comprehensive dashboard data"""
        dashboard = {
            "project": "Bangladesh Flood Detection System",
            "generated": datetime.now().isoformat(),
            "locations": [],
            "overall_statistics": {
                "total_locations": len(self.catalog['metadata']['locations']),
                "total_tiles_processed": self.catalog['metadata']['total_tiles'],
                "total_area_covered_km2": 0,
                "peak_flood_coverage": 0
            },
            "location_details": {}
        }
        
        # Process each location
        for location, stats in self.catalog['metadata']['locations'].items():
            location_data = {
                "name": location,
                "tiles": stats['total_tiles'],
                "sensors": stats['sensors'],
                "periods": stats['periods'],
                "water_detection_rate": stats.get('water_tiles', 0) / max(stats['total_tiles'], 1),
                "quality_score": stats.get('avg_variance', 0) * 100
            }
            
            dashboard["locations"].append(location_data)
            dashboard["location_details"][location] = location_data
            
            # Update overall stats
            dashboard["overall_statistics"]["total_area_covered_km2"] += stats['total_tiles'] * 0.25  # Estimate
            dashboard["overall_statistics"]["peak_flood_coverage"] = max(
                dashboard["overall_statistics"]["peak_flood_coverage"],
                location_data["water_detection_rate"]
            )
        
        # Add insights
        dashboard["insights"] = [
            f"Analyzed {len(dashboard['locations'])} major flood events across Bangladesh",
            f"Processed {dashboard['overall_statistics']['total_tiles_processed']} satellite image tiles",
            f"Peak flood coverage reached {dashboard['overall_statistics']['peak_flood_coverage']*100:.1f}%",
            f"Total monitored area: {dashboard['overall_statistics']['total_area_covered_km2']:.0f} km²"
        ]
        
        # Save dashboard data
        with open("assets/dashboard/multi_location_data.json", 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        print("   ✓ Dashboard data generated")
        
        return dashboard
    
    def predict_flood(self, tile):
        """Predict flood mask for a tile"""
        # Prepare input
        if len(tile.shape) == 2:
            x = torch.tensor(tile[None, None], dtype=torch.float32)
        else:
            x = torch.tensor(tile.transpose(2,0,1)[None], dtype=torch.float32)
        
        # Ensure minimum size
        if x.shape[-1] < 256:
            pad_h = max(0, 256 - x.shape[-2])
            pad_w = max(0, 256 - x.shape[-1])
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x)
            pred_mask = pred.argmax(1)[0].cpu().numpy()
        
        # Crop back to original size if padded
        if len(tile.shape) == 2:
            pred_mask = pred_mask[:tile.shape[0], :tile.shape[1]]
        
        return pred_mask
    
    def generate_hero_visuals(self):
        """Create hero visuals for website"""
        print("🎨 Generating hero visuals...")
        
        # Find the best tiles for hero imagery
        best_flood_tiles = [t for t in self.catalog['tiles'] 
                           if t['period'] == 'flood' and t.get('water_score', 0) > 0.3]
        
        if best_flood_tiles:
            # Sort by water score
            best_flood_tiles.sort(key=lambda x: x.get('water_score', 0), reverse=True)
            
            # Create hero image from best flood tile
            hero_tile = np.load(best_flood_tiles[0]['path'])
            pred_mask = self.predict_flood(hero_tile)
            
            # Create stunning visualization
            if len(hero_tile.shape) == 2:
                base = cv2.applyColorMap((hero_tile * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
            else:
                base = (hero_tile[:,:,:3] * 255).astype(np.uint8)
            
            # Water overlay with gradient
            water_overlay = np.zeros_like(base)
            water_overlay[:,:,0] = pred_mask * 100  # Blue channel
            water_overlay[:,:,1] = pred_mask * 150
            water_overlay[:,:,2] = pred_mask * 255
            
            # Blend
            hero_image = cv2.addWeighted(base, 0.6, water_overlay, 0.4, 0)
            
            # Add gradient overlay for depth effect
            h, w = hero_image.shape[:2]
            gradient = np.linspace(0, 1, h).reshape(h, 1)
            gradient = np.tile(gradient, (1, w))
            gradient = np.stack([gradient * 50]*3, axis=-1).astype(np.uint8)
            
            hero_image = cv2.addWeighted(hero_image, 0.9, gradient, 0.1, 0)
            
            # Save hero image
            cv2.imwrite("assets/hero/main_hero_flood.jpg", hero_image,
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print("   ✓ Hero visuals created")
    
    def run(self):
        """Generate all assets"""
        print("="*60)
        print("GENERATING MULTI-LOCATION VISUALIZATION ASSETS")
        print("="*60)
        
        self.create_location_comparison()
        self.generate_hero_visuals()
        dashboard_data = self.create_dashboard_data()
        
        print("\n✅ All assets generated successfully!")
        print(f"   Locations analyzed: {len(dashboard_data['locations'])}")
        print(f"   Files created in: assets/")
        
        return dashboard_data

if __name__ == "__main__":
    visualizer = MultiLocationVisualizer()
    visualizer.run()