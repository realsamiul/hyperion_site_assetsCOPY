"""
Discover available flood imagery in Bangladesh
Finds actual dates with good satellite coverage
"""
import ee
import sys
import json
from datetime import datetime, timedelta
sys.path.append('..')
from common.config import GCP_PROJECT

# Initialize Earth Engine
ee.Initialize(project=GCP_PROJECT)
print("🌍 Earth Engine initialized")

class FloodImageryDiscovery:
    def __init__(self):
        self.results = []
        
    def search_location(self, name, lat, lon, year, months):
        """Search for imagery around a location during flood season"""
        print(f"\n🔍 Searching {name} ({lat}, {lon}) for {year}...")
        
        # Create a point and buffer to search area
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(50000)  # 50km radius
        
        results = {
            "location": name,
            "coordinates": [lat, lon],
            "year": year,
            "periods": {}
        }
        
        # Check each month
        for month in months:
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year+1}-01-01"
            else:
                end_date = f"{year}-{month+1:02d}-01"
            
            # Check Sentinel-1
            s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            # Check Sentinel-2
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
            
            s1_count = s1.size().getInfo()
            s2_count = s2.size().getInfo()
            
            if s1_count > 0 or s2_count > 0:
                # Get actual dates of images
                s1_dates = []
                s2_dates = []
                
                if s1_count > 0 and s1_count < 20:  # Limit to avoid timeout
                    s1_list = s1.limit(10).aggregate_array('system:time_start').getInfo()
                    s1_dates = [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in s1_list]
                
                if s2_count > 0 and s2_count < 20:
                    s2_list = s2.limit(10).aggregate_array('system:time_start').getInfo()
                    s2_dates = [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in s2_list]
                
                month_name = datetime(year, month, 1).strftime('%B')
                results["periods"][month_name] = {
                    "s1_count": s1_count,
                    "s2_count": s2_count,
                    "s1_sample_dates": s1_dates[:3] if s1_dates else [],
                    "s2_sample_dates": s2_dates[:3] if s2_dates else [],
                    "date_range": [start_date, end_date]
                }
                
                print(f"   {month_name}: {s1_count} SAR, {s2_count} optical images")
        
        return results
    
    def find_best_floods(self):
        """Search known flood events in Bangladesh"""
        
        # Major flood events with verified locations
        flood_events = [
            # 2024 Recent floods
            {"name": "Sylhet_2024", "lat": 24.8949, "lon": 91.8687, "year": 2024, "months": [5, 6, 7, 8]},
            {"name": "Feni_2024", "lat": 23.0159, "lon": 91.3976, "year": 2024, "months": [7, 8, 9]},
            
            # 2023 floods
            {"name": "Chittagong_2023", "lat": 22.3569, "lon": 91.7832, "year": 2023, "months": [6, 7, 8]},
            {"name": "Cox_Bazar_2023", "lat": 21.4272, "lon": 92.0058, "year": 2023, "months": [5, 6, 7, 8]},
            
            # 2022 Major floods (well documented)
            {"name": "Sylhet_2022", "lat": 24.8949, "lon": 91.8687, "year": 2022, "months": [5, 6, 7]},
            {"name": "Sunamganj_2022", "lat": 25.0658, "lon": 91.3950, "year": 2022, "months": [5, 6, 7]},
            {"name": "Netrokona_2022", "lat": 24.8766, "lon": 90.7278, "year": 2022, "months": [6, 7, 8]},
            
            # 2020 floods
            {"name": "Kurigram_2020", "lat": 25.8074, "lon": 89.6362, "year": 2020, "months": [6, 7, 8]},
            {"name": "Gaibandha_2020", "lat": 25.3297, "lon": 89.5430, "year": 2020, "months": [6, 7, 8]},
        ]
        
        all_results = []
        best_locations = []
        
        for event in flood_events:
            result = self.search_location(
                event["name"], 
                event["lat"], 
                event["lon"],
                event["year"],
                event["months"]
            )
            all_results.append(result)
            
            # Calculate total available imagery
            total_s1 = sum(p["s1_count"] for p in result["periods"].values())
            total_s2 = sum(p["s2_count"] for p in result["periods"].values())
            
            if total_s1 > 10 or total_s2 > 5:
                best_locations.append({
                    "name": event["name"],
                    "year": event["year"],
                    "total_s1": total_s1,
                    "total_s2": total_s2,
                    "score": total_s1 + total_s2 * 2,  # Weight optical higher
                    "details": result
                })
        
        # Sort by score
        best_locations.sort(key=lambda x: x["score"], reverse=True)
        
        print("\n" + "="*60)
        print("🏆 BEST LOCATIONS WITH IMAGERY:")
        print("="*60)
        
        for i, loc in enumerate(best_locations[:5], 1):
            print(f"\n{i}. {loc['name']} ({loc['year']})")
            print(f"   Total: {loc['total_s1']} SAR, {loc['total_s2']} optical images")
            print(f"   Score: {loc['score']}")
            
            # Show best months
            best_month = max(loc['details']['periods'].items(), 
                           key=lambda x: x[1]['s1_count'] + x[1]['s2_count'])
            print(f"   Best month: {best_month[0]} with {best_month[1]['s1_count']} SAR, {best_month[1]['s2_count']} optical")
        
        # Save results
        with open('discovered_imagery.json', 'w') as f:
            json.dump({
                "search_time": datetime.now().isoformat(),
                "all_results": all_results,
                "best_locations": best_locations[:5]
            }, f, indent=2)
        
        print(f"\n✅ Results saved to discovered_imagery.json")
        
        return best_locations

if __name__ == "__main__":
    discovery = FloodImageryDiscovery()
    best = discovery.find_best_floods()
    
    if best:
        print("\n📋 RECOMMENDED CONFIGURATION:")
        print("Copy this to your config.py:\n")
        
        # Generate config for best location
        top = best[0]
        details = top['details']
        
        # Find pre, flood, and post months
        months = list(details['periods'].keys())
        if len(months) >= 3:
            pre_month = months[0]
            flood_month = months[len(months)//2]
            post_month = months[-1]
            
            print(f"""BANGLADESH_FLOODS = [
    {{
        "name": "{top['name']}",
        "aoi": {[details['coordinates'][0]-0.3, details['coordinates'][1]-0.3, 
                details['coordinates'][0]+0.3, details['coordinates'][1]+0.3]},
        "periods": {{
            "pre": {{"start": "{details['periods'][pre_month]['date_range'][0]}", 
                    "end": "{details['periods'][pre_month]['date_range'][1][:-3]}-15"}},
            "flood": {{"start": "{details['periods'][flood_month]['date_range'][0]}", 
                      "end": "{details['periods'][flood_month]['date_range'][1][:-3]}-15"}},
            "post": {{"start": "{details['periods'][post_month]['date_range'][0]}", 
                     "end": "{details['periods'][post_month]['date_range'][1][:-3]}-15"}}
        }}
    }}
]""")