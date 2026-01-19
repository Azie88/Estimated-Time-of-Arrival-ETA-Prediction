import gradio as gr
import numpy as np
import pandas as pd
import os, joblib
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import folium
from folium import plugins
import requests

# Load model pipeline
file_path = os.path.abspath('toolkit/pipeline.joblib')
pipeline = joblib.load(file_path)

# Global variables for dynamic bounds
current_bounds = {
    'lat_min': -90,
    'lat_max': 90,
    'lon_min': -180,
    'lon_max': 180,
    'center_lat': 0,
    'center_lon': 0
}

# Cluster centroids - these will be auto-generated based on location
CLUSTER_CENTROIDS = {}

def geocode_location(place_name):
    """
    Convert place name/address to coordinates using Nominatim (OpenStreetMap)
    Free and no API key required!
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'ETA-Prediction-App/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                display_name = data[0]['display_name']
                return lat, lon, display_name, None
            else:
                return None, None, None, "Location not found. Please try a different search term."
        else:
            return None, None, None, f"Geocoding service error: {response.status_code}"
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

def reverse_geocode(lat, lon):
    """
    Convert coordinates to place name
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }
        headers = {
            'User-Agent': 'ETA-Prediction-App/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('display_name', 'Unknown location')
        else:
            return f"Lat: {lat:.4f}, Lon: {lon:.4f}"
    except:
        return f"Lat: {lat:.4f}, Lon: {lon:.4f}"

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def generate_clusters_for_area(center_lat, center_lon, radius_km=50):
    """
    Generate cluster centroids around a center point
    Creates a grid of 10 clusters
    """
    clusters = {}
    # Create a 3x3 grid plus center point = 10 clusters
    positions = [
        (0, 0),      # Center - cluster 0
        (-1, -1), (-1, 0), (-1, 1),  # Top row - clusters 1,2,3
        (0, -1), (0, 1),              # Middle row - clusters 4,5
        (1, -1), (1, 0), (1, 1)       # Bottom row - clusters 6,7,8
    ]
    
    # Convert km to degrees (approximate)
    lat_offset = radius_km / 111.0  # 1 degree latitude ≈ 111 km
    lon_offset = radius_km / (111.0 * cos(radians(center_lat)))
    
    for i, (lat_mult, lon_mult) in enumerate(positions):
        if i < 10:  # Only 10 clusters
            clusters[i] = (
                center_lat + (lat_mult * lat_offset / 3),
                center_lon + (lon_mult * lon_offset / 3)
            )
    
    # Add 10th cluster if needed
    if len(clusters) < 10:
        clusters[9] = (center_lat + lat_offset/6, center_lon + lon_offset/6)
    
    return clusters

def assign_cluster(lat, lon, clusters):
    """
    Assign cluster ID based on nearest centroid
    """
    if not clusters:
        return 4  # default middle cluster
    
    min_dist = float('inf')
    assigned_cluster = 4
    
    for cluster_id, (c_lat, c_lon) in clusters.items():
        dist = haversine_distance(lat, lon, c_lat, c_lon)
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = cluster_id
    
    return assigned_cluster

def create_interactive_map(origin_coords=None, dest_coords=None, zoom=12):
    """
    Create an interactive Folium map for visualizing route
    """
    if origin_coords:
        center = origin_coords
    elif dest_coords:
        center = dest_coords
    else:
        center = [0, 0]
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    # Add markers if coordinates are provided
    if origin_coords:
        folium.Marker(
            origin_coords,
            popup=f"<b>Origin</b><br>{origin_coords[0]:.4f}, {origin_coords[1]:.4f}",
            tooltip="Pickup Location",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
    
    if dest_coords:
        folium.Marker(
            dest_coords,
            popup=f"<b>Destination</b><br>{dest_coords[0]:.4f}, {dest_coords[1]:.4f}",
            tooltip="Dropoff Location",
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)
    
    # Draw line between origin and destination
    if origin_coords and dest_coords:
        folium.PolyLine(
            [origin_coords, dest_coords],
            color='blue',
            weight=4,
            opacity=0.8,
            popup=f"Distance: {haversine_distance(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1])/1000:.2f} km"
        ).add_to(m)
        
        # Fit bounds to show both markers
        m.fit_bounds([origin_coords, dest_coords])
    
    return m._repr_html_()

def calculate_pickup_week_hour(pickup_hour, pickup_weekday):
    return pickup_weekday * 24 + pickup_hour

def format_eta_output(seconds):
    """
    Convert seconds to human-readable format
    """
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes} min {secs} sec" if secs > 0 else f"{minutes} min"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}min"

def search_origin(search_term):
    """Handle origin location search"""
    if not search_term.strip():
        return None, None, "Please enter a location to search"
    
    lat, lon, display_name, error = geocode_location(search_term)
    
    if error:
        return None, None, f"❌ {error}"
    
    return lat, lon, f"✅ Found: {display_name}"

def search_destination(search_term):
    """Handle destination location search"""
    if not search_term.strip():
        return None, None, "Please enter a location to search"
    
    lat, lon, display_name, error = geocode_location(search_term)
    
    if error:
        return None, None, f"❌ {error}"
    
    return lat, lon, f"✅ Found: {display_name}"

def predict(origin_lat, origin_lon, dest_lat, dest_lon,
            dewpoint_temp, min_temp, temp_range, rain,
            pickup_weekday, pickup_hour, manual_distance, auto_calc_distance, cluster_override):
    
    # Validate that coordinates are provided
    if origin_lat is None or origin_lon is None:
        return "❌ Please provide origin coordinates", ""
    if dest_lat is None or dest_lon is None:
        return "❌ Please provide destination coordinates", ""
    
    # Generate clusters around origin area
    clusters = generate_clusters_for_area(origin_lat, origin_lon)
    
    # Calculate or use manual distance
    if auto_calc_distance:
        trip_distance = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        distance_info = f"📏 Auto-calculated distance: {trip_distance:.0f}m ({trip_distance/1000:.2f}km)"
    else:
        trip_distance = manual_distance
        distance_info = f"📏 Manual distance: {trip_distance:.0f}m ({trip_distance/1000:.2f}km)"
    
    # Auto-assign cluster or use override
    if cluster_override == "Auto":
        cluster_id = assign_cluster(origin_lat, origin_lon, clusters)
        cluster_info = f"🗺️ Auto-assigned cluster: {cluster_id}"
    else:
        cluster_id = int(cluster_override)
        cluster_info = f"🗺️ Manual cluster: {cluster_id}"
    
    # Calculate pickup_week_hour
    pickup_week_hour = calculate_pickup_week_hour(pickup_hour, pickup_weekday)
    
    # Get location names
    origin_name = reverse_geocode(origin_lat, origin_lon)
    dest_name = reverse_geocode(dest_lat, dest_lon)
    
    # Prediction
    try:
        model_output = abs(int(pipeline.predict(pd.DataFrame([[
            origin_lat, origin_lon, dest_lat, dest_lon,
            trip_distance, dewpoint_temp, min_temp,
            pickup_weekday, pickup_hour, pickup_week_hour,
            cluster_id, temp_range, rain
        ]], columns=[
            'Origin_lat', 'Origin_lon', 'Destination_lat', 'Destination_lon',
            'Trip_distance', 'dewpoint_2m_temperature', 'minimum_2m_air_temperature',
            'pickup_weekday', 'pickup_hour', 'pickup_week_hour',
            'cluster_id', 'temperature_range', 'rain'
        ]))[0]))
        
        # Format output
        eta_formatted = format_eta_output(model_output)
        
        # Calculate average speed
        if model_output > 0:
            avg_speed_mps = trip_distance / model_output
            avg_speed_kmh = avg_speed_mps * 3.6
            speed_info = f"🚗 Average speed: {avg_speed_kmh:.1f} km/h"
        else:
            speed_info = "⚠️ Invalid ETA prediction"
        
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        result = f"""
# 🎯 Estimated Time of Arrival

## ⏱️ **ETA: {eta_formatted}** ({model_output} seconds)

---

### 📍 Route Information

**Origin:** {origin_name}  
*Coordinates:* {origin_lat:.6f}, {origin_lon:.6f}

**Destination:** {dest_name}  
*Coordinates:* {dest_lat:.6f}, {dest_lon:.6f}

{distance_info}  
{cluster_info}

---

### 🕐 Trip Details

- **Day:** {weekdays[pickup_weekday]}
- **Time:** {pickup_hour:02d}:00
- {speed_info}

---

### 🌤️ Weather Conditions

- **Dewpoint:** {dewpoint_temp:.1f}K ({dewpoint_temp-273.15:.1f}°C)
- **Min Temperature:** {min_temp:.1f}K ({min_temp-273.15:.1f}°C)  
- **Temperature Range:** {temp_range:.1f}K
- **Rain:** {"Yes ☔" if rain == 1 else "No ☀️"}
"""
        
        # Update map
        map_html = create_interactive_map(
            origin_coords=[origin_lat, origin_lon],
            dest_coords=[dest_lat, dest_lon]
        )
        
        return result, map_html
        
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}\n\nPlease check your inputs and ensure the model file is loaded correctly.", ""

def update_map_only(origin_lat, origin_lon, dest_lat, dest_lon):
    """Update map visualization without prediction"""
    if origin_lat and origin_lon and dest_lat and dest_lon:
        return create_interactive_map(
            origin_coords=[origin_lat, origin_lon],
            dest_coords=[dest_lat, dest_lon]
        )
    elif origin_lat and origin_lon:
        return create_interactive_map(origin_coords=[origin_lat, origin_lon])
    elif dest_lat and dest_lon:
        return create_interactive_map(dest_coords=[dest_lat, dest_lon])
    else:
        return create_interactive_map()

def use_current_time():
    """Get current day and hour"""
    now = datetime.now()
    return now.weekday(), now.hour

def use_sample_nairobi():
    """Load sample Nairobi coordinates"""
    return 2.950, 36.700, 3.000, 36.750, "Sample: Nairobi area loaded ✅"

def use_sample_newyork():
    """Load sample New York coordinates"""
    return 40.7580, -73.9855, 40.7128, -74.0060, "Sample: New York (Times Square → Downtown) loaded ✅"

def use_sample_london():
    """Load sample London coordinates"""
    return 51.5074, -0.1278, 51.5155, -0.0922, "Sample: London (Westminster → Tower Bridge) loaded ✅"

# UI Layout
with gr.Blocks(title="Universal ETA Prediction") as app:
    gr.Markdown("# 🌍 Universal ETA Prediction App")
    gr.Markdown("""
    Predict ride ETA for **any location worldwide**. Search for places by name or enter coordinates manually.
    """)
    
    # First Row - Two Columns for Inputs
    with gr.Row():
        # Left Column - Location & Time Settings
        with gr.Column(scale=1):
            gr.Markdown("### 📍 Location Settings")
            
            # Quick sample locations
            gr.Markdown("**Quick Load Samples:**")
            with gr.Row():
                sample_nairobi = gr.Button("📍 Nairobi", size="sm")
                sample_ny = gr.Button("📍 New York", size="sm")
                sample_london = gr.Button("📍 London", size="sm")
            
            sample_status = gr.Textbox(label="Status", interactive=False, visible=False)
            
            # Origin section
            gr.Markdown("**🟢 Origin (Pickup Location)**")
            origin_search = gr.Textbox(
                label="Search for origin", 
                placeholder="e.g., Times Square, New York OR Nairobi CBD",
                info="Type a place name, address, or landmark"
            )
            origin_search_btn = gr.Button("🔍 Search Origin", size="sm", variant="secondary")
            origin_search_status = gr.Textbox(label="Search Result", interactive=False, visible=False)
            
            with gr.Row():
                origin_lat = gr.Number(label='Origin Latitude', precision=6, value=None)
                origin_lon = gr.Number(label='Origin Longitude', precision=6, value=None)
            
            gr.Markdown("---")
            
            # Destination section
            gr.Markdown("**🔴 Destination (Dropoff Location)**")
            dest_search = gr.Textbox(
                label="Search for destination",
                placeholder="e.g., Central Park, New York OR JKIA Airport",
                info="Type a place name, address, or landmark"
            )
            dest_search_btn = gr.Button("🔍 Search Destination", size="sm", variant="secondary")
            dest_search_status = gr.Textbox(label="Search Result", interactive=False, visible=False)
            
            with gr.Row():
                dest_lat = gr.Number(label='Destination Latitude', precision=6, value=None)
                dest_lon = gr.Number(label='Destination Longitude', precision=6, value=None)
            
            gr.Markdown("---")
            
            with gr.Row():
                auto_calc_distance = gr.Checkbox(
                    value=True, 
                    label="✓ Auto-calculate distance",
                    info="Recommended for accuracy"
                )
                manual_distance = gr.Number(
                    value=5000, label='Manual Distance (meters)', 
                    visible=False
                )
            
            cluster_override = gr.Dropdown(
                choices=["Auto"] + [str(i) for i in range(10)],
                value="Auto",
                label="Cluster ID",
                info="Auto-assigns based on location"
            )
            
            gr.Markdown("### 🕐 Time Settings")
            with gr.Row():
                use_current = gr.Button("📅 Use Current Date/Time", size="sm")
            
            with gr.Row():
                pickup_weekday = gr.Dropdown(
                    choices=[
                        ("Monday", 0), ("Tuesday", 1), ("Wednesday", 2),
                        ("Thursday", 3), ("Friday", 4), ("Saturday", 5), ("Sunday", 6)
                    ],
                    value=datetime.now().weekday(),
                    label='Pickup Day'
                )
                pickup_hour = gr.Dropdown(
                    choices=list(range(24)),
                    value=datetime.now().hour,
                    label='Pickup Hour (24h)'
                )
        
        # Right Column - Weather Settings
        with gr.Column(scale=1):
            gr.Markdown("### 🌤️ Weather Conditions")
            gr.Markdown("**⚠️ Manual Input Required** - Adjust sliders based on current weather")
            
            dewpoint_temp = gr.Slider(
                minimum=279.129,
                maximum=286.327,
                step=0.1,
                value=282.201,
                label='Dewpoint Temperature (Kelvin)',
                info="Humidity indicator: 279K = 6°C | 286K = 13°C"
            )
            
            min_temp = gr.Slider(
                minimum=282.037,
                maximum=292.543,
                step=0.1,
                value=285.203,
                label='Minimum Air Temperature (Kelvin)',
                info="Daily minimum: 282K = 9°C | 293K = 20°C"
            )
            
            temp_range = gr.Slider(
                minimum=1.663,
                maximum=10.022,
                step=0.1,
                value=5.0,
                label='Temperature Range (Kelvin)',
                info="Daily variation: Typical = 5-8K"
            )
            
            rain = gr.Dropdown(
                choices=[("No Rain ☀️", 0), ("Raining ☔", 1)],
                value=0,
                label='Precipitation Status'
            )
            
            gr.Markdown("---")
            gr.Markdown("**Quick Temperature Conversions:**")
            gr.Markdown("""
            - **Dewpoint:** 282K ≈ 9°C ≈ 48°F
            - **Min Temp:** 285K ≈ 12°C ≈ 54°F
            - **To convert:** °C = K - 273.15
            """)
    
    # Action Buttons Row
    with gr.Row():
        update_map_btn = gr.Button("🗺️ Update Map", variant="secondary", scale=1)
        predict_btn = gr.Button("🚀 Predict ETA", variant="primary", size="lg", scale=2)
    
    # Results Row - Full Width
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Prediction Results")
            output = gr.Markdown(value="*Enter locations and click 'Predict ETA' to see results*")
    
    # Map Row - Full Width
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🗺️ Route Visualization")
            map_output = gr.HTML(value=create_interactive_map())
    
    # Expander for help
    with gr.Accordion("ℹ️ How to Use This App", open=False):
        gr.Markdown("""
        ### 🎯 Three Ways to Set Locations:
        
        1. **Search by Name (Easiest - AUTOMATIC):**
           - Type any place name: "Eiffel Tower", "Tokyo Station", "Central Park"
           - Type addresses: "123 Main St, Boston"
           - Click the search button to get coordinates automatically
           - ✅ Fully automated - no manual input needed!
        
        2. **Use Quick Samples (AUTOMATIC):**
           - Click Nairobi, New York, or London buttons for instant examples
           - ✅ One-click automation
        
        3. **Enter Coordinates Manually (MANUAL):**
           - If you know exact lat/lon, enter them directly
           - Useful for precise locations or GPS coordinates
           - ⚠️ Requires manual entry
        
        ### 🔍 Location Search Tips:
        - Be specific: "JFK Airport" is better than just "airport"
        - Include city/country for common names: "Central Park, New York"
        - Landmarks work great: "Statue of Liberty", "Big Ben"
        - The geocoder uses OpenStreetMap (free, no API key needed!)
        
        ### 📏 Distance & Clusters (AUTOMATIC):
        - **Distance**: Auto-calculated using Haversine formula (great circle distance)
        - **Clusters**: Automatically assigned based on your origin location
        - Works anywhere in the world - no geographic restrictions!
        - ✅ No manual input required
        
        ### 🕐 Time Settings (SEMI-AUTOMATIC):
        - Click "Use Current Date/Time" for instant population ✅
        - Or manually select day and hour if predicting for future trips ⚠️
        
        ### 🌤️ Weather Parameters (MANUAL - See Details Below):
        
        **⚠️ These require manual input currently:**
        
        1. **Dewpoint Temperature (K):**
           - The temperature at which air becomes saturated and dew forms
           - Indicates humidity level - higher dewpoint = more humid
           - Range: 279-286K (6-13°C) based on training data
           - To convert from Celsius: K = °C + 273.15
           - Example: 10°C = 283.15K
           - **Where to get it:** Weather websites/APIs (OpenWeatherMap, WeatherAPI)
        
        2. **Minimum 2m Air Temperature (K):**
           - The lowest temperature at 2 meters above ground for that day
           - Range: 282-293K (9-20°C) based on training data
           - Usually occurs early morning (5-7 AM)
           - **Where to get it:** Historical weather data or daily forecast minimums
        
        3. **Temperature Range (K):**
           - Daily temperature variation (max temp - min temp)
           - Range: 1.7-10K based on training data
           - Typical values: 5-8K for moderate climates
           - Example: If max=25°C and min=15°C, range = 10K
           - **Where to get it:** Calculate from daily max/min temps
        
        4. **Rain (0 or 1):**
           - Binary indicator: 0 = No rain, 1 = Rain/Precipitation
           - Check current weather or forecast
           - **Where to get it:** Weather apps, look outside, or weather APIs
        
        ### 🔧 Future Enhancement - Weather API Integration:
        For fully automated weather data, consider integrating:
        - **OpenWeatherMap API** (free tier: 1000 calls/day)
        - **WeatherAPI** (free tier: 1M calls/month)
        - **Tomorrow.io** (free tier available)
        
        These APIs can auto-populate all weather fields based on location and time!
        
        ### 📊 Summary - What's Automatic vs Manual:
        
        **✅ AUTOMATIC (No manual input needed):**
        - Location coordinates (via search)
        - Trip distance
        - Cluster assignment
        - Current date/time (optional button)
        
        **⚠️ MANUAL (Requires user input):**
        - Weather parameters (4 fields)
        - Future date/time (if not using current)
        - Coordinates (if not using search)
        
        ### ⚠️ Important Notes:
        - This model was trained on specific data (East Africa region)
        - Predictions for other cities are **extrapolations** - accuracy may vary
        - Weather values should match your location for best results
        - For production use, retrain the model on data from your target area
        - Temperature values are in Kelvin (K) - subtract 273.15 for Celsius
        """)
    
    # Event handlers
    auto_calc_distance.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=[auto_calc_distance],
        outputs=[manual_distance]
    )
    
    use_current.click(
        fn=use_current_time,
        outputs=[pickup_weekday, pickup_hour]
    )
    
    # Sample location buttons
    sample_nairobi.click(
        fn=use_sample_nairobi,
        outputs=[origin_lat, origin_lon, dest_lat, dest_lon, sample_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[sample_status]
    )
    
    sample_ny.click(
        fn=use_sample_newyork,
        outputs=[origin_lat, origin_lon, dest_lat, dest_lon, sample_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[sample_status]
    )
    
    sample_london.click(
        fn=use_sample_london,
        outputs=[origin_lat, origin_lon, dest_lat, dest_lon, sample_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[sample_status]
    )
    
    # Search functionality
    origin_search_btn.click(
        fn=search_origin,
        inputs=[origin_search],
        outputs=[origin_lat, origin_lon, origin_search_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[origin_search_status]
    )
    
    dest_search_btn.click(
        fn=search_destination,
        inputs=[dest_search],
        outputs=[dest_lat, dest_lon, dest_search_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[dest_search_status]
    )
    
    predict_btn.click(
        fn=predict,
        inputs=[
            origin_lat, origin_lon, dest_lat, dest_lon,
            dewpoint_temp, min_temp, temp_range, rain,
            pickup_weekday, pickup_hour, manual_distance,
            auto_calc_distance, cluster_override
        ],
        outputs=[output, map_output]
    )
    
    update_map_btn.click(
        fn=update_map_only,
        inputs=[origin_lat, origin_lon, dest_lat, dest_lon],
        outputs=[map_output]
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True, theme=gr.themes.Soft())