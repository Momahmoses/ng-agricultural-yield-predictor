"""
GIS analysis: NDVI spatial distribution, yield hotspots, and market access mapping.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.generate_data import generate_farm_records, FARMING_STATES


def build_farm_gdf(farms_df: pd.DataFrame) -> gpd.GeoDataFrame:
    geom = [Point(r.lon, r.lat) for _, r in farms_df.iterrows()]
    return gpd.GeoDataFrame(farms_df, geometry=geom, crs="EPSG:4326")


def compute_ndvi_zones(farm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    farm_gdf = farm_gdf.copy()
    farm_gdf["ndvi_class"] = pd.cut(
        farm_gdf["ndvi_score"],
        bins=[0, 0.3, 0.5, 0.65, 1.0],
        labels=["Poor", "Moderate", "Good", "Excellent"]
    )
    return farm_gdf


def state_yield_summary(farms_df: pd.DataFrame) -> pd.DataFrame:
    return (
        farms_df.groupby(["state", "crop"])
        .agg(avg_yield=("yield_tons_ha", "mean"),
             avg_ndvi=("ndvi_score", "mean"),
             farm_count=("farm_id", "count"))
        .reset_index()
        .sort_values("avg_yield", ascending=False)
    )


def build_yield_map(farm_gdf: gpd.GeoDataFrame) -> folium.Map:
    m = folium.Map(location=[9.08, 8.67], zoom_start=6, tiles="CartoDB positron")

    heat = [[r.geometry.y, r.geometry.x, r.yield_tons_ha]
            for _, r in farm_gdf.iterrows()]
    HeatMap(heat, radius=14, blur=12, min_opacity=0.4,
            gradient={"0.2": "blue", "0.5": "yellow", "0.8": "orange", "1.0": "red"}).add_to(m)

    ndvi_colors = {"Poor": "#d32f2f", "Moderate": "#f57c00",
                   "Good": "#66bb6a", "Excellent": "#1b5e20"}
    sample = farm_gdf.sample(min(200, len(farm_gdf)), random_state=42)
    for _, row in sample.iterrows():
        color = ndvi_colors.get(str(row.get("ndvi_class", "Moderate")), "#888")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4, color=color, fill=True, fill_opacity=0.7,
            popup=(f"<b>{row['state']}</b><br>Crop: {row['crop']}<br>"
                   f"Yield: {row['yield_tons_ha']} t/ha<br>NDVI: {row['ndvi_score']:.2f}"),
        ).add_to(m)

    for state_name, slat, slon, main_crop, eco_zone in FARMING_STATES:
        folium.Marker(
            location=[slat, slon],
            popup=f"<b>{state_name}</b><br>Main: {main_crop}<br>Zone: {eco_zone}",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(m)
    return m


if __name__ == "__main__":
    farms_df = generate_farm_records(500)
    farm_gdf = build_farm_gdf(farms_df)
    farm_gdf = compute_ndvi_zones(farm_gdf)
    m = build_yield_map(farm_gdf)
    os.makedirs("app", exist_ok=True)
    m.save("app/yield_map.html")
    print(state_yield_summary(farms_df).head(10).to_string())
