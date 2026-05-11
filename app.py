"""
Nigeria Agricultural Yield Predictor — Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data.generate_data import generate_farm_records, generate_market_prices, CROPS
from gis.spatial_analysis import build_farm_gdf, compute_ndvi_zones, build_yield_map

st.set_page_config(page_title="NG Agri Yield Predictor", page_icon="🌾", layout="wide")
st.markdown("""
<style>
.kpi{background:#1b5e20;color:white;padding:14px;border-radius:8px;text-align:center;}
.kpi-val{font-size:1.9rem;font-weight:700;}
.kpi-lbl{font-size:.8rem;opacity:.85;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    farms = generate_farm_records(1500)
    prices = generate_market_prices()
    return farms, prices


@st.cache_resource
def train_sklearn_model(farms_df):
    features = ["rainfall_mm", "soil_ph", "temp_avg_c", "fertilizer_kg_ha",
                 "planting_density_score", "ndvi_score", "farm_size_ha", "farmer_experience_yrs"]
    X = farms_df[features].fillna(0)
    y = farms_df["yield_tons_ha"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return model, features, rmse, r2


def main():
    farms_df, prices_df = load_data()
    model, feature_names, rmse, r2 = train_sklearn_model(farms_df)

    with st.sidebar:
        st.title("🌾 Yield Predictor")
        st.caption("Nigeria Agri Intelligence")
        st.divider()
        selected_crops = st.multiselect("Filter Crops", CROPS, default=CROPS[:5])
        season_filter = st.radio("Season", ["All", "Wet", "Dry"])
        st.divider()
        st.markdown("**Predict Your Yield**")
        p_rainfall = st.slider("Rainfall (mm)", 200, 2000, 900)
        p_soil_ph = st.slider("Soil pH", 4.5, 8.0, 6.2)
        p_temp = st.slider("Avg Temp (°C)", 18, 40, 28)
        p_fert = st.slider("Fertilizer (kg/ha)", 0, 200, 60)
        p_ndvi = st.slider("NDVI Score", 0.1, 0.9, 0.55)
        p_size = st.slider("Farm Size (ha)", 0.5, 20.0, 3.0)
        p_exp = st.slider("Farmer Experience (yrs)", 1, 40, 10)
        p_density = 0.7

        if st.button("Predict Yield", type="primary"):
            X_pred = pd.DataFrame([{
                "rainfall_mm": p_rainfall, "soil_ph": p_soil_ph, "temp_avg_c": p_temp,
                "fertilizer_kg_ha": p_fert, "planting_density_score": p_density,
                "ndvi_score": p_ndvi, "farm_size_ha": p_size, "farmer_experience_yrs": p_exp,
            }])
            pred = model.predict(X_pred[feature_names])[0]
            st.success(f"Predicted Yield: **{pred:.2f} t/ha**")

    farm_filtered = farms_df[farms_df["crop"].isin(selected_crops)]
    if season_filter != "All":
        farm_filtered = farm_filtered[farm_filtered["season"] == season_filter]

    st.title("🌾 Nigeria Agricultural Yield Predictor")
    st.caption("Crop yield prediction · NDVI mapping · Market price tracking · GIS + PySpark + Azure")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{farms_df['yield_tons_ha'].mean():.2f} t/ha",
         f"{r2:.3f}", f"{rmse:.3f}", f"{farms_df['farm_size_ha'].mean():.1f} ha"],
        ["Avg Yield", "Model R²", "RMSE", "Avg Farm Size"]
    ):
        col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        st.subheader("🗺 Yield & NDVI Spatial Map")
        farm_gdf = build_farm_gdf(farm_filtered)
        farm_gdf = compute_ndvi_zones(farm_gdf)
        m = build_yield_map(farm_gdf)
        st_folium(m, width=700, height=460)

    with chart_col:
        st.subheader("📊 Average Yield by Crop")
        crop_yield = farm_filtered.groupby("crop")["yield_tons_ha"].mean().sort_values()
        fig = px.bar(crop_yield.reset_index(), x="yield_tons_ha", y="crop",
                     orientation="h", color="yield_tons_ha",
                     color_continuous_scale="Greens",
                     labels={"yield_tons_ha": "Avg Yield (t/ha)", "crop": ""},
                     height=460)
        fig.update_layout(coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=10, t=5, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_feat, col_price = st.columns(2)
    with col_feat:
        st.subheader("🔍 Feature Importance")
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(importance, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="YlGn")
        fig_imp.update_layout(coloraxis_showscale=False,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=0, t=5, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_price:
        st.subheader("💰 Market Price Trend")
        selected_crop_price = st.selectbox("Select Crop", CROPS, key="price_crop")
        price_data = prices_df[prices_df["crop"] == selected_crop_price]
        fig_price = px.line(price_data, x="date", y="price_naira_per_ton",
                            color_discrete_sequence=["#2e7d32"],
                            labels={"price_naira_per_ton": "Price (₦/ton)", "date": ""})
        fig_price.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=5, b=0))
        st.plotly_chart(fig_price, use_container_width=True)

    st.divider()
    st.subheader("📋 Farm Records Sample")
    st.dataframe(
        farm_filtered[["farm_id", "state", "crop", "season", "yield_tons_ha",
                        "ndvi_score", "rainfall_mm", "fertilizer_kg_ha"]].head(50)
        .style.background_gradient(subset=["yield_tons_ha"], cmap="Greens"),
        use_container_width=True, height=280,
    )
    st.caption("Data: Synthetic — replace with FMARD, NAERLS, Sentinel-2 NDVI. "
               "Pipeline: Azure Databricks PySpark GBT. Storage: Azure Blob.")


if __name__ == "__main__":
    main()
