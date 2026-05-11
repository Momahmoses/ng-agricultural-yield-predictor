[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Momahmoses%2Fng-agricultural-yield-predictor&branch=main&mainModule=app.py)

# 🌾 Nigeria Agricultural Yield Predictor

ML-powered crop yield prediction for Nigerian smallholder farmers, combining **NDVI satellite data**, **weather and soil features**, **PySpark GBT models**, **Azure ML**, and a **Streamlit** farmer dashboard.

## Problem Statement
Nigeria's agricultural sector employs 70% of the rural population but suffers from poor productivity — average maize yield is 1.7 t/ha vs 6+ t/ha globally. This tool helps extension officers and agri-fintech platforms predict yields per farm and identify the biggest production constraints.

## Tech Stack
| Layer | Technology |
|---|---|
| Geospatial | GeoPandas, Folium, Sentinel-2 NDVI |
| Big Data | PySpark GBT on Azure Databricks |
| Cloud | Azure Blob Storage, Azure Machine Learning |
| Dashboard | Streamlit + Plotly + scikit-learn |

## Project Structure
```
ng-agricultural-yield-predictor/
├── app.py                        # Streamlit dashboard + live prediction
├── pipeline/spark_pipeline.py    # PySpark GBT training + cross-validation
├── gis/spatial_analysis.py       # NDVI zones + yield heatmap
├── data/generate_data.py         # Synthetic farm records + market prices
├── azure/azure_config.py         # Azure Blob + Databricks helpers
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Features
- Live yield prediction from sidebar sliders (rainfall, soil pH, NDVI, fertilizer)
- NDVI spatial heatmap across farming states
- Feature importance bar chart
- Market price trend line by crop
- Crop yield comparison bar chart
- Farm records table with gradient styling

## Data Sources (Production)
- **NAERLS** — National Agricultural Extension and Research Liaison Services
- **FMARD** — Federal Ministry of Agriculture data
- **Sentinel-2** — ESA NDVI rasters via Copernicus
- **NIMET** — Weather station data
- **NBS** — Agricultural survey data
