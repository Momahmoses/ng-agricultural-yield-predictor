import pandas as pd
import numpy as np
import os

FARMING_STATES = [
    ("Kano", 12.0022, 8.5920, "Groundnut/Sorghum", "Sudan Savanna"),
    ("Kaduna", 10.5222, 7.4383, "Maize/Soybean", "Guinea Savanna"),
    ("Benue", 7.3369, 8.7404, "Yam/Cassava", "Guinea Savanna"),
    ("Niger", 10.0008, 5.5981, "Rice/Sorghum", "Guinea Savanna"),
    ("Kebbi", 11.4943, 4.2333, "Rice/Wheat", "Sudan Savanna"),
    ("Sokoto", 13.0059, 5.2476, "Millet/Cowpea", "Sahel Savanna"),
    ("Zamfara", 12.1222, 6.2236, "Groundnut/Millet", "Sudan Savanna"),
    ("Katsina", 12.9908, 7.6018, "Sorghum/Millet", "Sudan Savanna"),
    ("Jigawa", 12.2280, 9.5616, "Rice/Groundnut", "Sudan Savanna"),
    ("Oyo", 7.3775, 3.9470, "Maize/Cassava", "Derived Savanna"),
    ("Ogun", 6.9980, 3.4737, "Cassava/Maize", "Forest Savanna"),
    ("Cross River", 5.9631, 8.3305, "Cocoa/Oil Palm", "Rainforest"),
    ("Anambra", 6.2104, 6.9623, "Cassava/Yam", "Rainforest"),
    ("Imo", 5.4527, 7.0201, "Oil Palm/Cassava", "Rainforest"),
    ("Taraba", 7.9993, 10.7741, "Yam/Cassava", "Guinea Savanna"),
    ("Plateau", 9.2182, 9.5179, "Irish Potato/Maize", "Guinea Savanna"),
    ("Nassarawa", 8.4994, 8.1997, "Soybean/Sorghum", "Guinea Savanna"),
    ("Delta", 5.5320, 5.8987, "Oil Palm/Cassava", "Mangrove"),
    ("Bauchi", 10.3158, 9.8442, "Maize/Groundnut", "Guinea Savanna"),
    ("Adamawa", 9.3265, 12.3984, "Groundnut/Maize", "Guinea Savanna"),
]

CROPS = ["Cassava", "Maize", "Yam", "Rice", "Sorghum", "Millet",
         "Groundnut", "Soybean", "Cowpea", "Oil Palm"]

CROP_YIELD_RANGE = {
    "Cassava": (8, 25), "Maize": (1.5, 6), "Yam": (10, 30), "Rice": (1, 5),
    "Sorghum": (0.8, 3.5), "Millet": (0.5, 2), "Groundnut": (0.8, 3),
    "Soybean": (1, 3.5), "Cowpea": (0.5, 2), "Oil Palm": (4, 18),
}


def generate_farm_records(n: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    records = []
    for i in range(n):
        state_info = FARMING_STATES[np.random.randint(len(FARMING_STATES))]
        state, slat, slon, main_crop, eco_zone = state_info
        crop = np.random.choice(CROPS)
        lo, hi = CROP_YIELD_RANGE[crop]
        rainfall_mm = np.random.uniform(400, 1800)
        soil_ph = np.random.uniform(5.0, 7.5)
        temp_c = np.random.uniform(20, 38)
        fert_kg_ha = np.random.uniform(0, 150)
        planting_density = np.random.uniform(0.3, 1.0)
        ndvi = np.random.uniform(0.2, 0.85)
        base_yield = lo + (hi - lo) * (
            0.3 * (rainfall_mm / 1800) +
            0.25 * ((soil_ph - 5) / 2.5) +
            0.20 * ndvi +
            0.15 * (fert_kg_ha / 150) +
            0.10 * planting_density
        )
        records.append({
            "farm_id": f"NG-FARM-{i+1:05d}",
            "state": state,
            "eco_zone": eco_zone,
            "lat": slat + np.random.uniform(-0.7, 0.7),
            "lon": slon + np.random.uniform(-0.7, 0.7),
            "crop": crop,
            "season": np.random.choice(["Dry", "Wet"], p=[0.3, 0.7]),
            "year": np.random.randint(2019, 2024),
            "farm_size_ha": round(np.random.exponential(3), 2),
            "rainfall_mm": round(rainfall_mm, 1),
            "soil_ph": round(soil_ph, 2),
            "temp_avg_c": round(temp_c, 1),
            "fertilizer_kg_ha": round(fert_kg_ha, 1),
            "irrigation": np.random.choice([True, False], p=[0.25, 0.75]),
            "planting_density_score": round(planting_density, 3),
            "ndvi_score": round(ndvi, 3),
            "yield_tons_ha": round(max(0.1, base_yield + np.random.normal(0, 0.5)), 2),
            "market_distance_km": round(np.random.exponential(30), 1),
            "farmer_experience_yrs": int(np.random.randint(1, 40)),
        })
    return pd.DataFrame(records)


def generate_market_prices() -> pd.DataFrame:
    np.random.seed(42)
    months = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
    records = []
    for crop in CROPS:
        base_price = np.random.uniform(150_000, 800_000)
        for dt in months:
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * dt.month / 12)
            records.append({
                "date": dt, "crop": crop,
                "price_naira_per_ton": round(base_price * seasonal * np.random.uniform(0.9, 1.1), 0),
            })
    return pd.DataFrame(records)


def save_all(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    generate_farm_records().to_csv(f"{output_dir}/farm_records.csv", index=False)
    generate_market_prices().to_csv(f"{output_dir}/market_prices.csv", index=False)
    print("Agricultural data generated.")


if __name__ == "__main__":
    save_all()
