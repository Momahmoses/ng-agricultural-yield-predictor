"""
Agricultural Yield Prediction Pipeline — PySpark on Azure Databricks
Trains a GBT regression model and scores new farm records.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os


def get_spark() -> SparkSession:
    return (SparkSession.builder.appName("AgriYieldPredictor")
            .config("fs.azure.account.key.<STORAGE_ACCOUNT>.blob.core.windows.net",
                    os.getenv("AZURE_STORAGE_KEY", ""))
            .getOrCreate())


def load_data(spark, path="data/"):
    farms = spark.read.csv(f"{path}farm_records.csv", header=True, inferSchema=True)
    prices = spark.read.csv(f"{path}market_prices.csv", header=True, inferSchema=True)
    return farms, prices


def feature_engineering(farms_df):
    crop_idx = StringIndexer(inputCol="crop", outputCol="crop_idx")
    crop_enc = OneHotEncoder(inputCol="crop_idx", outputCol="crop_enc")
    zone_idx = StringIndexer(inputCol="eco_zone", outputCol="zone_idx")
    zone_enc = OneHotEncoder(inputCol="zone_idx", outputCol="zone_enc")
    season_idx = StringIndexer(inputCol="season", outputCol="season_idx")

    numeric_features = [
        "rainfall_mm", "soil_ph", "temp_avg_c", "fertilizer_kg_ha",
        "planting_density_score", "ndvi_score", "farm_size_ha",
        "market_distance_km", "farmer_experience_yrs",
    ]
    assembler = VectorAssembler(
        inputCols=numeric_features + ["crop_enc", "zone_enc", "season_idx"],
        outputCol="raw_features"
    )
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withMean=True, withStd=True)
    return [crop_idx, crop_enc, zone_idx, zone_enc, season_idx, assembler, scaler]


def train_model(farms_df):
    stages = feature_engineering(farms_df)
    gbt = GBTRegressor(featuresCol="features", labelCol="yield_tons_ha",
                       maxIter=50, maxDepth=5, stepSize=0.1)
    pipeline = Pipeline(stages=stages + [gbt])

    train, test = farms_df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)
    preds = model.transform(test)

    evaluator = RegressionEvaluator(labelCol="yield_tons_ha",
                                    predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(preds)
    r2_eval = RegressionEvaluator(labelCol="yield_tons_ha",
                                  predictionCol="prediction", metricName="r2")
    r2 = r2_eval.evaluate(preds)
    print(f"RMSE: {rmse:.3f}  |  R²: {r2:.3f}")
    return model, preds


def compute_yield_by_state(farms_df):
    return (
        farms_df
        .groupBy("state", "crop", "season", "year")
        .agg(
            F.avg("yield_tons_ha").alias("avg_yield"),
            F.sum(F.col("yield_tons_ha") * F.col("farm_size_ha")).alias("total_yield_tons"),
            F.count("*").alias("farm_count"),
            F.avg("ndvi_score").alias("avg_ndvi"),
            F.avg("rainfall_mm").alias("avg_rainfall"),
        )
    )


if __name__ == "__main__":
    spark = get_spark()
    farms_df, prices_df = load_data(spark)
    model, preds = train_model(farms_df)
    state_summary = compute_yield_by_state(farms_df)
    state_summary.show(15)
    state_summary.toPandas().to_csv("data/yield_summary.csv", index=False)
    spark.stop()
