
# THIS CODE IS HOSTED IN EC2 AWS AND IS TRIGERRED WHEN A NEW DATA IS UPLOADED IN THE S3 BUCKET


import sys
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from datetime import timedelta

import mlflow
import mlflow.prophet
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "demand_forecasting_system"
VALIDATION_DAYS = 30

NEW_DATA_PATH = sys.argv[1] 

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

df = pd.read_csv(NEW_DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.rename(columns={"date": "ds", "sales": "y"})

for (store_id, item_id), group in df.groupby(["store", "item"]):

    print(f"\n🔄 Retraining Store {store_id} | Item {item_id}")

    group = group.sort_values("ds")

    cutoff_date = group["ds"].max() - timedelta(days=VALIDATION_DAYS)

    train_df = group[group["ds"] <= cutoff_date]
    val_df = group[group["ds"] > cutoff_date]

    if len(val_df) < 10:
        print("⚠️ Not enough validation data — skipping")
        continue


    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_df[["ds", "y"]])


    future = val_df[["ds"]]
    forecast = model.predict(future)

    rmse = np.sqrt(mean_squared_error(val_df["y"], forecast["yhat"]))
    print(f"📉 RMSE: {rmse:.4f}")


    model_name = f"demand_forecast_store_{store_id}_item_{item_id}"

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("store_id", store_id)
        mlflow.log_param("item_id", item_id)
        mlflow.log_param("validation_days", VALIDATION_DAYS)

        mlflow.log_metric("rmse", rmse)

        mlflow.prophet.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )

        run_id = mlflow.active_run().info.run_id

    # GET NEW VERSION

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    new_version = max(
        [int(v.version) for v in versions]
    )

    previous_rmse = None
    previous_version = None

    try:
        prev = client.get_model_version_by_alias(
            model_name, "champion"
        )
        previous_version = prev.version
        previous_rmse = float(prev.tags.get("rmse"))
    except:
        print("ℹ️ No existing champion model")

    client.set_model_version_tag(
        model_name,
        new_version,
        "rmse",
        str(rmse)
    )

    if previous_rmse is None or rmse < previous_rmse:

        print("🏆 New model is BETTER — promoting")

        client.set_registered_model_alias(
            model_name,
            "champion",
            new_version
        )

        if previous_version:
            client.set_model_version_tag(
                model_name,
                previous_version,
                "status",
                "replaced"
            )
    else:
        print("❌ New model is worse — keeping old champion")

        client.set_model_version_tag(
            model_name,
            new_version,
            "status",
            "discarded"
        )

print("\n✅ Retraining completed")
