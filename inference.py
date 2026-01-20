import mlflow
import pandas as pd
from datetime import timedelta

MLFLOW_TRACKING_URI = "http://107.21.193.74:5000"

MODEL_NAME = "demand_forecast_store_1_item_1"
MODEL_VERSION = "latest"     
FORECAST_HORIZON_DAYS = 30


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

print("Tracking URI:", mlflow.get_tracking_uri())
print("Registry URI:", mlflow.get_registry_uri())


if MODEL_VERSION.lower() == "latest":
    model_uri = f"models:/{MODEL_NAME}/latest"
else:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

print(f"Loading model from: {model_uri}")
model = mlflow.prophet.load_model(model_uri)
print(f"Done")
last_date = pd.Timestamp.today().normalize()

future_df = pd.DataFrame({
    "ds": pd.date_range(
        start=last_date + timedelta(days=1),
        periods=FORECAST_HORIZON_DAYS,
        freq="D"
    )
})

forecast = model.predict(future_df)

output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

print(output.head())
