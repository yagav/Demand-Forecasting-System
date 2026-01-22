import streamlit as st
import pandas as pd
import mlflow
import plotly.graph_objects as go
from datetime import date

MLFLOW_TRACKING_URI = "http://107.21.193.74:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(page_title="Demand Forecasting", layout="centered")

st.title("📈 Demand Forecasting System")
st.markdown("Forecast demand using Prophet models stored in MLflow")

store_id = st.selectbox("Select Store", options=[1, 2, 3])
item_id = st.selectbox("Select Item", options=list(range(1, 51)))

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=date(2017, 1, 1),
        min_value=date(2013, 1, 1),
        max_value=date(2018, 12, 31)
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=date(2017, 3, 31),
        min_value=date(2013, 1, 1),
        max_value=date(2018, 12, 31)
    )

if start_date > end_date:
    st.error("❌ Start date must be before end date")
    st.stop()

if st.button("🔮 Generate Forecast"):

    model_name = f"demand_forecast_store_{store_id}_item_{item_id}"
    model_uri = f"models:/{model_name}/latest"

    st.info(f"Loading model: `{model_name}`")

    try:
        model = mlflow.prophet.load_model(model_uri)
    except Exception as e:
        st.error("❌ Failed to load model from MLflow")
        st.exception(e)
        st.stop()

    future_dates = pd.date_range(
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date),
        freq="D"
    )

    future_df = pd.DataFrame({"ds": future_dates})

    forecast = model.predict(future_df)

    forecast = forecast[
        (forecast["ds"] >= pd.to_datetime(start_date)) &
        (forecast["ds"] <= pd.to_datetime(end_date))
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="blue", width=3)
    ))

    fig.update_layout(
        title=f"Demand Forecast | Store {store_id} | Item {item_id}",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 View Forecast Data"):
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .rename(columns={
                "ds": "Date",
                "yhat": "Forecast",
                "yhat_lower": "Lower Bound",
                "yhat_upper": "Upper Bound"
            })
        )
