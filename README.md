# **Demand Forecasting System with Automated Retraining and Cloud-Based MLOps**

---

## **Overview**

This project implements a **production-ready demand forecasting system** using **Facebook Prophet** and **MLflow**, designed to follow real-world **MLOps best practices**.

The system supports:

- Automated retraining  
- Model evaluation and versioning  
- Controlled model promotion  
- Cloud-based artifact storage  
- Remote inference  
- A user-facing web application for interactive forecasting  

The primary objective of this project is to demonstrate how a **time-series forecasting solution** can be **built, deployed, governed, and consumed** in a scalable and maintainable manner.

---

## **Problem Statement**

Accurate demand forecasting is critical for **inventory planning**, **logistics**, and **operational efficiency**.

This system forecasts **daily demand** for multiple **store–item combinations**, ensuring:

- Independent modeling per store and item  
- Controlled model evolution over time  
- Safe deployment of improved models  
- Reproducibility and traceability  

---

## **Key Features**

- Time-series forecasting using **Facebook Prophet**  
- Separate forecasting models per **Store × Item**  
- Automated retraining triggered by new data  
- Model evaluation using **RMSE**  
- Automatic promotion of superior models  
- **MLflow-based** experiment tracking and model registry  
- **AWS S3-backed** artifact storage  
- Remote inference through **MLflow Tracking Server**  
- Interactive forecasting interface using **Streamlit**

---

## **Data Description**

- **Time period:** 2013 – 2017  
- **Forecast horizon:** Up to 2018  
- **Stores:** 1 to 3  
- **Items:** 1 to 50  
- **Granularity:** Daily demand  

Each **store–item pair** is treated as an **independent time series** to preserve demand patterns and seasonality.

---

## **Model Training & Validation**

- **Model:** Facebook Prophet  
- **Validation strategy:** Last 30-day holdout  
- **Metric:** Root Mean Squared Error (**RMSE**)  

---

## **Model Promotion Logic**

### **Promotion Rules**

- If no existing model exists → **Promote new model**
- If new model RMSE is lower → **Promote new model**
- Otherwise → **Retain current model**

The promoted model is assigned the **`champion` alias** in MLflow.

---

## **Inference Design**

- Remote inference via **MLflow Tracking Server**
- Explicit user-defined forecast dates
- Forecasting restricted to **2013–2018**
- No reliance on system date

---

## **Streamlit Web Application**

### **User Capabilities**

- Select **Store ID (1–3)**
- Select **Item ID (1–50)**
- Choose date range within **2013–2018**
- Visualize forecasted demand
- View forecast data table

---

## **Deployment Architecture**

- Training & retraining: EC2  
- Tracking server: MLflow on EC2  
- Artifact storage: AWS S3  
- Inference: Remote via MLflow  
- UI: Streamlit

---

## **Engineering Principles Applied**

- End-to-end **MLOps lifecycle**
- Automated retraining & evaluation
- Version-controlled deployment
- Cloud-native artifacts
- Reproducible workflows
