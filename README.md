# 🌾 AI Crop Yield Prediction System  
**Satellite‑Driven, Cloud‑Safe, Industry‑Ready Machine Learning for Yield Forecasting**

This project is a full-stack, production-oriented machine learning system for crop yield prediction using satellite Earth observation data.  
It combines **Sentinel‑2 optical imagery**, **Sentinel‑1 SAR**, **weather variables**, and **XGBoost regression** to deliver:

- **✅ Per‑pixel yield maps**
- **✅ Spatial uncertainty maps**
- **✅ Field‑level yield estimates**  
- **✅ An interactive Streamlit dashboard**  

The system is designed for **cloudy regions**, **Africa‑ready deployments**, and **real agribusiness workflows**.

---

## 🚀 What This System Does
  ```bash
  | Component | Output |
  |--------|--------|
  | Sentinel‑2 | NDVI vegetation index |
  | Sentinel‑1 SAR | Cloud‑independent vegetation proxy |
  | Data fusion | Cloud‑safe NDVI composite |
  | Machine Learning | Yield prediction + uncertainty |
  | Raster engine | GeoTIFF yield & uncertainty maps |
  | Dashboard | Interactive web visualization |
  ```
---

## 🧠 AI, SAR–Optical Fusion & ML

The biggest limitation of satellite agriculture is simple:

> **Clouds break optical NDVI.**

This pipeline solves the problem using **multi‑sensor data fusion**:

1. Compute NDVI from **Sentinel‑2**
2. Compute a SAR vegetation proxy from **Sentinel‑1**
3. Fill cloudy pixels using SAR
4. Generate a **cloud‑safe NDVI composite**
5. Train an **XGBoost regression model** using NDVI, SAR, and weather data

This approach makes the system **robust, scalable, and usable in tropical and high‑cloud regions**, including much of Africa.

---

## 📡 Data Sources

  | Source               | Purpose                          |
  |---------------------|---------------------------------|
  | Sentinel-2 (10 m)   | NDVI vegetation index            |
  | Sentinel-1 SAR      | Cloud-independent crop signal   |
  | Field CSV           | Yield, rainfall, temperature    |
  | Google Earth Engine | Scalable satellite processing   |

---

## 🤖 Machine Learning
  ```bash
  - **Model:** XGBoost Regressor  
  - **Features:** NDVI_mean, NDVI_max, VV, VH, rainfall, temp  
  - **Target:** yield (t/ha)  
  - **Validation:** 5-fold cross-validation
```

---

**Example output:**
  ```text
    Predicted Yield: 7.39 ± 0.28 t/ha
    CV RMSE: 0.28
  ```
---

## 🚀 Features

- **Cloud-safe NDVI composites** using SAR data  
- **XGBoost ML model** for yield prediction  
- **GeoTIFF outputs** ready for QGIS/ArcGIS  
- **Interactive dashboard** for decision support  

---

## 🗂 Project Structure
  ```bash
  Yield_prediction/
  ├── AI_crop.py         # Satellite + ML pipeline
  ├── Dashboard.py       # Streamlit dashboard
  ├── DS_5_Fielddata.csv # Training data
  ├── output/            # NDVI, Yield, Uncertainty maps
  └── crop_env/          # Conda environment
  ```

---

## Setup environment
  ```bash
  conda create -n crop_env python=3.10
  conda activate crop_env
  chmod +x ./setup.sh
  ./setup.sh
  ```

---

## Run pipeline
```bash
python AI_crop.py
```

---

## 📜 License
This project is licensed under the **MIT License**.


