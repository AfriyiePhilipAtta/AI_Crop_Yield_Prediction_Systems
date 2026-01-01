# 🌾 AI Crop Yield Prediction System

**Satellite-Driven, Cloud-Safe, Industry-Grade Yield Forecasting**  

  This system uses **Sentinel-2 NDVI**, **Sentinel-1 SAR**, **weather data**, and **XGBoost** to produce:

  - Per-pixel yield maps  
  - Prediction uncertainty maps  
  - Field-level yield forecasts  
  - Interactive **Streamlit dashboard**

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


