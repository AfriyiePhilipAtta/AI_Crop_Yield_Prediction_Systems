import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# ============================================================
# 1️⃣ PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Crop Yield Dashboard",
    layout="wide"
)

st.title("🌾 AI Crop Yield Prediction Dashboard")

# ============================================================
# 2️⃣ PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

NDVI_MAP = os.path.join(OUTPUT_DIR, "NDVI_Map.tif")
YIELD_MAP = os.path.join(OUTPUT_DIR, "Yield_Map.tif")
UNCERTAINTY_MAP = os.path.join(OUTPUT_DIR, "Yield_Uncertainty_Map.tif")
FEATURE_PLOT = os.path.join(OUTPUT_DIR, "feature_importance.png")
METADATA_JSON = os.path.join(OUTPUT_DIR, "model_metadata.json")

# ============================================================
# 3️⃣ HELPER FUNCTIONS
# ============================================================
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def plot_raster(arr, title, cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7)
    st.pyplot(fig)

# ============================================================
# 4️⃣ CHECK FILES
# ============================================================
missing = [p for p in [NDVI_MAP, YIELD_MAP, UNCERTAINTY_MAP, FEATURE_PLOT, METADATA_JSON] if not os.path.exists(p)]
if missing:
    st.error(f"❌ Missing files: {missing}\nRun the AI crop yield pipeline first.")
    st.stop()

# ============================================================
# 5️⃣ LOAD DATA
# ============================================================
ndvi = load_raster(NDVI_MAP)
yield_map = load_raster(YIELD_MAP)
uncertainty = load_raster(UNCERTAINTY_MAP)

with open(METADATA_JSON, 'r') as f:
    metadata = json.load(f)

# ============================================================
# 6️⃣ METRICS
# ============================================================
mean_yield = np.nanmean(yield_map)
mean_uncertainty = np.nanmean(uncertainty)

c1, c2, c3 = st.columns(3)
c1.metric("Mean Field Yield (t/ha)", f"{mean_yield:.2f}")
c2.metric("Mean Uncertainty (t/ha)", f"± {mean_uncertainty:.2f}")
c3.metric("Relative Uncertainty (%)", f"{100*mean_uncertainty/mean_yield:.1f}")

st.markdown("---")

# ============================================================
# 7️⃣ MAPS
# ============================================================
st.subheader("Field Maps")
c1, c2, c3 = st.columns(3)
with c1:
    plot_raster(ndvi, "NDVI Composite", cmap="YlGn", vmin=0, vmax=1)
with c2:
    plot_raster(yield_map, "Predicted Yield (t/ha)", cmap="viridis")
with c3:
    plot_raster(uncertainty, "Yield Uncertainty (t/ha)", cmap="magma")

st.markdown("---")

# ============================================================
# 8️⃣ FEATURE IMPORTANCE
# ============================================================
st.subheader("Feature Importance")
st.image(FEATURE_PLOT, caption="XGBoost Feature Importance", use_column_width=True)

# ============================================================
# 9️⃣ TRAINING METRICS
# ============================================================
st.subheader("Training Metrics")
st.write(f"- RMSE: {metadata['train_rmse']:.3f}")
st.write(f"- R²: {metadata['train_r2']:.3f}")
st.write(f"- Cross-validation R²: {metadata['cv_r2']:.3f}")
st.write(f"- Cross-validation RMSE: {metadata['cv_rmse']:.3f}")

st.markdown("---")

# ============================================================
# 🔟 PREDICTED YIELD
# ============================================================
st.subheader("Predicted Yield")
st.write(f"🌾 Predicted Field Yield: {metadata['predicted_yield']:.2f} ± {metadata['uncertainty']:.2f} t/ha")

# ============================================================
# 1️⃣1️⃣ FOOTER
# ============================================================
st.caption(
    "NDVI derived from Sentinel‑2 (cloud‑free median composite). "
    "Yield predicted using XGBoost with weather and SAR features."
)
