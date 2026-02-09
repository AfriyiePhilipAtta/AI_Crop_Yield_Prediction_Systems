#!/usr/bin/env python
# ============================================================
# DFH Yield Prediction Using NDVI
# Sentinel-2 NDVI with Sentinel-1 Gap Filling
# ML (XGBoost) + Yield (tons/ha) + Uncertainty Mapping Pipeline
# CORRECTED: Proper separation of VI and environmental features
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import ee
import geemap
import matplotlib.pyplot as plt
from contextlib import contextmanager

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from xgboost import XGBRegressor   # ✅ XGBOOST

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONTEXT MANAGER TO SUPPRESS VERBOSE OUTPUT
# ============================================================

@contextmanager
def suppress_output():
    """Suppress stdout to hide verbose geemap logging"""
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

# ============================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# ============================================================

PROJECT_ID = "quiet-subset-447718-q0"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

print("✅ Google Earth Engine initialized")

# ============================================================
# PATHS
# ============================================================

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

AOI_SHP = os.path.join(BASE_DIRECTORY, "Farm", "witz_farm.shp")
PLOT_SHP = os.path.join(BASE_DIRECTORY, "GGE_vector", "GGE_Harvest_150_gcs.shp")
DATA_FILE_PATH = os.path.join(BASE_DIRECTORY, "plot_satellite_indices_cloud_robust.csv")

OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "output_dfh")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# ============================================================
# LOAD AOI & PLOTS
# ============================================================

AOI = geemap.shp_to_ee(AOI_SHP).geometry()
PLOTS = geemap.shp_to_ee(PLOT_SHP).geometry()

print("✅ AOI and plot layout loaded")

# ============================================================
# TIME RANGE & GROWTH STAGES
# ============================================================

START_DATE = "2025-03-01"
END_DATE = "2025-08-10"

GROWTH_STAGES = {
    "early": ("2025-03-01", "2025-04-30"),
    "mid": ("2025-05-01", "2025-06-30"),
    "late": ("2025-07-01", "2025-08-10")
}

STAGE_ORDER = ["early", "mid", "late"]

# ============================================================
# SENTINEL-2 NDVI
# ============================================================

def add_ndvi(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    scl = img.select("SCL")
    valid = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    return img.addBands(ndvi).updateMask(valid)

s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .map(add_ndvi)
)

# ============================================================
# SENTINEL-1 (NDVI GAP-FILL SUPPORT)
# ============================================================

def add_rvi(img):
    vv = img.select("VV")
    vh = img.select("VH")
    rvi = vh.multiply(4).divide(vv.add(vh)).rename("RVI")
    return img.addBands(rvi)

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .map(add_rvi)
)

# ============================================================
# STAGE-WISE FUSED NDVI
# ============================================================

stage_fused_ndvi = {}

for stage, (start, end) in GROWTH_STAGES.items():
    ndvi = s2.filterDate(start, end).select("NDVI").median().clip(AOI)
    rvi = s1.filterDate(start, end).select("RVI").median().clip(AOI)
    stage_fused_ndvi[stage] = ndvi.unmask(rvi.unitScale(0, 1))

# ============================================================
# EXPORT NDVI MAPS FOR EACH GROWTH STAGE
# ============================================================

print("\n🗺️  Exporting NDVI maps for each growth stage...")

for stage in STAGE_ORDER:
    ndvi_output_path = os.path.join(OUTPUT_DIRECTORY, f"NDVI_{stage.upper()}_stage.tif")
    with suppress_output():
        geemap.ee_export_image(
            stage_fused_ndvi[stage],
            ndvi_output_path,
            scale=10,
            region=AOI,
            file_per_band=False
        )

print("✅ All NDVI maps exported")

# ============================================================
# LOAD TABULAR DATA
# ============================================================

print("\nLoading input data...\n")
data = pd.read_csv(DATA_FILE_PATH)

TARGET = "dryYieldg"
LOCATION = "location"
STAGE = "growth_stage"

# ============================================================
# CORRECTED: PROPER FEATURE CATEGORIZATION
# ============================================================

# Vegetation indices (should end with _auc or _mean)
VI_COLUMNS = [c for c in data.columns if c.endswith("_auc") or c.endswith("_mean")]

# Environmental/plot variables (NOT vegetation indices)
ENV_COLUMNS = ["rainfall", "temp", "plot_area_m2"]

# All features for modeling
FEATURE_COLUMNS = VI_COLUMNS + ENV_COLUMNS

print(f"📊 Found {len(VI_COLUMNS)} vegetation indices")
print(f"📊 Found {len(ENV_COLUMNS)} environmental variables")
print(f"📊 Total features: {len(FEATURE_COLUMNS)}\n")

# ============================================================
# MODEL FUNCTION (XGBOOST)
# ============================================================

def build_model():
    return XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42
    )

# ============================================================
# STAGE-WISE ML + UNCERTAINTY + RMSE + R²
# ============================================================

stage_results = {}
stage_uncertainty = {}

for stage in STAGE_ORDER:

    stage_data = data[data[STAGE] == stage]
    if stage_data.empty:
        continue

    print(f"{'='*60}")
    print(f"Processing growth stage: {stage.upper()}")
    print(f"{'='*60}")

    X = stage_data[FEATURE_COLUMNS]
    y = stage_data[TARGET]
    groups = stage_data[LOCATION]

    # Remove zero-variance features
    X = X.loc[:, X.std() > 0]
    
    # Update VI_COLUMNS to only include those that survived variance filter
    vi_cols_available = [c for c in VI_COLUMNS if c in X.columns]
    
    # ============================================================
    # CORRECTED: Calculate correlations ONLY for vegetation indices
    # ============================================================
    
    vi_correlations = X[vi_cols_available].corrwith(y).abs().sort_values(ascending=False)
    best_single_index = vi_correlations.index[0]
    best_correlation = vi_correlations.iloc[0]
    
    print(f"\n📈 Best single vegetation index: {best_single_index}")
    print(f"   Correlation with yield: {best_correlation:.3f}")
    
    # Show top 5 VIs
    print(f"\n📊 Top 5 vegetation indices by correlation:")
    for i, (idx, corr) in enumerate(vi_correlations.head(5).items(), 1):
        print(f"   {i}. {idx}: {corr:.3f}")

    # ============================================================
    # For feature selection: use ALL features (VI + environmental)
    # ============================================================
    
    all_correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Get top 10 features (can include environmental variables)
    top_10 = all_correlations.head(10).index.tolist()
    
    print(f"\n📊 Top 10 features overall (for selection):")
    for i, feat in enumerate(top_10, 1):
        feat_type = "VI" if feat in vi_cols_available else "ENV"
        print(f"   {i}. [{feat_type}] {feat}: {all_correlations[feat]:.3f}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[top_10])

    # Sequential feature selection
    selector = SequentialFeatureSelector(
        build_model(),
        n_features_to_select=3,
        direction="forward",
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )

    selector.fit(X_scaled, y)
    selected_features = list(np.array(top_10)[selector.get_support()])

    print(f"\n✅ Features selected by sequential forward selection:")
    for i, feat in enumerate(selected_features, 1):
        feat_type = "VI" if feat in vi_cols_available else "ENV"
        print(f"   {i}. [{feat_type}] {feat}")

    final_features = selected_features

    # ---- Cross‑validated residuals (RMSE + uncertainty) ----
    residuals, y_true_all, y_pred_all = [], [], []
    train_rmse_list, train_r2_list = [], []

    gkf = GroupKFold(n_splits=min(3, groups.nunique()))
    for tr, te in gkf.split(X, y, groups):
        model = build_model()
        model.fit(X.iloc[tr][final_features], y.iloc[tr])
        
        # Validation predictions
        preds = model.predict(X.iloc[te][final_features])
        residuals.extend(y.iloc[te] - preds)
        y_true_all.extend(y.iloc[te])
        y_pred_all.extend(preds)
        
        # Training predictions
        train_preds = model.predict(X.iloc[tr][final_features])
        train_rmse = np.sqrt(mean_squared_error(y.iloc[tr], train_preds))
        train_r2 = r2_score(y.iloc[tr], train_preds)
        train_rmse_list.append(train_rmse)
        train_r2_list.append(train_r2)

    residuals = np.array(residuals)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)
    
    # Average training metrics
    avg_train_rmse = np.mean(train_rmse_list)
    avg_train_r2 = np.mean(train_r2_list)

    plot_area_ha = stage_data["plot_area_m2"].mean() / 10_000
    uncertainty_tons_ha = (residuals.std() / 1_000_000) / plot_area_ha

    print(f"\n📊 Model Performance:")
    print(f"   Training RMSE: {avg_train_rmse:.2f} g")
    print(f"   Training R²  : {avg_train_r2:.3f}")
    print(f"   Validation MAE : {mae:.2f} g")
    print(f"   Validation RMSE: {rmse:.2f} g")
    print(f"   Validation R²  : {r2:.3f}")
    print(f"   Estimated uncertainty: ±{uncertainty_tons_ha:.2f} tons/ha\n")

    stage_results[stage] = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_rmse": avg_train_rmse,
        "train_r2": avg_train_r2,
        "features": final_features,
        "best_vi": best_single_index
    }

    stage_uncertainty[stage] = uncertainty_tons_ha

# ============================================================
# BEST STAGE
# ============================================================

BEST_STAGE = min(stage_results, key=lambda s: stage_results[s]["mae"])
print(f"\n{'='*60}")
print(f"🏆 Best growth stage identified: {BEST_STAGE.upper()}")
print(f"{'='*60}")
print(f"   MAE: {stage_results[BEST_STAGE]['mae']:.2f} g")
print(f"   R²: {stage_results[BEST_STAGE]['r2']:.3f}")
print(f"   Best VI: {stage_results[BEST_STAGE]['best_vi']}")

# ============================================================
# FINAL MODEL
# ============================================================

print(f"\n🔧 Training final model using ONLY the {BEST_STAGE.upper()} stage...\n")

best_data = data[data[STAGE] == BEST_STAGE]
X_final = best_data[stage_results[BEST_STAGE]["features"]]
y_final = best_data[TARGET]

final_model = build_model()
final_model.fit(X_final, y_final)

# Training metrics
train_preds_g = final_model.predict(X_final)
train_preds_tons_ha = (train_preds_g / 1_000_000) / (best_data["plot_area_m2"] / 10_000)
y_train_tons_ha = (y_final / 1_000_000) / (best_data["plot_area_m2"] / 10_000)

train_rmse = np.sqrt(mean_squared_error(y_train_tons_ha, train_preds_tons_ha))
train_r2 = r2_score(y_train_tons_ha, train_preds_tons_ha)

# For validation, use cross-validation
residuals_val, y_true_val, y_pred_val = [], [], []
gkf = GroupKFold(n_splits=min(3, best_data[LOCATION].nunique()))
for tr, te in gkf.split(X_final, y_final, best_data[LOCATION]):
    model = build_model()
    model.fit(X_final.iloc[tr], y_final.iloc[tr])
    preds = model.predict(X_final.iloc[te])
    
    preds_tons_ha = (preds / 1_000_000) / (best_data.iloc[te]["plot_area_m2"] / 10_000)
    y_true_tons_ha_val = (y_final.iloc[te] / 1_000_000) / (best_data.iloc[te]["plot_area_m2"] / 10_000)
    
    y_true_val.extend(y_true_tons_ha_val)
    y_pred_val.extend(preds_tons_ha)

y_true_val = np.array(y_true_val)
y_pred_val = np.array(y_pred_val)

val_rmse = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
val_r2 = r2_score(y_true_val, y_pred_val)

mean_yield = train_preds_tons_ha.mean()
std_yield = train_preds_tons_ha.std()

print(f"📊 {BEST_STAGE.upper()} stage yield prediction:")
print(f"   Mean yield: {mean_yield:.2f} tons/ha")
print(f"   Std dev: {std_yield:.2f} tons/ha")
print(f"\n   Training RMSE: {train_rmse:.2f} tons/ha")
print(f"   Training R²: {train_r2:.3f}")
print(f"   Validation RMSE: {val_rmse:.2f} tons/ha")
print(f"   Validation R²: {val_r2:.3f}")

# ============================================================
# PREDICTED VS OBSERVED PLOT
# ============================================================

plt.figure(figsize=(8, 8))
plt.scatter(y_true_val, y_pred_val, alpha=0.6, edgecolors='k', s=80)

# 1:1 line
min_val = min(y_true_val.min(), y_pred_val.min())
max_val = max(y_true_val.max(), y_pred_val.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

plt.xlabel('Observed Yield (tons/ha)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Yield (tons/ha)', fontsize=12, fontweight='bold')
plt.title(f'Predicted vs Observed Yield - {BEST_STAGE.upper()} Stage', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

# Add R² and RMSE as text
textstr = f'Validation R² = {val_r2:.3f}\nValidation RMSE = {val_rmse:.2f} tons/ha\nTraining R² = {train_r2:.3f}\nTraining RMSE = {train_rmse:.2f} tons/ha'
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'Predicted_vs_Observed_{BEST_STAGE.upper()}.png'), dpi=300)
print(f"\n✅ Predicted vs Observed plot saved")
plt.close()

# ============================================================
# SPATIAL YIELD & UNCERTAINTY MAPS - INDIVIDUAL STAGES
# ============================================================

def yield_map_tons_ha(ndvi_img, plot_area_m2):
    return ndvi_img.multiply(1000).add(500).divide(1_000_000).divide(plot_area_m2 / 10_000)

print("\n🗺️  Exporting yield maps for each growth stage...")

for stage in STAGE_ORDER:
    stage_data = data[data[STAGE] == stage]
    if stage_data.empty:
        continue
    
    avg_plot_area = stage_data["plot_area_m2"].mean()
    
    # Yield map for this stage
    yield_map_stage = yield_map_tons_ha(stage_fused_ndvi[stage], avg_plot_area)
    
    with suppress_output():
        geemap.ee_export_image(
            yield_map_stage,
            os.path.join(OUTPUT_DIRECTORY, f"Yield_{stage.upper()}_stage_TONS_PER_HA.tif"),
            scale=10,
            region=AOI,
            file_per_band=False
        )
    
    # Uncertainty map for this stage
    uncertainty_map_stage = (
        stage_fused_ndvi[stage]
        .multiply(0)
        .add(stage_uncertainty[stage])
        .rename("yield_uncertainty_tons_ha")
    )
    
    with suppress_output():
        geemap.ee_export_image(
            uncertainty_map_stage,
            os.path.join(OUTPUT_DIRECTORY, f"Yield_Uncertainty_{stage.upper()}_stage_TONS_PER_HA.tif"),
            scale=10,
            region=AOI,
            file_per_band=False
        )

print("✅ All yield and uncertainty maps exported")

# ============================================================
# COMBINED NDVI AND YIELD MAPS (ALL STAGES)
# ============================================================

print("\n🗺️  Exporting COMBINED maps (all stages)...")

# Combined NDVI (mean of all stages)
combined_ndvi = stage_fused_ndvi["early"].add(stage_fused_ndvi["mid"]).add(stage_fused_ndvi["late"]).divide(3)

with suppress_output():
    geemap.ee_export_image(
        combined_ndvi,
        os.path.join(OUTPUT_DIRECTORY, "NDVI_COMBINED_all_stages.tif"),
        scale=10,
        region=AOI,
        file_per_band=False
    )

# Combined Yield map
avg_plot_area_all = data["plot_area_m2"].mean()
combined_yield_map = yield_map_tons_ha(combined_ndvi, avg_plot_area_all)

with suppress_output():
    geemap.ee_export_image(
        combined_yield_map,
        os.path.join(OUTPUT_DIRECTORY, "Yield_COMBINED_all_stages_TONS_PER_HA.tif"),
        scale=10,
        region=AOI,
        file_per_band=False
    )

# Combined Uncertainty map (mean uncertainty)
mean_uncertainty = np.mean(list(stage_uncertainty.values()))
combined_uncertainty_map = (
    combined_ndvi
    .multiply(0)
    .add(mean_uncertainty)
    .rename("yield_uncertainty_tons_ha")
)

with suppress_output():
    geemap.ee_export_image(
        combined_uncertainty_map,
        os.path.join(OUTPUT_DIRECTORY, "Yield_Uncertainty_COMBINED_all_stages_TONS_PER_HA.tif"),
        scale=10,
        region=AOI,
        file_per_band=False
    )

print("✅ All combined maps exported")

print("\n" + "="*60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)