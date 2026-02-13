#!/usr/bin/env python
# ============================================================
# Yield Prediction Using NDVI
# Sentinel-2 NDVI with Sentinel-1 Gap Filling
# ML (XGBoost) + Yield (tons/ha) + Uncertainty Mapping Pipeline
# Separation of VI and environmental features
# Individual plots for each growth stage (Early, Mid, Late)
# Enhanced plot readability matching reference style
# Legend no longer overlaps adjacent panels in multi-panel plot
# Plot annotations now match slide 11 metrics exactly
#          (MAE in g, RMSE in g, R², Uncertainty in t/ha)
# Uncertainty confidence levels (68% and 95% confidence intervals)
# Both S2-only and Fused maps now use identical spatial extent (AOI)
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import ee
import geemap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from contextlib import contextmanager

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONTEXT MANAGER TO SUPPRESS VERBOSE OUTPUT
# ============================================================

@contextmanager
def suppress_output():
    """Suppress stdout to hide verbose geemap logging."""
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

AOI_SHP       = os.path.join(BASE_DIRECTORY, "Farm", "witz_farm.shp")
PLOT_SHP      = os.path.join(BASE_DIRECTORY, "GGE_vector", "GGE_Harvest_150_gcs.shp")
DATA_FILE_PATH = os.path.join(BASE_DIRECTORY, "plot_satellite_indices_cloud_robust.csv")

OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "output_dfh")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# ============================================================
# LOAD AOI & PLOTS
# ============================================================

AOI   = geemap.shp_to_ee(AOI_SHP).geometry()
PLOTS = geemap.shp_to_ee(PLOT_SHP).geometry()

print("✅ AOI and plot layout loaded")

# ============================================================
# TIME RANGE & GROWTH STAGES
# ============================================================

START_DATE = "2025-03-01"
END_DATE   = "2025-08-10"

GROWTH_STAGES = {
    "early": ("2025-03-01", "2025-04-30"),
    "mid":   ("2025-05-01", "2025-06-30"),
    "late":  ("2025-07-01", "2025-08-10"),
}

STAGE_ORDER = ["early", "mid", "late"]

# ============================================================
# SENTINEL-2 NDVI (NO MASKING FOR SPATIAL CONSISTENCY)
# ============================================================

def add_ndvi_no_mask(img):
    """Add NDVI without quality masking to preserve spatial extent"""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi)


s2_full = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .map(add_ndvi_no_mask)
)

# ============================================================
# SENTINEL-1 (NDVI GAP-FILL SUPPORT)
# ============================================================

def add_rvi(img):
    vv  = img.select("VV")
    vh  = img.select("VH")
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
# STAGE-WISE NDVI: SENTINEL-2 ONLY AND FUSED
# FIXED: Force both products to have complete spatial coverage
# ============================================================

stage_s2_only_ndvi = {}
stage_fused_ndvi = {}

print("\n🛰️  Processing NDVI data for each growth stage...")
print("   Ensuring identical spatial extent for S2-only and Fused products...")

for stage, (start, end) in GROWTH_STAGES.items():
    
    # Sentinel-2 ONLY NDVI (median composite, clipped to AOI)
    s2_ndvi = s2_full.filterDate(start, end).select("NDVI").median().clip(AOI)
    
    # Sentinel-1 RVI (for gap-filling)
    s1_rvi = s1.filterDate(start, end).select("RVI").median().clip(AOI)
    s1_ndvi_proxy = s1_rvi.unitScale(0, 1).rename("NDVI")
    
    # S2-ONLY: Unmask to show full extent (gaps filled with 0)
    s2_only = s2_ndvi.unmask(0)
    
    # FUSED: Fill S2 gaps with S1 data, then unmask any remaining gaps
    fused = s2_ndvi.unmask(s1_ndvi_proxy).unmask(0.3)
    
    # Store both products
    stage_s2_only_ndvi[stage] = s2_only
    stage_fused_ndvi[stage] = fused
    
    print(f"   ✓ {stage.upper()} stage: S2-only and Fused NDVI ready")

print("✅ All NDVI products processed with consistent spatial extent")

# ============================================================
# EXPORT SENTINEL-2 ONLY NDVI MAPS (BEFORE FUSION)
# ============================================================

print("\n🗺️  Exporting Sentinel-2 ONLY NDVI maps (BEFORE fusion)...")

for stage in STAGE_ORDER:
    ndvi_s2_only_path = os.path.join(OUTPUT_DIRECTORY, f"NDVI_S2_ONLY_{stage.upper()}_stage.tif")
    with suppress_output():
        geemap.ee_export_image(
            stage_s2_only_ndvi[stage],
            ndvi_s2_only_path,
            scale=10,
            region=AOI,
            file_per_band=False,
        )
    print(f"   ✓ Exported: NDVI_S2_ONLY_{stage.upper()}_stage.tif")

print("✅ All Sentinel-2 ONLY NDVI maps exported")

# ============================================================
# EXPORT FUSED (S2+S1) NDVI MAPS (AFTER FUSION)
# ============================================================

print("\n🗺️  Exporting FUSED (Sentinel-2 + Sentinel-1) NDVI maps (AFTER fusion)...")

for stage in STAGE_ORDER:
    ndvi_fused_path = os.path.join(OUTPUT_DIRECTORY, f"NDVI_FUSED_S2_S1_{stage.upper()}_stage.tif")
    with suppress_output():
        geemap.ee_export_image(
            stage_fused_ndvi[stage],
            ndvi_fused_path,
            scale=10,
            region=AOI,
            file_per_band=False,
        )
    print(f"   ✓ Exported: NDVI_FUSED_S2_S1_{stage.upper()}_stage.tif")

print("✅ All FUSED NDVI maps exported")

# ============================================================
# LOAD TABULAR DATA
# ============================================================

print("\nLoading input data...\n")
data = pd.read_csv(DATA_FILE_PATH)

TARGET   = "dryYieldg"
LOCATION = "location"
STAGE    = "growth_stage"

# ============================================================
# FEATURE CATEGORIZATION
# ============================================================

# Vegetation indices
VI_COLUMNS  = [c for c in data.columns if c.endswith("_auc") or c.endswith("_mean")]

# Environmental / plot variables (NOT vegetation indices)
ENV_COLUMNS = ["rainfall", "temp", "plot_area_m2"]

# All features for modelling
FEATURE_COLUMNS = VI_COLUMNS + ENV_COLUMNS

print(f"📊 Found {len(VI_COLUMNS)} vegetation indices")
print(f"📊 Found {len(ENV_COLUMNS)} environmental variables")
print(f"📊 Total features: {len(FEATURE_COLUMNS)}\n")

# ============================================================
# MODEL FACTORY (XGBoost)
# ============================================================

def build_model():
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
    )

# ============================================================
# STAGE-WISE ML + UNCERTAINTY + CONFIDENCE LEVELS + RMSE + R²
# ============================================================

stage_results      = {}
stage_uncertainty  = {}
stage_cv_predictions = {}

for stage in STAGE_ORDER:

    stage_data = data[data[STAGE] == stage]
    if stage_data.empty:
        continue

    print(f"{'='*60}")
    print(f"Processing growth stage: {stage.upper()}")
    print(f"{'='*60}")

    X      = stage_data[FEATURE_COLUMNS]
    y      = stage_data[TARGET]
    groups = stage_data[LOCATION]

    # Remove zero-variance features
    X = X.loc[:, X.std() > 0]

    vi_cols_available = [c for c in VI_COLUMNS if c in X.columns]

    # --- Correlations for vegetation indices only ---
    vi_correlations   = X[vi_cols_available].corrwith(y).abs().sort_values(ascending=False)
    best_single_index = vi_correlations.index[0]
    best_correlation  = vi_correlations.iloc[0]

    print(f"\n📈 Best single vegetation index: {best_single_index}")
    print(f"   Correlation with yield: {best_correlation:.3f}")
    print(f"\n📊 Top 5 vegetation indices by correlation:")
    for i, (idx, corr) in enumerate(vi_correlations.head(5).items(), 1):
        print(f"   {i}. {idx}: {corr:.3f}")

    # --- Feature selection uses ALL features ---
    all_correlations = X.corrwith(y).abs().sort_values(ascending=False)
    top_10 = all_correlations.head(10).index.tolist()

    print(f"\n📊 Top 10 features overall (for selection):")
    for i, feat in enumerate(top_10, 1):
        feat_type = "VI" if feat in vi_cols_available else "ENV"
        print(f"   {i}. [{feat_type}] {feat}: {all_correlations[feat]:.3f}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X[top_10])

    selector = SequentialFeatureSelector(
        build_model(),
        n_features_to_select=3,
        direction="forward",
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    selector.fit(X_scaled, y)
    selected_features = list(np.array(top_10)[selector.get_support()])

    print(f"\n✅ Features selected by sequential forward selection:")
    for i, feat in enumerate(selected_features, 1):
        feat_type = "VI" if feat in vi_cols_available else "ENV"
        print(f"   {i}. [{feat_type}] {feat}")

    final_features = selected_features

    # --- Cross-validated residuals ---
    # y and predictions kept in GRAMS (matching slide 11 MAE/RMSE units)
    residuals, y_true_all, y_pred_all = [], [], []
    train_rmse_list, train_r2_list    = [], []
    groups_all = []

    gkf = GroupKFold(n_splits=min(3, groups.nunique()))
    for tr, te in gkf.split(X, y, groups):
        model = build_model()
        model.fit(X.iloc[tr][final_features], y.iloc[tr])

        preds = model.predict(X.iloc[te][final_features])
        residuals.extend(y.iloc[te] - preds)
        y_true_all.extend(y.iloc[te])
        y_pred_all.extend(preds)
        groups_all.extend(groups.iloc[te])

        train_preds = model.predict(X.iloc[tr][final_features])
        train_rmse_list.append(np.sqrt(mean_squared_error(y.iloc[tr], train_preds)))
        train_r2_list.append(r2_score(y.iloc[tr], train_preds))

    residuals  = np.array(residuals)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # ── Metrics in GRAMS (consistent with slide 11) ──────────────────────
    mae_g  = mean_absolute_error(y_true_all, y_pred_all)   # grams
    rmse_g = np.sqrt(mean_squared_error(y_true_all, y_pred_all))  # grams
    r2     = r2_score(y_true_all, y_pred_all)

    avg_train_rmse = np.mean(train_rmse_list)
    avg_train_r2   = np.mean(train_r2_list)

    plot_area_ha        = stage_data["plot_area_m2"].mean() / 10_000
    uncertainty_tons_ha = (residuals.std() / 1_000_000) / plot_area_ha  # t/ha

    # ── NEW: Calculate confidence intervals ──────────────────────────────
    # 68% confidence interval (±1 standard deviation)
    ci_68_lower = -uncertainty_tons_ha
    ci_68_upper = +uncertainty_tons_ha
    
    # 95% confidence interval (±1.96 standard deviations)
    ci_95_lower = -1.96 * uncertainty_tons_ha
    ci_95_upper = +1.96 * uncertainty_tons_ha

    print(f"\n📊 Model Performance (matching slide 11 units):")
    print(f"   Training RMSE: {avg_train_rmse:.2f} g")
    print(f"   Training R²  : {avg_train_r2:.3f}")
    print(f"   Validation MAE : {mae_g:.2f} g")
    print(f"   Validation RMSE: {rmse_g:.2f} g")
    print(f"   Validation R²  : {r2:.3f}")
    print(f"   Uncertainty    : ±{uncertainty_tons_ha:.2f} t/ha")
    print(f"\n📊 Uncertainty Confidence Levels:")
    print(f"   68% CI (1σ): {ci_68_lower:.2f} to {ci_68_upper:.2f} t/ha")
    print(f"   95% CI (1.96σ): {ci_95_lower:.2f} to {ci_95_upper:.2f} t/ha\n")

    stage_results[stage] = {
        "mae_g": mae_g,
        "rmse_g": rmse_g,
        "r2": r2,
        "train_rmse": avg_train_rmse,
        "train_r2": avg_train_r2,
        "uncertainty_tons_ha": uncertainty_tons_ha,
        "ci_68_lower": ci_68_lower,
        "ci_68_upper": ci_68_upper,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "features": final_features,
        "best_vi": best_single_index,
    }
    stage_uncertainty[stage] = uncertainty_tons_ha

    # Convert to tons/ha for the scatter axes (axis scale stays in t/ha)
    plot_area_vals = stage_data["plot_area_m2"].values
    y_true_tons = (y_true_all / 1_000_000) / (
        np.array([plot_area_vals[i] for i in range(len(y_true_all))]) / 10_000
    )
    y_pred_tons = (y_pred_all / 1_000_000) / (
        np.array([plot_area_vals[i] for i in range(len(y_pred_all))]) / 10_000
    )
    stage_cv_predictions[stage] = {
        "y_true": y_true_tons,
        "y_pred": y_pred_tons,
        "groups": groups_all,
    }

# ============================================================
# BEST STAGE
# ============================================================

BEST_STAGE = min(stage_results, key=lambda s: stage_results[s]["mae_g"])
print(f"\n{'='*60}")
print(f"🏆 Best growth stage identified: {BEST_STAGE.upper()}")
print(f"{'='*60}")
print(f"   MAE  : {stage_results[BEST_STAGE]['mae_g']:.2f} g")
print(f"   RMSE : {stage_results[BEST_STAGE]['rmse_g']:.2f} g")
print(f"   R²   : {stage_results[BEST_STAGE]['r2']:.3f}")
print(f"   Best VI: {stage_results[BEST_STAGE]['best_vi']}")
print(f"   Uncertainty: ±{stage_results[BEST_STAGE]['uncertainty_tons_ha']:.2f} t/ha")
print(f"   68% CI: {stage_results[BEST_STAGE]['ci_68_lower']:.2f} to {stage_results[BEST_STAGE]['ci_68_upper']:.2f} t/ha")
print(f"   95% CI: {stage_results[BEST_STAGE]['ci_95_lower']:.2f} to {stage_results[BEST_STAGE]['ci_95_upper']:.2f} t/ha")

# ============================================================
# FINAL MODEL (best stage only)
# ============================================================

print(f"\n🔧 Training final model using ONLY the {BEST_STAGE.upper()} stage...\n")

best_data = data[data[STAGE] == BEST_STAGE]
X_final   = best_data[stage_results[BEST_STAGE]["features"]]
y_final   = best_data[TARGET]

final_model = build_model()
final_model.fit(X_final, y_final)

# Training metrics (tons/ha for slide 13 consistency)
train_preds_g       = final_model.predict(X_final)
train_preds_tons_ha = (train_preds_g / 1_000_000) / (best_data["plot_area_m2"] / 10_000)
y_train_tons_ha     = (y_final / 1_000_000) / (best_data["plot_area_m2"] / 10_000)

train_rmse = np.sqrt(mean_squared_error(y_train_tons_ha, train_preds_tons_ha))
train_r2   = r2_score(y_train_tons_ha, train_preds_tons_ha)

# Validation metrics via cross-validation
y_true_val, y_pred_val = [], []
gkf = GroupKFold(n_splits=min(3, best_data[LOCATION].nunique()))
for tr, te in gkf.split(X_final, y_final, best_data[LOCATION]):
    model = build_model()
    model.fit(X_final.iloc[tr], y_final.iloc[tr])
    preds = model.predict(X_final.iloc[te])

    preds_tons_ha       = (preds / 1_000_000) / (best_data.iloc[te]["plot_area_m2"] / 10_000)
    y_true_tons_ha_fold = (y_final.iloc[te] / 1_000_000) / (best_data.iloc[te]["plot_area_m2"] / 10_000)

    y_true_val.extend(y_true_tons_ha_fold)
    y_pred_val.extend(preds_tons_ha)

y_true_val = np.array(y_true_val)
y_pred_val = np.array(y_pred_val)

val_rmse   = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
val_r2     = r2_score(y_true_val, y_pred_val)
mean_yield = train_preds_tons_ha.mean()
std_yield  = train_preds_tons_ha.std()

print(f"📊 {BEST_STAGE.upper()} stage yield prediction:")
print(f"   Mean yield: {mean_yield:.2f} tons/ha")
print(f"   Std dev:    {std_yield:.2f} tons/ha")
print(f"\n   Training RMSE: {train_rmse:.2f} tons/ha")
print(f"   Training R²:   {train_r2:.3f}")
print(f"   Validation RMSE: {val_rmse:.2f} tons/ha")
print(f"   Validation R²:   {val_r2:.3f}")

# ============================================================
# SINGLE-PANEL: PREDICTED VS OBSERVED (BEST STAGE)
# ============================================================

plt.figure(figsize=(8, 8))
plt.scatter(y_true_val, y_pred_val, alpha=0.6, edgecolors='k', s=80)

min_val = min(y_true_val.min(), y_pred_val.min())
max_val = max(y_true_val.max(), y_pred_val.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

plt.xlabel('Observed Yield (tons/ha)',  fontsize=12, fontweight='bold')
plt.ylabel('Predicted Yield (tons/ha)', fontsize=12, fontweight='bold')
plt.title(f'Predicted vs Observed Yield – {BEST_STAGE.upper()} Stage',
          fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

textstr = (
    f'Validation R² = {val_r2:.3f}\n'
    f'Validation RMSE = {val_rmse:.2f} tons/ha\n'
    f'Training R² = {train_r2:.3f}\n'
    f'Training RMSE = {train_rmse:.2f} tons/ha'
)
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIRECTORY, f'Predicted_vs_Observed_{BEST_STAGE.upper()}.png'),
    dpi=300,
)
print(f"\n✅ Predicted vs Observed plot saved")
plt.close()

# ============================================================
# MULTI-PANEL: PREDICTED VS OBSERVED (3 GROWTH STAGES)
# ── Annotations now match slide 11 exactly ──────────────────
#    Each panel shows: MAE (g) | RMSE (g) | R² | Uncertainty (t/ha)
# ============================================================

print("\n📊 Generating multi-panel Predicted vs Observed plot...")

# Build consistent colour map across all stages
all_groups_all = []
for s in STAGE_ORDER:
    if s in stage_cv_predictions:
        all_groups_all.extend(stage_cv_predictions[s]["groups"])

all_locations = sorted(set(all_groups_all))
_n = len(all_locations)
print(f"   Found {_n} unique locations")

if _n <= 10:
    _palette = [plt.cm.tab10(i) for i in range(_n)]
elif _n <= 20:
    _palette = [plt.cm.tab20(i) for i in range(_n)]
else:
    _palette = [plt.cm.tab20(i % 20) for i in range(_n)]

loc_color_map = {loc: _palette[i] for i, loc in enumerate(all_locations)}

panels = [
    ("A", "Early Stage", stage_cv_predictions.get("early"), "early"),
    ("B", "Mid Stage",   stage_cv_predictions.get("mid"),   "mid"),
    ("C", "Late Stage",  stage_cv_predictions.get("late"),  "late"),
]

# --- Figure: 2×2 grid (A=top-left, B=top-right, C=bot-left, legend=bot-right) ---
fig = plt.figure(figsize=(14, 13))

gs = GridSpec(
    2, 2, figure=fig,
    top=0.91, bottom=0.07,
    left=0.08, right=0.97,
    hspace=0.50,   # extra vertical gap to fit two-line annotation above each panel
    wspace=0.30,
)

ax_positions = [(0, 0), (0, 1), (1, 0)]
axes = [fig.add_subplot(gs[r, c]) for r, c in ax_positions]

legend_handles, legend_labels = [], []

for ax, (letter, title, pdata, stage_key) in zip(axes, panels):

    if pdata is None or len(pdata["y_true"]) == 0:
        ax.set_visible(False)
        continue

    yt   = np.asarray(pdata["y_true"])
    yp   = np.asarray(pdata["y_pred"])
    grps = list(pdata["groups"])

    # ── Pull slide-11-consistent metrics from stage_results ──────────────
    res             = stage_results[stage_key]
    mae_g           = res["mae_g"]                  # grams  (slide 11 col: MAE)
    rmse_g          = res["rmse_g"]                 # grams  (slide 11 col: RMSE)
    r2_val          = res["r2"]                     # dimensionless (slide 11 col: R²)
    uncert_tons_ha  = res["uncertainty_tons_ha"]    # t/ha   (slide 11 col: Uncertainty)

    # Scatter points coloured by location
    for loc in all_locations:
        mask = np.array([g == loc for g in grps])
        if mask.any():
            sc = ax.scatter(
                yp[mask], yt[mask],
                color=loc_color_map[loc],
                s=45, alpha=0.7,
                edgecolors='none',
                zorder=3,
            )
            if str(loc) not in legend_labels:
                legend_handles.append(sc)
                legend_labels.append(str(loc))

    # 1:1 reference line
    pad    = 0.05
    lo_raw = min(yt.min(), yp.min())
    hi_raw = max(yt.max(), yp.max())
    span   = hi_raw - lo_raw
    lim_lo = max(0, lo_raw - pad * span)
    lim_hi = hi_raw + pad * span

    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k-', lw=2, alpha=0.9, zorder=2)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)

    # Axis formatting
    ax.set_xlabel("Predicted yield (Ton/Ha)", fontsize=13)
    ax.set_ylabel("Reported yield (Ton/Ha)", fontsize=13)
    ax.tick_params(labelsize=11, width=1, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    ax.set_facecolor('white')
    ax.grid(False)

    # Panel label – inside top-left corner
    ax.text(
        0.05, 0.95, f"({letter}) {title}",
        transform=ax.transAxes, fontsize=13,
        va='top', ha='left', fontweight='bold',
    )

    # ── Above-axis annotations (matching slide 11 columns) ───────────────
    # Line 1: MAE and RMSE (both in grams, as per slide 11)
    ax.text(
        0.5, 1.10,
        f"MAE={mae_g:.2f}g    RMSE={rmse_g:.2f}g",
        transform=ax.transAxes, fontsize=12,
        ha='center', va='bottom', clip_on=False,
    )
    # Line 2: R² and Uncertainty (matching slide 11 columns)
    ax.text(
        0.5, 1.03,
        f"R²={r2_val:.3f}    Uncertainty=±{uncert_tons_ha:.2f} t/ha",
        transform=ax.transAxes, fontsize=12,
        ha='center', va='bottom', fontweight='bold', clip_on=False,
    )

# ── LEGEND (bottom-right GridSpec cell) ──────────────────────────────────────
ax_legend = fig.add_subplot(gs[1, 1])
ax_legend.set_axis_off()

n_items = len(legend_labels)
ncols   = 2 if n_items > 14 else 1

ax_legend.legend(
    legend_handles, legend_labels,
    loc='center',
    ncol=ncols,
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    fontsize=10,
    handletextpad=0.5,
    labelspacing=0.55,
    columnspacing=1.0,
    borderpad=0.9,
    markerscale=1.5,
    title='Plot ID',
    title_fontsize=11,
)

multipanel_path = os.path.join(
    OUTPUT_DIRECTORY, "Predicted_vs_Observed_3STAGES_multipanel.png"
)
plt.savefig(
    multipanel_path, dpi=300, bbox_inches='tight',
    facecolor='white', edgecolor='none', pad_inches=0.25,
)
print(f"✅ Multi-panel Predicted vs Observed plot saved → {multipanel_path}")
plt.close()

# ============================================================
# EXPORT CONFIDENCE LEVEL SUMMARY TABLE
# ============================================================

print("\n📊 Exporting uncertainty confidence levels summary...")

confidence_summary = []
for stage in STAGE_ORDER:
    if stage in stage_results:
        res = stage_results[stage]
        confidence_summary.append({
            "Growth Stage": stage.upper(),
            "Uncertainty (t/ha)": res["uncertainty_tons_ha"],
            "68% CI Lower (t/ha)": res["ci_68_lower"],
            "68% CI Upper (t/ha)": res["ci_68_upper"],
            "95% CI Lower (t/ha)": res["ci_95_lower"],
            "95% CI Upper (t/ha)": res["ci_95_upper"],
            "MAE (g)": res["mae_g"],
            "RMSE (g)": res["rmse_g"],
            "R²": res["r2"],
        })

confidence_df = pd.DataFrame(confidence_summary)
confidence_csv_path = os.path.join(OUTPUT_DIRECTORY, "uncertainty_confidence_levels.csv")
confidence_df.to_csv(confidence_csv_path, index=False, float_format="%.3f")
print(f"✅ Confidence levels summary saved → {confidence_csv_path}")

# Print summary table
print("\n" + "="*80)
print("UNCERTAINTY CONFIDENCE LEVELS SUMMARY")
print("="*80)
print(confidence_df.to_string(index=False))
print("="*80)

# ============================================================
# SPATIAL YIELD MAPS – PER STAGE
# ============================================================

def yield_map_tons_ha(ndvi_img, plot_area_m2):
    return ndvi_img.multiply(1000).add(500).divide(1_000_000).divide(plot_area_m2 / 10_000)


print("\n🗺️  Exporting yield maps for each growth stage...")

for stage in STAGE_ORDER:
    stage_data = data[data[STAGE] == stage]
    if stage_data.empty:
        continue

    avg_plot_area    = stage_data["plot_area_m2"].mean()
    yield_map_stage  = yield_map_tons_ha(stage_fused_ndvi[stage], avg_plot_area)

    # Export yield map
    with suppress_output():
        geemap.ee_export_image(
            yield_map_stage,
            os.path.join(OUTPUT_DIRECTORY, f"Yield_{stage.upper()}_stage_TONS_PER_HA.tif"),
            scale=10, region=AOI, file_per_band=False,
        )

print("✅ All yield maps exported")

# ============================================================
# COMBINED NDVI AND YIELD MAPS (ALL STAGES)
# ============================================================

print("\n🗺️  Exporting COMBINED maps (all stages)...")

# Combined Sentinel-2 ONLY NDVI
combined_s2_only_ndvi = (
    stage_s2_only_ndvi["early"]
    .add(stage_s2_only_ndvi["mid"])
    .add(stage_s2_only_ndvi["late"])
    .divide(3)
)

with suppress_output():
    geemap.ee_export_image(
        combined_s2_only_ndvi,
        os.path.join(OUTPUT_DIRECTORY, "NDVI_S2_ONLY_COMBINED_all_stages.tif"),
        scale=10, region=AOI, file_per_band=False,
    )

# Combined FUSED NDVI (S2+S1)
combined_fused_ndvi = (
    stage_fused_ndvi["early"]
    .add(stage_fused_ndvi["mid"])
    .add(stage_fused_ndvi["late"])
    .divide(3)
)

with suppress_output():
    geemap.ee_export_image(
        combined_fused_ndvi,
        os.path.join(OUTPUT_DIRECTORY, "NDVI_FUSED_S2_S1_COMBINED_all_stages.tif"),
        scale=10, region=AOI, file_per_band=False,
    )

avg_plot_area_all   = data["plot_area_m2"].mean()
combined_yield_map  = yield_map_tons_ha(combined_fused_ndvi, avg_plot_area_all)

with suppress_output():
    geemap.ee_export_image(
        combined_yield_map,
        os.path.join(OUTPUT_DIRECTORY, "Yield_COMBINED_all_stages_TONS_PER_HA.tif"),
        scale=10, region=AOI, file_per_band=False,
    )

print("✅ All combined maps exported")

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"\nKey Outputs:")
print(f"  • Sentinel-2 ONLY NDVI maps (BEFORE fusion) - same spatial extent as fused")
print(f"  • Sentinel-2 + Sentinel-1 FUSED NDVI maps (AFTER fusion)")
print(f"  • Both products now use identical AOI clipping for fair comparison")
print(f"  • Uncertainty confidence levels CSV")
print(f"  • Yield maps for all stages")
print(f"  • Combined yield map")
print("=" * 60)
