#!/usr/bin/env python
# ============================================================
# VEGETATION INDEX EXTRACTION – CLOUD-ROBUST (OPTICAL + SAR)
# Sentinel-1 used ONLY for cloud gap filling (feature-level)
# ============================================================

import ee
import geopandas as gpd
import pandas as pd

# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================
ee.Initialize(project="quiet-subset-447718-q0")

# ============================================================
# INPUTS
# ============================================================
PLOT_SHP = "GGE_Harvest_150.shp"
CROP_COL = "Field"
AREA_COL = "Shape_Area"
OUTPUT_CSV = "plot_satellite_indices_cloud_robust.csv"
BATCH_SIZE = 10
MIN_S2_IMAGES = 5   # Threshold for cloud gap handling

# ============================================================
# SEASONAL STAGES – WINTER WHEAT (CENTRAL GERMANY)
# ============================================================
STAGES = {
    "early": ("2025-03-01", "2025-04-30"),
    "mid":   ("2025-05-01", "2025-06-30"),
    "late":  ("2025-07-01", "2025-08-10"),
}

# ============================================================
# LOAD & FIX SHAPEFILE
# ============================================================
gdf = gpd.read_file(PLOT_SHP)
gdf["geometry"] = gdf["geometry"].buffer(0)

gdf_utm = gdf.to_crs(epsg=32632)
gdf_utm["geometry"] = gdf_utm.geometry.centroid
gdf = gdf_utm.to_crs(epsg=4326)

print(f"✅ Loaded {len(gdf)} plots")

# ============================================================
# VEGETATION INDICES (OPTICAL)
# ============================================================
def add_indices(img):
    nir = img.select("B8")
    red = img.select("B4")
    green = img.select("B3")
    rededge = img.select("B5")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = nir.subtract(red).divide(
        nir.add(red.multiply(6)).add(1)
    ).multiply(2.5).rename("EVI")
    ndre = nir.subtract(rededge).divide(nir.add(rededge)).rename("NDRE")
    gndvi = nir.subtract(green).divide(nir.add(green)).rename("GNDVI")
    ciredge = nir.divide(rededge).subtract(1).rename("CIrededge")

    return img.addBands([ndvi, evi, ndre, gndvi, ciredge])

# ============================================================
# SENTINEL-1 (SAR) – CLOUD GAP FILLING ONLY
# ============================================================
def add_vh_linear(img):
    return img.addBands(
        img.expression("pow(10, vh / 10)", {"vh": img.select("VH")})
        .rename("VH_linear")
    )

def safe_get(d, key):
    return ee.Algorithms.If(d.contains(key), d.get(key), None)

OPTICAL_INDICES = ["NDVI", "EVI", "NDRE", "GNDVI", "CIrededge"]

# ============================================================
# EXTRACTION FUNCTION
# ============================================================
def extract_plot(idx, row):

    geom = ee.Geometry.Point(row.geometry.x, row.geometry.y)
    features = []

    for stage, (start, end) in STAGES.items():

        record = {
            "location": f"plot_{idx}",
            "crop": row[CROP_COL],
            "growth_stage": stage,
            "plot_area_m2": row[AREA_COL]
        }

        # ------------------------------
        # Sentinel-2 (OPTICAL)
        # ------------------------------
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(add_indices)
        )

        s2_count = s2.size()
        record["S2_count"] = s2_count

        for idx_name in OPTICAL_INDICES:
            record[f"{idx_name}_mean"] = safe_get(
                s2.select(idx_name).mean().reduceRegion(
                    ee.Reducer.mean(), geom, 10
                ), idx_name
            )
            record[f"{idx_name}_auc"] = safe_get(
                s2.select(idx_name).sum().reduceRegion(
                    ee.Reducer.mean(), geom, 10
                ), idx_name
            )

        # ------------------------------
        # Sentinel-1 – ONLY if S2 sparse
        # ------------------------------
        use_sar = ee.Number(s2_count).lt(MIN_S2_IMAGES)

        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            ))
            .map(add_vh_linear)
        )

        vh_mean = safe_get(
            s1.select("VH_linear").mean().reduceRegion(
                ee.Reducer.mean(), geom, 10
            ), "VH_linear"
        )

        record["VH_gapfill"] = ee.Algorithms.If(use_sar, vh_mean, None)

        # ------------------------------
        # ERA5 WEATHER
        # ------------------------------
        era5 = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(geom)
            .filterDate(start, end)
        )

        record["rainfall"] = safe_get(
            era5.select("total_precipitation_sum").sum().reduceRegion(
                ee.Reducer.mean(), geom, 1000
            ), "total_precipitation_sum"
        )

        record["temp"] = safe_get(
            era5.select("temperature_2m").mean().reduceRegion(
                ee.Reducer.mean(), geom, 1000
            ), "temperature_2m"
        )

        features.append(ee.Feature(None, record))

    return features

# ============================================================
# BATCH PROCESSING
# ============================================================
dfs = []

for start in range(0, len(gdf), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(gdf))
    print(f"📦 Processing plots {start}–{end-1}")

    feats = []
    for i in range(start, end):
        feats.extend(extract_plot(i, gdf.iloc[i]))

    fc = ee.FeatureCollection(feats)

    df = ee.data.computeFeatures({
        "expression": fc,
        "fileFormat": "PANDAS_DATAFRAME"
    })

    dfs.append(df)

# ============================================================
# FINAL OUTPUT
# ============================================================
out_df = pd.concat(dfs, ignore_index=True)

# Order stages
out_df["growth_stage"] = pd.Categorical(
    out_df["growth_stage"],
    ["early", "mid", "late"],
    ordered=True
)

out_df = out_df.sort_values(
    ["growth_stage", "location"]
).reset_index(drop=True)

out_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Cloud-robust dataset exported")
print(out_df.head())
