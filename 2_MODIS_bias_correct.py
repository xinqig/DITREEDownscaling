import os
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
import gpflow
import tensorflow as tf

# ============================================================
# 1. CONFIGURATION
# ============================================================

landsat_dir = 'Landsat_wofs'
modis_dir = 'MODIS_wofs'
modis_val_dir = 'MODIS_filtered'        # folder to bias-correct
water_body_mask = 'large_water_bodies/large_water_bodies.shp'

# Only these 6 bands will be calibrated with GP
calibration_bands = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

# These indices will be computed AFTER calibration
index_bands = ["mNDWI1", "mNDWI2", "NDVI"]

crs_wgs84 = "EPSG:4326"

output_dir = "MODIS_corrected"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2. UTILITIES
# ============================================================

def assign_crs(ds):
    if not ds.rio.crs:
        ds = ds.rio.write_crs(crs_wgs84)
    return ds

def mask_water_bodies(ds, shapefile):
    ds = assign_crs(ds)
    if shapefile.crs != ds.rio.crs:
        shapefile = shapefile.to_crs(ds.rio.crs)
    return ds.rio.clip(shapefile.geometry, shapefile.crs,
                       drop=False, invert=True, all_touched=True)

def compute_index(b1, b2):
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = (b1 - b2) / (b1 + b2)
        idx[(b1 + b2) == 0] = np.nan
    return idx

# ============================================================
# 3. TRAINING GP MODEL
# ============================================================

def train_sgpr(modis_vals, landsat_vals, num_inducing=100):

    mask = (~np.isnan(modis_vals)) & (~np.isnan(landsat_vals))
    X = modis_vals[mask].reshape(-1, 1).astype(np.float64)
    Y = landsat_vals[mask].reshape(-1, 1).astype(np.float64)

    inducing = X[np.random.choice(len(X), size=min(num_inducing, len(X)),
                                  replace=False)]

    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.Gaussian()

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing,
        num_latent_gps=1
    )

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure((X, Y)),
                 variables=model.trainable_variables,
                 options=dict(maxiter=500),   
                 method="L-BFGS-B")

    return model

# ============================================================
# 4. LOAD TRAINING DATA (MODIS–LANDSAT PAIRS)
# ============================================================

water_bodies = gpd.read_file(water_body_mask)

landsat_files = sorted([f for f in os.listdir(landsat_dir) if f.endswith(".nc")])
modis_files  = sorted([f for f in os.listdir(modis_dir)  if f.endswith(".nc")])

train_pairs = {}

for lf, mf in zip(landsat_files, modis_files):
    date = lf.split("_")[-1].split(".nc")[0]
    l_ds = mask_water_bodies(xr.open_dataset(os.path.join(landsat_dir, lf)), water_bodies)
    m_ds = mask_water_bodies(xr.open_dataset(os.path.join(modis_dir,  mf)), water_bodies)
    train_pairs[date] = (l_ds, m_ds)

# ============================================================
# 5. CONSTRUCT TRAINING ARRAYS PER BAND
# ============================================================

training_data = {}

for band in calibration_bands:
    modis_vals = []
    landsat_vals = []

    for date, (lds, mds) in train_pairs.items():
        l_vals = lds[band].values.flatten()
        m_vals = mds[band].values.flatten()
        valid = (~np.isnan(l_vals)) & (~np.isnan(m_vals))

        landsat_vals.extend(l_vals[valid])
        modis_vals.extend(m_vals[valid])

    training_data[band] = (np.array(modis_vals), np.array(landsat_vals))

# ============================================================
# 6. TRAIN ONE GP MODEL PER BAND
# ============================================================

models = {}

for band in calibration_bands:
    print(f"\nTraining GP model for band: {band}")
    modis_vals, landsat_vals = training_data[band]
    models[band] = train_sgpr(modis_vals, landsat_vals, num_inducing=100)

print("\n=== All GP models trained ===")

# ============================================================
# 7. APPLY MODELS TO MODIS_FILTERED AND SAVE RESULTS
# ============================================================

# modis_val_files = sorted([f for f in os.listdir('MODIS_trial') if f.endswith(".nc")])
modis_val_files = sorted([f for f in os.listdir(modis_val_dir) if f.endswith(".nc")])

modis_to_band = {
    "sur_refl_b01_1": "Red",
    "sur_refl_b02_1": "NIR",
    "sur_refl_b03_1": "Blue",
    "sur_refl_b04_1": "Green",
    "sur_refl_b06_1": "SWIR1",
    "sur_refl_b07_1": "SWIR2"
}

for f in modis_val_files:
    date = f.split("_")[-1].split(".nc")[0]
    print(f"\nCorrecting MODIS for date: {date}")

    raw_modis = mask_water_bodies(xr.open_dataset(os.path.join(modis_val_dir, f)), water_bodies)
    # raw_modis = mask_water_bodies(xr.open_dataset(os.path.join('MODIS_trial', f)), water_bodies)
    # raw_modis loaded from MODIS_trial
    # raw_modis = wo_clip(raw_modis, clip_template)
    for var in raw_modis.data_vars:
        raw_modis[var] = raw_modis[var] * 10000
    
    translated = {}
    for var in raw_modis.data_vars:
        if var in modis_to_band:
            translated[modis_to_band[var]] = raw_modis[var]

    ds = xr.Dataset(translated, coords=raw_modis.coords, attrs=raw_modis.attrs)    
    corrected_bands = {}

    # -----------------------------------------------
    # STEP 1 — Calibrate ONLY raw bands (NO indices)
    # -----------------------------------------------
    for band in calibration_bands:
        arr = ds[band].values
        flat = arr.flatten().astype(np.float64)

        mask = ~np.isnan(flat)
        X_pred = flat[mask].reshape(-1, 1)

        mean, _ = models[band].predict_f(X_pred)
        mean = mean.numpy().flatten()

        corrected = np.full_like(flat, np.nan, dtype=np.float32)
        corrected[mask] = mean

        corrected_bands[band] = corrected.reshape(arr.shape)

    # -----------------------------------------------
    # STEP 2 — Compute indices AFTER band calibration
    # -----------------------------------------------

    corrected_bands["mNDWI1"] = compute_index(
        corrected_bands["Green"], corrected_bands["SWIR1"]
    )

    corrected_bands["mNDWI2"] = compute_index(
        corrected_bands["Green"], corrected_bands["SWIR2"]
    )

    corrected_bands["NDVI"] = compute_index(
        corrected_bands["NIR"], corrected_bands["Red"]
    )

    # -----------------------------------------------
    # STEP 3 — Save all (bands + indices)
    # -----------------------------------------------

    out_ds = xr.Dataset(
        {band: (ds['Green'].dims, corrected_bands[band])
         for band in calibration_bands + index_bands},
        coords=ds.coords,
        attrs=ds.attrs
    )

    # out_path = os.path.join(output_dir, f"calibrated_modis_{date}.nc")
    # out_ds.to_netcdf(out_path)
    np.savez(f"calibrated_modis_{date}.npz", **corrected_bands)
    # print(f"Saved → {out_path}")

print("\n=== DONE! All MODIS_filtered files have been corrected ===")
