import os
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from pyproj import CRS
from rasterio.enums import Resampling
import glob
from eofs.xarray import Eof
from affine import Affine
import re
# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the NetCDF data
tiff_file = 'DEM.tif'
ds = xr.open_rasterio(tiff_file)  # Use open_rasterio for GeoTIFF files
# Add Inundation Information

data_dates = ['2016-10-25','2022-09-24', '2022-10-10', '2022-11-11', '2022-12-13', '2022-12-29']

date = data_dates[4]

DEM = ds.sel(band=1).squeeze()

# modis_wf = sorted([f for f in os.listdir('MODIS_WF') if f.endswith(".nc")])
# data_dates = []
# for modis_file in modis_wf:
#     date = modis_file.split("_")[-1].split(".nc")[0]
#     data_dates.append(date)
    
# date = data_dates[0]

# predicted_wf = xr.open_dataset(f'MODIS_WF/water_fraction_{date}.nc')['water_fraction']
# predicted_wf = xr.open_dataset(f'WOs_Bourke_clipped_bounds/clipped_bound_wos_wgs84_{date}.nc')['__xarray_dataarray_variable__']
predicted_wf = xr.open_dataset(f'water_fraction_{date}.nc')['water_fraction']
predicted_wf = predicted_wf.rio.write_crs("EPSG:4326")
# predicted_wf = predicted_wf.rio.reproject(target_crs)

DEM_clipped = DEM.rio.clip_box(
    minx=float(predicted_wf.x.min()),
    maxx=float(predicted_wf.x.max()),
    miny=float(predicted_wf.y.min()),
    maxy=float(predicted_wf.y.max())
)

dem_4326 = DEM_clipped
modis_4326 = predicted_wf



def downscale_modis_to_dem(dem_4326, modis_4326):
    """
    dem_4326: 2D DataArray (y, x), EPSG:4326, fine (~30 m)
    modis_4326: 2D DataArray (y, x), EPSG:4326, coarse (~698 m), contains water fraction
    """
    # --- 0. Basic checks ---
    ny_f, nx_f = dem_4326.shape
    ny_c, nx_c = modis_4326.shape

    transform: Affine = modis_4326.rio.transform()
    inv = ~transform  # world -> pixel

    # --- 1. Compute parent MODIS row/col for each DEM pixel using the transform ---
    xs, ys = np.meshgrid(dem_4326["x"].values, dem_4326["y"].values)
    cols, rows = inv * (xs, ys)          # float pixel coords
    cols = cols.astype(int)
    rows = rows.astype(int)

    # mask out DEM pixels that fall outside the MODIS raster
    inside = (
        (cols >= 0) & (cols < nx_c) &
        (rows >= 0) & (rows < ny_c)
    )

    # --- 2. Build parent_id on DEM grid ---
    parent_id = np.full_like(cols, -1, dtype=int)
    parent_id[inside] = rows[inside] * nx_c + cols[inside]

    # --- 3. Flatten DEM + parent info ---
    elev = dem_4326.values
    valid_dem = np.isfinite(elev) & inside
    if not np.any(valid_dem):
        raise ValueError("No valid DEM pixels inside MODIS extent.")

    pid_flat = parent_id.flatten()[valid_dem.flatten()]
    e_flat   = elev.flatten()[valid_dem.flatten()]

    # --- 4. Flatten MODIS water fraction, align indexing with pid definition ---
    wf_parent = modis_4326.values.reshape(-1)

    # handle 0–100 vs 0–1
    if np.nanmax(wf_parent) > 1.1:
        wf_parent = wf_parent / 100.0

    # --- 5. Sort by parent_id then elevation ---
    order = np.lexsort((e_flat, pid_flat))  # first pid, then elevation
    pid_s = pid_flat[order]
    e_s   = e_flat[order]

    # --- 6. Group by parent_id to compute how many fine pixels should be water ---
    upids, start_idx, counts = np.unique(pid_s, return_index=True, return_counts=True)

    wf_target = wf_parent[upids]  # fraction for each parent_id
    n_target  = np.rint(wf_target * counts).astype(int)

    # --- 7. Assign water flags in sorted space ---
    water_s = np.zeros_like(e_s, dtype=bool)

    for s, c, nt, frac in zip(start_idx, counts, n_target, wf_target):
        if not np.isfinite(frac) or frac <= 0:
            continue
        if frac >= 1:
            water_s[s:s+c] = True
        else:
            nt = max(0, min(c, nt))
            water_s[s:s+nt] = True

    # --- 8. Scatter back to full DEM grid ---
    valid_flat = valid_dem.flatten()
    water_flat = np.zeros_like(valid_flat, dtype=bool)
    water_flat[np.where(valid_flat)[0][order]] = water_s

    water_mask = xr.zeros_like(dem_4326, dtype=bool)
    water_mask.values = water_flat.reshape(dem_4326.shape)

    return water_mask

water_mask = downscale_modis_to_dem(dem_4326, modis_4326)

plt.figure(figsize=(12, 8))
water_mask.plot.imshow(cmap="Blues")
plt.title("Downscaled 30 m Water Mask (DEM grid)")
plt.show()

water_int = water_mask.astype("int16")



########################## EOF Reconstruction ################################
modes = 500

fine_data_mean = np.load("seasonal_wos_mean.npy")
fine_data_mean_stacked = fine_data_mean.reshape(-1)

eofs = xr.open_dataset('seasonal_wos_eofs_500.nc')['eofs']
eofs_stacked = eofs.stack(space=('y', 'x'))
eofs_stacked_val = eofs_stacked.values
spatial_shape = fine_data_mean.shape


water_dir = "MODIS_WF_downscaled"
water_files = sorted(glob.glob(os.path.join(water_dir, "downscaled_water_mask_*.nc")))
ds_w = xr.open_dataset(water_files[0])
water_int = ds_w['water_mask_30m']
water_int = water_int.fillna(0)
# ensure CRS (if relevant to later steps; doesn't affect EOF math)
water_int = water_int.rio.write_crs("EPSG:4326")

# da_stack_matched = da_stack_matched.rio.write_crs("EPSG:4326", inplace=True)
# water_int = water_int.rio.write_crs("EPSG:4326", inplace=True)


# Align / reproject the original WO to the same grid as wo_wf
# original_wo_clipped = original_wo.rio.reproject_match(da_stack_matched[0])
dem_wo_stacked = np.reshape(water_int.values, (-1))

nan_mask = (
    np.isfinite(dem_wo_stacked) &
    np.isfinite(fine_data_mean_stacked) &
    np.isfinite(eofs_stacked_val).all(axis=0)
)
dem_wo_valid = dem_wo_stacked[nan_mask]
fine_data_mean_valid = fine_data_mean_stacked[nan_mask]
eofs_valid = eofs_stacked_val[:, nan_mask]


coarse_ec, _, _, _ =np.linalg.lstsq(eofs_valid.T, dem_wo_valid - fine_data_mean_valid, rcond=None)

# Reshape calculated_pc to have shape (1, M)
coarse_ec_reshaped = coarse_ec.reshape(1, modes)


reconstructed_data = coarse_ec_reshaped.dot(eofs_valid) + fine_data_mean_valid

reconstructed_valid = reconstructed_data.ravel()
reconstructed_flat = np.full(nan_mask.shape, np.nan)
reconstructed_flat[nan_mask] = reconstructed_valid


reconstructed_reshaped = reconstructed_flat.reshape(spatial_shape)
binary_reconstructed = xr.where(reconstructed_reshaped > 0.5, 1, 0)

reconstructed_da = xr.DataArray(
    binary_reconstructed,
    dims=("y", "x"),   # match order used in fine_profim_eofs
    coords={
        "y": water_int["y"],
        "x": water_int["x"]
    },
    name="reconstructed_flood"
)
reconstructed_ds = reconstructed_da.to_dataset()

reconstructed_ds['reconstructed_flood'].plot()
