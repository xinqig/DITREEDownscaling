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

# -----------------------
# 2. List all water_int files
# -----------------------
# adjust pattern/path as needed
water_dir = "MODIS_WF_downscaled_new"
# water_dir = "MODIS_shep_WF_downscaled"
water_files = sorted(glob.glob(os.path.join(water_dir, "downscaled_water_mask_*.nc")))

print(f"Found {len(water_files)} water_int files")

wf_dir = "MODIS_WF"
# wf_dir = "MODIS_WF_shep"
wf_files = sorted(glob.glob(os.path.join(wf_dir, "water_fraction_*.nc")))


# # folder to save reconstructions
# out_dir = "MODIS_WF_reconstructed_shep"
# os.makedirs(out_dir, exist_ok=True)

reconstructed_list = []
recon_dates = []
for wf in  wf_files:
    ds_wf = xr.open_dataset(wf)
    water_fraction = ds_wf['water_fraction']
    water_fraction_stacked = np.reshape(water_fraction.values, (-1))
    nan_mask = (np.isfinite(water_fraction_stacked))
# eofs_200 = eofs_stacked_val[:200, :]
modes = 500

fine_data_mean = np.load("seasonal_wos_mean.npy")
fine_data_mean_stacked = fine_data_mean.reshape(-1)

eofs = xr.open_dataset('seasonal_wos_eofs_500.nc')['eofs']
eofs_stacked = eofs.stack(space=('y', 'x'))
eofs_stacked_val = eofs_stacked.values
spatial_shape = fine_data_mean.shape

# -----------------------
# 3. Reconstruction loop
# -----------------------
for f in water_files:
    # extract date from filename, assuming pattern ..._<YYYY-MM-DD>.nc
    base = os.path.basename(f)
    date = base.split("_")[-1].split(".nc")[0]

    print(f"Reconstructing for {date} from {f}")

    ds_w = xr.open_dataset(f)
    water_int = ds_w['water_mask_30m']
    # water_int = water_int.fillna(0)
    

    # ensure CRS (if relevant to later steps; doesn't affect EOF math)
    water_int = water_int.rio.write_crs("EPSG:4326")

    # reconstruction
    dem_wo_stacked = np.reshape(water_int.values, (-1))

    nan_mask = (
        np.isfinite(dem_wo_stacked) &
        np.isfinite(fine_data_mean_stacked) &
        np.isfinite(eofs_stacked_val).all(axis=0)
    )

    dem_wo_valid = dem_wo_stacked[nan_mask]
    fine_data_mean_valid = fine_data_mean_stacked[nan_mask]
    eofs_valid = eofs_stacked_val[:, nan_mask]   # (modes, n_valid)

    # least squares solve for expansion coefficients
    coarse_ec, _, _, _ = np.linalg.lstsq(
        eofs_valid.T,              # (n_valid, modes)
        dem_wo_valid - fine_data_mean_valid,
        rcond=None
    )

    # reshape to (1, modes) just like you had
    coarse_ec_reshaped = coarse_ec.reshape(1, modes)

    # reconstruct on valid points
    reconstructed_data = coarse_ec_reshaped.dot(eofs_valid) + fine_data_mean_valid
    reconstructed_valid = reconstructed_data.ravel()

    # put back into full grid
    reconstructed_flat = np.full(nan_mask.shape, np.nan)
    reconstructed_flat[nan_mask] = reconstructed_valid
    reconstructed_reshaped = reconstructed_flat.reshape(spatial_shape)

    # threshold to binary
    binary_reconstructed = xr.where(reconstructed_reshaped > 0.5, 1, 0)

    # wrap in DataArray with coords taken from water_int
    reconstructed_da = xr.DataArray(
        binary_reconstructed,
        dims=("y", "x"),
        coords={
            "y": water_int["y"],
            "x": water_int["x"]
        },
        name="reconstructed_flood"
    )

    # # save each reconstructed field to NetCDF
    # out_path = os.path.join(out_dir, f"reconstructed_flood_{date}.nc")
    # reconstructed_da.to_dataset().to_netcdf(out_path)
    # print(f"Saved reconstructed map to {out_path}")

    # store for plotting
    reconstructed_list.append(reconstructed_da)
    recon_dates.append(date)

# -----------------------
# 4. Plot all reconstructions in a single multi-panel figure
# -----------------------
n = len(reconstructed_list)
if n > 0:
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, (recon_da, date) in enumerate(zip(reconstructed_list, recon_dates)):
        recon_da.plot.imshow(
            ax=axes[i],
            cmap="Blues",
            add_colorbar=False
        )
        axes[i].set_title(date)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # turn off unused axes (if any)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No reconstructions created.")













#################################### Time Series Animation ####################################


import glob, os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# ---------------------------
# 1) Load MODIS WF (middle column)
# ---------------------------
wf_dir = "MODIS_WF"
wf_files = sorted(glob.glob(os.path.join(wf_dir, "*.nc")))
wf_list = []
wf_dates = []

for file in wf_files:
    ds = xr.open_dataset(file)
    wf = ds["water_fraction"]
    wf_list.append(wf)
    # adjust if your naming is different
    date_str = os.path.basename(file).split("_")[-1].split(".nc")[0]
    wf_dates.append(date_str)

recon_dir = "MODIS_WF_reconstructed_new"
recon_files = sorted(glob.glob(os.path.join(recon_dir, "*.nc")))
recon_list = []
recon_dates = []

for file in recon_files:
    ds = xr.open_dataset(file)
    recon = ds["reconstructed_flood"]
    recon_list.append(recon)
    # adjust if your naming is different
    recon_date_str = os.path.basename(file).split("_")[-1].split(".nc")[0]
    recon_dates.append(recon_date_str)
# ---------------------------
# 2) Load sparse dataset (first column)
#    e.g. Landsat WOs or ProFIM, only 5 dates
# ---------------------------
sparse_dir = "Clipped_WOs"   # <-- change to your folder
sparse_files = sorted(glob.glob(os.path.join(sparse_dir, "*.nc")))

sparse_list = []
sparse_dates = []
for file in sparse_files:
    ds = xr.open_dataset(file)
    da = list(ds.data_vars.values())[0]  # or ds["var_name"]
    sparse_list.append(da)
    date_str = os.path.basename(file).split("_")[-1].split(".nc")[0]
    sparse_dates.append(date_str)

# Make a dict for quick lookup by date
sparse_dict = {d: da for d, da in zip(sparse_dates, sparse_list)}

# ---------------------------
# 3) You already have reconstructed_list & recon_dates
#    Make sure they line up with wf_list
# ---------------------------
n_frames = min(len(wf_list), len(recon_list), len(recon_dates))

# ---------------------------
# 4) Animation setup
# ---------------------------
vmin = 0
vmax = 1
cmap = "Blues"

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
cbar.set_label("Water fraction / reconstructed flood")
fps = 2
writer = FFMpegWriter(fps=fps)   # 1 frame per second

# ⏱ control how long each frame stays
base_seconds_per_frame = 1   # non-sparse dates
highlight_seconds = 4.0        # sparse dates (with sparse data available)

base_repeats = max(1, int(round(base_seconds_per_frame * fps)))
highlight_repeats = max(1, int(round(highlight_seconds * fps)))

out_name = "rq3_results.mp4"

with writer.saving(fig, out_name, dpi=150):
    for i in range(n_frames):
        date   = recon_dates[i]
        da_wf  = wf_list[i]
        da_rec = recon_list[i]
        # how many times to repeat this frame?
        if date in sparse_dict:
            repeats = highlight_repeats   # slow / paused
        else:
            repeats = base_repeats        # faster

        for _ in range(repeats):
            # Clear only the axes, keep colorbar
            for ax in axes:
                ax.clear()
        # # Clear all axes each frame
        # for ax in axes:
        #     ax.clear()

            # -------- Column 1: sparse dataset (only for 5 dates) --------
            ax0 = axes[0]
            if date in sparse_dict:
                sparse_da = sparse_dict[date]
                sparse_da.plot.imshow(
                    ax=ax0,
                    cmap=cmap,
                    add_colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax0.set_title(f"Water Observations {date}")
            else:
                ax0.set_title(f"No Landsat data\nfor {date}")
                ax0.set_xticks([])
                ax0.set_yticks([])

            ax0.set_xlabel("")
            ax0.set_ylabel("")

            # -------- Column 2: MODIS WF --------
            ax1 = axes[1]
            da_wf.plot.imshow(
                ax=ax1,
                cmap=cmap,
                add_colorbar=False,
                vmin=vmin,
                vmax=vmax,
            )
            ax1.set_title(f"MODIS Water Fraction {date}")
            ax1.set_xlabel("")
            ax1.set_ylabel("")

            # -------- Column 3: reconstructed flood --------
            ax2 = axes[2]
            da_rec.plot.imshow(
                ax=ax2,
                cmap=cmap,
                add_colorbar=False,
                vmin=vmin,
                vmax=vmax,
            )
            ax2.set_title(f"Downscaled MODIS {date}")
            ax2.set_xlabel("")
            ax2.set_ylabel("")

            writer.grab_frame()

plt.close(fig)
print(f"Saved animation to {out_name}")






######### Water Fraction, DEM Downscaled MODIS, EOF Reconstructed MODIS #########

wf_dir = "MODIS_WF"
wf_files = sorted(glob.glob(os.path.join(wf_dir, "*.nc")))
wf = {}
for f in wf_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    wf[d] = ds["water_fraction"]

recon_dir = "MODIS_WF_reconstructed_new"
recon_files = sorted(glob.glob(os.path.join(recon_dir, "*.nc")))
rec = {}
for f in recon_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    rec[d] = ds["reconstructed_flood"]

dem_dir = "MODIS_WF_downscaled_new"
dem_files = sorted(glob.glob(os.path.join(dem_dir, "*.nc")))
dem = {}
for f in dem_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    dem[d] = ds["water_mask_30m"]

sparse_dir = "Clipped_WOs"
sparse_files = sorted(glob.glob(os.path.join(sparse_dir, "*.nc")))
sparse = {}
for f in sparse_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    da = list(ds.data_vars.values())[0]
    sparse[d] = da
    

dates = ["2022-11-11","2022-11-28",]

# ---------------------------
# Plot grid: rows=dates, cols=3
# ---------------------------
vmin, vmax = 0, 1
cmap = "Blues"

n = len(dates)
fig, axes = plt.subplots(n, 3, figsize=(11, 3.2*n), constrained_layout=True)

if n == 1:
    axes = np.array([axes])  # keep indexing consistent

titles = ["MODIS Water Fraction", "DEM Downscaled MODIS", "EOF Reconstructed MODIS"]
date_label = ["11/11/2022", "28/11/2022"]
for i, (d, label) in enumerate(zip(dates, date_label)):
    das = [wf[d], dem[d], rec[d]]
    for j in range(3):
        ax = axes[i, j]
        im = das[j].plot.imshow(
            ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False
        )
        ax.set_title("" if i > 0 else titles[j], fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    # Date label on the left of each row
    axes[i, 0].text(
        -0.02, 0.5, label, transform=axes[i, 0].transAxes,
        va="center", ha="right", fontsize=12, rotation=90
    )

# Shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
cbar.set_label("Water fraction", fontsize=12)

import string

letters = list(string.ascii_lowercase)  

nrows, ncols = axes.shape
k = 0
for i in range(nrows):
    for j in range(ncols):
        axes[i, j].text(
            0.02, 0.98,
            f"({letters[k]})",
            transform=axes[i, j].transAxes,
            va="top", ha="left",
            fontsize=14, fontweight="normal"
        )
        k += 1

plt.show()



########################### Significant Modes Plot ###########################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

eof = xr.open_dataset('seasonal_wos_eofs_500.nc')

eof_da = eof["eofs"]

subset = eof_da.isel(mode=slice(0, 4))

# symmetric color scale (important for EOFs)
vmax = np.nanmax(np.abs(subset.values))
vmin = -vmax

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
axes = axes.ravel()

# --- plot ---
for i in range(4):
    ax = axes[i]
    im = eof_da.isel(mode=i).plot(
        ax=ax,
        cmap="RdBu",   # FIXED: proper EOF diverging (not reversed)
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    ax.set_title(f"Mode {i+1}", fontsize=10)

    if i % 2 == 0:
        ax.set_ylabel("Latitude", fontsize=9)
    else:
        ax.set_ylabel("")
    if i >= 2:
        ax.set_xlabel("Longitude", fontsize=9)
    else:
        ax.set_xlabel("")

    ax.tick_params(labelsize=8)


cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("EOF values", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# leave space for colorbar
plt.subplots_adjust(right=0.9)
# plt.savefig(
#     "EOF_modes.jpg",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor="white",
#     format="jpg"
# )
plt.show()



########################### Difference Maps Comparison ###########################
import glob, os
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

dem_downscale_dir = "MODIS_WF_downscaled_new"
dem_downscale_files = sorted(glob.glob(os.path.join(dem_downscale_dir, "downscaled_water_mask_*.nc")))
eof_dir = "MODIS_WF_reconstructed_new"
eof_files = sorted(glob.glob(os.path.join(water_dir, "reconstructed_flood_*.nc")))
wo_dir = "Clipped_WOs"
wo_files = sorted(glob.glob(os.path.join(sparse_dir, "*.nc")))

date = '2022-11-11'

# DEM downscaled
dem_file = [f for f in dem_downscale_files if date in f][0]

# reconstructed water
eof_file = [f for f in eof_files if date in f][0]

# sparse WO
wo_file = [f for f in wo_files if date in f][0]

dem_ds = xr.open_dataset(dem_file)['water_mask_30m']
eof_ds = xr.open_dataset(eof_file)['reconstructed_flood']
wo_ds  = xr.open_dataset(wo_file)['__xarray_dataarray_variable__']

def make_confusion_map(pred, obs):
    valid = (~np.isnan(pred)) & (~np.isnan(obs))
    out = xr.full_like(obs, np.nan, dtype=float)

    out = xr.where(valid & (pred == 0) & (obs == 0), 0, out)  # TN
    out = xr.where(valid & (pred == 1) & (obs == 1), 1, out)  # TP
    out = xr.where(valid & (pred == 1) & (obs == 0), 2, out)  # FP
    out = xr.where(valid & (pred == 0) & (obs == 1), 3, out)  # FN

    return out

dem_cmp = make_confusion_map(dem_ds, wo_ds)
eof_cmp = make_confusion_map(eof_ds, wo_ds)

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 18))

# ---- manual axes positions: [left, bottom, width, height] ----
ax0 = fig.add_axes([0.29, 0.68, 0.37, 0.22])   # top center
ax1 = fig.add_axes([0.08, 0.41, 0.37, 0.22], sharex=ax0, sharey=ax0)
ax2 = fig.add_axes([0.48, 0.41, 0.37, 0.22], sharex=ax0, sharey=ax0)
ax3 = fig.add_axes([0.08, 0.14, 0.37, 0.22], sharex=ax0, sharey=ax0)
ax4 = fig.add_axes([0.48, 0.14, 0.37, 0.22], sharex=ax0, sharey=ax0)

axes = [ax0, ax1, ax2, ax3, ax4]

# flood maps: grey dry, blue wet
flood_cmap = ListedColormap([plt.cm.Blues(0.05), plt.cm.Blues(0.9)])
flood_norm = BoundaryNorm([-0.5, 0.5, 1.5], flood_cmap.N)

# confusion maps
colors = [plt.cm.Blues(0.05), plt.cm.Blues(0.9), "#fdae61", "#f03b20"]
cmp_cmap = ListedColormap(colors)
cmp_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmp_cmap.N)

# (a) WO
im0 = ax0.pcolormesh(
    wo_ds["x"], wo_ds["y"], wo_ds.values,
    cmap=flood_cmap, norm=flood_norm, shading="auto"
)
ax0.set_title("WO Observation on 11/11/2022", fontsize=16, pad=10)
ax0.set_ylabel("Latitude", fontsize=15)
ax0.set_xlabel("Longitude", fontsize=15)
# (b) DEM
im1 = ax1.pcolormesh(
    dem_ds["x"], dem_ds["y"], dem_ds.values,
    cmap=flood_cmap, norm=flood_norm, shading="auto"
)
ax1.set_title("DEM-downscaled MODIS on 11/11/2022", fontsize=15, pad=8)
ax1.set_ylabel("Latitude", fontsize=15)


# (c) EOF
im2 = ax2.pcolormesh(
    eof_ds["x"], eof_ds["y"], eof_ds.values,
    cmap=flood_cmap, norm=flood_norm, shading="auto"
)
ax2.set_title("EOF Reconstructed MODIS on 11/11/2022", fontsize=15, pad=8)

# (d) DEM vs WO
im3 = ax3.pcolormesh(
    dem_cmp["x"], dem_cmp["y"], dem_cmp.values,
    cmap=cmp_cmap, norm=cmp_norm, shading="auto"
)
ax3.set_title("Difference Map between DEM-downscaled\nMODIS and WO Observation on 11/11/2022",
              fontsize=15, pad=8)
ax3.set_xlabel("Longitude", fontsize=15)
ax3.set_ylabel("Latitude", fontsize=15)

# (e) EOF vs WO
im4 = ax4.pcolormesh(
    eof_cmp["x"], eof_cmp["y"], eof_cmp.values,
    cmap=cmp_cmap, norm=cmp_norm, shading="auto"
)
ax4.set_title("Difference Map between EOF Reconstructed\nMODIS and WO Observation on 11/11/2022",
              fontsize=15, pad=8)
ax4.set_xlabel("Longitude", fontsize=15)

# tick sizes
for ax in axes:
    ax.tick_params(labelsize=12)

# hide x tick labels on top and middle rows
# ax0.tick_params(labelbottom=False)
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)

# optional: hide y tick labels on right column
ax2.tick_params(labelleft=False)
ax4.tick_params(labelleft=False)

# panel labels
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
for ax, lab in zip(axes, panel_labels):
    ax.text(0.02, 0.96, lab,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left")
plt.tight_layout()
# colorbars
cax1 = fig.add_axes([0.87, 0.5, 0.02, 0.24])
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.set_ticks([0, 1])
cbar1.set_ticklabels(["Dry", "Inundated"])
cbar1.ax.tick_params(labelsize=14)

cax2 = fig.add_axes([0.87, 0.15, 0.02, 0.24])
cbar2 = fig.colorbar(im3, cax=cax2)
cbar2.set_ticks([0, 1, 2, 3])
cbar2.set_ticklabels(["Dry", "Inundated", "False Alarm", "Missed"])
cbar2.ax.tick_params(labelsize=14)

plt.show()



######################## Comparison with Sentinel-2 NDWI ##################################

import os
import glob
import re
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import mapping

# --------------------------------------------------
# paths
# --------------------------------------------------
s2_dir = "Sentinel_NDWI"
wf_dir = "MODIS_WF"
recon_dir = "MODIS_WF_reconstructed_new"
study_area_path = "clipped_boundary/clipped_boundary.shp"

# --------------------------------------------------
# collect files by date
# --------------------------------------------------
date_pattern = r"\d{4}-\d{2}-\d{2}"

# Sentinel-2
s2_files = sorted(glob.glob(os.path.join(s2_dir, "*.tiff")))
s2_dict = {}
for f in s2_files:
    fname = os.path.basename(f)
    m = re.search(date_pattern, fname)
    if not m:
        continue
    date = m.group(0)
    if "B03" in fname:
        s2_dict.setdefault(date, {})["B03"] = f
    elif "B08" in fname:
        s2_dict.setdefault(date, {})["B08"] = f
s2_dict = {d: v for d, v in s2_dict.items() if "B03" in v and "B08" in v}

# reconstructed
recon_files = sorted(glob.glob(os.path.join(recon_dir, "reconstructed_flood_*.nc")))
recon_dict = {}
for f in recon_files:
    m = re.search(date_pattern, os.path.basename(f))
    if m:
        recon_dict[m.group(0)] = f

# water fraction
wf_files = sorted(glob.glob(os.path.join(wf_dir, "water_fraction_*.nc")))
wf_dict = {}
for f in wf_files:
    m = re.search(date_pattern, os.path.basename(f))
    if m:
        wf_dict[m.group(0)] = f

# common dates
common_dates = sorted(set(s2_dict.keys()) & set(recon_dict.keys()) & set(wf_dict.keys()))
print("Common dates:", common_dates)

# --------------------------------------------------
# helpers
# --------------------------------------------------
study_area = gpd.read_file(study_area_path).to_crs("EPSG:4326")

def compute_ndwi(b03_path, b08_path):
    green = rioxarray.open_rasterio(b03_path).squeeze("band", drop=True).astype("float32")
    nir   = rioxarray.open_rasterio(b08_path).squeeze("band", drop=True).astype("float32")

    # scale if values are reflectance * 10000
    if float(np.nanmax(green.values)) > 1.5:
        green = green / 10000.0
        nir   = nir / 10000.0

    ndwi = (green - nir) / (green + nir)
    ndwi = ndwi.where(np.abs(green + nir) > 1e-6)

    if ndwi.rio.crs is None:
        ndwi = ndwi.rio.write_crs("EPSG:4326")

    return ndwi

def clip_to_study_area(da):
    return da.rio.clip(study_area.geometry.apply(mapping), study_area.crs)

def open_single_var(path):
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]]
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)

    # convert fill value to NaN if present
    fill_value = da.attrs.get("_FillValue", None)
    if fill_value is not None:
        da = da.where(da != fill_value)

    return da

# --------------------------------------------------
# colormaps
# --------------------------------------------------
base_cmap = plt.cm.Blues

# binary map uses same blue family
binary_cmap = ListedColormap([base_cmap(0.05), base_cmap(0.9)])
binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)

# --------------------------------------------------
# figure
# --------------------------------------------------
n = len(common_dates)
fig, axes = plt.subplots(
    n, 3,
    figsize=(12, 3.6 * n),
    squeeze=False,
    sharex=True,
    sharey=True
)

panel_labels = [chr(97 + i) for i in range(n * 3)]  # a, b, c, ...

for i, date in enumerate(common_dates):
    # Sentinel-2 NDWI
    ndwi = compute_ndwi(s2_dict[date]["B03"], s2_dict[date]["B08"])
    ndwi_clip = clip_to_study_area(ndwi)

    # reconstructed + water fraction
    recon = open_single_var(recon_dict[date])
    wf = open_single_var(wf_dict[date])

    # ensure CRS
    if recon.rio.crs is None:
        recon = recon.rio.write_crs("EPSG:4326")
    if wf.rio.crs is None:
        wf = wf.rio.write_crs("EPSG:4326")
    if ndwi_clip.rio.crs is None:
        ndwi_clip = ndwi_clip.rio.write_crs("EPSG:4326")

    # align WF and NDWI to reconstructed grid
    wf_match = wf.rio.reproject_match(recon)
    ndwi_match = ndwi_clip.rio.reproject_match(recon)

    # --------------------------------------------------
    # IMPORTANT: mask EOF-refined inundation wherever
    # water fraction is blank, so blank stays blank
    # --------------------------------------------------
    recon_masked = recon.where(np.isfinite(wf_match))

    # ---------------- column 1: water fraction ----------------
    im0 = axes[i, 0].pcolormesh(
        wf_match["x"], wf_match["y"], wf_match.values,
        cmap=base_cmap,
        vmin=0, vmax=1,
        shading="auto"
    )

    # ---------------- column 2: reconstructed inundation ----------------
    im1 = axes[i, 1].pcolormesh(
        recon_masked["x"], recon_masked["y"], recon_masked.values,
        cmap=binary_cmap,
        norm=binary_norm,
        shading="auto"
    )

    # ---------------- column 3: Sentinel-2 NDWI ----------------
    im2 = axes[i, 2].pcolormesh(
        ndwi_match["x"], ndwi_match["y"], ndwi_match.values,
        cmap=base_cmap,
        vmin=-0.5, vmax=0.5,
        shading="auto"
    )

    # panel labels
    for j, ax in enumerate(axes[i]):
        label = panel_labels[i * 3 + j]
        ax.text(
            0.02, 0.96, f"({label})",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left"
        )
        ax.tick_params(labelsize=10)

    # date label on the left of each row
    axes[i, 0].text(
        -0.33, 0.5, date,
        transform=axes[i, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=14
    )

# --------------------------------------------------
# column titles (top row only)
# --------------------------------------------------
axes[0, 0].set_title("MODIS Water Fraction (500m)", fontsize=15, pad=8)
axes[0, 1].set_title("EOF-refined inundation (30m)", fontsize=15, pad=8)
axes[0, 2].set_title("Sentinel-2 NDWI (10m)", fontsize=15, pad=8)

# --------------------------------------------------
# axis labels
# --------------------------------------------------
for i in range(n):
    axes[i, 0].set_ylabel("Latitude", fontsize=13)
    axes[i, 1].set_ylabel("")
    axes[i, 2].set_ylabel("")

for j in range(3):
    axes[-1, j].set_xlabel("Longitude", fontsize=13)

# --------------------------------------------------
# layout
# --------------------------------------------------
fig.subplots_adjust(
    left=0.14,
    right=0.96,
    top=0.92,
    bottom=0.16,
    wspace=0.06,
    hspace=0.10
)

# --------------------------------------------------
# horizontal colorbars
# --------------------------------------------------
# water fraction
cax0 = fig.add_axes([0.17, 0.09, 0.20, 0.018])
cbar0 = fig.colorbar(im0, cax=cax0, orientation="horizontal")
cbar0.set_label("Water fraction", fontsize=14)
cbar0.ax.tick_params(labelsize=12)

# reconstructed inundation
cax1 = fig.add_axes([0.45, 0.09, 0.20, 0.018])
cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
cbar1.set_ticks([0, 1])
cbar1.set_ticklabels(["Dry", "Inundated"])
cbar1.ax.tick_params(labelsize=14)

# Sentinel-2 NDWI
cax2 = fig.add_axes([0.73, 0.09, 0.20, 0.018])
cbar2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
cbar2.set_label("NDWI", fontsize=14)
cbar2.ax.tick_params(labelsize=12)

plt.show()
































########################### Difference Maps Comparison ###########################

import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# 1. LOAD FILES INTO DICTIONARIES
# ============================================================

wf_dir = "MODIS_WF"
wf_files = sorted(glob.glob(os.path.join(wf_dir, "*.nc")))
wf = {}
for f in wf_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    wf[d] = ds["water_fraction"]

recon_dir = "MODIS_WF_reconstructed_new"
recon_files = sorted(glob.glob(os.path.join(recon_dir, "*.nc")))
rec = {}
for f in recon_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    rec[d] = ds["reconstructed_flood"]

dem_dir = "MODIS_WF_downscaled_new"
dem_files = sorted(glob.glob(os.path.join(dem_dir, "*.nc")))
dem = {}
for f in dem_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    dem[d] = ds["water_mask_30m"]

sparse_dir = "Clipped_WOs"
sparse_files = sorted(glob.glob(os.path.join(sparse_dir, "*.nc")))
sparse = {}
for f in sparse_files:
    ds = xr.open_dataset(f)
    d = os.path.basename(f).split("_")[-1].split(".nc")[0]
    sparse[d] = ds['__xarray_dataarray_variable__']

# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def make_confusion_map(pred, obs, dem_base):
    valid = (~np.isnan(pred)) & (~np.isnan(obs)) & (~np.isnan(dem_base))
    out = xr.full_like(obs, np.nan, dtype=float)

    out = xr.where(valid & (pred == 0) & (obs == 0), 0, out)  # TN / Dry
    out = xr.where(valid & (pred == 1) & (obs == 1), 1, out)  # TP / Inundated
    out = xr.where(valid & (pred == 1) & (obs == 0), 2, out)  # FP / False Alarm
    out = xr.where(valid & (pred == 0) & (obs == 1), 3, out)  # FN / Missed

    return out

# ============================================================
# 3. PLOTTING FUNCTION FOR ONE DATE
# ============================================================

def plot_six_panel_figure(date, date_label=None, save_path=None):
    if date_label is None:
        date_label = date

    # --- get data ---
    wf_ds = wf[date]
    dem_ds = dem[date]
    eof_ds = rec[date]
    wo_ds = sparse[date]

    # --- make EOF blank where WF is blank ---
    # eof_ds = eof_ds.where(np.isfinite(wf_ds))

    # --- confusion maps ---
    dem_cmp = make_confusion_map(dem_ds, wo_ds, dem_ds)
    eof_cmp = make_confusion_map(eof_ds, wo_ds, dem_ds)

    fig, axes = plt.subplots(3, 2, figsize=(11, 14), sharex=True, sharey=True)

    axes = axes.ravel()

    # --------------------------------------------------------
    # colormaps
    # --------------------------------------------------------
    base_cmap = plt.cm.Blues

    flood_cmap = ListedColormap([base_cmap(0.05), base_cmap(0.9)])
    flood_norm = BoundaryNorm([-0.5, 0.5, 1.5], flood_cmap.N)

    cmp_cmap = ListedColormap([base_cmap(0.05), base_cmap(0.9), "#fdae61", "#f03b20"])
    cmp_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmp_cmap.N)

    # --------------------------------------------------------
    # (a) WO
    # --------------------------------------------------------
    im0 = axes[0].pcolormesh(
        wo_ds["x"], wo_ds["y"], wo_ds.values,
        cmap=flood_cmap, norm=flood_norm, shading="auto"
    )
    axes[0].set_title(f"WO Observation {date_label}", fontsize=14)

    # --------------------------------------------------------
    # (b) WF
    # --------------------------------------------------------
    im1 = axes[1].pcolormesh(
        wf_ds["x"], wf_ds["y"], wf_ds.values,
        cmap="Blues", vmin=0, vmax=1, shading="auto"
    )
    axes[1].set_title(f"MODIS Water Fraction {date_label}", fontsize=14)

    # --------------------------------------------------------
    # (c) DEM
    # --------------------------------------------------------
    im2 = axes[2].pcolormesh(
        dem_ds["x"], dem_ds["y"], dem_ds.values,
        cmap=flood_cmap, norm=flood_norm, shading="auto"
    )
    axes[2].set_title(f"DEM-downscaled MODIS {date_label}", fontsize=14)

    # --------------------------------------------------------
    # (d) EOF
    # --------------------------------------------------------
    im3 = axes[3].pcolormesh(
        eof_ds["x"], eof_ds["y"], eof_ds.values,
        cmap=flood_cmap, norm=flood_norm, shading="auto"
    )
    axes[3].set_title(f"EOF-refined MODIS {date_label}", fontsize=14)

    # --------------------------------------------------------
    # (e) DEM vs WO
    # --------------------------------------------------------
    im4 = axes[4].pcolormesh(
        dem_cmp["x"], dem_cmp["y"], dem_cmp.values,
        cmap=cmp_cmap, norm=cmp_norm, shading="auto"
    )
    axes[4].set_title("Difference map between DEM-downscaled \nMODIS and WO", fontsize=14)

    # --------------------------------------------------------
    # (f) EOF vs WO
    # --------------------------------------------------------
    im5 = axes[5].pcolormesh(
        eof_cmp["x"], eof_cmp["y"], eof_cmp.values,
        cmap=cmp_cmap, norm=cmp_norm, shading="auto"
    )
    axes[5].set_title("Difference map between EOF-refined \nMODIS and WO", fontsize=14)

    # --------------------------------------------------------
    # formatting
    # --------------------------------------------------------
    for ax in axes:
        ax.tick_params(labelsize=11)

    # labels only outer
    for i in [0, 2, 4]:
        axes[i].set_ylabel("Latitude", fontsize=12)

    for i in [4, 5]:
        axes[i].set_xlabel("Longitude", fontsize=12)

    # panel labels
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, lab in zip(axes, labels):
        ax.text(0.02, 0.96, lab,
                transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                va="top", ha="left")

    # date title
    # fig.suptitle(f"{date_label}", fontsize=16, y=0.98)

    # --------------------------------------------------------
    # colorbars (vertical)
    # --------------------------------------------------------

    fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.08)

    # --------------------------------------------------------
    # (1) WATER FRACTION COLORBAR — next to (b)
    # --------------------------------------------------------
    pos_b = axes[1].get_position()

    cax_wf = fig.add_axes([
        pos_b.x1 + 0.03,   # right of panel (b)
        pos_b.y0,
        0.015,
        pos_b.height
    ])

    cbar_wf = fig.colorbar(im1, cax=cax_wf)
    cbar_wf.set_label("Water fraction", fontsize=13)
    cbar_wf.ax.tick_params(labelsize=11)

    # --------------------------------------------------------
    # (2) CONFUSION COLORBAR — next to bottom panels (e,f)
    # --------------------------------------------------------
    pos_d = axes[3].get_position()  # (d)
    pos_f = axes[5].get_position()  # (f)

    # vertical span from bottom of (f) to top of (d)
    bottom = pos_f.y0
    top = pos_d.y1
    height = top - bottom

    # horizontal position: just right of right column
    right_edge = max(pos_d.x1, pos_f.x1)

    cax_cmp = fig.add_axes([
        right_edge + 0.03,   # slight gap to the right
        bottom,
        0.018,
        height
    ])

    cbar_cmp = fig.colorbar(im4, cax=cax_cmp)
    cbar_cmp.set_ticks([0, 1, 2, 3])
    cbar_cmp.set_ticklabels([
        "Dry",
        "Inundated",
        "False alarm",
        "Missed"
    ])
    cbar_cmp.ax.tick_params(labelsize=11)
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
    for ax in [axes[4], axes[5]]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.015, pos.width, pos.height])
    xmin = float(wo_ds["x"].min())
    xmax = float(wo_ds["x"].max())
    ymin = float(wo_ds["y"].min())
    ymax = float(wo_ds["y"].max())

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    plt.show()
# ============================================================
# 4. MAKE THE TWO FIGURES
# ============================================================

plot_six_panel_figure("2022-12-13", date_label="13/12/2022")
# plot_six_panel_figure("2022-11-28", date_label="28/11/2022")




########################### Seasonal WOs Frequency Check ###########################
def wo_clip(dataset, wo_reference):

    # Ensure both have CRS defined
    if not dataset.rio.crs:
        dataset = dataset.rio.write_crs("EPSG:4326")
    if not wo_reference.rio.crs:
        wo_reference = wo_reference.rio.write_crs("EPSG:4326")
    clipped = dataset.rio.reproject_match(wo_reference)

    return clipped

file_name = 'Seasonal_WOs/ga_ls_wo_fq_apr_oct_3_x59y37_1998-04--P7M_final_frequency.tif'
multiyr_wo = xr.open_rasterio(file_name)
multiyr_wo = multiyr_wo.rio.reproject("EPSG:4326")

clip_template = xr.open_dataset('Clipped_WOs/clipped_original_wos_2022-11-11.nc')

clipped_multiyr_wo = wo_clip(multiyr_wo[0], clip_template)

# wo_frequency = clipped_multiyr_wo.fillna(0)

wet_map = clipped_multiyr_wo>0.05
wet_map.plot()



######################## Seasonal WOs Frequency Example Plot ########################
import os
import glob
import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --------------------------------------------------
# helper
# --------------------------------------------------
def wo_clip(dataset, wo_reference):
    if not dataset.rio.crs:
        dataset = dataset.rio.write_crs("EPSG:4326")
    if not wo_reference.rio.crs:
        wo_reference = wo_reference.rio.write_crs("EPSG:4326")
    clipped = dataset.rio.reproject_match(wo_reference)
    return clipped

# --------------------------------------------------
# paths
# --------------------------------------------------
seasonal_dir = "Seasonal_WOs"
seasonal_files = sorted(glob.glob(os.path.join(seasonal_dir, "*.tif")))

# keep files 2–5 (Python index 1:5)
plot_files = seasonal_files[1:5]

clip_template = xr.open_dataset("Clipped_WOs/clipped_original_wos_2022-11-11.nc")
clip_template_da = list(clip_template.data_vars.values())[0]

# --------------------------------------------------
# colormap: binary, same blue family
# --------------------------------------------------
base_cmap = plt.cm.Blues
binary_cmap = ListedColormap([base_cmap(0.05), base_cmap(0.9)])
binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)

plot_files = [plot_files[i] for i in [2, 1, 0, 3]]
# --------------------------------------------------
# figure
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.ravel()

titles = [
    "Nov-Mar 2011 (Frequency > 0.2)",
    "Nov-Mar 2000 (Frequency > 0.1)",
    "Apr–Oct 1998 (Frequency > 0.2)",
    "Nov-Mar 2022 (Frequency > 0.1)"
]
frequency = [0.2, 0.1, 0.2, 0.1]
for i, file_name in enumerate(plot_files):
    # open + reproject
    multiyr_wo = rioxarray.open_rasterio(file_name).squeeze("band", drop=True)
    multiyr_wo = multiyr_wo.rio.reproject("EPSG:4326")

    # clip/match to template
    clipped_multiyr_wo = wo_clip(multiyr_wo, clip_template_da)

    # binary wet/dry map
    wet_map = xr.where(clipped_multiyr_wo > frequency[i], 1, 0)
    wet_map = wet_map.where(~np.isnan(clipped_multiyr_wo))

    im = axes[i].pcolormesh(
        wet_map["x"], wet_map["y"], wet_map.values,
        cmap=binary_cmap, norm=binary_norm, shading="auto"
    )

    axes[i].set_title(titles[i], fontsize=12)
    axes[i].tick_params(labelsize=10)

# axis labels only on outer panels
axes[0].set_ylabel("Latitude", fontsize=11)
axes[2].set_ylabel("Latitude", fontsize=11)
axes[2].set_xlabel("Longitude", fontsize=11)
axes[3].set_xlabel("Longitude", fontsize=11)

# panel labels
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes, panel_labels):
    ax.text(
        0.02, 0.96, lab,
        transform=ax.transAxes,
        fontsize=13, fontweight="bold",
        va="top", ha="left"
    )


# leave space
fig.subplots_adjust(right=0.85)

# add colorbar axis
cax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  # [left, bottom, width, height]

cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Dry", "Inundation"])
cbar.ax.tick_params(labelsize=10)

fig.subplots_adjust(wspace=0.06, hspace=0.12)
# plt.suptitle("Seasonal surface water occurrence patterns", fontsize=14, y=0.98)
plt.show()