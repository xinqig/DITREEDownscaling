import os
import numpy as np
import xarray as xr
import rioxarray
import warnings
from pyproj import CRS
from rasterio.enums import Resampling
import glob
from eofs.xarray import Eof
import re

# WOs frequency maps
def wo_clip(dataset, wo_reference):

    # Ensure both have CRS defined
    if not dataset.rio.crs:
        dataset = dataset.rio.write_crs("EPSG:4326")
    if not wo_reference.rio.crs:
        wo_reference = wo_reference.rio.write_crs("EPSG:4326")
    clipped = dataset.rio.reproject_match(wo_reference)

    return clipped

water_int = xr.open_dataset('dem_filled_int.nc')['__xarray_dataarray_variable__'].astype("int16")
water_int = water_int.rio.write_crs("EPSG:4326", inplace=True)

seasonal_file_list = sorted(glob.glob("Seasonal_WO_ex2022/*.tif"))
seasonal_wo_list = [xr.open_rasterio(f) for f in seasonal_file_list]
clip_template = xr.open_dataset('Original_WOs_Bourke/original_wos_product_2022-12-29.nc')
thresholds = np.linspace(0.01, 0.4, 200)
maps = []
year = []
for f in seasonal_file_list:
    file_year = int(re.search(r"_(\d{4})-", f).group(1))
    year.append(file_year)
for seasonal_wo, yr in zip(seasonal_wo_list, year):
    seasonal_wo = seasonal_wo.rio.reproject("EPSG:4326")
    clipped_multiyr_wo = wo_clip(seasonal_wo[0], clip_template)
    wo_frequency = clipped_multiyr_wo.fillna(0)

    for t in thresholds:
        mask = (wo_frequency >= t).astype(np.int16)
        mask = mask.expand_dims(dim={"time": [float(t+yr)]})  # label each with threshold value
        maps.append(mask)


wo_frequency_stack = xr.concat(maps, dim="time")

# wo_wf_stacked = np.reshape(original_wo['__xarray_dataarray_variable__'].values, (-1))
wo_frequency_stack = wo_frequency_stack.rio.write_crs("EPSG:4326", inplace=True)
da_stack_matched = wo_frequency_stack.rio.reproject_match(water_int)

modes = 400

# --- Compute mean and EOFs ---
fine_data_mean = da_stack_matched.mean(dim='time').values
fine_data_mean_stacked = fine_data_mean.reshape(-1)
np.save('seasonal_wos_mean_400.npy', fine_data_mean)

solver = Eof(da_stack_matched, center=True)
eofs = solver.eofs(neofs=modes)

variance_fraction = solver.varianceFraction().values
cumulative_variance = np.cumsum(variance_fraction)

n_modes = 400
explained_500 = cumulative_variance[n_modes - 1]

print("====================================")
print(f"Variance explained by first {n_modes} modes: {explained_500:.4f}")

# Modes needed for 90%, 95%
n90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
n95 = np.where(cumulative_variance >= 0.95)[0][0] + 1

print(f"Modes needed for 90% variance: {n90}")
print(f"Modes needed for 95% variance: {n95}")
print("====================================")

# ===============================
# Save outputs (VERY IMPORTANT)
# ===============================
np.save("variance_fraction.npy", variance_fraction)
np.save("cumulative_variance.npy", cumulative_variance)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(cumulative_variance)
plt.axvline(500, linestyle='--')
plt.xlabel("Number of EOF modes")
plt.ylabel("Cumulative explained variance")
plt.title("EOF variance explained")
plt.savefig("eof_variance_explained_400.png", dpi=300)

eofs_stacked = eofs.stack(space=('y', 'x'))

eofs.to_netcdf(f'seasonal_wos_eofs_{modes}_400.nc')