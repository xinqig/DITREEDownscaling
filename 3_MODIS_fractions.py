import os
import xarray as xr
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from datetime import datetime
import geopandas as gpd
import warnings
import pickle
import rasterio
from rasterio import features
import matplotlib.colors as mcolors
import math
from rasterio.enums import Resampling

warnings.filterwarnings("ignore")

# date_pattern = r'\d{4}-\d{2}-\d{2}'  # Pattern to extract date from filenames
water_body_mask = 'large_water_bodies/large_water_bodies.shp'
crs_wgs84 = 'EPSG:4326'

def assign_crs(dataset):
    if not dataset.rio.crs:
        dataset.rio.write_crs(crs_wgs84, inplace=True)
    return dataset
def mask_water_bodies(dataset, shapefile):
    dataset = assign_crs(dataset)
    if shapefile.crs != dataset.rio.crs:
        shapefile = shapefile.to_crs(dataset.rio.crs)
    return dataset.rio.clip(shapefile.geometry, shapefile.crs, drop=False, invert=True, all_touched=True)
water_bodies = gpd.read_file(water_body_mask)

variables_to_use = ["SWIR1", "SWIR2", "Blue", "Red", "mNDWI1", "mNDWI2", "NDVI"]

with open(f"svgp_100_pts_all.pkl", "rb") as f:
    model = pickle.load(f)

prev_day_wo = xr.open_dataset('WOs_Bourke_clipped_bounds/clipped_bound_wos_wgs84_2022-10-10.nc')['__xarray_dataarray_variable__']
# prev_day_wo = xr.open_dataset('Original_WOs_Bourke/original_wos_product_2022-11-11.nc')['__xarray_dataarray_variable__']
prev_day_wo_flattened = prev_day_wo.values.flatten()
###############################################################
prev_day_wo_masked = mask_water_bodies(prev_day_wo, water_bodies)

# Use lat/lon coordinates from the dataset directly
lat = prev_day_wo.coords['y']
lon = prev_day_wo.coords['x']

water_mask = prev_day_wo_masked.isnull()
water_mask_flat = water_mask.values.flatten()

modis_val_files = sorted([f for f in os.listdir('MODIS_corrected') if f.endswith(".npz")])
n_files = len(modis_val_files)

# Choose subplot grid size
n_cols = 5
n_rows = math.ceil(n_files / n_cols)

cmap_flood = mcolors.LinearSegmentedColormap.from_list(
        "flood_cmap", 
        [(0.0, "#EAF1FA"),    # Light grey (dry)
        (0.3, "#62A2DD"),    # Light blue (emergent water)
        (1.0, "#053AAC")]    # Dark blue (deep/full water)
    )

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), 
                         subplot_kw={'aspect': 'equal'})
axes = axes.ravel()  # flatten axes array

for idx, f in enumerate(modis_val_files):
    date = f.split("_")[-1].split(".npz")[0]
    # date = '2022-11-14'
    loaded = np.load(f"MODIS_corrected/calibrated_modis_{date}.npz")
    # loaded = np.load(f"calibrated_modis_0.npz")
    predictions = {key: loaded[key] for key in loaded.files}

    validation_feature_list = []
    for band in variables_to_use:
        # Flatten each band (assuming each prediction is in a 2D array)
        band_data = predictions[band].astype(np.float64).flatten()
        validation_feature_list.append(band_data)

    # Stack them column-wise: each column is one band, shape becomes (n_pixels, n_bands)
    X_validation_index_multi = np.column_stack(validation_feature_list)
    # 1. Mask out invalid input rows (NaNs anywhere)
    valid_input_mask = ~np.any(np.isnan(X_validation_index_multi), axis=1)
    X_valid = X_validation_index_multi[valid_input_mask]

    # 2. Predict only valid inputs
    y_mean, y_var = model.predict_y(X_valid)
    # Clip predictions: values > 1 set to 1, values < 0 set to 0
    y_pred = tf.clip_by_value(y_mean, 0.0, 1.0)

    # 3. Insert predictions back into full grid
    y_pred_np = np.full(X_validation_index_multi.shape[0], np.nan)
    y_pred_np[valid_input_mask] = y_pred.numpy().squeeze()
    
    # # rebuild 2D prediction on MODIS grid
    # y_pred_2d = np.full(predictions[variables_to_use[0]].shape, np.nan)
    # y_pred_2d.flat[valid_input_mask] = y_pred.numpy().squeeze()

    # # reproject to prev_day_wo grid
    # y_pred_da = (
    #     xr.DataArray(y_pred_2d, dims=("y", "x"), coords={"y": modis_template["y"], "x": modis_template["x"]})
    #     .rio.write_crs(prev_day_wo.rio.crs)
    #     .rio.reproject_match(prev_day_wo, resampling=Resampling.nearest)
    # )
    # # y_pred_da = y_pred_da.rio.reproject_match(prev_day_wo)
    # y_pred_np = y_pred_da.values.flatten()

    # Create a final output array for validation predictions filled with NaNs
    final_pred = np.full_like(y_pred_np, np.nan)
    # final_pred = np.full(n_target, np.nan)
    final_pred[~water_mask_flat] = y_pred_np[~water_mask_flat]
    final_pred[water_mask_flat] = prev_day_wo_flattened[water_mask_flat]

    # Reshape the predicted array to match the spatial shape (y, x)
    y_pred_reshaped = final_pred.reshape(len(lat), len(lon))

    # Wrap in a DataArray with geocoordinates
    predicted_water_da = xr.DataArray(
        y_pred_reshaped,
        dims=("y", "x"),
        coords={"y": lat, "x": lon},
        # name="Predicted Water Fraction"
        name="water_fraction"
    )

    # Plot in subplot
    ax = axes[idx]
    im = predicted_water_da.plot(
        ax=ax,
        cmap=cmap_flood,
        add_colorbar=False,   # we add one global colorbar later
        vmin=0, vmax=1
    )
    ax.set_title(f"{date}", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # # Plot with geospatial axes
    # plt.figure(figsize=(8, 6))
    # plot = predicted_water_da.plot(
    #     cmap=cmap_flood,
    #     cbar_kwargs={
    #         'label': "Predicted Water Fraction",
    #         'ticks': np.linspace(0, 1, 6)})
    # plt.title(f"Predicted Water Fraction Using Reference Model {date}", fontsize=14)
    # plt.xlabel("Longitude", fontsize=14)
    # plt.ylabel("Latitude", fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # if hasattr(plot, 'colorbar') and plot.colorbar is not None:
    #     plot.colorbar.set_label("Predicted Water Fraction", fontsize=14)
    #     plot.colorbar.ax.tick_params(labelsize=12)
    # plt.show()

    predicted_water_da.to_netcdf(f'water_fraction_{date}.nc')



# turn off unused panels
for j in range(idx + 1, len(axes)):
    axes[j].axis("off")
# optional: one shared colorbar
# fig.colorbar(im, ax=axes[:idx+1], fraction=0.02, pad=0.02)

plt.tight_layout()
plt.show()


# Plot with latitude and longitude axes
wo_original = y_validation_array['__xarray_dataarray_variable__']
plt.figure(figsize=(8, 6))
plot2 = wo_original.plot(
    cmap=cmap_flood,
    cbar_kwargs={
        'label': "Actual Water Fraction",
        'ticks': np.linspace(0, 1, 6)})
plt.title(f"Actual Water Fraction {data_dates[validation_index]}", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
if hasattr(plot2, 'colorbar') and plot2.colorbar is not None:
    plot2.colorbar.set_label("Actual Water Fraction", fontsize=14)
    plot2.colorbar.ax.tick_params(labelsize=12)
plt.show()


















# %%
# Figure 6 - MODIS spectral bands and indices on 13/12/2022
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

calibrated_modis = xr.open_dataset('MODIS_wofs/modis_wofs_2022-12-13.nc')

    
variables = list(calibrated_modis.data_vars.keys())

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 11), constrained_layout=True)

remove_yticks = [1, 2, 4, 5, 7, 8]
remove_xticks = [0, 1, 2, 3, 4, 5]

for idx, (ax, var) in enumerate(zip(axes.flat, variables)):
    da = calibrated_modis[var].squeeze()

    # Choose colormap
    if var in ['NDVI', 'mNDWI1', 'mNDWI2']:
        cmap = 'RdBu_r'

        # Make color scale symmetric around 0
        data_min = float(da.min().values)
        data_max = float(da.max().values)
        abs_max = max(abs(data_min), abs(data_max))
        vmin, vmax = -abs_max, abs_max

    else:
        cmap = 'viridis'
        vmin = float(da.min().values)
        vmax = float(da.max().values)

    # Plot (continuous color scale)
    im = da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    # Axis label logic
    if idx in remove_yticks:
        ax.set_ylabel("")
        ax.set_yticks([])
    else:
        ax.set_ylabel("Latitude", fontsize=13)

    if idx in remove_xticks:
        ax.set_xlabel("")
        ax.set_xticks([])
    else:
        ax.set_xlabel("Longitude", fontsize=13)

    # Consistent axis formatting
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(var, fontsize=15)
    ax.tick_params(labelsize=11)

    # Continuous colorbar with 0-centered ticks
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    if var in ['NDVI', 'mNDWI1', 'mNDWI2']:
        tick_vals = np.linspace(vmin, vmax, 7)
        # Ensure 0 is exactly in the middle
        mid_idx = len(tick_vals) // 2
        tick_vals[mid_idx] = 0.0
        cbar.set_label("Index Value", fontsize=13)
    else:
        tick_vals = np.linspace(vmin, vmax, 6)
        cbar.set_label("Surface Reflectance", fontsize=13)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{t:.2f}" for t in tick_vals])
    cbar.ax.tick_params(labelsize=11)


plt.suptitle("MODIS Spectral Bands and Indices on 13/12/2022", fontsize=20)
plt.savefig("Paper2_Figure_6_updated.jpg", dpi=300, format='jpg', bbox_inches='tight')
plt.show()




# %%
# Figure 7 - Scatter plots of predicted vs actual water fraction against each Landsat band/index
import matplotlib.pyplot as plt
import numpy as np


# Variable names
variables = ["SWIR1", "SWIR2", "Blue", "Red", "mNDWI1", "mNDWI2", "NDVI"]

# Create subplot grid: 2 rows × 4 columns
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), dpi=150)
axes = axes.flatten()

# Plot each variable
for i, var in enumerate(variables):
    ax = axes[i]
    x = X_validation_index_valid[:, i]
    ax.scatter(x, y_validation_filtered, label='Actual', color='grey', s=50)
    ax.scatter(x, y_pred, label='Predicted', color='blue', alpha=0.6, s=50)

    # Axis labels
    ax.set_xlabel(f'{var}', fontsize=18)
    if i % 4 == 0:
        ax.set_ylabel('Surface Water Fraction', fontsize=22)
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])

    # Panel label (a), (b), etc.
    if i < 4:  # First row → upper right
        ax.text(0.98, 0.98, f'({chr(97 + i)})', transform=ax.transAxes,
                fontsize=25, fontweight='normal', va='top', ha='right')
    else:  # Second row → upper left
        ax.text(0.02, 0.98, f'({chr(97 + i)})', transform=ax.transAxes,
                fontsize=25, fontweight='normal', va='top', ha='left')


    # Axis formatting
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

from matplotlib.lines import Line2D

# Final subplot: manual legend using Line2D handles
axes[-1].axis('off')
legend_elements = [
    Line2D([0], [0], marker='o', color='grey', label='Actual', linestyle='None', markersize=10),
    Line2D([0], [0], marker='o', color='blue', label='Predicted', linestyle='None', markersize=10, alpha=0.6)
]
axes[-1].legend(handles=legend_elements, loc='center left', fontsize=20, frameon=False)

# Adjust spacing
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.suptitle('Estimated Water Fractions vs Aggregated Landsat 8 Bands and Indices', fontsize=25, y=1.02)
plt.savefig("Paper2_Figure_7.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Figure 8 - Comparison of actual vs estimated water fraction for two dates
import pickle
import matplotlib.pyplot as plt

# Load both saved plot data
with open("water_fraction_comparison_2022-12-13.pkl", "rb") as f:
    data1 = pickle.load(f)

with open("water_fraction_comparison_2022-12-29.pkl", "rb") as f:
    data2 = pickle.load(f)

# Create 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

for ax, data in zip(axes, [data1, data2]):
    y_true = data["y_val_filtered"]
    y_pred = data["y_pred_filtered"]
    date = data["date"]

    ax.scatter(y_true, y_pred, alpha=0.5, s=100, color='blue', label='Estimation')

    # Perfect prediction line
    ax.plot([0, 1], [0, 1], 'r--', label='1:1 Line')

    ax.set_xlabel('Actual Water Fraction', fontsize=20)
    ax.set_ylabel('Estimated Water Fraction', fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=18)
    ax.set_title(f'{date}', fontsize=20)
    ax.legend(fontsize=18)
# Remove y-axis label and ticks for the second subplot
axes[1].set_ylabel("")
axes[1].set_yticklabels([])
axes[1].set_yticks([])

# Add a shared suptitle
fig.suptitle("Comparison of Actual and Estimated Water Fraction", fontsize=25)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
# plt.savefig("comparison_actual_vs_estimated_two_dates.jpg", dpi=300, bbox_inches='tight')
plt.show()








#####################################################
# Figure 9
# Load datasets
datasets = [ref_wf_1312, dir_wf_1312, actual_wf_1312, ref_wf_2912, dir_wf_2912, actual_wf_2912]
title = ['Reference Model - 13/12/2022', 'Direct Model - 13/12/2022', 'Actual - 13/12/2022',
         'Reference Model - 29/12/2022', 'Direct Model - 29/12/2022', 'Actual - 29/12/2022']
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
colors = ["#EAF1FA", "#053AAC"]
boundaries = [-0.1, 0.05, 0.2, 0.4, 0.6, 0.8, 1.1]  # small epsilon between 0 and >0
norm = BoundaryNorm(boundaries, ncolors=len(boundaries)-1)

# Define custom colors for each bin (6 bins = 6 colors)
colors = [
    "#f7f7f7",  # bin 0 = exact 0 (e.g., land or no water)
    "#e0ecfc",  # >0 to 0.2
    "#b3cde3",  # 0.2 to 0.4
    "#70a1cf",  # 0.4 to 0.6
    "#2e78c2",  # 0.6 to 0.8
    "#053aac"   # 0.8 to 1.0
]
cmap_discrete = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=len(colors), extend='neither')

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), constrained_layout=True)

for idx, (ax, da, date) in enumerate(zip(axes.flat, datasets, data_dates)):
    img = da.plot(
        ax=ax,
        cmap=cmap_discrete,
        norm=norm,
        add_colorbar=False,
    )

    # Remove x-axis ticks & label for subplots 0, 1, 2 (1-based: figs 1,2,3)
    if idx in [0, 1, 2]:
        ax.set_xlabel("")
        ax.set_xticks([])

    else:
        ax.set_xlabel("Longitude", fontsize=18)

    # Remove y-axis ticks & label for subplots 1, 2, 4, 5 (1-based: figs 2,3,5,6)
    if idx in [1, 2, 4, 5]:
        ax.set_ylabel("")
        ax.set_yticks([])

    else:
        ax.set_ylabel("Latitude", fontsize=18)
    ax.set_title(title[idx], fontsize=18)
    ax.tick_params(labelsize=18)
tick_positions = boundaries[1:]  # from 0.00001 onwards
# Add a single colorbar for all plots
cbar = fig.colorbar(
    img, ax=axes,
    orientation='vertical',
    fraction=0.02,
    pad=0.02,
    # ticks=[0, 1]
    ticks=tick_positions
)
# cbar.ax.set_yticklabels(['Land', 'Water'])
cbar.ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
cbar.set_label("Surface Water Fraction", fontsize=22)
cbar.ax.tick_params(labelsize=20)
plt.savefig("Paper2_Figure_9.jpg", dpi=300, bbox_inches='tight', format='jpg')
# plt.suptitle("Water Fractions Calculated from WOs Product", fontsize=25)
plt.show()



















plt.figure(figsize=(10, 6))
plt.imshow(y_pred_reshaped[0], cmap=cmap_flood, origin="upper")
plt.colorbar(label="Predicted Water Fraction")
plt.title(f"Predicted Water Fraction val_idx_{validation_index}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Reshape wo_values for visualization (if necessary)
wo_original = y_validation_array['__xarray_dataarray_variable__']
plt.figure(figsize=(10, 6))
plt.imshow(wo_original, cmap=cmap_flood, origin="upper")
plt.colorbar(label="Actual Water Fraction")
plt.title(f"Actual Water Fraction val_idx_{validation_index}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Calculate differences
difference = y_validation_array['__xarray_dataarray_variable__'] - y_pred_reshaped[0]

# Visualize differences
plt.figure(figsize=(10, 6))
plt.imshow(difference, cmap="coolwarm", origin="upper")
plt.colorbar(label="Difference (Actual - Predicted)")
plt.title(f"Difference Between Actual and Predicted val_idx_{validation_index}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



# Generate scatter plot for expected vs predicted water fraction
plt.figure(figsize=(8, 8))
plt.scatter(y_val_filtered, y_pred_filtered, alpha=0.5, s=120, color='blue', label='Estimaged Water Fraction')

# Plot a line of perfect prediction for reference
min_val = min(y_val_filtered.min(), min(y_pred_filtered))
max_val = max(y_val_filtered.max(), min(y_pred_filtered))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')

# Set labels and title
plt.xlabel('Actual Water Fraction', fontsize=22)
plt.ylabel('Estimated Water Fraction', fontsize=22)
plt.title(f'Comparison of Actual and Estimated Water Fraction on 29/12/2022', fontsize=25)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Display R² score on the plot
legend = plt.legend(loc='upper left', fontsize=20)

# Display R² score right below the legend box
# r2 = r2_score(y_val_final, y_pred_final)
# plt.text(legend.get_frame().get_x() + 0.1, legend.get_frame().get_y() + 0.8, f'R² = {r2:.2f}', 
#          fontsize=12, color='green', ha='left', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()








# Generate scatter plot for expected vs predicted water fraction
plt.figure(figsize=(10, 10))
plt.scatter(y_val_final, y_pred_final, alpha=0.5, color='blue', label='Predicted vs. Actual')

# Plot a line of perfect prediction for reference
min_val = min(y_val_final.min(), y_pred_final.min())
max_val = max(y_val_final.max(), y_pred_final.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')

# Set labels and title
plt.xlabel('Actual Water Fraction')
plt.ylabel('Predicted Water Fraction')
plt.title('Comparison of Actual vs. Predicted Water Fraction')

# Display R² score on the plot
legend = plt.legend(loc='lower right')

# Display R² score right below the legend box
r2 = r2_score(y_val_final, y_pred_final)
plt.text(legend.get_frame().get_x() + 0.1, legend.get_frame().get_y() + 0.8, f'R² = {r2:.2f}', 
         fontsize=12, color='green', ha='left', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

###################################################################################################
modis1213 = xr.open_dataset('MODIS_wofs/modis_wofs_2022-12-13.nc')# Define the bands you want to plot
bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
titles = ['Blue Band', 'Green Band', 'Red Band', 'NIR Band', 'SWIR1 Band', 'SWIR2 Band']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

for i, ax in enumerate(axes.flat):
    img = modis1213[bands[i]].plot(ax=ax, add_colorbar=False)
    ax.set_title(f'{titles[i]} on 13/12/2022', fontsize=12)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label("Surface Reflectance", fontsize=10)

plt.show()


# Define the bands you want to plot
bands = ['mNDWI1', 'mNDWI2', 'NDVI']
titles = ['mNDWI1', 'mNDWI2', 'NDVI']

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for i, ax in enumerate(axes.flat):
    img = modis1213[bands[i]].plot(ax=ax, add_colorbar=False)
    ax.set_title(f'{titles[i]} on 13/12/2022', fontsize=12)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label("Normalised Index Value", fontsize=10)

plt.show()