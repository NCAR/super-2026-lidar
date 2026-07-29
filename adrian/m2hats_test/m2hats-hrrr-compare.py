# Data Manipulation and Analysis
import xarray as xr
import numpy as np
import pandas as pd
import scipy as sp

# Data Visualization / Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Helpful for formatting the time axis smoothly


# ------------------------------
# Load 30-minute VAD lidar data
# ------------------------------
ds_lidar = xr.open_dataset('30min_winds_20230820.nc')


# print(ds_lidar)


# ------------------------------
# Load HRRR data
# ------------------------------
ds_hrrr = xr.open_dataset('/scr/isf_apg/models/m2hats/hrrr/20230820/hrrr_profile_20230820_23_ISS1.nc');


# ------------------------------
# Clean time coordinates
# ------------------------------
lidar_times = pd.to_datetime(ds_lidar.time.values)
hrrr_times = pd.to_datetime(ds_hrrr.time.values)

ds_lidar = ds_lidar.assign_coords(time=lidar_times)
hrrr_clean = ds_hrrr.assign_coords(time=hrrr_times)

# Fix height coordinates
# Tonopah, NV airport elevation is roughly 1655 meters MSL
tonopah_elevation = 1641.0

# Create a clean height_msl coordinate for the lidar without relying on 'alt'
ds_lidar['height_msl'] = ds_lidar.height + tonopah_elevation

# Two-step interpolation
# Step A: Interpolate time first
hrrr_time_aligned = hrrr_clean.interp(time=ds_lidar.time, method='linear')

# Step B: Interpolate height using extrapolation to prevent edge clipping 
# (This handles lidar points that sit below HRRR's lowest 1841m floor)
hrrr_aligned = hrrr_time_aligned.interp(
    height_msl=ds_lidar.height_msl, 
    method='linear',
    kwargs={"fill_value": "extrapolate"}
)

# Quick validation check
final_nans = np.isnan(hrrr_aligned.wspd.values).sum()
print(f"Success! Final NaN count is now: {final_nans} out of 4800")

# Map the lidar variable name (key) to the corresponding HRRR variable name (value)
# UPDATE: Verify your lidar uses 'u', 'v', 'w' by checking list(ds_lidar.data_vars)
vars_to_compare = {
    "u": "u_wind",
    "v": "v_wind",
    "w": "w_wind"
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (lidar_var, hrrr_var) in zip(axes, vars_to_compare.items()):
    
    # 1. Pull the data cleanly from your aligned datasets
    # .ravel() flattens the matrix into a 1D vector just like .flatten()
    x = ds_lidar[lidar_var].values.ravel()
    y = hrrr_aligned[hrrr_var].values.ravel()

    # 2. Mask nan values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # 3. Scatter plot
    ax.scatter(x, y, s=5, alpha=0.5)

    # 4. Add x = y reference line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "r--", label="x=y line")

    # 5. Correlation and R²
    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}\nR² = {r2:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    # 6. Update labels to accurately reflect your instruments
    ax.set_xlabel(f"Lidar {lidar_var} component (m/s)")
    ax.set_ylabel(f"HRRR {hrrr_var} (m/s)")
    ax.set_title(f"Comparison of {hrrr_var}")
    ax.legend()

plt.tight_layout()
plt.show()