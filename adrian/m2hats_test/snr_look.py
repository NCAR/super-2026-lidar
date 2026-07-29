import netCDF4 as nc
import numpy as np
import glob

vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
vad_files = sorted(glob.glob(vad_base + '30min_winds_*.nc'))

if not vad_files:
    raise SystemExit("No VAD files found - check vad_base path")

print(f"Found {len(vad_files)} files, using: {vad_files[0]}")

vad = nc.Dataset(vad_files[0])
snr = vad.variables['mean_snr'][:]
wind_speed = vad.variables['wind_speed'][:]
vad.close()

# see what fraction of data falls below various thresholds
for thresh in [-20, -23, -25, -27]:
    frac = (snr < thresh).sum() / snr.size * 100
    print(f"SNR < {thresh} dB: {frac:.1f}% of points filtered")

# check if low SNR correlates with missing wind data
already_masked = (wind_speed == -9999.0).sum()
print(f"\nAlready masked by -9999: {already_masked} points")
snr_would_add = ((snr < -23) & (wind_speed != -9999.0)).sum()
print(f"Additional points SNR < -23 would filter: {snr_would_add}")