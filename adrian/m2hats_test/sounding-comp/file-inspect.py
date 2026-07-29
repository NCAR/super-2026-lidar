import netCDF4 as nc
import numpy as np
import glob, os

vad_base   = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/'
sonde_base = '/net/isf/radiosonde_archive/2023_m2hats/qc/ncdf_v1/'

# --- check a sample VAD file -------------------------------------------------
vad_files = sorted(glob.glob(vad_base + '30min_winds_*.nc'))
print("VAD files found: %d" % len(vad_files))

if vad_files:
    vad = nc.Dataset(vad_files[0])
    base_t   = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    vad_epoch = base_t + time_vad
    print("VAD sample epoch range: %d to %d" % (vad_epoch[0], vad_epoch[-1]))
    vad.close()

# --- check the sonde directory structure -------------------------------------
# Print the first few subdirectories and files to see the actual layout
print("\nSonde base contents (first 5 entries):")
try:
    entries = sorted(os.listdir(sonde_base))[:5]
    for e in entries:
        print("  ", e)
except Exception as ex:
    print("  ERROR listing sonde_base:", ex)

# --- try the glob pattern for the first VAD date -----------------------------
if vad_files:
    date = os.path.basename(vad_files[0]).replace('30min_winds_', '').replace('.nc', '')
    print("\nFirst VAD date: %s" % date)

    # show what the glob is looking for
    pattern = os.path.join(sonde_base, date, 'NCAR_M2HATS_ISS1_RS41_v1_*_*' + date + '_*.nc')
    print("Sonde glob pattern: %s" % pattern)
    s_files = sorted(glob.glob(pattern))
    print("Sonde files matched: %d" % len(s_files))

    # if nothing matched, show what IS in that date subdirectory
    date_dir = os.path.join(sonde_base, date)
    if os.path.isdir(date_dir):
        print("Files in %s:" % date_dir)
        for f in sorted(os.listdir(date_dir))[:10]:
            print("  ", f)
    else:
        print("Date subdirectory does not exist: %s" % date_dir)
        # maybe sondes aren't in subdirectories - check for flat layout
        flat = sorted(glob.glob(os.path.join(sonde_base, '*' + date + '*.nc')))
        print("Flat glob (*%s*.nc) found: %d" % (date, len(flat)))
        for f in flat[:5]:
            print("  ", os.path.basename(f))

# --- if a sonde file is found, inspect its variables and time ----------------
all_sondes = sorted(glob.glob(os.path.join(sonde_base, '**', 'NCAR_M2HATS_ISS1_RS41_v1_*.nc'), recursive=True))
print("\nTotal sonde files found (recursive): %d" % len(all_sondes))
if all_sondes:
    sf = all_sondes[0]
    print("Inspecting: %s" % os.path.basename(sf))
    ds = nc.Dataset(sf)
    print("Variables:", list(ds.variables.keys()))
    for v in ['base_time', 'time', 'height', 'alt', 'altitude', 'wind_speed', 'wspd', 'wind_direction', 'wdir']:
        if v in ds.variables:
            arr = ds.variables[v][:]
            print("  %s: shape=%s  first value=%s" % (v, arr.shape, arr.flat[0]))
    ds.close()