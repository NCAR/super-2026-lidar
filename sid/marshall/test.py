import netCDF4 as nc
import numpy as np, glob, os, warnings
warnings.filterwarnings('ignore')

era5_base = '/scr/isf_apg/models/lotos2025/era5/'
vad_base  = '/scr/isf_apg/projects/lotos2025/iss2/reprocessed/windcube/vad_cnr35/'

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
    p_files = sorted(glob.glob(era5_base + date + '/era5_pressure_' + date + '_*.nc'))
    s_files = sorted(glob.glob(era5_base + date + '/era5_surface_'  + date + '_*.nc'))
    if p_files and s_files:
        break
print("First overlapping date:", date)
print("pressure files:", len(p_files), " surface files:", len(s_files))

sds = nc.Dataset(s_files[0])
surf_alt = float(sds.variables['z'][0, 0, 0]) / 9.80665
sds.close()
print("surf_alt (MSL):", surf_alt)

ds = nc.Dataset(p_files[0])
z = ds.variables['z'][0, :, 0, 0]
et = int(ds.variables['valid_time'][0])
ds.close()
agl = z/9.80665 - surf_alt
print("era5 AGL:", np.round(agl, 0))
print("levels in [100,2000]:", np.sum((agl >= 100) & (agl <= 2000)))

vad = nc.Dataset(vad_file)
ws = vad.variables['wind_speed'][:] if 'wind_speed' in vad.variables else None
corr = np.asarray(vad.variables['correlation'][:])
resid = np.asarray(vad.variables['residual'][:])
snr = np.asarray(vad.variables['mean_snr'][:])
base_t = int(vad.variables['base_time'][:]); tv = vad.variables['time'][:]
vad.close()

print("nearest VAD time diff (s):", np.min(np.abs(base_t + tv - et)))
print("corr  range:", np.nanmin(corr),  np.nanmax(corr))
print("resid range:", np.nanmin(resid), np.nanmax(resid))
print("snr   range:", np.nanmin(snr),   np.nanmax(snr))
print("corr>=0.8 count:",  np.sum(corr >= 0.8))
print("resid<=2.25 count:", np.sum(resid <= 2.25))
print("snr>=-28 count:",    np.sum(snr >= -28))

