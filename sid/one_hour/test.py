import netCDF4 as nc
import numpy as np, glob, os, warnings
warnings.filterwarnings('ignore')

vad_file = sorted(glob.glob('/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_consensus/30min_winds_*.nc'))[0]
date = os.path.basename(vad_file).replace('30min_winds_','').replace('.nc','')
era5_f = sorted(glob.glob('/scr/isf_apg/models/m2hats/era5/'+date+'/era5_pressure_'+date+'_*_ISS1.nc'))[0]

vad = nc.Dataset(vad_file)
station_alt = float(vad.variables['alt'][:])
height = vad.variables['height'][:]
print("station_alt (MSL):", station_alt)
print("VAD height AGL range:", height.min(), "to", height.max())
vad.close()

ds = nc.Dataset(era5_f)
z = ds.variables['z'][0,:,0,0]
ds.close()
era5_msl = z/9.80665
print("ERA5 MSL range:", era5_msl.min(), "to", era5_msl.max())
print("ERA5 AGL (MSL - station_alt):", np.round(era5_msl - station_alt, 1))
