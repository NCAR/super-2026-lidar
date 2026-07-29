import netCDF4 as nc
import numpy as np
import glob, os, warnings
warnings.filterwarnings('ignore')

hrrr_base = '/scr/isf_apg/models/m2hats/hrrr/'
vad_base = '/scr/isf_apg/projects/m2hats/iss1/reprocessed/windcube/vad_cnr33_witherror/'

thresholds = [None, -27.0, -25.0, -23.0, -20.0]

all_snr, all_sdiff, all_ddiff_snr, all_ws_pair = [], [], [], []

for vad_file in sorted(glob.glob(vad_base + 'VAD_*.nc')):
    date = os.path.basename(vad_file).replace('VAD_', '').replace('.nc', '')
    h_files = sorted(glob.glob(hrrr_base + date + '/hrrr_profile_' + date + '_*_ISS1.nc'))

    if not h_files:
        continue

    vad = nc.Dataset(vad_file)
    ws_vad = vad.variables['wind_speed'][:]
    wd_vad = vad.variables['wind_direction'][:]
    height = vad.variables['height'][:]
    base_t = int(vad.variables['base_time'][:])
    time_vad = vad.variables['time'][:]
    snr_vad = vad.variables['mean_snr'][:]
    vad.close()

    ws_vad = np.ma.masked_where(ws_vad == -9999.0, ws_vad)
    wd_vad = np.ma.masked_where(wd_vad == -9999.0, wd_vad)
    vad_epoch = base_t + time_vad

    for f in h_files:
        ds = nc.Dataset(f)
        hrrr_ws = ds.variables['wspd'][:]
        hrrr_dir = ds.variables['wdir'][:]
        hrrr_agl = ds.variables['height'][:]
        et = int(ds.variables['time'][0])
        ds.close()

        ti = np.argmin(np.abs(vad_epoch - et))
        if abs(vad_epoch[ti] - et) > 900:
            continue

        valid = ~np.ma.getmaskarray(ws_vad[ti])
        if not valid.any():
            continue

        h_valid = height[valid]
        idx_valid = np.where(valid)[0]

        for k in range(len(hrrr_agl)):
            if not (100 <= hrrr_agl[k] <= 2000):
                continue
            j = np.argmin(np.abs(h_valid - hrrr_agl[k]))
            if np.abs(h_valid[j] - hrrr_agl[k]) > 25:
                continue
            idx = idx_valid[j]

            # GUARD 1: reject any masked VAD element that slipped through
            if np.ma.is_masked(ws_vad[ti, idx]) or np.ma.is_masked(wd_vad[ti, idx]):
                continue
            # GUARD 2: reject non-finite values
            if not (np.isfinite(ws_vad[ti, idx]) and np.isfinite(snr_vad[ti, idx])):
                continue

            all_snr.append(float(snr_vad[ti, idx]))
            all_sdiff.append(float(ws_vad[ti, idx] - hrrr_ws[k]))
            all_ws_pair.append((float(hrrr_ws[k]), float(ws_vad[ti, idx])))
            dd = ((float(wd_vad[ti, idx]) - hrrr_dir[k] + 180) % 360) - 180
            all_ddiff_snr.append(dd)

all_snr = np.array(all_snr)
all_sdiff = np.array(all_sdiff)
all_ddiff_snr = np.array(all_ddiff_snr)
all_ws_pair = np.array(all_ws_pair)

print(f"Total matched points collected: {len(all_snr)}")
print(f"sdiff min/max: {all_sdiff.min():.1f} / {all_sdiff.max():.1f}")
print(f"vad_ws min/max: {all_ws_pair[:,1].min():.1f} / {all_ws_pair[:,1].max():.1f}\n")

dir_speed_ok = (all_ws_pair[:, 0] > 2.0) & (all_ws_pair[:, 1] > 2.0)

print(f"{'SNR cutoff':>12} | {'ws pts':>7} | {'ws mad':>7} | {'ws sd':>7} | {'dir pts':>8} | {'dir mad':>8} | {'dir sd':>7}")
print("-" * 78)

for thresh in thresholds:
    if thresh is None:
        keep = np.ones(len(all_snr), dtype=bool)
        label = "none"
    else:
        keep = all_snr >= thresh
        label = f"{thresh:.0f} dB"

    sd = all_sdiff[keep]
    ws_mad, ws_sd = np.mean(np.abs(sd)), np.std(sd)

    dkeep = keep & dir_speed_ok
    dd = all_ddiff_snr[dkeep]
    dir_mad, dir_sd = np.mean(np.abs(dd)), np.std(dd)

    print(f"{label:>12} | {keep.sum():>7} | {ws_mad:>7.2f} | {ws_sd:>7.2f} | "
          f"{dkeep.sum():>8} | {dir_mad:>8.2f} | {dir_sd:>7.2f}")