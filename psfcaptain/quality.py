"""Per-image field-quality measurements used by the lens quality report.

Adds fixed-aperture photometry with a background-limited signal-to-noise ratio to
every star, and samples the smooth sky background across the field (vignetting).
The results go into the per-image CSV (new columns) and a small JSON sidecar
(`<image>_quality.json`) so the report can be rebuilt later from CSVs/ alone.
"""

import json
import os

import numpy as np
from photutils.aperture import ApertureStats, CircularAperture
from photutils.background import Background2D, MedianBackground

# Resolution of the normalized sky-background grid stored in the sidecar.
SKY_GRID_SIZE = 32


def _saturation_level(data, header):
    """Best guess at the saturation level of the raw image."""
    if header is not None and 'SATURATE' in header:
        return float(header['SATURATE'])
    peak = float(np.nanmax(data))
    for full_well in (255.0, 4095.0, 16383.0, 65535.0):
        if peak <= full_well:
            return full_well
    return peak


def _pixel_scale(wcs, x, y):
    """Local pixel scale in arcsec/pixel at the given pixel positions."""
    sky_c = wcs.pixel_to_world(x, y)
    sky_x = wcs.pixel_to_world(x + 1, y)
    sky_y = wcs.pixel_to_world(x, y + 1)
    return np.sqrt(sky_c.separation(sky_x).arcsec * sky_c.separation(sky_y).arcsec)


def measure_field_quality(data, header, df, wcs, nx, ny):
    """Adds aperture photometry columns to `df` and returns (df, quality_meta).

    New columns:
      ap_flux   - background-subtracted flux in a circular aperture of radius
                  max(1.5, median FWHM) pixels, the same radius for every star so
                  that a locally blurred PSF loses light, as it does in practice
      sky       - local sky level under the star (raw counts)
      sky_rms   - local sky noise per pixel
      snr       - background-limited S/N: ap_flux / (sky_rms * sqrt(aperture area))
      mag_ap    - instrumental aperture magnitude, -2.5 log10(ap_flux)
      saturated - True when the star's brightest pixel is at the saturation level
    """
    box = int(max(16, min(nx, ny) // 16))
    bkg = Background2D(data, (box, box), filter_size=(3, 3), bkg_estimator=MedianBackground())

    meta = {
        'nx': int(nx),
        'ny': int(ny),
        'exptime': float(header['EXPTIME']) if header is not None and 'EXPTIME' in header else None,
        'date_obs': str(header['DATE-OBS']) if header is not None and 'DATE-OBS' in header else None,
        'sky_median': float(np.median(bkg.background)),
        'sky_rms_median': float(np.median(bkg.background_rms)),
        'saturation_level': _saturation_level(data, header),
    }

    # Sky background sampled on a normalized grid (cell centers), for vignetting.
    gu = (np.arange(SKY_GRID_SIZE) + 0.5) / SKY_GRID_SIZE
    gx = np.clip((gu * nx).astype(int), 0, nx - 1)
    gy = np.clip((gu * ny).astype(int), 0, ny - 1)
    meta['sky_grid'] = bkg.background[np.ix_(gy, gx)].tolist()

    if wcs:
        cx, cy = nx / 2.0, ny / 2.0
        meta['pixel_scale_center'] = float(_pixel_scale(wcs, np.array([cx]), np.array([cy]))[0])
        corners_x = np.array([0.0, nx - 1.0, 0.0, nx - 1.0])
        corners_y = np.array([0.0, 0.0, ny - 1.0, ny - 1.0])
        meta['pixel_scale_corners'] = float(np.mean(_pixel_scale(wcs, corners_x, corners_y)))

    if len(df) == 0:
        meta['aperture_radius'] = None
        return df, meta

    fwhm = df['fwhm'].values
    good_fwhm = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
    radius = float(max(1.5, np.median(good_fwhm))) if len(good_fwhm) else 1.5 * 3.0
    meta['aperture_radius'] = radius

    xy = np.column_stack((df['xcentroid'].values, df['ycentroid'].values))
    aperture = CircularAperture(xy, r=radius)
    stats = ApertureStats(data - bkg.background, aperture)
    raw_stats = ApertureStats(data, aperture)

    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, nx - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, ny - 1)
    sky_rms = bkg.background_rms[yi, xi]
    noise = sky_rms * np.sqrt(np.pi * radius ** 2)

    ap_flux = np.asarray(stats.sum, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['ap_flux'] = ap_flux
        df['sky'] = bkg.background[yi, xi]
        df['sky_rms'] = sky_rms
        df['snr'] = np.where(noise > 0, ap_flux / noise, np.nan)
        df['mag_ap'] = np.where(ap_flux > 0, -2.5 * np.log10(ap_flux), np.nan)
    df['saturated'] = np.asarray(raw_stats.max, dtype=float) >= 0.98 * meta['saturation_level']

    if wcs:
        df['pix_scale'] = _pixel_scale(wcs, xy[:, 0], xy[:, 1])

    return df, meta


def write_quality_meta(meta, csvs_dir, image_name):
    """Writes the per-image quality sidecar JSON next to the results CSV."""
    path = os.path.join(csvs_dir, f'{image_name}_quality.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=1)
    return path
