"""Per-star PSF morphology measurement (FWHM, elongation, orientation)."""

import concurrent.futures
import os

import numpy as np
from photutils.morphology import data_properties

from psfcaptain.wcs_utils import safe_pixel_to_world


def process_single_star(i, x, y, mag_instr, cutout, std, wcs=None):
    """Worker function to process a single star's morphology and coordinates."""
    try:
        # Get RA/Dec if WCS is available
        ra, dec = -999.0, -999.0
        local_scale = None
        fwhm_arcsec = -999.0
        
        if wcs:
            try:
                sky_coord = safe_pixel_to_world(wcs, x, y)
                ra, dec = float(sky_coord.ra.deg), float(sky_coord.dec.deg)
                
                # Sample local pixel scale
                sky1 = sky_coord
                sky2 = safe_pixel_to_world(wcs, x + 1, y)
                sky3 = safe_pixel_to_world(wcs, x, y + 1)
                local_scale = float(np.sqrt(sky1.separation(sky2).arcsec * sky1.separation(sky3).arcsec))
            except:
                pass

        # Morphology (FWHM, elongation, theta)
        # Apply a threshold mask to isolate the star from background noise
        mask = cutout < (2 * std)
        props = data_properties(cutout, mask=mask)
        
        fwhm = props.fwhm.value if hasattr(props.fwhm, 'value') else props.fwhm
        elongation = props.elongation.value if hasattr(props.elongation, 'value') else props.elongation
        theta = props.orientation.deg if hasattr(props.orientation, 'deg') else (
                props.orientation.value if hasattr(props.orientation, 'value') else props.orientation)
        
        fwhm = float(np.array(fwhm))
        elongation = float(np.array(elongation))
        theta = float(np.array(theta))
        
        if local_scale is not None:
            fwhm_arcsec = fwhm * local_scale
            
        return {
            'index': i,
            'x': float(x),
            'y': float(y),
            'ra': ra,
            'dec': dec,
            'mag_instr': float(mag_instr),
            'fwhm': fwhm,
            'fwhm_arcsec': float(fwhm_arcsec),
            'elongation': elongation,
            'theta': theta,
            'local_scale': local_scale
        }
    except Exception as e:
        return {
            'index': i,
            'x': float(x),
            'y': float(y),
            'ra': -999.0,
            'dec': -999.0,
            'mag_instr': float(mag_instr),
            'fwhm': -999.0,
            'fwhm_arcsec': -999.0,
            'elongation': -999.0,
            'theta': -999.0,
            'local_scale': None
        }


def measure_morphology(df, data_sub, std, wcs, nx, ny, args):
    """Measures PSF morphology for every star in `df`, updating it in place."""
    # PSF Morphometry
    print(f"Measuring morphology...")
    star_tasks = []
    for i, (idx, row) in enumerate(df.iterrows()):
        x, y = row['xcentroid'], row['ycentroid']
        size = int(args.fwhm * 5)
        x_min, x_max = max(0, int(x - size)), min(nx, int(x + size))
        y_min, y_max = max(0, int(y - size)), min(ny, int(y + size))
        cutout = data_sub[y_min:y_max, x_min:x_max]
        star_tasks.append((i, x, y, row['mag_instr'], cutout, std, wcs))

    results = [None] * len(df)
    
    # Adaptive multiprocessing: Only use a pool for stars if we are NOT already in an image-level pool
    # We can check if 'IMAGE_POOL' is in the environment
    is_image_parallel = os.environ.get("IMAGE_POOL", "0") == "1"
    
    if args.cores > 1 and not is_image_parallel and len(df) > 50:
        print(f"  (Processing {len(df)} stars using {args.cores} cores)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as executor:
            futures = [executor.submit(process_single_star, *task) for task in star_tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results[res['index']] = res
    else:
        # Sequential morphology
        for task in star_tasks:
            res = process_single_star(*task)
            results[res['index']] = res
    
    # Merge results
    for i, res in enumerate(results):
        if res is not None:
            for key in res:
                if key != 'index': # index is internal
                    df.loc[df.index[i], key] = res[key]

    # Check specifically for RMS morphology parameters
    had_rms_fwhm = 'rms_fwhm' in df.columns
    had_rms_elongation = 'rms_elongation' in df.columns
    had_rms_theta = 'rms_theta' in df.columns

    # If RMS finder was used, overwrite the respective array blocks mapped dynamically locally
    if had_rms_fwhm:
        mask = ~df['rms_fwhm'].isna()
        df.loc[mask, 'fwhm'] = df.loc[mask, 'rms_fwhm']
        
    if had_rms_elongation:
        mask = ~df['rms_elongation'].isna()
        df.loc[mask, 'elongation'] = df.loc[mask, 'rms_elongation']
        
    if had_rms_theta:
        mask = ~df['rms_theta'].isna()
        df.loc[mask, 'theta'] = df.loc[mask, 'rms_theta']

    return df


def compute_fwhm_arcsec(df, wcs):
    """Vectorized local pixel-scale sampling; fills in df['fwhm_arc'] in place.

    No-op (leaves df['fwhm_arc'] as NaN) when there is no WCS solution.
    """
    if not wcs or len(df) == 0:
        return df

    all_x = df['xcentroid'].values
    all_y = df['ycentroid'].values

    sky_c = wcs.pixel_to_world(all_x, all_y)
    sky_x = wcs.pixel_to_world(all_x + 1, all_y)
    sky_y = wcs.pixel_to_world(all_x, all_y + 1)

    # separation returns Angle instances; we want arcseconds
    sep_x = sky_c.separation(sky_x).arcsec
    sep_y = sky_c.separation(sky_y).arcsec
    local_scales = np.sqrt(sep_x * sep_y)

    df['fwhm_arc'] = df['fwhm'] * local_scales
    return df
