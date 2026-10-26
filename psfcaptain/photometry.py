"""Instrumental magnitudes, outlier rejection and absolute photometric calibration."""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from psfcaptain.catalogs import query_catalog_tiled
from psfcaptain.wcs_utils import safe_pixel_to_world

# Catalog cross-match: search radius, and how far the catalog magnitude may sit from the
# zero-point prediction (fainter / brighter) before the pair is rejected as a false match.
MATCH_RADIUS_PX = 2.0
MATCH_FAINT_TOL = 0.75
MATCH_BRIGHT_TOL = 2.5


def to_dataframe(sources):
    """Converts a source table to a DataFrame with instrumental magnitudes, clipping non-positive flux."""
    df = sources.to_pandas()

    # photutils >= 3.0 renamed xcentroid/ycentroid to x_centroid/y_centroid; the rest
    # of the pipeline relies on the legacy names, so normalize them back.
    rename_map = {'x_centroid': 'xcentroid', 'y_centroid': 'ycentroid'}
    df = df.rename(columns={src: dst for src, dst in rename_map.items()
                            if src in df.columns and dst not in df.columns})

    df['mag_instr'] = -2.5 * np.log10(df['flux'])
    df['mag_abs'] = np.nan
    df['fwhm_arc'] = np.nan

    # Keep only stars bright enough to give a negative instrumental magnitude (flux > 1).
    count_before = len(df)
    df = df[df['mag_instr'] < 0].copy()
    num_clipped = count_before - len(df)
    if num_clipped > 0:
        print(f"Clipped {num_clipped} stars with non-positive flux. {len(df)} stars remaining.")
    return df


def reject_fwhm_outliers(df):
    """Splits stars whose FWHM deviates by more than 2 sigma, returning (df, df_outliers)."""
    # Filter outliers based on FWHM (> 2 sigma from mean)
    valid_fwhm_mask = df['fwhm'] > 0
    df_outliers = pd.DataFrame()
    if valid_fwhm_mask.sum() > 0:
        mean_fwhm = df.loc[valid_fwhm_mask, 'fwhm'].mean()
        std_fwhm = df.loc[valid_fwhm_mask, 'fwhm'].std()
        
        # Calculate mask for stars where FWHM is > 2 sigma from mean
        # Using fillna(False) to ensure no NaN issues
        outlier_mask = valid_fwhm_mask & (np.abs(df['fwhm'] - mean_fwhm) > 2 * std_fwhm)
        df_outliers = df[outlier_mask].copy()
        
        count_before = len(df)
        df = df[~outlier_mask].copy()
        num_outliers = count_before - len(df)
        if num_outliers > 0:
            print(f"Removed {num_outliers} stars identified as FWHM outliers (>2 sigma from mean). {len(df)} stars remaining.")
    return df, df_outliers


def reject_psf_limit(df, df_outliers, psf_limit):
    """Moves stars whose FWHM exceeds `psf_limit` pixels into df_outliers."""
    if 'fwhm' not in df.columns:
        return df, df_outliers

    psf_limit_mask = df['fwhm'] > psf_limit
    num_psf_limited = psf_limit_mask.sum()
    if num_psf_limited > 0:
        df_rejected_psf = df[psf_limit_mask].copy()
        df = df[~psf_limit_mask].copy()
        # Add to df_outliers so they are drawn as red circles in the summary overlay
        df_outliers = pd.concat([df_outliers, df_rejected_psf], ignore_index=True)
        print(f"Removed {num_psf_limited} stars exceeding PSF limit of {psf_limit} pixels. {len(df)} stars remaining.")
    return df, df_outliers


def _fit_zero_point(mag_instr, cat_mag, mag_label, verbose=True):
    """Zero point (catalog - instrumental) from matched stars, RANSAC-robust when enough are available."""
    log = print if verbose else (lambda *a, **k: None)
    # Selective RANSAC Fitting: Use stars with mag_instr < 0
    fit_mask = mag_instr < 0
    if fit_mask.sum() > 5:
        log(f"Applying RANSAC fit specifically to {fit_mask.sum()} stars with mag_instr < 0...")
        x_fit = mag_instr[fit_mask].reshape(-1, 1)
        y_fit = cat_mag[fit_mask]
    else:
        if fit_mask.sum() > 0:
            log(f"Warning: Only {fit_mask.sum()} stars found with mag_instr < 0. Using all {len(mag_instr)} matched stars for fit.")
        x_fit = mag_instr.reshape(-1, 1)
        y_fit = cat_mag

    if len(x_fit) >= 3:
        from sklearn.linear_model import RANSACRegressor
        ransac = RANSACRegressor()
        ransac.fit(x_fit, y_fit)
        inlier_mask = ransac.inlier_mask_

        # RANSAC fits y = slope*x + intercept with y = cat_mag, x = mag_instr. We expect a slope
        # near 1 and want cat_mag = mag_instr + ZP, so ZP is the median offset of the inliers.
        zero_point = np.median(y_fit[inlier_mask] - x_fit[inlier_mask].flatten())
        log(f"RANSAC Zero-Point ({mag_label}): {zero_point:.3f} mag (based on {inlier_mask.sum()} inliers)")
    else:
        log(f"Warning: Only {len(x_fit)} matched stars available. Skipping RANSAC and using strict median offset.")
        zero_point = np.median(y_fit - x_fit.flatten())
        log(f"Median Zero-Point ({mag_label}): {zero_point:.3f} mag")
    return zero_point


def calibrate_absolute(df, wcs, nx, ny, args):
    """Matches stars to a reference catalog and derives the photometric zero point."""
    # Absolute Photometry
    zero_point = None
    if args.absolute:
        if not wcs:
            print("!!! Absolute photometry skipped: Astrometry must be solved first. Use --astrometry.")
        else:
            from astropy.coordinates import SkyCoord
            center_sky = safe_pixel_to_world(wcs, df['xcentroid'].mean(), df['ycentroid'].mean())
            # Estimate radius needed to cover the image
            p0 = safe_pixel_to_world(wcs, 0, 0)
            p1 = safe_pixel_to_world(wcs, nx, ny)
            radius = p0.separation(p1).deg / 2.0 * 1.2
            
            print(f"Performing absolute photometry via {args.catalog.upper()} catalog...")
            catalog_results = query_catalog_tiled(center_sky, radius, args.catalog)
            
            if catalog_results is not None and len(catalog_results) > 0:
                if args.catalog == 'gaia':
                    # Filter by magnitude limit (default 12.0 or user-specified)
                    mag_mask = catalog_results['phot_g_mean_mag'] < args.gaia_mag_limit
                    catalog_results = catalog_results[mag_mask]
                    print(f"Filtered Gaia catalog to {len(catalog_results)} stars brighter than magnitude {args.gaia_mag_limit}.")

                if len(catalog_results) > 0:
                    cat_ra = catalog_results['ra'] if 'ra' in catalog_results.colnames else catalog_results['RA(ICRS)']
                    cat_dec = catalog_results['dec'] if 'dec' in catalog_results.colnames else catalog_results['DE(ICRS)']

                    # Magnitude column depends on catalog
                    if args.catalog == 'gaia':
                        cat_mag = catalog_results['phot_g_mean_mag']
                        mag_label = "Gaia G"
                    else: # tycho2
                        cat_mag = catalog_results['VTmag']
                        mag_label = "Tycho-2 VT"

                    catalog_coords = SkyCoord(ra=cat_ra, dec=cat_dec, unit=u.deg)
                    star_coords = safe_pixel_to_world(wcs, df['xcentroid'], df['ycentroid'])

                    # Candidate pairs within MATCH_RADIUS_PX. The catalog is much denser than
                    # our detections, so the nearest catalog star is often an unrelated faint one.
                    scale_guess = wcs.pixel_to_world(0,0).separation(wcs.pixel_to_world(1,0)).arcsec
                    match_limit_arcsec = MATCH_RADIUS_PX * scale_guess
                    print(f"Matching stars with a limit of {MATCH_RADIUS_PX:g} pixels ({match_limit_arcsec:.2f} arcsec)...")

                    cat_mag = np.asarray(cat_mag, dtype=float)
                    idx_cat, idx_star, d2d, _ = star_coords.search_around_sky(
                        catalog_coords, match_limit_arcsec * u.arcsec)

                    df['catalog_mag'] = np.nan

                    if len(idx_star) > 0:
                        # Pass 1: nearest candidate per star, only to get a preliminary zero point.
                        order = np.argsort(d2d.arcsec)
                        first = pd.DataFrame({'star': idx_star[order], 'cat': idx_cat[order]}).drop_duplicates('star')
                        zero_point = _fit_zero_point(df['mag_instr'].values[first['star']],
                                                     cat_mag[first['cat']], mag_label, verbose=False)

                        # Pass 2: per star, the candidate closest in magnitude to the prediction.
                        # Vignetting only dims a star, so a catalog star much fainter than the
                        # measurement is a false match; a brighter one may be a vignetted true match.
                        dm = cat_mag[idx_cat] - (df['mag_instr'].values[idx_star] + zero_point)
                        ok = (dm < MATCH_FAINT_TOL) & (dm > -MATCH_BRIGHT_TOL)
                        pairs = pd.DataFrame({'star': idx_star[ok], 'cat': idx_cat[ok], 'adm': np.abs(dm[ok])})
                        pairs = pairs.sort_values('adm').drop_duplicates('star').drop_duplicates('cat')
                        print(f"Kept {len(pairs)} magnitude-consistent matches "
                              f"(of {len(first)} stars with a catalog star within {MATCH_RADIUS_PX:g} px).")

                    if len(idx_star) > 0 and len(pairs) > 0:
                        star_rows = df.index[pairs['star'].values]
                        df.loc[star_rows, 'catalog_mag'] = cat_mag[pairs['cat'].values]
                        zero_point = _fit_zero_point(df.loc[star_rows, 'mag_instr'].values,
                                                     cat_mag[pairs['cat'].values], mag_label)
                        df['mag_abs'] = df['mag_instr'] + zero_point

                        # We will generate the mag_comparison plot later from the saved CSV
                    else:
                        zero_point = None
                        print(f"No {mag_label} stars matched within {match_limit_arcsec:.2f} arcseconds.")
                else:
                    print(f"No {args.catalog.upper()} catalog stars remain after magnitude filtering.")
            else:
                print(f"{args.catalog.upper()} query returned no sources in this field.")
    return zero_point
