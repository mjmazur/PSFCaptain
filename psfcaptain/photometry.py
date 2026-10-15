"""Instrumental magnitudes, outlier rejection and absolute photometric calibration."""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from psfcaptain.catalogs import query_catalog_tiled
from psfcaptain.wcs_utils import safe_pixel_to_world


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

                    # Match
                    # We use a 5 pixel match limit. Calculate that in arcsec based on image scale
                    scale_guess = wcs.pixel_to_world(0,0).separation(wcs.pixel_to_world(1,0)).arcsec
                    match_limit_arcsec = 5 * scale_guess
                    print(f"Matching stars with a limit of 5 pixels ({match_limit_arcsec:.2f} arcsec)...")

                    idx, d2d, _ = star_coords.match_to_catalog_sky(catalog_coords)
                    matches_mask = d2d < match_limit_arcsec * u.arcsec

                    df['catalog_mag'] = np.nan

                    if matches_mask.any():
                        matched_df = df[matches_mask].copy()
                        matched_cat_mag = cat_mag[idx[matches_mask]]
                        df.loc[matches_mask, 'catalog_mag'] = np.asarray(matched_cat_mag)

                        # Selective RANSAC Fitting: Use stars with mag_instr < 0
                        fit_mask = matched_df['mag_instr'] < 0

                        if fit_mask.sum() > 5:
                            print(f"Applying RANSAC fit specifically to {fit_mask.sum()} stars with mag_instr < 0...")
                            x_fit = matched_df[fit_mask]['mag_instr'].values.reshape(-1, 1)
                            y_fit = np.asarray(matched_cat_mag[fit_mask])
                        else:
                            if fit_mask.sum() > 0:
                                print(f"Warning: Only {fit_mask.sum()} stars found with mag_instr < 0. Using all {len(matched_df)} matched stars for fit.")
                            x_fit = matched_df['mag_instr'].values.reshape(-1, 1)
                            y_fit = np.asarray(matched_cat_mag)

                        if len(x_fit) >= 3:
                            from sklearn.linear_model import RANSACRegressor
                            ransac = RANSACRegressor()
                            ransac.fit(x_fit, y_fit)
                            inlier_mask = ransac.inlier_mask_

                            # The offset (zero point) is the intercept if we assume slope=1
                            # Actually, we want phot_cat = mag_instr + ZP, so ZP = phot_cat - mag_instr
                            # RANSAC fits y = slope*x + intercept. Here y is cat_mag, x is mag_instr.
                            # We expect slope close to 1.
                            zero_point = np.median(y_fit[inlier_mask] - x_fit[inlier_mask].flatten())
                            print(f"RANSAC Zero-Point ({mag_label}): {zero_point:.3f} mag (based on {inlier_mask.sum()} inliers)")
                        else:
                            print(f"Warning: Only {len(x_fit)} matched stars available. Skipping RANSAC and using strict median offset.")
                            zero_point = np.median(y_fit - x_fit.flatten())
                            print(f"Median Zero-Point ({mag_label}): {zero_point:.3f} mag")

                        df['mag_abs'] = df['mag_instr'] + zero_point

                        # We will generate the mag_comparison plot later from the saved CSV
                    else:
                        print(f"No Gaia stars matched within {match_limit_arcsec:.2f} arcseconds.")
                else:
                    print(f"No {args.catalog.upper()} catalog stars remain after magnitude filtering.")
            else:
                print(f"{args.catalog.upper()} query returned no sources in this field.")
    return zero_point
