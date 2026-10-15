"""Background estimation and star detection."""

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder, IRAFStarFinder

from psfcaptain.finders import RMSStarFinder


def subtract_background(data, use_sky=False):
    """Removes the sky background, returning (data_sub, mean, median, std)."""
    if use_sky:
        print("Estimating local background...")
        bkg_estimator = MedianBackground()
        # box size of ~50 is a reasonable default for star fields
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        mean, median, std = np.median(bkg.background), np.median(bkg.background_median), np.median(bkg.background_rms)
        data_sub = data - bkg.background
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median
    return data_sub, mean, median, std


def detect_stars(data_sub, std, args, nx, ny):
    """Runs the configured finder(s) over the image, returning a source table or None."""
    # Star Detection
    finder_args = args.finder if isinstance(args.finder, list) else [args.finder]
    primary_finder = finder_args[0]
    secondary_finder = finder_args[1] if len(finder_args) > 1 and primary_finder == 'rms' else None

    print(f"Finding stars (using {primary_finder.upper()} finder)...")
    if primary_finder == 'dao':
        finder = DAOStarFinder(fwhm=args.fwhm, threshold=args.threshold * std,
                               sharplo=args.sharplo, sharphi=args.sharphi,
                               roundlo=args.roundlo, roundhi=args.roundhi,
                               exclude_border=args.exclude_border)
    elif primary_finder == 'iraf':
        finder = IRAFStarFinder(fwhm=args.fwhm, threshold=args.threshold * std,
                                sharplo=args.sharplo, sharphi=args.sharphi,
                                roundlo=args.roundlo, roundhi=args.roundhi,
                                exclude_border=args.exclude_border)
    else:
        finder = RMSStarFinder(fwhm=args.fwhm, threshold=args.threshold * std, gamma=args.gamma, exclude_border=args.exclude_border)
        
    sources = finder(data_sub)

    if sources is None or len(sources) == 0:
        print("No stars found.")
        return

    # For the dao and iraf finders, ignore all stars found within 25 pixels of the edge
    if 'dao' in finder_args or 'iraf' in finder_args:
        edge_mask = (sources['xcentroid'] >= 25) & (sources['xcentroid'] <= nx - 25) & \
                    (sources['ycentroid'] >= 25) & (sources['ycentroid'] <= ny - 25)
        count_before = len(sources)
        sources = sources[edge_mask]
        if len(sources) < count_before:
            print(f"Ignored {count_before - len(sources)} stars within 25 pixels of the edge. {len(sources)} remain.")

    if len(sources) == 0:
        print("No stars found after edge filtering.")
        return

    if secondary_finder:
        print(f"Running secondary photometry using {secondary_finder.upper()} finder at RMS coordinates...")
        xycoords = np.transpose((sources['xcentroid'], sources['ycentroid']))
        mean_rms_fwhm = np.median(sources['rms_fwhm']) if 'rms_fwhm' in sources.colnames else args.fwhm
        
        if secondary_finder == 'dao':
            sec_finder = DAOStarFinder(fwhm=mean_rms_fwhm, threshold=-1e9,
                                       xycoords=xycoords,
                                       exclude_border=args.exclude_border)
        elif secondary_finder == 'iraf':
            sec_finder = IRAFStarFinder(fwhm=mean_rms_fwhm, threshold=-1e9,
                                        xycoords=xycoords,
                                        exclude_border=args.exclude_border)
        elif secondary_finder == 'rms':
            sec_finder = RMSStarFinder(fwhm=mean_rms_fwhm, threshold=-1e9, gamma=args.gamma,
                                       xycoords=xycoords,
                                       exclude_border=args.exclude_border)
        else:
            print(f"Unknown secondary finder: {secondary_finder}")
            return
            
        sec_sources = sec_finder(data_sub)
        
        if sec_sources is not None and len(sec_sources) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.transpose((sec_sources['xcentroid'], sec_sources['ycentroid'])))
            dists, idxs = tree.query(xycoords)
            
            match_mask = dists < 1.0
            sources['flux'][match_mask] = sec_sources['flux'][idxs[match_mask]]
            if 'peak' in sec_sources.colnames and 'peak' in sources.colnames:
                sources['peak'][match_mask] = sec_sources['peak'][idxs[match_mask]]
                
            if 'rms_fwhm' in sec_sources.colnames:
                if 'rms_fwhm' not in sources.colnames:
                    sources['rms_fwhm'] = np.nan
                sources['rms_fwhm'][match_mask] = sec_sources['rms_fwhm'][idxs[match_mask]]
                
            if 'rms_elongation' in sec_sources.colnames:
                if 'rms_elongation' not in sources.colnames:
                    sources['rms_elongation'] = np.nan
                sources['rms_elongation'][match_mask] = sec_sources['rms_elongation'][idxs[match_mask]]
                
            if 'rms_theta' in sec_sources.colnames:
                if 'rms_theta' not in sources.colnames:
                    sources['rms_theta'] = np.nan
                sources['rms_theta'][match_mask] = sec_sources['rms_theta'][idxs[match_mask]]
                
            print(f"Updated fluxes and morphology properties for {np.sum(match_mask)} out of {len(sources)} stars using {secondary_finder.upper()}.")

    print(f"Found {len(sources)} stars.")
    return sources
