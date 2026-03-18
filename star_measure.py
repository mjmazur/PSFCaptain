import os
# Set environment variables to limit threading in MKL-dependent libraries
# This must be done BEFORE importing numpy, scipy, etc. to prevent memory exhaustion in multiprocessing.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DOMAIN_NUM_THREADS"] = "1"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.morphology import data_properties
from PIL import Image
import concurrent.futures
import multiprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable

def twoDGaussian(params, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """ Defines a 2D Gaussian distribution. """
    x, y, saturation = params

    if isinstance(saturation, np.ndarray):
        saturation = saturation[0, 0]
    
    xo = float(xo)
    yo = float(yo)

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2)))

    # Limit values to saturation level
    g[g > saturation] = saturation

    return g.ravel()

@np.vectorize
def gamma_correction(intensity, gamma, bp=0, wp=255):
    """ Correct the given intensity for gamma. """
    if intensity < 0:
        intensity = 0
    x = (intensity - bp) / (wp - bp)
    if x > 0:
        return bp + (wp - bp) * (x**(1.0/gamma))
    else:
        return bp

class RMSStarFinder:
    def __init__(self, fwhm, threshold, gamma=1.0, neighborhood_size=10, segment_radius=15, roundness_threshold=0.1, max_feature_ratio=2.0, exclude_border=False):
        self.fwhm = fwhm
        self.threshold = threshold
        self.gamma = gamma
        self.neighborhood_size = neighborhood_size
        self.segment_radius = segment_radius
        self.roundness_threshold = roundness_threshold
        self.max_feature_ratio = max_feature_ratio
        self.exclude_border = exclude_border

    def __call__(self, data):
        import scipy.ndimage as ndimage
        import scipy.optimize as opt
        from astropy.table import Table
        
        # We need a positive data for RMS finder. In star_measure.py, data is background subtracted
        # calculate image mean
        avepixel_mean = 0.0 # because it's already background subtracted
        
        # Apply a mean filter to the image to reduce noise
        data_filtered = ndimage.convolve(data.astype(np.float32), weights=np.full((2, 2), 1.0/4))
        
        # Locate local maxima on the image
        data_max = ndimage.maximum_filter(data_filtered, self.neighborhood_size)
        maxima = (data_filtered == data_max)
        data_min = ndimage.minimum_filter(data_filtered, self.neighborhood_size)
        diff = ((data_max - data_min) > self.threshold)
        maxima[diff == 0] = 0
        
        if self.exclude_border:
            border = int(self.fwhm * 2)
            border_mask = np.ones_like(maxima)*255
            border_mask[:border,:] = 0
            border_mask[-border:,:] = 0
            border_mask[:,:border] = 0
            border_mask[:,-border:] = 0
            maxima[border_mask == 0] = 0
            
        # Find and label the maxima
        labeled, num_objects = ndimage.label(maxima)
        if num_objects == 0:
            return None
            
        # Find centres of mass of each labeled objects
        xy = np.array(ndimage.center_of_mass(data_filtered, labeled, range(1, num_objects+1)))
        
        if len(xy) == 0:
            return None
            
        # Unpack star coordinates
        y, x = np.hsplit(xy, 2)
        y = y.flatten()
        x = x.flatten()
        
        x_fitted = []
        y_fitted = []
        amplitude_fitted = []
        intensity_fitted = []
        fwhm_fitted = []
        
        segment_radius = self.segment_radius
        ny, nx = data.shape
        
        initial_guess = (30.0, segment_radius, segment_radius, 1.0, 1.0, 0.0, avepixel_mean)
        
        for yi, xi in zip(y, x):
            y_min = int(yi - segment_radius)
            y_max = int(yi + segment_radius)
            x_min = int(xi - segment_radius)
            x_max = int(xi + segment_radius)
            
            if y_min < 0 or x_min < 0 or y_max > ny or x_max > nx:
                continue
                
            star_seg = data[y_min:y_max, x_min:x_max]
            y_ind, x_ind = np.indices(star_seg.shape)
            saturation = np.max(data) * 2 * np.ones_like(y_ind) # Prevent clamping
            
            try:
                import warnings
                from scipy.optimize import OptimizeWarning
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', OptimizeWarning)
                    popt, pcov = opt.curve_fit(twoDGaussian, (y_ind, x_ind, saturation), star_seg.ravel(), 
                                               p0=initial_guess, maxfev=200)
            except RuntimeError:
                continue
                
            amplitude, yo, xo, sigma_y, sigma_x, theta, offset = popt
            
            if min(sigma_y/sigma_x, sigma_x/sigma_y) < self.roundness_threshold:
                continue
                
            if (4*sigma_x*sigma_y/segment_radius**2 > self.max_feature_ratio):
                continue
                
            crop_y_min = int(yo - 3*sigma_y) + 1
            if crop_y_min < 0: crop_y_min = 0
            crop_y_max = int(yo + 3*sigma_y) + 1
            if crop_y_max >= star_seg.shape[0]: crop_y_max = star_seg.shape[0] - 1
            crop_x_min = int(xo - 3*sigma_x) + 1
            if crop_x_min < 0: crop_x_min = 0
            crop_x_max = int(xo + 3*sigma_x) + 1
            if crop_x_max >= star_seg.shape[1]: crop_x_max = star_seg.shape[1] - 1
            
            if (y_max - y_min) < 3:
                crop_y_min = int(yo - 2)
                crop_y_max = int(yo + 2)
            if (x_max - x_min) < 3:
                crop_x_min = int(xo - 2)
                crop_x_max = int(xo + 2)
                
            star_seg_crop = star_seg[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            if star_seg_crop.shape[0] == 0 or star_seg_crop.shape[1] == 0:
                continue
                
            # Gamma correct the star segment and background offset
            star_seg_crop_corr = gamma_correction(star_seg_crop.astype(np.float32), self.gamma)
            offset_corr = gamma_correction(offset, self.gamma)
            
            intensity = np.sum(star_seg_crop_corr - offset_corr)
            if intensity <= 0:
                continue
                
            sigma_fitted = np.sqrt(sigma_x**2 + sigma_y**2)
            fwhm_val = 2.355 * sigma_fitted
            
            x_fitted.append(x_min + xo)
            y_fitted.append(y_min + yo)
            amplitude_fitted.append(amplitude)
            intensity_fitted.append(intensity)
            fwhm_fitted.append(fwhm_val)
            
        if not x_fitted:
            return None
            
        t = Table()
        t['id'] = np.arange(1, len(x_fitted) + 1)
        t['xcentroid'] = x_fitted
        t['ycentroid'] = y_fitted
        t['flux'] = intensity_fitted
        t['peak'] = amplitude_fitted
        t['rms_fwhm'] = fwhm_fitted
        return t

def process_single_star(i, x, y, mag_instr, cutout, std, wcs=None):
    """Worker function to process a single star's morphology and coordinates."""
    try:
        # Get RA/Dec if WCS is available
        ra, dec = -999.0, -999.0
        local_scale = None
        fwhm_arcsec = -999.0
        
        if wcs:
            try:
                sky_coord = wcs.pixel_to_world(x, y)
                ra, dec = float(sky_coord.ra.deg), float(sky_coord.dec.deg)
                
                # Sample local pixel scale
                sky1 = sky_coord
                sky2 = wcs.pixel_to_world(x + 1, y)
                sky3 = wcs.pixel_to_world(x, y + 1)
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


def load_image(file_path):
    """Loads FITS or PNG image and returns data as a 2D numpy array."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.fits', '.fit']:
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            # FITS data can be 3D (e.g., [1, Y, X]), take the first slice if so
            if data.ndim == 3:
                data = data[0]
            header = hdul[0].header
            return data.astype(float), header
    elif ext in ['.png', '.bmp']:
        img = Image.open(file_path).convert('L')
        data = np.array(img).astype(float)
        header = None
        
        # Check if background is light (e.g., inverted image)
        # If median is closer to max than min, it's likely light background
        if np.median(data) > 127:
            print("Detected light background image. Inverting for processing...")
            data = np.max(data) - data
            
        return data, header
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def solve_astrometry(image_path, sources=None, width=None, height=None, api_key=None):
    """Solves astrometry using astrometry.net via astroquery."""
    try:
        from astroquery.astrometry_net import AstrometryNet
    except ImportError:
        print("astroquery is not installed. Astrometry will be skipped.")
        return None

    ast = AstrometryNet()
    # Set a local cache directory to avoid permission issues
    ast.cache_location = os.path.join(os.getcwd(), '.astrometry_cache')
    if not os.path.exists(ast.cache_location):
        os.makedirs(ast.cache_location)

    if api_key:
        ast.api_key = api_key
    elif 'ASTROMETRY_NET_API_KEY' in os.environ:
        ast.api_key = os.environ['ASTROMETRY_NET_API_KEY']
    else:
        ast.api_key = 'aifriketqrtctpor'

    try:
        from astroquery.exceptions import LoginError, TimeoutError
    except ImportError:
        # Fallback for older versions or different import structures
        class LoginError(Exception): pass
        class TimeoutError(Exception): pass

    try:
        if sources is not None and width is not None and height is not None:
            num_sources = min(len(sources), 100)
            if num_sources < 10:
                print(f"Warning: Only {num_sources} stars found. Astrometry may fail without more stars.")
            
            print(f"Submitting top {num_sources} sources to Astrometry.net...")
            sorted_indices = np.argsort(sources['peak'])[::-1]
            x_sorted = sources['xcentroid'][sorted_indices][:num_sources]
            y_sorted = sources['ycentroid'][sorted_indices][:num_sources]
            
            wcs_header = ast.solve_from_source_list(x_sorted, y_sorted, width, height, 
                                                   solve_timeout=300,
                                                   scale_units='arcsecperpix',
                                                   scale_lower=0.1,
                                                   scale_upper=100.0)
        else:
            print("Submitting full image to Astrometry.net (this may be slower)...")
            wcs_header = ast.solve_from_image(image_path, solve_timeout=300,
                                             scale_units='arcsecperpix',
                                             scale_lower=0.1,
                                             scale_upper=100.0)
            
        if wcs_header:
            from astropy.wcs import WCS
            return WCS(wcs_header)
        else:
            print("!!! Astrometry failed: The field could not be solved by Astrometry.net.")
            print("    Possible reasons: fuzzy stars, wrong scale hints, or wrong coordinates.")
    except LoginError:
        print("!!! Astrometry failed: Invalid API key. Please check your key at nova.astrometry.net.")
    except TimeoutError:
        print("!!! Astrometry failed: The connection to Astrometry.net timed out.")
    except Exception as e:
        err_str = str(e)
        print(f"!!! Astrometry failed with error: {err_str}")
        if "RemoteDisconnected" in err_str or "Max retries exceeded" in err_str:
            print("    Network Issue: Could not connect to nova.astrometry.net. Please check your internet connection or server status.")
        elif "api_key" in err_str.lower():
            print("    API Key Issue: Ensure your API key is valid and has not expired.")
        else:
            print("    Hint: Check if the image has enough sharp stars and the field is not too crowded/sparse.")
    
    return None

def query_catalog_tiled(center_sky, radius_deg, catalog_name="gaia", tile_size_deg=2.0):
    """
    Queries a catalog using a tiled approach to cover large fields without server timeouts.
    """
    from astropy.table import vstack, unique
    
    # Calculate the number of tiles needed
    # We use a grid that covers a square of 2*radius x 2*radius
    num_tiles_side = int(np.ceil((2 * radius_deg) / tile_size_deg))
    if num_tiles_side <= 1:
        # Single query is enough
        return _perform_single_query(center_sky, radius_deg, catalog_name)
    
    print(f"Field is large ({radius_deg:.2f} deg radius). Dividing into {num_tiles_side}x{num_tiles_side} tiles...")
    
    all_results = []
    # Grid spacing (with 10% overlap)
    spacing = tile_size_deg * 0.9
    
    # Start from center - radius and go to center + radius
    # Note: This is an approximation for RA/Dec but sufficient for tiling a small-ish region of the sky
    # For very large regions or near poles, this would need more care.
    ra_start = center_sky.ra.deg - radius_deg
    dec_start = center_sky.dec.deg - radius_deg
    
    for i in range(num_tiles_side):
        for j in range(num_tiles_side):
            # Calculate tile center
            tile_ra = ra_start + (i + 0.5) * spacing
            tile_dec = dec_start + (j + 0.5) * spacing
            
            # Wrap RA
            tile_ra = tile_ra % 360.0
            # Clip Dec
            tile_dec = np.clip(tile_dec, -90.0, 90.0)
            
            tile_center = SkyCoord(ra=tile_ra, dec=tile_dec, unit=(u.deg, u.deg))
            # Search radius for each tile covers the tile area
            tile_radius = (tile_size_deg / np.sqrt(2)) * 1.1 # slightly larger to ensure coverage
            
            res = _perform_single_query(tile_center, tile_radius, catalog_name)
            if res is not None and len(res) > 0:
                all_results.append(res)
    
    if not all_results:
        return None
    
    combined = vstack(all_results)
    # Remove duplicates based on coordinates (rounded to avoid precision issues)
    # Different catalogs have different ID columns, so coordinates are a safe bet.
    ra_col = combined['ra'] if 'ra' in combined.colnames else combined['RA(ICRS)']
    dec_col = combined['dec'] if 'dec' in combined.colnames else combined['DE(ICRS)']
    combined['tmp_ra_round'] = np.round(np.asarray(ra_col), 6)
    combined['tmp_dec_round'] = np.round(np.asarray(dec_col), 6)
    
    final_table = unique(combined, keys=['tmp_ra_round', 'tmp_dec_round'])
    final_table.remove_columns(['tmp_ra_round', 'tmp_dec_round'])
    
    return final_table

def process_image(image_path, args, figures_dir, csvs_dir):
    """Processes a single image file."""
    print(f"\n--- Processing: {os.path.basename(image_path)} ---")
    individuals_dir = os.path.join(figures_dir, 'Individuals')
    try:
        data, header = load_image(image_path)
        ny, nx = data.shape
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return

    # Background subtraction
    if args.sky:
        print("Estimating local background...")
        bkg_estimator = MedianBackground()
        # box size of ~50 is a reasonable default for star fields
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        mean, median, std = np.median(bkg.background), np.median(bkg.background_median), np.median(bkg.background_rms)
        data_sub = data - bkg.background
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median

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
            print(f"Updated fluxes for {np.sum(match_mask)} out of {len(sources)} stars using {secondary_finder.upper()}.")

    print(f"Found {len(sources)} stars.")

    # Solve Astrometry
    wcs = None
    if args.astrometry:
        print("Attempting to solve astrometry...")
        ext_check = os.path.splitext(image_path)[1].lower()
        if ext_check not in ['.fits', '.fit']:
            temp_fits = os.path.join(individuals_dir, f'temp_astrometry_{os.path.basename(image_path)}.fits')
            from astropy.io import fits
            fits.writeto(temp_fits, data, overwrite=True)
            wcs = solve_astrometry(temp_fits, sources, api_key=args.api_key)
            try:
                os.remove(temp_fits)
            except OSError:
                pass
        else:
            wcs = solve_astrometry(image_path, sources, api_key=args.api_key)
            
        if not wcs:
            print("!!! Astrometry failed. Some features will be limited.")

    # Convert sources to pandas for easier handling
    df = sources.to_pandas()
    df['mag_instr'] = -2.5 * np.log10(df['flux'])
    df['mag_abs'] = np.nan
    df['fwhm_arc'] = np.nan
    
    # If the finding method produced its own fwhm, save it to be restored after morphology
    had_rms_fwhm = 'rms_fwhm' in df.columns

    # Initial clip: Only stars with positive flux (mag < 0)
    # The flux output by the finder should be strictly positive, making mag_instr < 0 when there's enough flux
    # Wait, mag_instr < 0 means flux > 1. Some faint stars might have flux < 1, giving mag > 0.
    # Let's keep the existing logic.
    count_before = len(df)
    # df = df[df['mag_instr'] < 0].copy() # Existing logic... Wait! If flux < 1, mag > 0. Actually the code clips mag_instr < 0.
    df = df[df['mag_instr'] < 0].copy()
    num_clipped = count_before - len(df)
    if num_clipped > 0:
        print(f"Clipped {num_clipped} stars with non-positive flux. {len(df)} stars remaining.")

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

    # If RMS finder was used, overwrite the FWHM morphology with the RMS fitted fwhm
    if had_rms_fwhm:
        df['fwhm'] = df['rms_fwhm']

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

    # Absolute Photometry
    zero_point = None
    if args.absolute:
        if not wcs:
            print("!!! Absolute photometry skipped: Astrometry must be solved first. Use --astrometry.")
        else:
            from astropy.coordinates import SkyCoord
            center_sky = wcs.pixel_to_world(df['xcentroid'].mean(), df['ycentroid'].mean())
            # Estimate radius needed to cover the image
            p0 = wcs.pixel_to_world(0, 0)
            p1 = wcs.pixel_to_world(nx, ny)
            radius = p0.separation(p1).deg / 2.0 * 1.2
            
            print(f"Performing absolute photometry via {args.catalog.upper()} catalog...")
            catalog_results = query_catalog_tiled(center_sky, radius, args.catalog)
            
            if catalog_results is not None and len(catalog_results) > 0:
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
                star_coords = wcs.pixel_to_world(df['xcentroid'], df['ycentroid'])
                
                # Match
                # We use a 5 pixel match limit. Calculate that in arcsec based on image scale
                scale_guess = wcs.pixel_to_world(0,0).separation(wcs.pixel_to_world(1,0)).arcsec
                match_limit_arcsec = 5 * scale_guess
                print(f"Matching stars with a limit of 5 pixels ({match_limit_arcsec:.2f} arcsec)...")
                
                idx, d2d, _ = star_coords.match_to_catalog_sky(catalog_coords)
                matches_mask = d2d < match_limit_arcsec * u.arcsec
                
                if matches_mask.any():
                    matched_df = df[matches_mask].copy()
                    matched_cat_mag = cat_mag[idx[matches_mask]]
                    
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
                print(f"{args.catalog.upper()} query returned no sources in this field.")

    # Save Results
    results_csv = os.path.join(csvs_dir, f'{os.path.basename(image_path)}_results.csv')
    df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

    # Generate Catalog Plots from the generic CSV ensuring we capture the explicit population
    if args.absolute and os.path.exists(results_csv):
        csv_df = pd.read_csv(results_csv)
        if 'mag_abs' in csv_df.columns and 'mag_instr' in csv_df.columns:
            valid_mags = csv_df.dropna(subset=['mag_abs', 'mag_instr'])
            if not valid_mags.empty:
                plt.figure(figsize=(10, 6))
                plt.scatter(valid_mags['mag_instr'], valid_mags['mag_abs'], c='blue', s=10, alpha=0.5, label='All Extracted Stars')
                
                # Plot the theoretical fit line
                x_range = np.linspace(valid_mags['mag_instr'].min(), valid_mags['mag_instr'].max(), 100)
                # Since mag_abs = mag_instr + ZP, plot x_range vs x_range + ZP. ZP isn't easily grabbed here, but we can compute it
                zp_calc = valid_mags['mag_abs'].iloc[0] - valid_mags['mag_instr'].iloc[0]
                plt.plot(x_range, x_range + zp_calc, 'r--', alpha=0.8, label=f'Fit (ZP={zp_calc:.2f})')
                
                plt.xlabel("Instrumental Magnitude")
                plt.ylabel("Absolute Magnitude")
                plt.title(f"Absolute Photometry Overview - {os.path.basename(image_path)}")
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                
                mag_plot_name = os.path.join(individuals_dir, f'{os.path.basename(image_path)}_mag_comparison.png')
                plt.savefig(mag_plot_name, bbox_inches='tight')
                plt.close()
                
                # Also generate the abs_mag_hist plot since it's in the README
                plt.figure(figsize=(10, 6))
                plt.hist(valid_mags['mag_abs'], bins=30, color='purple', alpha=0.7, edgecolor='black')
                plt.xlabel("Absolute Magnitude")
                plt.ylabel("Count")
                plt.title(f"Absolute Magnitude Distribution - {os.path.basename(image_path)}")
                plt.grid(axis='y', alpha=0.3)
                hist_plot_name = os.path.join(individuals_dir, f'{os.path.basename(image_path)}_abs_mag_hist.png')
                plt.savefig(hist_plot_name, bbox_inches='tight')
                plt.close()

    # Visualizations
    print("Generating figures...")
    
    # Simple star field plot
    plt.figure(figsize=(10, 8))
    plt.imshow(data, origin='lower', cmap='gray', vmax=np.percentile(data, 99))
    plt.scatter(df['xcentroid'], df['ycentroid'], s=20, edgecolor='red', facecolor='none', alpha=0.5)
    plt.title(f"Detected Stars - {os.path.basename(image_path)}")
    plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_detected.png'), bbox_inches='tight')
    plt.close()

    # Interpolated Maps
    if len(df) > 10:
        print("Generating interpolated maps...")
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[0:nx:100j, 0:ny:100j]
        
        # FWHM Map
        mean_fwhm = np.nanmean(df['fwhm'])
        std_fwhm = np.nanstd(df['fwhm'])
        fwhm_vmin = mean_fwhm - (2 * std_fwhm)
        fwhm_vmax = mean_fwhm + (2 * std_fwhm)
        
        plt.figure(figsize=(10, 8))
        grid_fwhm = griddata((df['xcentroid'], df['ycentroid']), df['fwhm'], (grid_x, grid_y), method='cubic')
        ax = plt.gca()
        im = ax.imshow(grid_fwhm.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis', vmin=fwhm_vmin, vmax=fwhm_vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='FWHM (pixels)')
        ax.set_title(f"FWHM Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_fwhm_map.png'), bbox_inches='tight')
        plt.close()

        if wcs:
            print("Generating PSF arcsec map (vectorized)...")
            # Vectorized local pixel scale calculation
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
            
            mean_arc = np.nanmean(df['fwhm_arc'])
            std_arc = np.nanstd(df['fwhm_arc'])
            arc_vmin = mean_arc - (2 * std_arc)
            arc_vmax = mean_arc + (2 * std_arc)
            
            plt.figure(figsize=(10, 8))
            grid_fwhm_arc = griddata((df['xcentroid'], df['ycentroid']), df['fwhm_arc'], (grid_x, grid_y), method='cubic')
            ax = plt.gca()
            im = ax.imshow(grid_fwhm_arc.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis', vmin=arc_vmin, vmax=arc_vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='FWHM (arcsec)')
            ax.set_title(f"PSF Size Map (arcsec) - {os.path.basename(image_path)}")
            plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_psf_arcsec_map.png'), bbox_inches='tight')
            plt.close()

        # Theta Map
        plt.figure(figsize=(10, 8))
        grid_theta = griddata((df['xcentroid'], df['ycentroid']), df['theta'], (grid_x, grid_y), method='cubic')
        ax = plt.gca()
        im = ax.imshow(grid_theta.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Theta (degrees)')
        ax.set_title(f"PSF Orientation Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_theta_map.png'), bbox_inches='tight')
        plt.close()

    if wcs:
        print("Generating distortion map...")
        # Create a grid of points for WCS sampling
        gy, gx = np.mgrid[0:ny:20j, 0:nx:20j]
        scales = np.zeros_like(gx)
        
        # Calculate reference scale at the image center (zero distortion reference)
        cx, cy = nx / 2, ny / 2
        s1_c = wcs.pixel_to_world(cx, cy)
        s2_c = wcs.pixel_to_world(cx + 1, cy)
        s3_c = wcs.pixel_to_world(cx, cy + 1)
        center_scale_x = s1_c.separation(s2_c).arcsec
        center_scale_y = s1_c.separation(s3_c).arcsec
        center_scale_avg = np.sqrt(center_scale_x * center_scale_y)
        
        # Vectorized grid sampling
        all_gx = gx.flatten()
        all_gy = gy.flatten()
        
        sky1_all = wcs.pixel_to_world(all_gx, all_gy)
        sky2_all = wcs.pixel_to_world(all_gx + 1, all_gy)
        sky3_all = wcs.pixel_to_world(all_gx, all_gy + 1)
        
        scale_x = sky1_all.separation(sky2_all).arcsec
        scale_y = sky1_all.separation(sky3_all).arcsec
        scales = np.sqrt(scale_x * scale_y).reshape(gx.shape)
        
        # Calculate the gradient of the scale field
        grad_y, grad_x = np.gradient(scales)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.where(grad_mag == 0, 1e-10, grad_mag)
        dir_x = -grad_x / grad_mag
        dir_y = -grad_y / grad_mag
        dist_mag = np.abs(scales - center_scale_avg)
        u_raw = dist_mag * dir_x
        v_raw = dist_mag * dir_y
        
        raw_mag = np.sqrt(u_raw**2 + v_raw**2)
        max_raw = np.max(raw_mag)
        
        if max_raw > 0:
            u_comp = (u_raw / max_raw) * 75
            v_comp = (v_raw / max_raw) * 75
            key_length_px = 50
            key_length_arcsec = (50.0 / 75.0) * max_raw
        else:
            u_comp = u_raw
            v_comp = v_raw
            key_length_arcsec = 0.0
            key_length_px = 50 
        
        # Interpolate heatmap
        grid_x_f, grid_y_f = np.mgrid[0:nx:100j, 0:ny:100j]
        grid_dist = griddata((gx.flatten(), gy.flatten()), scales.flatten(), 
                             (grid_x_f, grid_y_f), method='cubic')
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        im = ax.imshow(grid_dist.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        q = ax.quiver(gx, gy, u_comp, v_comp, color='white', alpha=0.8, 
                      scale_units='xy', angles='xy', scale=1.0,
                      headwidth=2.25, headlength=3.75, headaxislength=3.375)
        ax.quiverkey(q, 0.9, 1.05, 50, f'{key_length_arcsec:.3f}"/pix relative to center', 
                     labelpos='E', coordinates='axes', color='black')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Local Pixel Scale (arcsec/pixel)')
        ax.set_title(f"Astrometric Distortion Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_distortion_map.png'), bbox_inches='tight')
        plt.close()

    # Summary Figure
    print("Generating summary overlay figure...")
    plt.figure(figsize=(12, 10))
    plt.imshow(data, origin='lower', cmap='gray', vmax=np.percentile(data, 99))
    
    # Draw circles around stars: radius = 2 * FWHM_pixels
    for _, row in df.iterrows():
        circ = plt.Circle((row['xcentroid'], row['ycentroid']), 2 * row['fwhm'], 
                          color='green', fill=False, linewidth=0.8, alpha=0.6)
        plt.gca().add_patch(circ)

    # Draw red circles around outlier stars
    if not df_outliers.empty:
        for _, row in df_outliers.iterrows():
            circ = plt.Circle((row['xcentroid'], row['ycentroid']), 2 * row['fwhm'], 
                              color='red', fill=False, linewidth=0.8, alpha=0.6)
            plt.gca().add_patch(circ)

    # Info box
    fwhm_px = df['fwhm'].values
    fwhm_arc = df['fwhm_arc'].values
    def fmt_stats(px, arc, func):
        v_px = func(px)
        v_arc = func(arc)
        if np.isnan(v_arc):
            return f"{v_px:.2f} px (N/A)"
        return f"{v_px:.2f} px ({v_arc:.2f}\")"

    if wcs:
        if 'all_local_scales' in locals():
            avg_scale = np.mean(all_local_scales)
        elif 'scales' in locals():
            avg_scale = np.mean(scales)
        else:
            try:
                sky1_ref = wcs.pixel_to_world(nx/2, ny/2)
                sky2_ref = wcs.pixel_to_world(nx/2 + 1, ny/2)
                sky3_ref = wcs.pixel_to_world(nx/2, ny/2 + 1)
                avg_scale = np.sqrt(sky1_ref.separation(sky2_ref).arcsec * sky1_ref.separation(sky3_ref).arcsec)
            except: avg_scale = -999.0
    else: avg_scale = -999.0

    info_text = (
        f"Image: {os.path.basename(image_path)}\n"
        f"Size: {nx} x {ny} px\n"
        f"Scale: {avg_scale:.3f}\"/pix\n"
        f"Zero Point: {zero_point if zero_point else 'N/A'}\n\n"
        f"FWHM:\n"
        f"  Min: {fmt_stats(fwhm_px, fwhm_arc, np.min)}\n"
        f"  Max: {fmt_stats(fwhm_px, fwhm_arc, np.max)}\n"
        f"  Avg: {fmt_stats(fwhm_px, fwhm_arc, np.mean)}\n"
    )
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10, fontfamily='monospace')
    
    plt.title(f"Star Detection Summary - {os.path.basename(image_path)}")
    plt.savefig(os.path.join(figures_dir, f'{os.path.basename(image_path)}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

def _perform_single_query(center_sky, radius_deg, catalog_name):
    """Helper for a single catalog query."""
    try:
        if catalog_name == "gaia":
            from astroquery.gaia import Gaia
            job = Gaia.cone_search_async(center_sky, radius=radius_deg * u.deg)
            return job.get_results()
        elif catalog_name == "tycho2":
            from astroquery.vizier import Vizier
            v = Vizier(catalog=['I/259/tyc2'])
            v.ROW_LIMIT = -1
            viz_results = v.query_region(center_sky, radius=radius_deg * u.deg)
            if viz_results:
                return viz_results[0]
    except Exception as e:
        print(f"  Query at {center_sky.to_string('hmsdms')} failed: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Find and measure stars.")
    parser.add_argument("image", help="Path to an image file or a directory of images.")
    parser.add_argument("--astrometry", action="store_true")
    parser.add_argument("--api-key", default="aifriketqrtctpor")
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction factor for RMS finder (default 1.0).")
    parser.add_argument("--finder", nargs="+", default=["dao"], 
                        help="Star detection algorithm (dao, iraf, or rms). For two-pass photometry, provide two arguments (e.g. 'rms iraf' or 'rms dao').")
    parser.add_argument("--sharplo", type=float, default=0.2, help="Lower bound for sharpness (default 0.2).")
    parser.add_argument("--sharphi", type=float, default=1.0, help="Upper bound for sharpness (default 1.0).")
    parser.add_argument("--roundlo", type=float, default=-1.0, help="Lower bound for roundness (default -1.0).")
    parser.add_argument("--roundhi", type=float, default=1.0, help="Upper bound for roundness (default 1.0).")
    parser.add_argument("--exclude-border", action="store_true", help="Exclude stars found near image border.")
    parser.add_argument("--sky", action="store_true", help="Use local background estimation instead of global median.")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() - 2), help="Cores for general parallel tasks.")
    parser.add_argument("--max-image-workers", type=int, default=None, help="Max parallel images (defaults to min(cores, 4) to prevent out-of-memory).")
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--catalog", choices=["gaia", "tycho2"], default="gaia")
    args = parser.parse_args()

    # Determine input type
    image_list = []
    if os.path.isdir(args.image):
        print(f"Directory detected: {args.image}")
        extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.fits', '.fit')
        for f in os.listdir(args.image):
            if f.lower().endswith(extensions):
                image_list.append(os.path.join(args.image, f))
        
        output_base = args.image
        image_list.sort()
    else:
        if os.path.exists(args.image):
            image_list = [args.image]
            output_base = os.path.dirname(args.image) if os.path.dirname(args.image) else "."
        else:
            print(f"Error: Path {args.image} does not exist.")
            return

    if not image_list:
        print("No valid images found to process.")
        return

    # Create output directories
    figures_dir = os.path.join(output_base, 'Figures')
    csvs_dir = os.path.join(output_base, 'CSVs')
    
    for d in [figures_dir, csvs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Subdirectory for individual plot maps
    individuals_dir = os.path.join(figures_dir, 'Individuals')
    if not os.path.exists(individuals_dir):
        os.makedirs(individuals_dir)

    print(f"Found {len(image_list)} images to process.")
    
    image_workers = args.max_image_workers if args.max_image_workers is not None else min(args.cores, 4)
    
    if image_workers > 1 and len(image_list) > 1:
        print(f"Processing images in parallel using {image_workers} workers...")
        # Mark that we are in a parallel image loop to avoid nested pools
        os.environ["IMAGE_POOL"] = "1"
        with concurrent.futures.ProcessPoolExecutor(max_workers=image_workers) as executor:
            # We must use wraps or partial to pass extra args
            from functools import partial
            worker = partial(process_image, args=args, figures_dir=figures_dir, csvs_dir=csvs_dir)
            list(executor.map(worker, image_list))
    else:
        for img in image_list:
            process_image(img, args, figures_dir, csvs_dir)

    print("\nBatch processing complete.")

    if args.absolute:
        print("Generating composite magnitude plot...")
        all_dfs = []
        for img in image_list:
            csv_path = os.path.join(csvs_dir, f'{os.path.basename(img)}_results.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if 'mag_instr' in df.columns and 'mag_abs' in df.columns:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            valid_df = combined_df.dropna(subset=['mag_instr', 'mag_abs'])
            
            if not valid_df.empty:
                plt.figure(figsize=(10, 6))
                plt.scatter(valid_df['mag_instr'], valid_df['mag_abs'], alpha=0.1, s=5, c='blue', label='All Matched Stars')
                
                median_zp = (valid_df['mag_abs'] - valid_df['mag_instr']).median()
                x_range = np.linspace(valid_df['mag_instr'].min(), valid_df['mag_instr'].max(), 100)
                plt.plot(x_range, x_range + median_zp, 'r--', alpha=0.8, label=f'Fit (Median ZP={median_zp:.2f})')
                
                plt.xlabel("Instrumental Magnitude")
                plt.ylabel("Absolute Magnitude")
                plt.title("Composite Absolute Photometry - All Images")
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                
                stats_text = f"Total Matched: {len(valid_df)}\nMedian ZP: {median_zp:.3f}"
                plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                composite_path = os.path.join(figures_dir, "composite_mag_comparison.png")
                plt.savefig(composite_path, bbox_inches='tight')
                plt.close()
                print(f"Composite plot saved to {composite_path}")

                # Composite Histogram
                plt.figure(figsize=(10, 6))
                plt.hist(valid_df['mag_abs'], bins=40, color='purple', alpha=0.7, edgecolor='black')
                plt.xlabel("Absolute Magnitude")
                plt.ylabel("Frequency")
                plt.title("Composite Absolute Magnitude Distribution - All Images")
                plt.grid(True, alpha=0.3)
                
                # Add stats box for the histogram
                histo_stats = (
                    f"Count: {len(valid_df)}\n"
                    f"Min: {valid_df['mag_abs'].min():.2f}\n"
                    f"Max: {valid_df['mag_abs'].max():.2f}\n"
                    f"Median: {valid_df['mag_abs'].median():.2f}"
                )
                plt.text(0.95, 0.95, histo_stats, transform=plt.gca().transAxes, 
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                composite_hist_path = os.path.join(figures_dir, "composite_mag_histogram.png")
                plt.savefig(composite_hist_path, bbox_inches='tight')
                plt.close()
                print(f"Composite histogram saved to {composite_hist_path}")
            else:
                print("No valid magnitude data found across images for composite plot.")

if __name__ == "__main__":
    main()
