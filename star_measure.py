import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.morphology import data_properties
from PIL import Image

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

def main():
    parser = argparse.ArgumentParser(description="Find and measure stars in an image.")
    parser.add_argument("image", help="Path to FITS or PNG image.")
    parser.add_argument("--astrometry", action="store_true", help="Attempt to solve astrometry via astrometry.net")
    parser.add_argument("--api-key", help="Astrometry.net API key (default: aifriketqrtctpor)")
    parser.add_argument("--fwhm", type=float, default=3.0, help="Estimated FWHM in pixels (default 3.0)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Source detection threshold in sigma (default 5.0)")
    args = parser.parse_args()

    # Create Figures directory
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    print(f"Processing image: {args.image}")
    data, header = load_image(args.image)

    # Basic stats for background subtraction
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    data_sub = data - median

    # Star Detection
    print("Finding stars...")
    daofind = DAOStarFinder(fwhm=args.fwhm, threshold=args.threshold * std)
    sources = daofind(data_sub)

    if sources is None or len(sources) == 0:
        print("No stars found.")
        return

    print(f"Found {len(sources)} stars.")

    # Astrometry
    wcs = None
    if args.astrometry:
        print("Attempting to solve astrometry...")
        ny, nx = data.shape
        wcs = solve_astrometry(args.image, sources=sources, width=nx, height=ny, api_key=args.api_key)
        if wcs:
            print("Astrometry solved successfully.")
        else:
            print("Astrometry matching failed or skipped.")

    # Measurements
    results = []
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=args.fwhm * 1.5)
    annulus_aperture = CircularAnnulus(positions, r_in=args.fwhm * 2, r_out=args.fwhm * 3)
    
    # Photometry
    print("Performing photometry...")
    phot_table = aperture_photometry(data_sub, apertures)
    
    # Background estimation for photometry (simple local background subtraction already done globally but refined here)
    # Ensure flux is a plain array for log calculation
    flux = phot_table['aperture_sum']
    if hasattr(flux, 'value'):
        flux = flux.value
    
    # mag = -2.5 * log10(flux)
    mag_instr = -2.5 * np.log10(np.maximum(flux, 1e-6))
    phot_table['mag_instr'] = mag_instr

    # Morphology (FWHM, Elongation, Angle)
    print("Measuring morphology...")
    for i, row in enumerate(sources):
        x, y = row['xcentroid'], row['ycentroid']
        
        # Get RA/Dec if WCS is available
        ra, dec = -999, -999
        if wcs:
            try:
                sky_coord = wcs.pixel_to_world(x, y)
                if hasattr(sky_coord.ra, 'deg'):
                    ra, dec = sky_coord.ra.deg, sky_coord.dec.deg
                else:
                    ra, dec = sky_coord.ra, sky_coord.dec
            except:
                pass

        # Estimate local morphology (FWHM, elongation, theta)
        size = int(args.fwhm * 5)
        x_min, x_max = max(0, int(x - size)), min(data.shape[1], int(x + size))
        y_min, y_max = max(0, int(y - size)), min(data.shape[0], int(y + size))
        cutout = data_sub[y_min:y_max, x_min:x_max]
        
        try:
            props = data_properties(cutout)
            # Use getattr to safely get values from Quantities
            fwhm = props.fwhm
            if hasattr(fwhm, 'value'): fwhm = fwhm.value
            
            elongation = props.elongation
            if hasattr(elongation, 'value'): elongation = elongation.value
            
            theta = props.orientation
            if hasattr(theta, 'deg'): 
                theta = theta.deg
            elif hasattr(theta, 'value'):
                theta = theta.value
            
            # Final conversion to float, handling any remaining Quantity issues
            fwhm = float(np.array(fwhm))
            elongation = float(np.array(elongation))
            theta = float(np.array(theta))
            
        except Exception as e:
            if i < 5: # Only print for first 5 stars to avoid spam
                print(f"Morphology failed for star at ({x:.1f}, {y:.1f}): {e}")
            fwhm, elongation, theta = -999, -999, -999

        results.append({
            'x': float(x),
            'y': float(y),
            'ra': float(ra),
            'dec': float(dec),
            'mag_instr': float(mag_instr[i]),
            'fwhm': float(fwhm),
            'elongation': float(elongation),
            'theta': float(theta)
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_name = f"{args.image}_results.csv"
    df.to_csv(csv_name, index=False)
    print(f"Results saved to {csv_name}")

    # Plotting
    print("Generating figures...")
    
    # Histogram of instrumental magnitudes
    plt.figure(figsize=(8, 6))
    mag_data = df['mag_instr'].values
    mag_data = mag_data[np.isfinite(mag_data)]
    if len(mag_data) > 0:
        plt.hist(mag_data, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Instrumental Magnitudes - {os.path.basename(args.image)}")
        plt.xlabel("Instrumental Magnitude")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join('Figures', f'{os.path.basename(args.image)}_mag_hist.png'))
    plt.close()

    # Histogram of direction of elongation
    plt.figure(figsize=(8, 6))
    theta_data = df[df['theta'] != -999]['theta'].values
    theta_data = theta_data[np.isfinite(theta_data)]
    if len(theta_data) > 0:
        plt.hist(theta_data, bins=30, color='salmon', edgecolor='black')
        plt.title(f"Direction of Elongation - {os.path.basename(args.image)}")
        plt.xlabel("Orientation (degrees)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join('Figures', f'{os.path.basename(args.image)}_elong_angle_hist.png'))
    plt.close()

    # Interpolated Maps
    # Filter for valid measurements
    valid_mask = (df['fwhm'] != -999) & (df['theta'] != -999)
    valid_df = df[valid_mask]

    if len(valid_df) >= 4: # Need at least 4 points for cubic interpolation
        print("Generating interpolated maps...")
        ny, nx = data.shape
        grid_x, grid_y = np.mgrid[0:nx:100j, 0:ny:100j] # 100x100 resolution for speed
        
        # PSF Size Map
        grid_fwhm = griddata((valid_df['x'], valid_df['y']), valid_df['fwhm'], 
                             (grid_x, grid_y), method='linear')
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_fwhm.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        plt.colorbar(im, label='FWHM (pixels)')
        plt.scatter(valid_df['x'], valid_df['y'], c='red', s=5, alpha=0.3)
        plt.title(f"Interpolated PSF Size (FWHM) - {os.path.basename(args.image)}")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.savefig(os.path.join('Figures', f'{os.path.basename(args.image)}_psf_size_map.png'))
        plt.close()

        # Theta Map
        grid_theta = griddata((valid_df['x'], valid_df['y']), valid_df['theta'], 
                              (grid_x, grid_y), method='linear')
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_theta.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        plt.colorbar(im, label='Orientation (degrees)')
        plt.scatter(valid_df['x'], valid_df['y'], c='red', s=5, alpha=0.3)
        plt.title(f"Interpolated PSF Orientation (Theta) - {os.path.basename(args.image)}")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.savefig(os.path.join('Figures', f'{os.path.basename(args.image)}_theta_map.png'))
        plt.close()
    else:
        print("Insufficient valid stars for interpolation mapping.")

    # Distortion Map (Local Pixel Scale)
    if wcs:
        print("Generating distortion map...")
        ny, nx = data.shape
        # Create a grid for sampling the WCS
        gy, gx = np.mgrid[0:ny:20j, 0:nx:20j]
        scales = np.zeros_like(gx)
        
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                # Sample local pixel scale
                # We move 1 pixel in X and Y to find the local scale
                x0, y0 = gx[i, j], gy[i, j]
                sky1 = wcs.pixel_to_world(x0, y0)
                sky2 = wcs.pixel_to_world(x0 + 1, y0)
                sky3 = wcs.pixel_to_world(x0, y0 + 1)
                
                # Separation is in degrees, convert to arcsec
                scale_x = sky1.separation(sky2).arcsec
                scale_y = sky1.separation(sky3).arcsec
                scales[i, j] = np.sqrt(scale_x * scale_y)
        
        # Interpolate sampled scales to a finer grid for plotting
        grid_x_f, grid_y_f = np.mgrid[0:nx:100j, 0:ny:100j]
        grid_dist = griddata((gx.flatten(), gy.flatten()), scales.flatten(), 
                             (grid_x_f, grid_y_f), method='cubic')
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_dist.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        plt.colorbar(im, label='Local Pixel Scale (arcsec/pixel)')
        plt.title(f"Field Distortion (Local Pixel Scale) - {os.path.basename(args.image)}")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.savefig(os.path.join('Figures', f'{os.path.basename(args.image)}_distortion_map.png'))
        plt.close()
    else:
        print("Astrometry not solved; cannot generate distortion map.")

    print("Done.")

if __name__ == "__main__":
    main()
