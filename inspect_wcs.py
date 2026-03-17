import os
import pickle
import numpy as np
from astropy.wcs import WCS
import glob

def check_wcs_scale():
    cache_dir = '.astrometry_cache'
    if not os.path.exists(cache_dir):
        print("No cache found.")
        return

    # Find the latest pickle in cache
    files = glob.glob(os.path.join(cache_dir, "*.pickle"))
    if not files:
        print("No pickle files in cache.")
        return
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading WCS from {latest_file}...")
    
    with open(latest_file, 'rb') as f:
        wcs_header = pickle.load(f)
    
    wcs = WCS(wcs_header)
    print("\nWCS Information:")
    print(wcs)
    
    # Calculate scale
    # Average pixel scale is roughly the CD matrix determinant or similar
    # Using the method from star_measure.py
    x, y = 1000, 1000
    sky1 = wcs.pixel_to_world(x, y)
    sky2 = wcs.pixel_to_world(x + 1, y)
    sky3 = wcs.pixel_to_world(x, y + 1)
    
    scale_x = sky1.separation(sky2).arcsec
    scale_y = sky1.separation(sky3).arcsec
    
    print(f"\nLocal Scale at (1000, 1000):")
    print(f"Scale X: {scale_x:.6f} arcsec/pix")
    print(f"Scale Y: {scale_y:.6f} arcsec/pix")
    print(f"Resultant Scale: {np.sqrt(scale_x * scale_y):.6f} arcsec/pix")
    
    # Check CD matrix if present
    if hasattr(wcs.wcs, 'cd'):
        print("\nCD Matrix:")
        print(wcs.wcs.cd)
        det = np.abs(np.linalg.det(wcs.wcs.cd))
        print(f"CD Determinant Scale: {np.sqrt(det) * 3600:.6f} arcsec/pix")
    elif hasattr(wcs.wcs, 'pc') and hasattr(wcs.wcs, 'cdelt'):
        print("\nPC Matrix + CDELT:")
        print("PC:", wcs.wcs.pc)
        print("CDELT:", wcs.wcs.cdelt)
        scale = np.abs(wcs.wcs.cdelt[0]) * 3600
        print(f"CDELT[0] Scale: {scale:.6f} arcsec/pix")

if __name__ == "__main__":
    check_wcs_scale()
