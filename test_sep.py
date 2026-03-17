from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

def test_separation():
    # Create a dummy WCS with 0.12 arcsec/pix
    # 0.12 arcsec = 0.12/3600 degrees = 3.3333e-5 degrees
    w = WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = [3.333333333e-5, 3.333333333e-5]
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    print("WCS Scale (CDELT):", w.wcs.cdelt[0] * 3600, "arcsec/pix")
    
    x, y = 100, 100
    sky1 = w.pixel_to_world(x, y)
    sky2 = w.pixel_to_world(x + 1, y)
    
    sep = sky1.separation(sky2)
    print(f"Separation: {sep}")
    print(f"Separation in arcsec: {sep.arcsec}")
    
    # Test with a real Header-like dict if possible
    # But this covers the logic.

if __name__ == "__main__":
    test_separation()
