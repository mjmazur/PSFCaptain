"""Image loading for FITS and standard raster formats."""

import os

import numpy as np
from PIL import Image
from astropy.io import fits


def load_image(file_path):
    """Loads FITS or PNG image and returns data as a 2D numpy array."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.fits', '.fit']:
        with fits.open(file_path) as hdul:
            data = None
            header = None
            for hdu in hdul:
                if hdu.data is not None and getattr(hdu.data, 'size', 0) > 0:
                    data = hdu.data
                    header = hdu.header
                    break
            
            if data is None:
                raise ValueError(f"No valid image data found in FITS: {file_path}")
            
            # FITS data can be 3D (e.g., [1, Y, X]), take the first slice if so
            if data.ndim == 3:
                data = data[0]
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
