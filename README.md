# PSFCaptain - Star Field Measurement Tool

PSFCaptain is a Python-based utility designed for automated star detection, photometry, and morphology analysis in astronomical images. It supports common formats like FITS and PNG, providing a streamlined workflow from raw image to structured data and visualization.

## Features

- **Multi-Format Support**: Process `.fits`, `.fit`, `.png`, and `.bmp` images.
- **Robust Star Detection**: Utilizes `photutils.detection.DAOStarFinder` for accurate source identification.
- **Precision Photometry**: Performs aperture photometry to calculate instrumental magnitudes.
- **Morphology Analysis**: Measures Point Spread Function (PSF) characteristics including:
  - FWHM (Full Width at Half Maximum)
  - Elongation
  - Orientation Angle (Theta)
- **Automatic Image Inversion**: Features intelligent detection of light-background images (common in inverted PNG exports), automatically inverting them for correct source detection.
- **Astrometric Integration**: Optional integration with `astrometry.net` via `astroquery` for RA/Dec solving.
- **Visualization**: Automatically generates histograms for instrumental magnitudes and elongation angles.

## Installation

### Prerequisites
- Python 3.8+ (Conda environment recommended)
- `numpy < 2.0.0` (required for Astropy compatibility)

### Setup Environment
```bash
# Create a new environment
conda create -n rms python=3.10
conda activate rms

# Install dependencies
pip install numpy==1.26.4 pandas matplotlib astropy photutils astroquery pillow
```

## Usage

### Basic Processing
Run the script by providing the path to your image:
```bash
python star_measure.py path/to/your_image.fits
```

### With Astrometry Solving
To get RA/Dec coordinates, use the `--astrometry` flag. A default API key is provided, but you can override it with your own:
```bash
# Using default API key
python star_measure.py path/to/your_image.png --astrometry

# Using your own API key
python star_measure.py path/to/your_image.png --astrometry --api-key YOUR_API_KEY
```

## Utility Tools

### FITS Binning
Use `bin_fits.py` to downsample large FITS files. This is useful for speeding up astrometry or reducing noise.
```bash
# Default 2x2 binning (averaging)
python bin_fits.py path/to/your_image.fits

# Specific 4x4 binning
python bin_fits.py path/to/your_image.fits --bin 4

# Using summation instead of averaging
python bin_fits.py path/to/your_image.fits --method sum
```

## Output

The script generates the following outputs:

### 1. CSV Results
A file named `[image_name]_results.csv` containing:
- `x`, `y`: Centered pixel coordinates.
- `ra`, `dec`: Right Ascension and Declination (if astrometry solved).
- `mag_instr`: Instrumental magnitude.
- `fwhm`: Full Width at Half Maximum.
- `elongation`: Ratio of major to minor axis.
- `theta`: Orientation angle in degrees.

### 2. Figures
Saved in the `Figures/` directory:
- `[image_name]_mag_hist.png`: Histogram of star brightness.
- `[image_name]_elong_angle_hist.png`: Distribution of star orientation angles.
- `[image_name]_psf_size_map.png`: 2D heatmap of FWHM variation.
- `[image_name]_theta_map.png`: 2D heatmap of PSF orientation.
- `[image_name]_distortion_map.png`: 2D heatmap of local pixel scale (distortion).

## Parameters
- `--fwhm`: Expected FWHM of stars in pixels (default: 3.0).
- `--threshold`: Detection threshold in standard deviations above background (default: 5.0).
- `--api-key`: Astrometry.net API key (can also be set via `ASTROMETRY_NET_API_KEY` environment variable).

## License
MIT
