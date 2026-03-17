# PSFCaptain - Star Field Measurement Tool

PSFCaptain is a Python-based utility designed for automated star detection, photometry, and morphology analysis in astronomical images. It supports common formats like FITS and PNG, providing a streamlined workflow from raw image to structured data and visualization.

## Features

- **Multi-Format Support**: Process `.fits`, `.fit`, `.png`, and `.bmp` images.
- **Robust Star Detection**: Utilizes `photutils.detection.DAOStarFinder` for accurate source identification.
- **High-Performance Processing**: Built-in multiprocessing support (`--cores`) for fast analysis on multi-core systems.
- **Precision Photometry**: Performs aperture photometry to calculate instrumental magnitudes.
- **Absolute Photometry**: Calibrate your data against international catalogs:
  - **Gaia G-band**: Best for medium to faint stars.
  - **Tycho-2 VT-band**: Optimized for bright stars (magnitude 0-9).
  - **Robust Fitting**: Uses RANSAC algorithm to automatically exclude outliers for high-precision zero-point calibration.
  - **Smart Tiling**: Automatically handles large fields-of-view (up to 10+ degrees) by dividing queries into manageable tiles.
- **Morphology Analysis**: Measures Point Spread Function (PSF) characteristics:
  - FWHM (Full Width at Half Maximum) in pixels and arcseconds.
  - Elongation and Orientation Angle (Theta).
- **Automatic Image Inversion**: Intelligent detection and inversion of light-background images.
- **Astrometric Integration**: Seamless integration with `astrometry.net` for RA/Dec solving.
- **Comprehensive Visualization**: Generates heatmaps, histograms, and star-field summary overlays.

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
```bash
python star_measure.py image.fits --cores 4
```

### Absolute Photometry (Calibration)
To calibrate your magnitudes, you must first solve for astrometry. You can choose between Gaia and Tycho-2 catalogs:
```bash
# Calibrate against Gaia (Default)
python star_measure.py image.fits --astrometry --absolute

# Calibrate against Tycho-2 (Recommended for bright stars mag 0-9)
python star_measure.py image.fits --astrometry --absolute --catalog tycho2
```

### Multiprocessing
Speed up morphology measurements on large files by using multiple CPU cores:
```bash
python star_measure.py image.fits --cores 8
```

## Output

### 1. CSV Results
A file named `[image_name]_results.csv` containing:
- `x`, `y`: Centered pixel coordinates.
- `ra`, `dec`: Right Ascension and Declination.
- `mag_instr`: Instrumental magnitude.
- `mag_abs`: Calibrated absolute magnitude (if `--absolute` is used).
- `fwhm`, `fwhm_arcsec`: PSF size.
- `elongation`, `theta`: Star shape characteristics.

### 2. Figures
Saved in the `Figures/` directory:
- `[image_name]_summary.png`: Star field overlay with detection circles and image statistics.
- `[image_name]_mag_comparison.png`: Diagnostic plot of Catalog vs Instrumental magnitude.
- `[image_name]_abs_mag_hist.png`: Histogram of calibrated magnitudes.
- `[image_name]_psf_arcsec_map.png`: 2D heatmap of PSF size in arcseconds.
- `[image_name]_distortion_map.png`: 2D heatmap of local pixel scale.

## Parameters
- `--fwhm`: Expected FWHM in pixels (default: 3.0).
- `--threshold`: Detection threshold in sigma (default: 5.0).
- `--cores`: Number of CPU cores to use.
- `--astrometry`: Solve for RA/Dec via Astrometry.net.
- `--absolute`: Perform absolute photometry calibration.
- `--catalog`: Choose `gaia` or `tycho2`.
- `--api-key`: Astrometry.net API key.

## Performance Note
The script automatically limits Intel MKL and OpenMP threading to 1 thread per process when using multiprocessing. This prevents "Paging file is too small" errors and memory exhaustion on Windows systems.

## License
MIT
