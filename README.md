# PSFCaptain - Star Field Measurement Tool

PSFCaptain is a Python-based utility designed for automated star detection, photometry, and morphology analysis in astronomical images. It supports common formats like FITS and PNG, providing a streamlined workflow from raw image to structured data and visualization.

## Features

- **Multi-Format Support**: Process `.fits`, `.fit`, `.png`, `.jpg`, and `.bmp` images.
- **Batch Processing**: Automatically process entire directories of images. Results are neatly organized into `Figures/` and `CSVs/` subfolders.
- **Advanced Star Detection**:
  - **RMSStarFinder**: Full geometric bounding, optimal flux calculations mapped directly from Pi in the Sky methodologies.
  - **DAOStarFinder**: Standard Gaussian-PSF detection.
  - **IRAFStarFinder**: Robust detection for bright or saturated stars.
  - **Two-Pass Photometry**: Cascade two finding methods together (e.g. `--finder rms iraf`) to derive structural mappings from the first pass and high-efficiency fluxes from the second.
  - **2D Background Suppression**: Optional local sky estimation (`--sky`) for improved detection in varied fields.
- **High-Performance Processing**:
  - **Intelligent Parallelism**: Parallel image processing for batches; star-level parallel morphology for single images.
  - **Vectorized WCS**: Blazing fast coordinate and distortion map generation.
- **Precision Photometry**: Performs aperture photometry to calculate instrumental magnitudes.
- **Absolute Photometry**: Calibrate your data against international catalogs:
  - **Gaia G-band**: Best for medium to faint stars.
  - **Tycho-2 VT-band**: Optimized for bright stars (magnitude 0-9).
  - **Robust Fitting**: Uses RANSAC algorithm to automatically exclude outliers for high-precision zero-point calibration.
  - **Smart Tiling**: Automatically handles large fields-of-view (up to 10+ degrees) by dividing queries into manageable tiles (online queries only; skipped entirely once a local catalog is available, see below).
  - **Local Catalog Caching**: Gaia and Tycho-2 both prefer a local, in-memory catalog over per-field network queries:
    - **Gaia**: Loaded from an [RMS](https://github.com/CroatianMeteorNetwork/RMS) installation's pre-filtered catalog file (`gaia_dr2_mag_11.5.npy`), if one is found on the system.
    - **Tycho-2**: The full ~2.5 million star catalog is downloaded from VizieR once and cached under `.catalog_cache/` in the working directory; every run after that reads the local cache instead of querying online.
    - Either way, once a local catalog is in play, a field is matched with a single in-memory box+circle filter instead of a network round trip per tile.
- **Morphology Analysis**: Measures Point Spread Function (PSF) characteristics:
  - FWHM (Full Width at Half Maximum) in pixels and arcseconds.
  - Elongation and Orientation Angle (Theta).
- **Automatic Image Inversion**: Intelligent detection and inversion of light-background images.
- **Astrometric Integration**: Seamless integration with `astrometry.net` for RA/Dec solving.
- **Comprehensive Visualization**: Generates heatmaps, histograms, and star-field summary overlays.
- **Lens Quality Report**: A single-page PDF summarizing optical quality across the whole field of view (PSF size and shape, 5σ limiting magnitude, vignetting, plate-scale distortion), averaged over all frames in a run. See [Lens Quality Report](#lens-quality-report).

## Project Layout

```
psfcaptain/          Core package
  cli.py             Argument parsing and batch orchestration
  pipeline.py        Per-image pipeline (detect -> measure -> calibrate -> plot)
  image_io.py        FITS / PNG / BMP loading
  detection.py       Background subtraction and star detection (incl. two-pass)
  finders.py         RMSStarFinder and its Gaussian fitting helpers
  morphology.py      Per-star FWHM, elongation and orientation measurement
  astrometry.py      Plate solving (local package, solve-field CLI, or API)
  catalogs.py        Gaia / Tycho-2 queries with tiling for large fields
  photometry.py      Instrumental magnitudes, outlier rejection, zero-point fit
  plots.py           All figure generation
  quality.py         Per-image aperture S/N and sky-background map for the report
  report.py          Single-page lens quality PDF (also runnable on an existing CSVs/ dir)
  wcs_utils.py       Shared WCS helpers
tools/               Standalone FITS utilities (binning, stacking, header edits)
tests/               Tests
data/                Sample and session input images
results/             Archived outputs from previous runs
star_measure.py      Thin wrapper kept for backwards compatibility
```

## Installation

### Prerequisites
- Python 3.8+ (Conda environment recommended)
- `numpy < 2.0.0` (required for Astropy compatibility)

### Setup Environment
```bash
# Create a new environment
conda create -n rms python=3.10
conda activate rms

# Install the package and its dependencies
pip install -e .

# Optional: offline plate solving with cached index files
pip install -e ".[local-solver]"
```

### Running

All three invocations are equivalent:

```bash
python -m psfcaptain data/test.fits      # module entry point
psfcaptain data/test.fits                # console script (after pip install)
python star_measure.py data/test.fits    # legacy wrapper
```

## Usage

### Basic Processing
```bash
python -m psfcaptain image.fits --cores 4
```

### Batch Processing (Directory)
Process all images in a folder and organize outputs:
```bash
python -m psfcaptain path/to/images/ --cores 8 --sky --finder iraf
```

### Absolute Photometry (Calibration)
To calibrate your magnitudes, you must first solve for astrometry. You can choose between Gaia and Tycho-2 catalogs:
```bash
# Calibrate against Gaia (Default)
python -m psfcaptain image.fits --astrometry --absolute

# Calibrate against Tycho-2 (Recommended for bright stars mag 0-9)
python -m psfcaptain image.fits --astrometry --absolute --catalog tycho2
```

### Multiprocessing
Speed up morphology measurements on large files by using multiple CPU cores:
```bash
python -m psfcaptain image.fits --cores 8
```

The script parallelizes based on the input: a directory processes multiple images
concurrently, a single image processes its stars concurrently. The default core
count is `CPU_COUNT - 2` to maintain system stability.

### FITS Utilities
Standalone helpers in `tools/`, each with its own `--help`:
```bash
python tools/bin_fits.py input.fits --bin 2 --method average  # Bin pixels, updating the WCS
python tools/stack_images.py frames/ --method median --output stack.fits
python tools/transform_fits.py input.fits --rotate 90 --flip-rows
python tools/edit_fits_header.py input.fits EXPTIME 10.0
python tools/check_fits.py input.fits                         # Shape/dtype of each HDU
python tools/inspect_wcs.py                                   # Pixel scales in the solver cache
```

## Lens Quality Report

Every run ends by writing `lens_quality_report.pdf` to the output directory, together with
`lens_quality_report_metrics.csv` (one row of numbers per frame). With several frames, the
table shows the **mean ± standard deviation across frames**; the maps and radial profiles
pool the stars of all frames in normalized sensor coordinates.

```bash
# Best results: astrometry + catalog calibration, several frames of the same lens
python -m psfcaptain frames/ --astrometry --absolute --catalog gaia --report-title "Lens 25mm f/1.4"

# Rebuild just the report from an existing CSVs/ directory
python -m psfcaptain.report frames/CSVs -o lens_25mm.pdf --title "Lens 25mm f/1.4"
```

What the page contains:

| Quantity | How it is measured |
|---|---|
| **PSF size** | FWHM per star (px and arcsec), as a binned field map and versus field radius |
| **PSF shape** | Elongation (a/b) and ellipticity whiskers (e = 1 − b/a, along the major axis), which show coma, astigmatism and tilt patterns |
| **Limiting magnitude** | 5σ point-source limit, `m + 2.5 log10(S/N / 5)`, from fixed-aperture photometry (r = median FWHM) with a background-limited S/N. With ≥ 30 catalog matches it is anchored to catalog magnitudes and includes vignetting; otherwise it uses the zero point (or stays instrumental) |
| **Vignetting** | Relative illumination from catalog-matched stars (aperture counts per unit catalog flux, normalized to the center) and from the smooth sky background |
| **Tilt** | Spread of the four corner FWHMs relative to the center FWHM |
| **Distortion** | Plate-scale change from center to corners (needs `--astrometry`) |

Field zones: *center* < 1/3, *middle* 1/3–2/3 and *edge* > 2/3 of the center-to-corner
distance; *corners* are the corner cells of a 3 × 3 grid.

Tips: the sky-based vignetting curve also picks up sky gradients and any camera bias offset,
so evenly lit frames (twilight flats) give the cleanest curve. The star-based curve and the
catalog-anchored limiting magnitude need many matched stars; Gaia gives far more than Tycho-2.

Report options: `--report PATH`, `--report-title TEXT`, `--no-report`.

## Output

### 1. CSV Results
A file named `[image_name]_results.csv` containing:
- `x`, `y`: Centered pixel coordinates.
- `ra`, `dec`: Right Ascension and Declination.
- `mag_instr`: Instrumental magnitude.
- `mag_abs`: Calibrated absolute magnitude (if `--absolute` is used).
- `fwhm`, `fwhm_arcsec`: PSF size.
- `elongation`, `theta`: Star shape characteristics (`theta` is the major-axis angle in degrees counter-clockwise from +x).
- `ap_flux`, `mag_ap`, `snr`, `sky`, `sky_rms`, `saturated`: Fixed-aperture photometry used by the lens quality report.

A `[image_name]_quality.json` sidecar stores per-image sky statistics, the sky-background grid and plate scales.

### 2. Figures
Saved in the `Figures/` output directory:
- `[image_name]_summary.png`: Master star field overlay with red/green detection circles and metrics box.

Saved in the `Figures/Individuals/` rendering directory:
- `[image_name]_mag_comparison.png`: Diagnostic plot of Catalog vs Instrumental magnitude.
- `[image_name]_abs_mag_hist.png`: Histogram of calibrated magnitudes.
- `[image_name]_psf_arcsec_map.png`: 2D heatmap of PSF size in arcseconds.
- `[image_name]_distortion_map.png`: 2D heatmap of local pixel scale.
- Single metric maps (e.g. FWHM pixels, Theta maps, simple detection outlines) are also pushed here.

### Detection & Processing
- `--fwhm`: Expected FWHM in pixels (default: 3.0).
- `--threshold`: Detection threshold in sigma (default: 5.0).
- `--gamma`: Corrects linear flux for gamma (default 1.0, specifically for `--finder rms`).
- `--finder`: Choose detection algorithm (`dao`, `iraf`, `rms`). You can queue two algorithms separated by a space (e.g., `--finder rms iraf`) to trigger **Two-Pass Photometry** cross matching.
- `--sky`: Enable 2D local background estimation (improves detection in uneven fields).
- `--cores`: Number of CPU cores to use (default: CPU_COUNT - 2).
- `--exclude-border`: Ignore stars detected near the image edges.
- `--output-dir`: Where to write `Figures/` and `CSVs/` (defaults to the input image's directory).

### Calibration & Analysis
- `--astrometry`: Solve for RA/Dec via Astrometry.net.
- `--absolute`: Perform absolute photometry calibration.
- `--catalog`: Choose `gaia` or `tycho2`. For Gaia, append a magnitude limit as `gaia:12.0` (or pass it as a trailing space-separated number); default limit is 12.0. Both catalogs prefer a local source over the network when available (see Local Catalog Caching above).
- `--api-key`: Astrometry.net API key.

### Fine-Tuning Detection
- `--sharplo`/`--sharphi`: Sharpness bounds for star detection (default 0.2 to 1.0).
- `--roundlo`/`--roundhi`: Roundness bounds for star detection (default -1.0 to 1.0).

## Performance Note
The script automatically limits Intel MKL and OpenMP threading to 1 thread per process when using multiprocessing. This prevents "Paging file is too small" errors and memory exhaustion on Windows systems.

## License
MIT
