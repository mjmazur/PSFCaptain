"""Single-image processing pipeline: detect, measure, calibrate and plot."""

import os

from psfcaptain import plots
from psfcaptain.astrometry import solve_for_image
from psfcaptain.detection import detect_stars, subtract_background
from psfcaptain.image_io import load_image
from psfcaptain.morphology import compute_fwhm_arcsec, measure_morphology
from psfcaptain.photometry import calibrate_absolute, reject_fwhm_outliers, reject_psf_limit, to_dataframe
from psfcaptain.quality import measure_field_quality, write_quality_meta


def process_image(image_path, args, figures_dir, csvs_dir):
    """Processes a single image file end to end, writing a CSV and figures."""
    name = os.path.basename(image_path)
    print(f"\n--- Processing: {name} ---")
    individuals_dir = os.path.join(figures_dir, 'Individuals')

    try:
        data, header = load_image(image_path)
        ny, nx = data.shape
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return

    data_sub, mean, median, std = subtract_background(data, use_sky=args.sky)

    sources = detect_stars(data_sub, std, args, nx, ny)
    if sources is None:
        return

    wcs = solve_for_image(image_path, data, sources, nx, ny, args, individuals_dir)

    df = to_dataframe(sources)
    df = measure_morphology(df, data_sub, std, wcs, nx, ny, args)
    df, df_outliers = reject_fwhm_outliers(df)
    df, df_outliers = reject_psf_limit(df, df_outliers, args.psf_limit)
    df = compute_fwhm_arcsec(df, wcs)

    zero_point = calibrate_absolute(df, wcs, nx, ny, args)

    df, quality_meta = measure_field_quality(data, header, df, wcs, nx, ny)
    quality_meta.update({
        'image': name,
        'finder': ' '.join(args.finder) if isinstance(args.finder, list) else args.finder,
        'catalog': args.catalog if args.absolute else None,
        'zero_point': float(zero_point) if zero_point is not None else None,
        'n_detected': int(len(df) + len(df_outliers)),
        'n_valid': int(len(df)),
    })
    write_quality_meta(quality_meta, csvs_dir, name)

    results_csv = os.path.join(csvs_dir, f'{name}_results.csv')
    df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

    if args.absolute:
        plots.plot_catalog_diagnostics(results_csv, image_path, individuals_dir)

    print("Generating figures...")
    plots.plot_detected(data, df, image_path, individuals_dir)
    plots.plot_psf_maps(df, wcs, nx, ny, image_path, individuals_dir)
    scales = plots.plot_distortion_map(wcs, nx, ny, image_path, individuals_dir)
    plots.plot_summary(data, df, df_outliers, wcs, scales, nx, ny, mean, std,
                       zero_point, image_path, figures_dir)
