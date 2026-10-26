"""Command line interface for PSFCaptain."""

import argparse
import concurrent.futures
import multiprocessing
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

from psfcaptain import plots, report
from psfcaptain.pipeline import process_image


def _preprocess_catalog_argv(argv):
    """Combines a space-separated catalog name and magnitude limit into 'name:limit'.

    e.g. ['--catalog', 'gaia', '12.0'] -> ['--catalog', 'gaia:12.0']
    """
    new_argv = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--catalog' and i + 1 < len(argv):
            catalog_name = argv[i + 1]
            if i + 2 < len(argv):
                next_arg = argv[i + 2]
                try:
                    float(next_arg)
                    combined = f"{catalog_name}:{next_arg}"
                    new_argv.extend([arg, combined])
                    i += 3
                    continue
                except ValueError:
                    pass
            new_argv.extend([arg, catalog_name])
            i += 2
        else:
            new_argv.append(arg)
            i += 1
    return new_argv


def build_parser():
    """Builds the command line argument parser."""
    parser = argparse.ArgumentParser(description="Find and measure stars.")
    parser.add_argument("image", help="Path to an image file or a directory of images.")
    parser.add_argument("--astrometry", action="store_true")
    parser.add_argument("--api-key", default="aifriketqrtctpor")
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--psf-limit", type=float, default=5.0,
                        help="Maximum size of the PSF in pixels used for magnitude calculations and plots (default 5.0).")
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
    parser.add_argument("--catalog", default="gaia",
                        help="Catalog for absolute calibration: 'gaia' or 'tycho2'. For Gaia, a magnitude "
                             "limit can be appended as 'gaia:12.0' or given as a trailing number (default limit 12.0).")
    parser.add_argument("--scale-low", type=float, default=20.0, help="Lower limit of pixel scale in arcsec/pixel (default 20.0).")
    parser.add_argument("--scale-high", type=float, default=50.0, help="Upper limit of pixel scale in arcsec/pixel (default 50.0).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write Figures/ and CSVs/ into (defaults to the input image's directory).")
    parser.add_argument("--report", default=None,
                        help="Path of the single-page lens quality PDF (default: lens_quality_report.pdf in the output directory).")
    parser.add_argument("--report-title", default=None, help="Title printed on the report (e.g. the lens name).")
    parser.add_argument("--no-report", action="store_true", help="Skip the lens quality PDF report.")
    return parser


def collect_images(path):
    """Returns (image_list, output_base) for a file or directory path."""
    extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.fits', '.fit')

    if os.path.isdir(path):
        print(f"Directory detected: {path}")
        image_list = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(extensions)
        )
        return image_list, path

    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return [], None

    return [path], os.path.dirname(path) or "."


def print_psf_summary(image_list, csvs_dir):
    """Prints a table of average PSF (FWHM) values per image across a batch."""
    print("\n=====================================================================================")
    print("                      SUMMARY OF AVERAGE PSF VALUES (FWHM)")
    print("=====================================================================================")

    psf_data = []
    for img in image_list:
        csv_path = os.path.join(csvs_dir, f'{os.path.basename(img)}_results.csv')
        if os.path.exists(csv_path):
            try:
                df_results = pd.read_csv(csv_path)
                if not df_results.empty and 'fwhm' in df_results.columns:
                    mean_fwhm_px = df_results['fwhm'].mean()

                    # Check if 'fwhm_arc' exists and contains non-NaN values
                    if 'fwhm_arc' in df_results.columns and not df_results['fwhm_arc'].isna().all():
                        mean_fwhm_arc = df_results['fwhm_arc'].mean()
                    else:
                        mean_fwhm_arc = np.nan

                    psf_data.append({
                        'image': os.path.basename(img),
                        'fwhm_px': mean_fwhm_px,
                        'fwhm_arc': mean_fwhm_arc,
                        'count': len(df_results)
                    })
            except Exception as e:
                print(f"Error reading results for {os.path.basename(img)}: {e}")

    if psf_data:
        print(f"{'Image Name':<45} | {'Stars':<5} | {'Avg FWHM (px)':<14} | {'Avg FWHM (arcsec)':<17}")
        print("-" * 92)
        all_px = []
        all_arc = []
        for item in psf_data:
            px_str = f"{item['fwhm_px']:.2f}"
            arc_str = f"{item['fwhm_arc']:.2f}\"" if not np.isnan(item['fwhm_arc']) and item['fwhm_arc'] > 0 else "N/A"
            print(f"{item['image']:<45} | {item['count']:<5d} | {px_str:<14} | {arc_str:<17}")
            all_px.append(item['fwhm_px'])
            if not np.isnan(item['fwhm_arc']) and item['fwhm_arc'] > 0:
                all_arc.append(item['fwhm_arc'])

        print("-" * 92)
        overall_px = np.mean(all_px)
        overall_arc = np.mean(all_arc) if all_arc else np.nan
        overall_arc_str = f"{overall_arc:.2f}\"" if not np.isnan(overall_arc) else "N/A"
        print(f"{'OVERALL AVERAGE':<45} | {'-':<5} | {overall_px:<14.2f} | {overall_arc_str:<17}")
    else:
        print("No valid PSF measurements found.")
    print("=====================================================================================\n")


def main():
    sys.argv = [sys.argv[0]] + _preprocess_catalog_argv(sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args()

    # Parse catalog and magnitude limit (e.g. "gaia:12.0")
    catalog_str = args.catalog
    if ":" in catalog_str:
        catalog_name, mag_limit_str = catalog_str.split(":", 1)
        try:
            gaia_mag_limit = float(mag_limit_str)
        except ValueError:
            parser.error(f"argument --catalog: invalid magnitude limit: '{mag_limit_str}'")
    else:
        catalog_name = catalog_str
        gaia_mag_limit = 12.0

    catalog_name = catalog_name.lower()
    if catalog_name not in ["gaia", "tycho2"]:
        parser.error(f"argument --catalog: invalid choice: '{catalog_name}' (choose from 'gaia', 'tycho2')")

    args.catalog = catalog_name
    args.gaia_mag_limit = gaia_mag_limit

    image_list, output_base = collect_images(args.image)
    if not image_list:
        if output_base is not None:
            print("No valid images found to process.")
        return

    if args.output_dir:
        output_base = args.output_dir

    # Output directories: Figures/ (with Individuals/ for per-metric plots) and CSVs/
    figures_dir = os.path.join(output_base, 'Figures')
    csvs_dir = os.path.join(output_base, 'CSVs')
    individuals_dir = os.path.join(figures_dir, 'Individuals')
    for d in (figures_dir, csvs_dir, individuals_dir):
        os.makedirs(d, exist_ok=True)

    print(f"Found {len(image_list)} images to process.")

    image_workers = args.max_image_workers if args.max_image_workers is not None else min(args.cores, 4)

    if image_workers > 1 and len(image_list) > 1:
        print(f"Processing images in parallel using {image_workers} workers...")
        # Mark that we are in a parallel image loop to avoid nested pools
        os.environ["IMAGE_POOL"] = "1"
        with concurrent.futures.ProcessPoolExecutor(max_workers=image_workers) as executor:
            worker = partial(process_image, args=args, figures_dir=figures_dir, csvs_dir=csvs_dir)
            list(executor.map(worker, image_list))
    else:
        for img in image_list:
            process_image(img, args, figures_dir, csvs_dir)

    print("\nBatch processing complete.")

    print_psf_summary(image_list, csvs_dir)

    if args.absolute:
        plots.plot_composite(image_list, csvs_dir, figures_dir)

    if not args.no_report:
        report_path = args.report or os.path.join(output_base, 'lens_quality_report.pdf')
        report.build_report(csvs_dir, report_path,
                            image_names=[os.path.basename(img) for img in image_list],
                            title=args.report_title)


if __name__ == "__main__":
    main()
