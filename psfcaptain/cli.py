"""Command line interface for PSFCaptain."""

import argparse
import concurrent.futures
import multiprocessing
import os
from functools import partial

from psfcaptain import plots
from psfcaptain.pipeline import process_image


def build_parser():
    """Builds the command line argument parser."""
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
    parser.add_argument("--scale-low", type=float, default=20.0, help="Lower limit of pixel scale in arcsec/pixel (default 20.0).")
    parser.add_argument("--scale-high", type=float, default=50.0, help="Upper limit of pixel scale in arcsec/pixel (default 50.0).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write Figures/ and CSVs/ into (defaults to the input image's directory).")
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


def main():
    args = build_parser().parse_args()

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

    if args.absolute:
        plots.plot_composite(image_list, csvs_dir, figures_dir)


if __name__ == "__main__":
    main()
