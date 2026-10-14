import os
import argparse
import numpy as np
from PIL import Image
from astropy.io import fits

def load_image(file_path):
    """Loads FITS or standard image and returns data (and header if FITS)."""
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
            if data.ndim == 3:
                data = data[0]
            return data.astype(float), header
    else:
        # PNG, BMP, JPG
        img = Image.open(file_path).convert('L')
        return np.array(img).astype(float), None

def save_image(file_path, data, header=None, is_fits=True):
    """Saves stacked data to FITS or standard image format."""
    if is_fits:
        # Save to FITS
        fits.writeto(file_path, data, header=header, overwrite=True)
    else:
        # Save to PNG/BMP/JPG
        clipped_data = np.clip(data, 0, 255).astype(np.uint8)
        img = Image.fromarray(clipped_data)
        img.save(file_path)

def main():
    parser = argparse.ArgumentParser(description="Stack a series of images in a directory.")
    parser.add_argument("directory", help="Path to the directory containing images to stack.")
    parser.add_argument("--method", choices=["max", "median", "mean"], default="median",
                        help="Combine method: max, median (default), or mean.")
    parser.add_argument("--discard-high", type=int, default=0,
                        help="Number of highest values to discard per pixel before combining.")
    parser.add_argument("--discard-low", type=int, default=0,
                        help="Number of lowest values to discard per pixel before combining.")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: stacked.fits or stacked.png).")
    parser.add_argument("--pattern", default=None,
                        help="Glob pattern to filter files (e.g. '*.fits' or 'image_*.png').")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory.")
        return
        
    # Supported extensions
    valid_exts = ('.fits', '.fit', '.png', '.jpg', '.jpeg', '.bmp')
    
    # List files
    files = []
    if args.pattern:
        import glob
        pattern_path = os.path.join(args.directory, args.pattern)
        files = glob.glob(pattern_path)
    else:
        for f in os.listdir(args.directory):
            if f.lower().endswith(valid_exts):
                files.append(os.path.join(args.directory, f))
                
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    
    if not files:
        print("No valid images found to stack.")
        return
        
    print(f"Found {len(files)} files to stack.")
    
    # Load first image to get dimensions and format
    first_file = files[0]
    first_ext = os.path.splitext(first_file)[1].lower()
    is_fits = first_ext in ['.fits', '.fit']
    
    # Read first image
    first_data, first_header = load_image(first_file)
    ny, nx = first_data.shape
    print(f"Image shape: {nx} x {ny}")
    
    # Pre-allocate stack array
    # Shape: [num_images, height, width]
    num_images = len(files)
    stack = np.empty((num_images, ny, nx), dtype=float)
    stack[0] = first_data
    
    # Load remaining images
    valid_count = 1
    for i in range(1, num_images):
        f = files[i]
        try:
            data, _ = load_image(f)
            if data.shape != (ny, nx):
                print(f"Warning: Skipping {os.path.basename(f)} because its shape {data.shape[::-1]} differs from {nx}x{ny}")
                continue
            stack[valid_count] = data
            valid_count += 1
        except Exception as e:
            print(f"Warning: Skipping {os.path.basename(f)} due to load error: {e}")
            
    if valid_count == 0:
        print("Error: No images were loaded successfully.")
        return
        
    # Resize stack array to actual loaded count
    if valid_count < num_images:
        stack = stack[:valid_count]
        print(f"Proceeding with {valid_count} successfully loaded images.")
        num_images = valid_count
        
    # Validate discard counts
    total_discard = args.discard_high + args.discard_low
    if total_discard >= num_images:
        print(f"Error: Cannot discard {total_discard} values when only {num_images} images are available.")
        return
        
    print(f"Combining via {args.method} (discarding {args.discard_low} lowest and {args.discard_high} highest)...")
    
    # Apply discard and combine logic
    if total_discard > 0:
        # Sort along the image (depth) axis
        print("Sorting pixel values to discard extreme values...")
        sorted_stack = np.sort(stack, axis=0)
        # Slice to exclude discarded indices
        start_idx = args.discard_low
        end_idx = num_images - args.discard_high
        sliced_stack = sorted_stack[start_idx:end_idx, :, :]
        
        # Combine
        if args.method == "median":
            stacked_data = np.median(sliced_stack, axis=0)
        elif args.method == "mean":
            stacked_data = np.mean(sliced_stack, axis=0)
        else: # max
            stacked_data = np.max(sliced_stack, axis=0)
    else:
        # Fast path when no discarding is needed
        if args.method == "median":
            stacked_data = np.median(stack, axis=0)
        elif args.method == "mean":
            stacked_data = np.mean(stack, axis=0)
        else: # max
            stacked_data = np.max(stack, axis=0)
            
    # Determine output path
    output_path = args.output
    if not output_path:
        default_name = "stacked.fits" if is_fits else "stacked.png"
        output_path = os.path.join(args.directory, default_name)
        
    print(f"Saving stacked image to {output_path}...")
    save_image(output_path, stacked_data, header=first_header, is_fits=is_fits)
    print("Stacking complete!")

if __name__ == "__main__":
    main()
