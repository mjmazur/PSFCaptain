import argparse
import os
import glob
import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate

def transform_fits(file_path, rot_angle, flip_r, flip_c, ext=0):
    """
    Transforms the array data of a specific HDU extension in a FITS file.
    """
    try:
        with fits.open(file_path, mode='update') as hdul:
            if ext >= len(hdul):
                print(f"Error: {file_path} only has {len(hdul)} extensions. Cannot target extension {ext}.")
                return False
                
            data = hdul[ext].data
            if data is None:
                print(f"Error: Extension {ext} in {file_path} contains no data.")
                return False
                
            # Perform transformations
            if flip_r:
                data = np.flipud(data)
            if flip_c:
                data = np.fliplr(data)
            
            if rot_angle != 0.0:
                # If exact multiples of 90, use basic rot90 for lossless pixel translation
                if rot_angle % 90 == 0:
                    k = int((rot_angle / 90) % 4)
                    data = np.rot90(data, k=k)
                else:
                    # Bilinear interpolation for arbitrary degrees
                    data = rotate(data, rot_angle, reshape=True, order=1)
            
            # Reassign the transformed data back to the HDU
            hdul[ext].data = data
            
            # Astropy automatically updates NAXIS1 and NAXIS2 upon flush if the shape changed.
            # NOTE: Any embedded WCS coordinates in the header will NOT be automatically re-rotated.
            hdul.flush()
            
        print(f"Transformed {os.path.basename(file_path)}:")
        if flip_r: print("  - Flipped rows (top-bottom)")
        if flip_c: print("  - Flipped columns (left-right)")
        if rot_angle != 0.0: print(f"  - Rotated {rot_angle} degrees CCW")
        
        return True
    
    except Exception as e:
        print(f"Failed to transform {os.path.basename(file_path)}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Rotate or flip a FITS image data array.")
    parser.add_argument("path", help="Path to a single FITS file or a directory containing FITS files.")
    parser.add_argument("--rotate", type=float, default=0.0, help="Rotate the image by this many degrees counter-clockwise.")
    parser.add_argument("--flip-rows", action="store_true", help="Flip the image top to bottom (up-down).")
    parser.add_argument("--flip-columns", action="store_true", help="Flip the image left to right.")
    parser.add_argument("--ext", type=int, default=0, help="The HDU extension index to modify (default: 0). Use 1 for Rice compressed file data blocks.")
    
    args = parser.parse_args()

    if args.rotate == 0.0 and not args.flip_rows and not args.flip_columns:
        print("No transformations specified. Use --rotate, --flip-rows, or --flip-columns.")
        return

    files_to_process = []
    
    if os.path.isdir(args.path):
        for f in os.listdir(args.path):
            if f.lower().endswith(('.fits', '.fit', '.fz')):
                files_to_process.append(os.path.join(args.path, f))
        files_to_process.sort()
    elif os.path.isfile(args.path):
        files_to_process.append(args.path)
    else:
        files_to_process = glob.glob(args.path)

    if not files_to_process:
        print(f"No FITS files found for path: {args.path}")
        return

    print(f"Found {len(files_to_process)} FITS file(s) to process...")
    success_count = 0
    
    for fpath in files_to_process:
        if transform_fits(fpath, args.rotate, args.flip_rows, args.flip_columns, args.ext):
            success_count += 1
            
    print(f"\nFinished! Successfully transformed {success_count} out of {len(files_to_process)} files.")

if __name__ == "__main__":
    main()
