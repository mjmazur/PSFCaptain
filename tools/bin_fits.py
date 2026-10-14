import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def bin_image(data, bin_factor, method='average'):
    """Bins the 2D or 3D image data by a factor of bin_factor x bin_factor spatially."""
    if data.ndim == 2:
        ny, nx = data.shape
        new_ny = ny // bin_factor
        new_nx = nx // bin_factor
        cropped_data = data[:new_ny * bin_factor, :new_nx * bin_factor]
        reshaped = cropped_data.reshape(new_ny, bin_factor, new_nx, bin_factor)
        if method == 'average':
            return reshaped.mean(axis=(1, 3))
        else:
            return reshaped.sum(axis=(1, 3))
    elif data.ndim == 3:
        nz, ny, nx = data.shape
        new_ny = ny // bin_factor
        new_nx = nx // bin_factor
        cropped_data = data[:, :new_ny * bin_factor, :new_nx * bin_factor]
        reshaped = cropped_data.reshape(nz, new_ny, bin_factor, new_nx, bin_factor)
        if method == 'average':
            return reshaped.mean(axis=(2, 4))
        else:
            return reshaped.sum(axis=(2, 4))
    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")

def update_header(header, bin_factor, method='average'):
    """Updates the FITS header for the new dimensions and WCS scale."""
    new_header = header.copy()
    
    # Update NAXIS
    if 'NAXIS1' in new_header:
        new_header['NAXIS1'] = new_header['NAXIS1'] // bin_factor
    if 'NAXIS2' in new_header:
        new_header['NAXIS2'] = new_header['NAXIS2'] // bin_factor
        
    # Update WCS if present
    # CRPIX: (pixel coords) -> CRPIX_new = (CRPIX_old - 0.5) / bin_factor + 0.5
    # CDELT / CD: (deg/pix) -> CDELT_new = CDELT_old * bin_factor
    
    # CRPIX
    for i in [1, 2]:
        key = f'CRPIX{i}'
        if key in new_header:
            new_header[key] = (new_header[key] - 0.5) / bin_factor + 0.5
            
    # CDELT
    for i in [1, 2]:
        key = f'CDELT{i}'
        if key in new_header:
            new_header[key] *= bin_factor
            
    # CD Matrix
    for i in [1, 2]:
        for j in [1, 2]:
            key = f'CD{i}_{j}'
            if key in new_header:
                new_header[key] *= bin_factor
                
    # PC Matrix (often used with CDELT)
    # CDELT is scaled above, PC matrix (rotation) remains the same
    
    new_header.add_history(f"Binned by factor {bin_factor} using {method}")
    return new_header

def process_file(input_path, output_path, bin_factor, method):
    print(f"Processing {input_path}...")
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            if data is None:
                # Check other HDUs
                for hdu in hdul[1:]:
                    if hdu.data is not None:
                        data = hdu.data
                        header = hdu.header
                        break
            
            if data is None:
                print("Error: No image data found in FITS file.")
                return

            print(f"Original shape: {data.shape} ({data.dtype})")
            binned_data = bin_image(data, bin_factor, method)
            
            # By default, cast back to 16-bit if it was integer-like, or if requested.
            # This ensures 2x2 binning results in 1/4 file size.
            original_is_int = np.issubdtype(data.dtype, np.integer)
            
            if original_is_int:
                # Round and clip to original integer range to prevent overflow/artifacts
                # Most astronomical images are uint16 (0-65535)
                if data.dtype == np.uint16:
                    binned_data = np.clip(np.round(binned_data), 0, 65535).astype(np.uint16)
                else:
                    binned_data = np.round(binned_data).astype(data.dtype)
                print(f"Binned shape: {binned_data.shape} (Casted to {binned_data.dtype})")
            else:
                # If original was float, keep as float32 to save space over float64
                if binned_data.dtype == np.float64:
                    binned_data = binned_data.astype(np.float32)
                print(f"Binned shape: {binned_data.shape} ({binned_data.dtype})")
            
            new_header = update_header(header, bin_factor, method)
            
            if not output_path:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}_bin{bin_factor}{ext}"
                
            new_hdu = fits.PrimaryHDU(data=binned_data, header=new_header)
            new_hdu.writeto(output_path, overwrite=True)
            print(f"Binned image saved to {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Bin a FITS image or directory of FITS images.")
    parser.add_argument("input", help="Input FITS file path or directory path")
    parser.add_argument("--bin", type=int, default=2, help="Binning factor (default: 2)")
    parser.add_argument("--method", choices=['average', 'sum'], default='average', 
                        help="Binning method (default: average)")
    parser.add_argument("-o", "--output", help="Output FITS file path (default for single file: [input]_binned.fits)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Path {args.input} not found.")
        return

    if os.path.isdir(args.input):
        print(f"Directory detected: {args.input}")
        extensions = ('.fits', '.fit')
        
        image_list = []
        for f in os.listdir(args.input):
            if f.lower().endswith(extensions):
                image_list.append(os.path.join(args.input, f))
                
        if not image_list:
            print(f"No FITS images found in {args.input}.")
            return
            
        print(f"Found {len(image_list)} images to process.")
        
        output_dir = os.path.join(args.input, f"binned{args.bin}x{args.bin}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for img_path in sorted(image_list):
            out_file = os.path.join(output_dir, os.path.basename(img_path))
            process_file(img_path, out_file, args.bin, args.method)
            
        print("\nBatch processing complete.")
    else:
        process_file(args.input, args.output, args.bin, args.method)

if __name__ == "__main__":
    main()
