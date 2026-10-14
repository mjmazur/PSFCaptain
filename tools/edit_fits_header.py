import argparse
import os
import glob
from astropy.io import fits

def update_fits_header(file_path, keyword, value, ext=0):
    """
    Updates or adds a keyword to a specific HDU extension in a FITS file.
    """
    try:
        # Intelligently parse the value type
        if value.lower() == 'true':
            parsed_val = True
        elif value.lower() == 'false':
            parsed_val = False
        else:
            try:
                if '.' in value:
                    parsed_val = float(value)
                else:
                    parsed_val = int(value)
            except ValueError:
                parsed_val = value  # leave as string
            
        with fits.open(file_path, mode='update') as hdul:
            if ext >= len(hdul):
                print(f"Error: {file_path} only has {len(hdul)} extensions. Cannot target extension {ext}.")
                return False
                
            hdul[ext].header[keyword] = parsed_val
            hdul.flush()
            
        print(f"Updated {os.path.basename(file_path)}: [{keyword}] = {parsed_val}")
        return True
    
    except Exception as e:
        print(f"Failed to update {os.path.basename(file_path)}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add or modify a keyword in a FITS header.")
    parser.add_argument("path", help="Path to a single FITS file or a directory containing FITS files.")
    parser.add_argument("keyword", help="The header keyword to change/add (e.g., EXPTIME, FILTER).")
    parser.add_argument("value", help="The value to set the keyword to (automatically parsed to int/float/bool if matched).")
    parser.add_argument("--ext", type=int, default=0, help="The HDU extension index to modify (default: 0). Use 1 for Rice compressed file data blocks.")
    
    args = parser.parse_args()

    # Validate keyword length
    clean_keyword = args.keyword.upper().strip()
    if len(clean_keyword) > 8 and not clean_keyword.startswith("HIERARCH"):
        print(f"Warning: FITS keywords are strictly 8 characters or less without HIERARCH. '{clean_keyword}' is {len(clean_keyword)} chars.")

    files_to_process = []
    
    # Check if a directory was given
    if os.path.isdir(args.path):
        for f in os.listdir(args.path):
            if f.lower().endswith(('.fits', '.fit', '.fz')):
                files_to_process.append(os.path.join(args.path, f))
        files_to_process.sort()
    elif os.path.isfile(args.path):
        files_to_process.append(args.path)
    else:
        # Attempt glob evaluation if wildcarded
        files_to_process = glob.glob(args.path)

    if not files_to_process:
        print(f"No FITS files found for path: {args.path}")
        return

    print(f"Found {len(files_to_process)} FITS file(s) to process...")
    success_count = 0
    
    for fpath in files_to_process:
        if update_fits_header(fpath, clean_keyword, args.value, args.ext):
            success_count += 1
            
    print(f"\nFinished! Successfully updated {success_count} out of {len(files_to_process)} files.")

if __name__ == "__main__":
    main()
