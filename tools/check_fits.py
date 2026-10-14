"""Prints the shape, dtype and size of each image HDU in the given FITS files."""

import argparse
import os

from astropy.io import fits

def check(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found")
        return
    with fits.open(filename) as hdul:
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None:
                print(f"File: {filename} HDU {i}")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  BITPIX: {hdu.header.get('BITPIX')}")
    print(f"  File Size: {os.path.getsize(filename)} bytes")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="FITS files to inspect.")
    args = parser.parse_args()
    for filename in args.files:
        check(filename)


if __name__ == "__main__":
    main()
