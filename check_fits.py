from astropy.io import fits
import os

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

check('test.fit')
check('test_bin2.fit')
