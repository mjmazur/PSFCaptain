"""PSFCaptain - star detection, photometry and PSF morphology for astronomical images."""

import os

# Limit threading in MKL-dependent libraries. This must happen BEFORE numpy, scipy
# or any submodule that imports them is loaded, otherwise multiprocessing workers
# can exhaust memory ("Paging file is too small" on Windows).
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DOMAIN_NUM_THREADS"] = "1"

# Disable astroquery caching to prevent race conditions during parallel processing.
try:
    from astroquery import cache_conf
    cache_conf.cache_active = False
except ImportError:
    pass

__version__ = "0.1.0"
