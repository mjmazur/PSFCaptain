"""Astrometric plate solving via local solvers or the astrometry.net API."""

import os
import time
from functools import wraps

import numpy as np


class SimpleFileLock:
    """Directory-based lock serializing astrometry solves across worker processes."""

    def __init__(self, lock_dir=".astrometry_lock", timeout=300):
        self.lock_dir = lock_dir
        self.timeout = timeout

    def __enter__(self):
        printed_waiting = False
        while True:
            try:
                os.mkdir(self.lock_dir)
                break
            except FileExistsError:
                if not printed_waiting:
                    print("Astrometry solver is currently in use by another process. Waiting for lock...")
                    printed_waiting = True
                try:
                    mtime = os.path.getmtime(self.lock_dir)
                    if time.time() - mtime > self.timeout:
                        print(f"Warning: Lock '{self.lock_dir}' appears expired. Cleaning up...")
                        try:
                            os.rmdir(self.lock_dir)
                        except OSError:
                            pass
                        continue
                except OSError:
                    pass
                time.sleep(0.5)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.rmdir(self.lock_dir)
        except OSError:
            pass


def serialize_astrometry(func):
    """Decorator serializing calls to the wrapped solver via SimpleFileLock."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with SimpleFileLock():
            return func(*args, **kwargs)
    return wrapper


def detect_local_astrometry_server():
    """Checks standard localhost ports for a running local Astrometry.net API server."""
    import urllib.request
    import urllib.parse
    import json
    
    ports = [8000, 8080]
    hosts = ['localhost', '127.0.0.1']
    
    for port in ports:
        for host in hosts:
            url = f"http://{host}:{port}/api"
            try:
                # To verify it is indeed an astrometry server, we test the login endpoint
                login_url = f"{url}/login"
                payload = {'request-json': json.dumps({"apikey": "test"})}
                data = urllib.parse.urlencode(payload).encode('utf-8')
                
                req = urllib.request.Request(login_url, data=data, method='POST')
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    res_data = response.read().decode('utf-8')
                    res_json = json.loads(res_data)
                    # A real astrometry server will return a JSON with a 'status' field
                    if 'status' in res_json:
                        return url
            except Exception:
                pass
    return None

def solve_astrometry_local_cli(image_path, solve_field_path, scale_low=20.0, scale_high=50.0):
    """Solves astrometry using local solve-field CLI solver."""
    import subprocess
    import tempfile
    from astropy.wcs import WCS
    
    print("Found local solve-field. Solving locally...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build command: output WCS file to solved.wcs in the temp directory
        cmd = [
            solve_field_path,
            image_path,
            "--dir", tmpdir,
            "--out", "solved",
            "--overwrite",
            "--no-plots",
            "--scale-units", "arcsecperpix",
            "--scale-low", str(scale_low),
            "--scale-high", str(scale_high),
        ]
        
        try:
            print(f"Running command: {' '.join(cmd)}")
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            if res.returncode == 0:
                wcs_file = os.path.join(tmpdir, "solved.wcs")
                if os.path.exists(wcs_file):
                    wcs = WCS(wcs_file)
                    print("Local astrometry solve successful!")
                    return wcs
            else:
                print(f"Local solve-field failed with exit code {res.returncode}")
                if res.stderr:
                    print(res.stderr)
        except Exception as e:
            print(f"Error running local solve-field: {e}")
            
    return None

@serialize_astrometry
def solve_astrometry(image_path, sources=None, width=None, height=None, api_key=None, scale_low=20.0, scale_high=50.0):
    """Solves astrometry using local astrometry.net if available, falling back to remote."""
    import shutil

    if sources is not None and len(sources) > 0:
        # Filter out NaN/Inf values from centroid columns and peak/flux column to prevent solver crashes
        x_raw = np.asarray(sources['xcentroid'])
        y_raw = np.asarray(sources['ycentroid'])
        peak_raw = np.asarray(sources['peak']) if 'peak' in sources.colnames else np.ones_like(x_raw)

        valid_mask = (
            ~np.isnan(x_raw) & ~np.isinf(x_raw) &
            ~np.isnan(y_raw) & ~np.isinf(y_raw) &
            ~np.isnan(peak_raw) & ~np.isinf(peak_raw)
        )
        sources = sources[valid_mask]

    # 1. Try python 'astrometry' package (installed via pip install astrometry)
    try:
        import astrometry
        from astropy.wcs import WCS

        if sources is not None and len(sources) > 0:
            num_sources = min(len(sources), 50)
            sorted_indices = np.argsort(sources['peak'])[::-1]
            x_sorted = np.asarray(sources['xcentroid'])[sorted_indices][:num_sources]
            y_sorted = np.asarray(sources['ycentroid'])[sorted_indices][:num_sources]
            stars = np.column_stack((x_sorted, y_sorted))
            
            print("Attempting local solve using python 'astrometry' package...")
            cache_dir = os.path.join(os.getcwd(), '.astrometry_cache')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # If the astrometry package has SizeHint, we can construct and pass it
            size_hint = None
            if hasattr(astrometry, 'SizeHint'):
                size_hint = astrometry.SizeHint(
                    lower_arcsec_per_pixel=scale_low,
                    upper_arcsec_per_pixel=scale_high
                )
                
            with astrometry.Solver(
                astrometry.series_5200.index_files(
                    cache_directory=cache_dir,
                    scales={2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
                )
            ) as solver:
                if size_hint:
                    solution = solver.solve(stars=stars, size_hint=size_hint)
                else:
                    solution = solver.solve(stars=stars)
                    
                if solution.has_match():
                    match = solution.best_match()
                    wcs = WCS(match.wcs_fields)
                    print("Local astrometry package solve successful!")
                    return wcs
                else:
                    print("Local astrometry package solve failed to find a match.")
    except Exception as e:
        # Ignore errors if package not installed or fails
        pass

    # 2. Try local CLI solver (solve-field)
    solve_field_path = shutil.which("solve-field")
    if solve_field_path:
        wcs = solve_astrometry_local_cli(image_path, solve_field_path, scale_low=scale_low, scale_high=scale_high)
        if wcs:
            return wcs
        print("Local solve-field failed. Trying local/remote API...")

    # 3. Fallback: Try astroquery with local/remote API
    try:
        from astroquery.astrometry_net import AstrometryNet
    except ImportError:
        print("astroquery is not installed. Astrometry will be skipped.")
        return None

    # Detect if a local ANSVR or API server is running
    local_url = detect_local_astrometry_server()
    servers_to_try = []
    if local_url:
        servers_to_try.append((local_url, None)) # local server doesn't need key
    servers_to_try.append(('https://nova.astrometry.net/api', api_key))

    for server_url, key in servers_to_try:
        is_local = "localhost" in server_url or "127.0.0.1" in server_url
        server_type = "LOCAL" if is_local else "ONLINE"
        print(f"Trying astrometry solve via {server_type} API at {server_url}...")
        try:
            # Set server URL in conf
            from astroquery.astrometry_net import conf
            conf.server = server_url

            ast = AstrometryNet()
            ast.cache_location = os.path.join(os.getcwd(), '.astrometry_cache')
            if not os.path.exists(ast.cache_location):
                os.makedirs(ast.cache_location)

            if key:
                ast.api_key = key
            elif 'ASTROMETRY_NET_API_KEY' in os.environ:
                ast.api_key = os.environ['ASTROMETRY_NET_API_KEY']
            else:
                ast.api_key = 'aifriketqrtctpor'

            # Disable Keep-Alive to prevent RemoteDisconnected errors when polling status
            if hasattr(ast, '_session') and hasattr(ast._session, 'headers'):
                ast._session.headers['Connection'] = 'close'

            # Run solving with retries for transient network/connection errors
            max_retries = 3
            wcs_header = None
            for attempt in range(max_retries):
                try:
                    if sources is not None and width is not None and height is not None:
                        num_sources = min(len(sources), 50)
                        if num_sources < 10:
                            print(f"Warning: Only {num_sources} stars found. Astrometry may fail without more stars.")

                        if attempt == 0:
                            print(f"Submitting top {num_sources} sources to Astrometry...")
                        else:
                            print(f"Submitting top {num_sources} sources to Astrometry (Attempt {attempt+1}/{max_retries})...")

                        sorted_indices = np.argsort(sources['peak'])[::-1]
                        x_sorted = np.asarray(sources['xcentroid'])[sorted_indices][:num_sources]
                        y_sorted = np.asarray(sources['ycentroid'])[sorted_indices][:num_sources]

                        try:
                            wcs_header = ast.solve_from_source_list(x_sorted, y_sorted, width, height,
                                                                   solve_timeout=180,
                                                                   scale_units='arcsecperpix',
                                                                   scale_lower=scale_low,
                                                                   scale_upper=scale_high)
                        except Exception as e_src:
                            print(f"Source list solve encountered error/timeout: {e_src}.")

                        if not wcs_header:
                            print("Source list solve failed or timed out. Extracting astrometry-optimized sources using DAOStarFinder...")
                            try:
                                from photutils.background import Background2D, MedianBackground
                                from photutils.detection import DAOStarFinder

                                from psfcaptain.image_io import load_image

                                # Load the image data
                                data, header = load_image(image_path)
                                ny_img, nx_img = data.shape

                                # Estimate local background (same as in process_image)
                                bkg_estimator = MedianBackground()
                                bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
                                data_sub = data - bkg.background
                                std_local = np.median(bkg.background_rms)

                                # Run DAOStarFinder with a robust 3.5 sigma threshold to get clean bright stars
                                dao_f = DAOStarFinder(fwhm=3.0, threshold=3.5 * std_local)
                                dao_src = dao_f(data_sub)

                                if dao_src is not None and len(dao_src) >= 10:
                                    # Filter edge stars (within 25 pixels)
                                    edge_m = (dao_src['xcentroid'] >= 25) & (dao_src['xcentroid'] <= nx_img - 25) & \
                                             (dao_src['ycentroid'] >= 25) & (dao_src['ycentroid'] <= ny_img - 25)
                                    dao_src_filt = dao_src[edge_m]

                                    if len(dao_src_filt) >= 10:
                                        num_dao = min(len(dao_src_filt), 50)
                                        print(f"Submitting top {num_dao} DAO-extracted sources to Astrometry...")
                                        s_indices = np.argsort(dao_src_filt['peak'])[::-1]
                                        x_dao = np.asarray(dao_src_filt['xcentroid'])[s_indices][:num_dao]
                                        y_dao = np.asarray(dao_src_filt['ycentroid'])[s_indices][:num_dao]

                                        wcs_header = ast.solve_from_source_list(x_dao, y_dao, nx_img, ny_img,
                                                                               solve_timeout=180,
                                                                               scale_units='arcsecperpix',
                                                                               scale_lower=scale_low,
                                                                               scale_upper=scale_high)
                            except Exception as e_dao:
                                print(f"DAO fallback extraction failed: {e_dao}.")

                        if not wcs_header:
                            print("DAO-optimized solve failed or timed out. Falling back to full image solve on the same server...")
                            wcs_header = ast.solve_from_image(image_path, solve_timeout=180,
                                                             scale_units='arcsecperpix',
                                                             scale_lower=scale_low,
                                                             scale_upper=scale_high)
                    else:
                        if attempt == 0:
                            print("Submitting full image to Astrometry...")
                        else:
                            print(f"Submitting full image to Astrometry (Attempt {attempt+1}/{max_retries})...")
                        wcs_header = ast.solve_from_image(image_path, solve_timeout=180,
                                                         scale_units='arcsecperpix',
                                                         scale_lower=scale_low,
                                                         scale_upper=scale_high)
                    break  # Success!
                except Exception as e:
                    if "Login failed" in str(e) or "invalid API key" in str(e).lower():
                        raise e
                    # Do not retry on timeout errors; raise immediately to try next server
                    if "timeout" in str(e).lower():
                        raise e
                    if attempt < max_retries - 1:
                        print(f"Astrometry solver encountered error: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        raise e

            if wcs_header:
                from astropy.wcs import WCS
                wcs = WCS(wcs_header)
                print(f"Astrometry solve successful via {server_url}!")
                return wcs
            else:
                print(f"Astrometry failed via {server_url}.")
        except Exception as e:
            print(f"Astrometry API solver failed for {server_url}: {e}")
            import traceback
            traceback.print_exc()

    return None


def solve_for_image(image_path, data, sources, nx, ny, args, individuals_dir):
    """Solves astrometry for an image, writing a temporary FITS for non-FITS inputs."""
    # Solve Astrometry
    wcs = None
    if args.astrometry:
        print("Attempting to solve astrometry...")
        ext_check = os.path.splitext(image_path)[1].lower()
        if ext_check not in ['.fits', '.fit']:
            temp_fits = os.path.join(individuals_dir, f'temp_astrometry_{os.path.basename(image_path)}.fits')
            from astropy.io import fits
            fits.writeto(temp_fits, data, overwrite=True)
            wcs = solve_astrometry(temp_fits, sources, width=nx, height=ny, api_key=args.api_key,
                                   scale_low=args.scale_low, scale_high=args.scale_high)
            try:
                os.remove(temp_fits)
            except OSError:
                pass
        else:
            wcs = solve_astrometry(image_path, sources, width=nx, height=ny, api_key=args.api_key,
                                   scale_low=args.scale_low, scale_high=args.scale_high)
            
        if not wcs:
            print("!!! Astrometry failed. Some features will be limited.")
    return wcs
