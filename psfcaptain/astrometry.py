"""Astrometric plate solving via local solvers or the astrometry.net API."""

import os

import numpy as np


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

def solve_astrometry(image_path, sources=None, width=None, height=None, api_key=None, scale_low=20.0, scale_high=50.0):
    """Solves astrometry using local astrometry.net if available, falling back to remote."""
    import shutil
    
    # 1. Try python 'astrometry' package (installed via pip install astrometry)
    try:
        import astrometry
        from astropy.wcs import WCS
        
        if sources is not None and len(sources) > 0:
            num_sources = min(len(sources), 100)
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
        print(f"Trying astrometry solve via API at {server_url}...")
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

            # Run solving
            if sources is not None and width is not None and height is not None:
                num_sources = min(len(sources), 100)
                if num_sources < 10:
                    print(f"Warning: Only {num_sources} stars found. Astrometry may fail without more stars.")
                
                print(f"Submitting top {num_sources} sources to Astrometry...")
                sorted_indices = np.argsort(sources['peak'])[::-1]
                x_sorted = np.asarray(sources['xcentroid'])[sorted_indices][:num_sources]
                y_sorted = np.asarray(sources['ycentroid'])[sorted_indices][:num_sources]
                
                wcs_header = ast.solve_from_source_list(x_sorted, y_sorted, width, height, 
                                                       solve_timeout=300,
                                                       scale_units='arcsecperpix',
                                                       scale_lower=scale_low,
                                                       scale_upper=scale_high)
            else:
                print("Submitting full image to Astrometry...")
                wcs_header = ast.solve_from_image(image_path, solve_timeout=300,
                                                 scale_units='arcsecperpix',
                                                 scale_lower=scale_low,
                                                 scale_upper=scale_high)
                
            if wcs_header:
                from astropy.wcs import WCS
                wcs = WCS(wcs_header)
                print(f"Astrometry solve successful via {server_url}!")
                return wcs
            else:
                print(f"Astrometry failed via {server_url}.")
        except Exception as e:
            print(f"Astrometry API solver failed for {server_url}: {e}")

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
