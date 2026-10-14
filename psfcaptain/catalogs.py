"""Reference catalog queries (Gaia, Tycho-2) with tiling for large fields.

Both catalogs prefer a local source over the network when one is available:
Gaia via an optional RMS installation's pre-filtered catalog file, Tycho-2 via
a one-time full-catalog download cached under .catalog_cache/. Local lookups
are a single box+circle filter over the whole field, so tiling (which exists
only to keep individual *online* queries small) is skipped whenever a local
catalog is in play.
"""

import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

# In-process memoization so a batch run (many tiles/images) loads each local
# catalog array from disk at most once per worker process.
_LOCAL_CATALOG_CACHE = {}

TYCHO2_CACHE_FILENAME = "tycho2_full.npy"


def _catalog_cache_dir():
    """Directory used to cache full catalog downloads, created on demand."""
    cache_dir = os.path.join(os.getcwd(), '.catalog_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _box_circle_filter(local_data, center_sky, radius_deg):
    """Filters an [ra, dec, mag] array to a cone: a fast box pre-filter, then a precise separation check."""
    ra_min = center_sky.ra.deg - radius_deg
    ra_max = center_sky.ra.deg + radius_deg
    dec_min = center_sky.dec.deg - radius_deg
    dec_max = center_sky.dec.deg + radius_deg

    if ra_min < 0 or ra_max > 360:
        ra_min_wrapped = ra_min % 360
        ra_max_wrapped = ra_max % 360
        if ra_min_wrapped > ra_max_wrapped:
            ra_mask = (local_data[:, 0] >= ra_min_wrapped) | (local_data[:, 0] <= ra_max_wrapped)
        else:
            ra_mask = (local_data[:, 0] >= ra_min_wrapped) & (local_data[:, 0] <= ra_max_wrapped)
    else:
        ra_mask = (local_data[:, 0] >= ra_min) & (local_data[:, 0] <= ra_max)

    dec_mask = (local_data[:, 1] >= dec_min) & (local_data[:, 1] <= dec_max)
    filtered = local_data[ra_mask & dec_mask]

    if len(filtered) > 0:
        catalog_coords = SkyCoord(ra=filtered[:, 0] * u.deg, dec=filtered[:, 1] * u.deg)
        sep = center_sky.separation(catalog_coords)
        filtered = filtered[sep.deg <= radius_deg]
    return filtered


def load_local_gaia_catalog(lim_mag=None):
    """Try to locate and load the local Gaia catalog from the RMS installation."""
    rms_dirs = [
        r"C:\Users\mjmaz\Documents\GitHub\RMS",
        os.path.expanduser("~/source/RMS"),
        os.path.expanduser("~/Documents/GitHub/RMS"),
    ]
    rms_dir = None
    for d in rms_dirs:
        if os.path.exists(d):
            rms_dir = d
            break

    if not rms_dir:
        return None

    cat_dir = os.path.join(rms_dir, "Catalogs")
    cat_file = "gaia_dr2_mag_11.5.npy"
    cat_path = os.path.join(cat_dir, cat_file)
    if not os.path.exists(cat_path):
        return None

    try:
        import sys
        if rms_dir not in sys.path:
            sys.path.append(rms_dir)
        from RMS.Formats.StarCatalog import loadGaiaCatalog
        data = loadGaiaCatalog(cat_dir, cat_file, lim_mag=lim_mag)
        return data
    except Exception as e:
        print(f"Warning: Failed to import/run loadGaiaCatalog: {e}. Falling back to direct numpy load.")
        try:
            results = np.load(cat_path, allow_pickle=False)
            if lim_mag is not None:
                results = results[results[:, 2] <= lim_mag]
            return results
        except Exception as e2:
            print(f"Error loading catalog directly: {e2}")
            return None


def fetch_and_cache_tycho2_catalog(cache_path):
    """Downloads the full Tycho-2 catalog once via VizieR and caches it as a local [ra, dec, VTmag] array."""
    from astroquery.vizier import Vizier

    print("Tycho-2 catalog not cached locally. Downloading the full catalog from VizieR "
          "(~2.5 million stars, one-time only; this may take several minutes)...")
    v = Vizier(columns=['RA(ICRS)', 'DE(ICRS)', 'VTmag'], catalog=['I/259/tyc2'])
    v.ROW_LIMIT = -1
    v.TIMEOUT = 1800
    tabs = v.get_catalogs('I/259/tyc2')
    if not tabs:
        raise RuntimeError("VizieR returned no Tycho-2 tables.")
    table = tabs[0]

    ra = np.asarray(table['RA(ICRS)'], dtype=float)
    dec = np.asarray(table['DE(ICRS)'], dtype=float)
    vtmag = np.asarray(table['VTmag'], dtype=float)

    # Drop stars with no VTmag measurement (masked/NaN in the source catalog)
    valid = ~np.isnan(vtmag)
    data = np.column_stack((ra[valid], dec[valid], vtmag[valid]))

    np.save(cache_path, data)
    print(f"Cached {len(data)} Tycho-2 stars to {cache_path}.")
    return data


def load_local_tycho2_catalog():
    """Loads the cached full Tycho-2 catalog, fetching and caching it from VizieR on first use."""
    cache_path = os.path.join(_catalog_cache_dir(), TYCHO2_CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=False)
        except Exception as e:
            print(f"Warning: Failed to load cached Tycho-2 catalog ({e}). Re-fetching...")

    try:
        return fetch_and_cache_tycho2_catalog(cache_path)
    except Exception as e:
        print(f"Warning: Failed to fetch full Tycho-2 catalog ({e}). Falling back to per-field VizieR queries.")
        return None


def _get_local_catalog_array(catalog_name):
    """Returns the memoized local [ra, dec, mag] array for `catalog_name`, or None if unavailable."""
    if catalog_name in _LOCAL_CATALOG_CACHE:
        return _LOCAL_CATALOG_CACHE[catalog_name]

    if catalog_name == "gaia":
        data = load_local_gaia_catalog()
        if data is None:
            print("Local Gaia catalog not found. Falling back to online Gaia...")
    elif catalog_name == "tycho2":
        data = load_local_tycho2_catalog()
    else:
        data = None

    _LOCAL_CATALOG_CACHE[catalog_name] = data
    return data


def _query_local_catalog(local_data, center_sky, radius_deg, catalog_name):
    """Filters a local catalog array to the requested cone and formats it like the matching astroquery result."""
    filtered = _box_circle_filter(local_data, center_sky, radius_deg)

    from astropy.table import Table
    table = Table()
    if catalog_name == "gaia":
        columns = {'ra': filtered[:, 0], 'dec': filtered[:, 1], 'phot_g_mean_mag': filtered[:, 2]}
    else:  # tycho2
        columns = {'RA(ICRS)': filtered[:, 0], 'DE(ICRS)': filtered[:, 1], 'VTmag': filtered[:, 2]}

    for name, values in columns.items():
        table[name] = values if len(filtered) > 0 else np.array([], dtype=float)

    label = "Gaia" if catalog_name == "gaia" else "Tycho-2"
    print(f"Loaded {len(table)} stars from local {label} catalog.")
    return table


def _perform_single_query(center_sky, radius_deg, catalog_name):
    """Helper for a single catalog query."""
    try:
        local_data = _get_local_catalog_array(catalog_name)
        if local_data is not None:
            return _query_local_catalog(local_data, center_sky, radius_deg, catalog_name)

        if catalog_name == "gaia":
            from astroquery.gaia import Gaia
            job = Gaia.cone_search_async(center_sky, radius=radius_deg * u.deg)
            return job.get_results()
        elif catalog_name == "tycho2":
            from astroquery.vizier import Vizier
            v = Vizier(catalog=['I/259/tyc2'])
            v.ROW_LIMIT = -1
            viz_results = v.query_region(center_sky, radius=radius_deg * u.deg)
            if viz_results:
                return viz_results[0]
    except Exception as e:
        print(f"  Query at {center_sky.to_string('hmsdms')} failed: {e}")
    return None

def query_catalog_tiled(center_sky, radius_deg, catalog_name="gaia", tile_size_deg=2.0):
    """
    Queries a catalog using a tiled approach to cover large fields without server timeouts.

    Tiling only helps keep *online* queries small, so it is skipped whenever a local
    catalog is available: the whole field is filtered from the in-memory array in one pass.
    """
    from astropy.table import vstack, unique

    if _get_local_catalog_array(catalog_name) is not None:
        return _perform_single_query(center_sky, radius_deg, catalog_name)

    # Calculate the number of tiles needed
    # We use a grid that covers a square of 2*radius x 2*radius
    num_tiles_side = int(np.ceil((2 * radius_deg) / tile_size_deg))
    if num_tiles_side <= 1:
        # Single query is enough
        return _perform_single_query(center_sky, radius_deg, catalog_name)

    print(f"Field is large ({radius_deg:.2f} deg radius). Dividing into {num_tiles_side}x{num_tiles_side} tiles...")

    all_results = []
    # Grid spacing (with 10% overlap)
    spacing = tile_size_deg * 0.9

    # Start from center - radius and go to center + radius
    # Note: This is an approximation for RA/Dec but sufficient for tiling a small-ish region of the sky
    # For very large regions or near poles, this would need more care.
    ra_start = center_sky.ra.deg - radius_deg
    dec_start = center_sky.dec.deg - radius_deg

    for i in range(num_tiles_side):
        for j in range(num_tiles_side):
            # Calculate tile center
            tile_ra = ra_start + (i + 0.5) * spacing
            tile_dec = dec_start + (j + 0.5) * spacing

            # Wrap RA
            tile_ra = tile_ra % 360.0
            # Clip Dec
            tile_dec = np.clip(tile_dec, -90.0, 90.0)

            tile_center = SkyCoord(ra=tile_ra, dec=tile_dec, unit=(u.deg, u.deg))
            # Search radius for each tile covers the tile area
            tile_radius = (tile_size_deg / np.sqrt(2)) * 1.1 # slightly larger to ensure coverage

            res = _perform_single_query(tile_center, tile_radius, catalog_name)
            if res is not None and len(res) > 0:
                all_results.append(res)

    if not all_results:
        return None

    combined = vstack(all_results)
    # Remove duplicates based on coordinates (rounded to avoid precision issues)
    # Different catalogs have different ID columns, so coordinates are a safe bet.
    ra_col = combined['ra'] if 'ra' in combined.colnames else combined['RA(ICRS)']
    dec_col = combined['dec'] if 'dec' in combined.colnames else combined['DE(ICRS)']
    combined['tmp_ra_round'] = np.round(np.asarray(ra_col), 6)
    combined['tmp_dec_round'] = np.round(np.asarray(dec_col), 6)

    final_table = unique(combined, keys=['tmp_ra_round', 'tmp_dec_round'])
    final_table.remove_columns(['tmp_ra_round', 'tmp_dec_round'])

    return final_table
