"""Reference catalog queries (Gaia, Tycho-2) with tiling for large fields."""

import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


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


def _perform_single_query(center_sky, radius_deg, catalog_name):
    """Helper for a single catalog query."""
    try:
        if catalog_name == "gaia":
            # Try loading local Gaia catalog first
            local_data = load_local_gaia_catalog()
            if local_data is not None:
                # Filter by simple box mask first for speed
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
                box_mask = ra_mask & dec_mask
                filtered = local_data[box_mask]

                # Apply precise separation check
                if len(filtered) > 0:
                    catalog_coords = SkyCoord(ra=filtered[:, 0]*u.deg, dec=filtered[:, 1]*u.deg)
                    sep = center_sky.separation(catalog_coords)
                    circle_mask = sep.deg <= radius_deg
                    filtered = filtered[circle_mask]

                # Format as astropy Table to match astroquery interface
                from astropy.table import Table
                table = Table()
                if len(filtered) > 0:
                    table['ra'] = filtered[:, 0]
                    table['dec'] = filtered[:, 1]
                    table['phot_g_mean_mag'] = filtered[:, 2]
                else:
                    table['ra'] = np.array([], dtype=float)
                    table['dec'] = np.array([], dtype=float)
                    table['phot_g_mean_mag'] = np.array([], dtype=float)
                print(f"Loaded {len(table)} stars from local Gaia catalog.")
                return table

            print("Local Gaia catalog not found. Falling back to online Gaia...")
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
    """
    from astropy.table import vstack, unique
    
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
