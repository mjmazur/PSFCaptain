"""Reference catalog queries (Gaia, Tycho-2) with tiling for large fields."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def _perform_single_query(center_sky, radius_deg, catalog_name):
    """Helper for a single catalog query."""
    try:
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
