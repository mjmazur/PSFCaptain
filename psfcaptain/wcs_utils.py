"""Small WCS helpers shared across the pipeline."""


def safe_pixel_to_world(wcs_obj, x, y):
    """Safely handles WCS pixel_to_world conversions which may return lists of coords for multi-axis FITS."""
    coord = wcs_obj.pixel_to_world(x, y)
    if isinstance(coord, (list, tuple)):
        return coord[0]
    return coord
