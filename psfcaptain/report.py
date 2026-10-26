"""Single-page lens quality report (PDF) built from the per-image CSVs and quality sidecars.

Numbers in the summary table are measured per frame and then averaged across
frames (mean, with the frame-to-frame standard deviation when there is more than
one frame). Field maps and radial profiles pool the stars of all frames in
normalized sensor coordinates, so every frame contributes to every cell.

Can be run on its own to rebuild a report from an existing CSVs/ directory:

    python -m psfcaptain.report path/to/CSVs -o report.pdf
"""

import argparse
import datetime
import glob
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

# Palette: one-hue sequential ramp for maps, neutral ink for text and scatter,
# two categorical slots for the binned profile lines.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'psf_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
SEQUENTIAL.set_bad('#eeeeec')
SERIES_1 = '#2a78d6'
SERIES_2 = '#eb6834'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8984'
GRID = '#e4e3df'

# Radial zones, as a fraction of the center-to-corner distance.
ZONES = (('Center', 0.0, 1 / 3), ('Middle', 1 / 3, 2 / 3), ('Edge', 2 / 3, 1.01))
CORNERS = ('BL', 'BR', 'TL', 'TR')
MIN_PER_CELL = 3
MIN_MATCHED_FOR_LIMIT = 30


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_frames(csvs_dir, image_names=None):
    """Returns a list of (meta, df) pairs for frames with a quality sidecar and results CSV."""
    if image_names is None:
        sidecars = sorted(glob.glob(os.path.join(csvs_dir, '*_quality.json')))
        image_names = [os.path.basename(p)[:-len('_quality.json')] for p in sidecars]

    frames = []
    for name in image_names:
        meta_path = os.path.join(csvs_dir, f'{name}_quality.json')
        csv_path = os.path.join(csvs_dir, f'{name}_results.csv')
        if not (os.path.exists(meta_path) and os.path.exists(csv_path)):
            print(f"Report: skipping {name} (no quality sidecar or results CSV).")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(csv_path)
        if df.empty or 'snr' not in df.columns:
            print(f"Report: skipping {name} (no measured stars).")
            continue
        frames.append((meta, _derive_columns(meta, df)))
    return frames


def _derive_columns(meta, df):
    """Adds normalized position, zone, shape and depth columns to a frame's stars."""
    nx, ny = meta['nx'], meta['ny']
    cx, cy = nx / 2.0, ny / 2.0
    df = df.copy()
    df['u'] = df['xcentroid'] / nx
    df['v'] = df['ycentroid'] / ny
    df['rho'] = np.hypot(df['xcentroid'] - cx, df['ycentroid'] - cy) / np.hypot(cx, cy)
    df['zone'] = None
    for label, lo, hi in ZONES:
        df.loc[(df['rho'] >= lo) & (df['rho'] < hi), 'zone'] = label
    third_u = np.where(df['u'] < 1 / 3, 'L', np.where(df['u'] > 2 / 3, 'R', ''))
    third_v = np.where(df['v'] < 1 / 3, 'B', np.where(df['v'] > 2 / 3, 'T', ''))
    df['corner'] = [v + u if u and v else None for u, v in zip(third_u, third_v)]

    good = (df['fwhm'] > 0) & np.isfinite(df['fwhm'])
    df = df[good].copy()
    if 'fwhm_arc' not in df.columns or df['fwhm_arc'].isna().all():
        df['fwhm_arc'] = df['fwhm'] * df['pix_scale'] if 'pix_scale' in df.columns else np.nan
    df['ellipticity'] = 1.0 - 1.0 / df['elongation'].clip(lower=1.0)

    unsaturated = ~df['saturated'].astype(bool) if 'saturated' in df.columns else True
    usable = unsaturated & (df['snr'] > 0) & (df['ap_flux'] > 0)
    has_cat = 'catalog_mag' in df.columns
    matched = usable & df['catalog_mag'].notna() if has_cat else pd.Series(False, index=df.index)
    df['matched'] = matched

    # Relative illumination from stars: aperture counts per unit catalog flux, normalized
    # to the median of the frame's central stars (all matched stars if none are central).
    df['rel_illum'] = np.nan
    if matched.any():
        throughput = 10 ** (-0.4 * (df.loc[matched, 'mag_ap'] - df.loc[matched, 'catalog_mag']))
        central = df.loc[matched, 'zone'] == 'Center'
        reference = np.median(throughput[central]) if central.sum() >= 3 else np.median(throughput)
        df.loc[matched, 'rel_illum'] = throughput / reference

    # 5-sigma limiting magnitude implied by each star: its magnitude plus 2.5 log10(S/N / 5).
    # The S/N is background-limited, so this extrapolation is exact for faint sources.
    # With a catalog magnitude the estimate includes vignetting; otherwise it uses the
    # frame's zero point (or stays instrumental) and reflects PSF size and sky noise only.
    snr_term = 2.5 * np.log10(df['snr'].where(usable) / 5.0)
    df['mlim_cat'] = (df['catalog_mag'] + snr_term).where(matched) if has_cat else np.nan
    base = df['mag_abs'] if df['mag_abs'].notna().any() else df['mag_instr']
    df['mlim_field'] = base + snr_term
    return df


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def _median(series, min_count=MIN_PER_CELL):
    series = pd.Series(series).dropna()
    return float(series.median()) if len(series) >= min_count else np.nan


def _sky_relative(meta):
    """Sky-background grid normalized to the median of its central cells."""
    grid = np.asarray(meta['sky_grid'], dtype=float)
    n = grid.shape[0]
    c0, c1 = int(n * 0.4), int(np.ceil(n * 0.6))
    center = np.median(grid[c0:c1, c0:c1])
    # A black-clipped or bias-subtracted camera can leave the sky at zero: nothing to normalize.
    if not np.isfinite(center) or center <= 0:
        return np.full_like(grid, np.nan)
    return grid / center


def frame_metrics(meta, df, limit_column):
    """Scalar quality metrics for one frame, keyed '<metric>|<region>'."""
    m = {'frame': meta['image'], 'stars|Field': len(df)}
    regions = [('Field', df)] + [(label, df[df['zone'] == label]) for label, _, _ in ZONES]
    regions += [(c, df[df['corner'] == c]) for c in CORNERS]

    for region, sub in regions:
        m[f'fwhm|{region}'] = _median(sub['fwhm'])
        m[f'fwhm_arc|{region}'] = _median(sub['fwhm_arc'])
        m[f'elongation|{region}'] = _median(sub['elongation'])
        m[f'mlim|{region}'] = _median(sub[limit_column])
        m[f'rel_illum|{region}'] = _median(sub['rel_illum']) * 100
        if region != 'Field':
            m[f'stars|{region}'] = len(sub)

    corner_fwhm = [m[f'fwhm|{c}'] for c in CORNERS]
    m['fwhm|Corners'] = np.nanmean(corner_fwhm) if np.isfinite(corner_fwhm).any() else np.nan
    for key in ('fwhm_arc', 'elongation', 'mlim', 'rel_illum'):
        vals = [m[f'{key}|{c}'] for c in CORNERS]
        m[f'{key}|Corners'] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
    m['stars|Corners'] = sum(m[f'stars|{c}'] for c in CORNERS)
    if np.isfinite(corner_fwhm).sum() >= 2:
        m['corner_fwhm_spread'] = (np.nanmax(corner_fwhm) - np.nanmin(corner_fwhm)) / m['fwhm|Center'] * 100 \
            if np.isfinite(m['fwhm|Center']) else np.nan

    # Sky relative illumination by zone, from the normalized background grid.
    sky = _sky_relative(meta)
    n = sky.shape[0]
    g = (np.arange(n) + 0.5) / n
    gu, gv = np.meshgrid(g, g)
    nx, ny = meta['nx'], meta['ny']
    grho = np.hypot((gu - 0.5) * nx, (gv - 0.5) * ny) / np.hypot(nx / 2, ny / 2)
    m['sky_illum|Field'] = float(np.median(sky)) * 100
    for label, lo, hi in ZONES:
        m[f'sky_illum|{label}'] = float(np.median(sky[(grho >= lo) & (grho < hi)])) * 100
    k = max(1, n // 6)
    m['sky_illum|Corners'] = float(np.mean([sky[:k, :k].mean(), sky[:k, -k:].mean(),
                                            sky[-k:, :k].mean(), sky[-k:, -k:].mean()])) * 100

    m['scale|Center'] = meta.get('pixel_scale_center', np.nan)
    m['scale|Corners'] = meta.get('pixel_scale_corners', np.nan)
    if np.isfinite(m['scale|Center']) and np.isfinite(m['scale|Corners']):
        m['distortion'] = (m['scale|Corners'] / m['scale|Center'] - 1) * 100
    mags = df['mag_abs'] if df['mag_abs'].notna().any() else df['mag_instr']
    m['detection_limit'] = float(np.nanpercentile(mags, 95)) if len(mags.dropna()) else np.nan
    m['zero_point'] = meta.get('zero_point') if meta.get('zero_point') is not None else np.nan
    m['sky_median'] = meta['sky_median']
    m['sky_rms'] = meta['sky_rms_median']
    m['n_detected'] = meta.get('n_detected', np.nan)
    m['matched'] = int(df['matched'].sum())
    return m


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _style_axis(ax, title):
    ax.set_title(title, loc='left', fontsize=8, color=INK, fontweight='bold', pad=4)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=6, length=2)


def _grid_shape(n_stars, aspect):
    """Number of map cells (x, y): aims for ~8 stars per cell, 3..12 cells a side."""
    cells = max(n_stars / 8.0, 6)
    nbx = int(np.clip(round(np.sqrt(cells * aspect)), 3, 12))
    nby = int(np.clip(round(nbx / aspect), 2, 12))
    return nbx, nby


def _binned_map(stars, column, nbx, nby):
    """Per-cell median and count of `column` on an (nby, nbx) grid in normalized coordinates."""
    sub = stars.dropna(subset=[column])
    if sub.empty:
        return np.full((nby, nbx), np.nan), np.zeros((nby, nbx))
    bins = [np.linspace(0, 1, nbx + 1), np.linspace(0, 1, nby + 1)]
    med = binned_statistic_2d(sub['u'], sub['v'], sub[column], 'median', bins=bins).statistic
    cnt = binned_statistic_2d(sub['u'], sub['v'], sub[column], 'count', bins=bins).statistic
    med[cnt < MIN_PER_CELL] = np.nan
    return med.T, cnt.T


def _draw_map(fig, ax, grid, nx, ny, title, label, fmt, empty_note):
    _style_axis(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])
    if not np.isfinite(grid).any():
        ax.set_facecolor('#f6f6f4')
        ax.text(0.5, 0.5, empty_note, ha='center', va='center', fontsize=6.5, color=INK_MUTED,
                transform=ax.transAxes, wrap=True)
        ax.set_aspect(ny / nx)
        return
    im = ax.imshow(grid, origin='lower', extent=(0, nx, 0, ny), cmap=SEQUENTIAL, aspect='equal',
                   interpolation='nearest')
    nby, nbx = grid.shape
    if nbx * nby <= 72:
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        for j in range(nby):
            for i in range(nbx):
                val = grid[j, i]
                if not np.isfinite(val):
                    continue
                dark = vmax > vmin and (val - vmin) / (vmax - vmin) > 0.55
                ax.text((i + 0.5) * nx / nbx, (j + 0.5) * ny / nby, fmt.format(val), ha='center',
                        va='center', fontsize=5.5, color='white' if dark else INK)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.06, pad=0.03, aspect=30)
    cbar.set_label(label, fontsize=6, color=INK_2)
    cbar.ax.tick_params(labelsize=5.5, colors=INK_2, length=2)
    cbar.outline.set_edgecolor(GRID)


def _draw_whiskers(ax, stars, nbx, nby, nx, ny):
    """Ellipticity whiskers per cell: spin-2 mean of (e cos 2theta, e sin 2theta)."""
    _style_axis(ax, 'PSF shape (ellipticity whiskers)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    sub = stars.dropna(subset=['ellipticity', 'theta'])
    if sub.empty:
        return
    angle = np.radians(sub['theta'].values)
    e1 = sub['ellipticity'].values * np.cos(2 * angle)
    e2 = sub['ellipticity'].values * np.sin(2 * angle)
    bins = [np.linspace(0, 1, nbx + 1), np.linspace(0, 1, nby + 1)]
    m1 = binned_statistic_2d(sub['u'], sub['v'], e1, 'median', bins=bins).statistic.T
    m2 = binned_statistic_2d(sub['u'], sub['v'], e2, 'median', bins=bins).statistic.T
    cnt = binned_statistic_2d(sub['u'], sub['v'], e1, 'count', bins=bins).statistic.T
    emag = np.hypot(m1, m2)
    emag[cnt < MIN_PER_CELL] = np.nan
    if not np.isfinite(emag).any():
        return
    cell = min(nx / nbx, ny / nby)
    e_ref = max(0.05, float(np.round(np.nanmax(emag), 2)))
    scale = 0.85 * cell / e_ref
    for j in range(nby):
        for i in range(nbx):
            if not np.isfinite(emag[j, i]):
                continue
            phi = 0.5 * np.arctan2(m2[j, i], m1[j, i])
            half = 0.5 * emag[j, i] * scale
            x0, y0 = (i + 0.5) * nx / nbx, (j + 0.5) * ny / nby
            ax.plot([x0 - half * np.cos(phi), x0 + half * np.cos(phi)],
                    [y0 - half * np.sin(phi), y0 + half * np.sin(phi)],
                    color=SERIES_1, lw=2, solid_capstyle='round')
    for k in range(1, nbx):
        ax.axvline(k * nx / nbx, color=GRID, lw=0.5)
    for k in range(1, nby):
        ax.axhline(k * ny / nby, color=GRID, lw=0.5)
    ax.plot([0.05 * nx, 0.05 * nx + e_ref * scale], [-0.07 * ny] * 2, color=SERIES_1, lw=2,
            solid_capstyle='round', clip_on=False)
    ax.text(0.07 * nx + e_ref * scale, -0.07 * ny, f'e = {e_ref:.2f}  (e = 1 - b/a)', va='center',
            fontsize=6, color=INK_2, clip_on=False)


def _radial_profile(rho, values, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    centers, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = values[(rho >= a) & (rho < b)]
        sel = sel[np.isfinite(sel)]
        if len(sel) >= MIN_PER_CELL:
            centers.append(0.5 * (a + b))
            med.append(np.median(sel))
            lo.append(np.percentile(sel, 25))
            hi.append(np.percentile(sel, 75))
    return np.array(centers), np.array(med), np.array(lo), np.array(hi)


def _draw_radial(ax, stars, column, title, ylabel, color=SERIES_1, label=None, scatter=True):
    _style_axis(ax, title)
    sub = stars.dropna(subset=[column])
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Field radius (0 = center, 1 = corner)', fontsize=6, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=6, color=INK_2)
    if sub.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=6.5, color=INK_MUTED,
                transform=ax.transAxes)
        return
    if scatter:
        ax.scatter(sub['rho'], sub[column], s=3, color=INK_MUTED, alpha=0.35, lw=0)
    n_bins = int(np.clip(len(sub) // 15, 4, 10))
    c, med, lo, hi = _radial_profile(sub['rho'].values, sub[column].values, n_bins)
    if len(c):
        ax.fill_between(c, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(c, med, color=color, lw=2, marker='o', ms=3, label=label or 'Median (band: IQR)')
        y_lo, y_hi = np.nanpercentile(sub[column], [1, 99])
        pad = 0.1 * (y_hi - y_lo or 1)
        ax.set_ylim(min(y_lo, lo.min()) - pad, max(y_hi, hi.max()) + pad)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(values, fmt, multi):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return '–'
    text = fmt.format(values.mean())
    if multi and len(values) > 1:
        text += ' ± ' + fmt.format(values.std(ddof=1))
    return text


def _draw_table(ax, metrics, limit_label):
    ax.axis('off')
    multi = len(metrics) > 1
    cols = ['Field', 'Center', 'Middle', 'Edge', 'Corners']
    rows = [
        ('Stars measured / frame', 'stars', '{:.0f}'),
        ('FWHM (px)', 'fwhm', '{:.2f}'),
        ('FWHM (arcsec)', 'fwhm_arc', '{:.1f}'),
        ('Elongation (a/b)', 'elongation', '{:.2f}'),
        (limit_label, 'mlim', '{:.2f}'),
        ('Rel. illumination, stars (%)', 'rel_illum', '{:.0f}'),
        ('Rel. illumination, sky (%)', 'sky_illum', '{:.0f}'),
    ]
    cell_text = []
    for label, key, fmt in rows:
        row = [_fmt([m.get(f'{key}|{c}') for m in metrics], fmt, multi) for c in cols]
        if all(t == '–' for t in row):
            continue
        cell_text.append([label] + row)
    widths = [0.25] + [0.15] * len(cols)
    table = ax.table(cellText=cell_text, colLabels=[''] + cols, colWidths=widths, cellLoc='center',
                     edges='horizontal', bbox=[0.0, 0.0, 1.0, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(6.3 if multi else 7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK_2 if c == 0 else INK)
        if c == 0:
            cell.get_text().set_horizontalalignment('left')
            cell.PAD = 0.02
        if r == 0:
            cell.get_text().set_fontweight('bold')
            cell.get_text().set_color(INK)


def _key_numbers(metrics, limit_label):
    multi = len(metrics) > 1
    get = lambda k: [m.get(k) for m in metrics]
    items = [
        (limit_label + ', field', _fmt(get('mlim|Field'), '{:.2f}', multi)),
        ('FWHM, field', _fmt(get('fwhm|Field'), '{:.2f}', multi) + ' px'),
        ('FWHM edge/center', _fmt([m['fwhm|Edge'] / m['fwhm|Center'] if m['fwhm|Center'] else np.nan
                                   for m in metrics], '{:.2f}', multi) + '×'),
        ('Corner FWHM spread (tilt)', _fmt(get('corner_fwhm_spread'), '{:.0f}', multi) + ' %'),
        ('Corner illumination, sky', _fmt(get('sky_illum|Corners'), '{:.0f}', multi) + ' %'),
    ]
    return items


def _secondary_numbers(metrics):
    multi = len(metrics) > 1
    get = lambda k: [m.get(k) for m in metrics]
    return [
        ('Detection limit (95th pct mag)', _fmt(get('detection_limit'), '{:.2f}', multi)),
        ('Photometric zero point', _fmt(get('zero_point'), '{:.2f}', multi)),
        ('Catalog-matched stars / frame', _fmt(get('matched'), '{:.0f}', multi)),
        ('Pixel scale center ("/px)', _fmt(get('scale|Center'), '{:.2f}', multi)),
        ('Scale change center→corner (%)', _fmt(get('distortion'), '{:+.1f}', multi)),
        ('Sky level (counts)', _fmt(get('sky_median'), '{:.0f}', multi)),
        ('Sky noise (counts/px)', _fmt(get('sky_rms'), '{:.1f}', multi)),
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _header_value(metas, *keys):
    """Distinct values of the first header keyword in `keys` found in any frame."""
    for key in keys:
        values = [m.get('header', {}).get(key) for m in metas]
        values = [v for v in values if v not in (None, '')]
        if key == 'EXPTIME' and not values:
            values = [m['exptime'] for m in metas if m.get('exptime') is not None]
        if values:
            return sorted(set(values), key=str)
    return []


def _num(value):
    """Compact number: 0.04, 15, 327 (strings pass through)."""
    try:
        return f'{float(value):g}'
    except (TypeError, ValueError):
        return str(value)


def _exposure_lines(metas):
    """Header lines describing exposure, camera and site, from whichever FITS keywords exist."""
    def show(values, unit='', fmt=_num):
        if not values:
            return None
        if len(values) == 1:
            return fmt(values[0]) + unit
        try:
            return f'{fmt(min(values, key=float))}–{fmt(max(values, key=float))}{unit}'
        except (TypeError, ValueError):
            return 'varies'

    def exposure(v):
        v = float(v)
        return f'{v * 1000:g} ms' if v < 1 else f'{v:g} s'

    exposure_parts = [
        ('Exposure', show(_header_value(metas, 'EXPTIME', 'EXPOSURE'), fmt=exposure)),
        ('Gain', show(_header_value(metas, 'GAIN', 'EGAIN'))),
        ('ISO', show(_header_value(metas, 'ISOSPEED'))),
        ('Frame rate', show(_header_value(metas, 'FRAMERAT'), ' fps')),
        ('Filter', show(_header_value(metas, 'FILTER'))),
        ('Sensor temp', show(_header_value(metas, 'CCD-TEMP'), ' °C', fmt=lambda v: f'{float(v):.1f}')),
    ]
    xbin, ybin = _header_value(metas, 'XBINNING'), _header_value(metas, 'YBINNING')
    if xbin or ybin:
        exposure_parts.append(('Binning', f'{show(xbin) or "?"} × {show(ybin) or "?"}'))
    exposure_parts.append(('Pixel format', show(_header_value(metas, 'PIXFMT'))))

    camera = _header_value(metas, 'INSTRUME', 'TELESCOP')
    camera_parts = [('Camera', ', '.join(map(str, camera)) if camera else None),
                    ('Focal length', show(_header_value(metas, 'FOCALLEN'), ' mm')),
                    ('Aperture', show(_header_value(metas, 'APTDIA'), ' mm'))]
    lat, lon = _header_value(metas, 'SITELAT'), _header_value(metas, 'SITELONG')
    if len(lat) == 1 and len(lon) == 1:
        elev = _header_value(metas, 'SITEELEV')
        site = f'{float(lat[0]):.4f}°, {float(lon[0]):.4f}°'
        camera_parts.append(('Site', site + (f', {_num(elev[0])} m' if len(elev) == 1 else '')))

    lines = []
    for parts in (exposure_parts, camera_parts):
        shown = [f'{label}: {value}' for label, value in parts if value]
        if shown:
            lines.append('   ·   '.join(shown))
    return lines


def build_report(csvs_dir, output_path, image_names=None, title=None):
    """Builds the single-page PDF report. Returns the output path, or None if nothing to report."""
    frames = load_frames(csvs_dir, image_names)
    if not frames:
        print("Report: no frames with quality measurements; no report written.")
        return None

    metas = [meta for meta, _ in frames]
    stars = pd.concat([df for _, df in frames], ignore_index=True)
    nx, ny = metas[0]['nx'], metas[0]['ny']

    # Limiting magnitude: catalog-anchored (includes vignetting) when enough stars match.
    use_catalog_limit = stars['matched'].sum() >= MIN_MATCHED_FOR_LIMIT
    limit_column = 'mlim_cat' if use_catalog_limit else 'mlim_field'
    calibrated = any(m.get('zero_point') is not None for m in metas)
    if use_catalog_limit:
        limit_label = 'Limiting mag (5σ)'
        limit_note = ('5σ limiting magnitude from catalog-matched stars: m_cat + 2.5 log10(S/N / 5), '
                      'so it includes PSF size, sky noise and vignetting.')
    elif calibrated:
        limit_label = 'Limiting mag (5σ)'
        limit_note = (f'5σ limiting magnitude from all stars using the frame zero point (fewer than '
                      f'{MIN_MATCHED_FOR_LIMIT} catalog matches): includes PSF size and sky noise but '
                      'not vignetting - see the illumination panels.')
    else:
        limit_label = 'Limiting mag (5σ, instr.)'
        limit_note = ('5σ limiting magnitude on the instrumental scale (no absolute calibration; '
                      'run with --astrometry --absolute for catalog magnitudes).')

    metrics = [frame_metrics(meta, df, limit_column) for meta, df in frames]
    pd.DataFrame(metrics).to_csv(os.path.splitext(output_path)[0] + '_metrics.csv', index=False)

    plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'axes.titlesize': 8})
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(5, 3, figure=fig, height_ratios=[0.8, 1.05, 1.25, 1.25, 1.4],
                  left=0.07, right=0.97, top=0.965, bottom=0.09, hspace=0.5, wspace=0.28)

    # Header
    head = fig.add_subplot(gs[0, :])
    head.axis('off')
    head.text(0, 1.0, title or 'Lens Quality Report', fontsize=15, fontweight='bold', color=INK, va='top')
    dates = sorted(m['date_obs'] for m in metas if m.get('date_obs'))
    finder = metas[0].get('finder', '?')
    catalog = metas[0].get('catalog') or 'none'
    scales = [m['pixel_scale_center'] for m in metas if m.get('pixel_scale_center')]
    scale = float(np.mean(scales)) if scales else np.nan
    names = [os.path.splitext(m['image'])[0] for m in metas]
    frame_text = names[0] if len(names) == 1 else f'{len(names)} ({names[0]} … {names[-1]})'
    info = [
        f'Frames: {frame_text}',
        f'Sensor: {nx} × {ny} px' + (f'   ·   Scale: {scale:.2f}"/px   ·   Field: '
                                    f'{nx * scale / 3600:.1f}° × {ny * scale / 3600:.1f}°' if np.isfinite(scale) else ''),
        f'Finder: {finder}   ·   Catalog: {catalog}   ·   Aperture radius: '
        f'{np.nanmean([m.get("aperture_radius") or np.nan for m in metas]):.1f} px   ·   '
        f'Stars used: {len(stars)}',
        *_exposure_lines(metas),
        (f'Observed: {dates[0][:19]}' + (f' → {dates[-1][:19]}' if len(dates) > 1 else '') + '   ·   ' if dates else '')
        + f'Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}',
    ]
    head.text(0, 0.7, '\n'.join(info), fontsize=6.8, color=INK_2, va='top', linespacing=1.6)

    # Key numbers (right side of header)
    key_ax = head.inset_axes([0.62, -0.05, 0.38, 1.05])
    key_ax.axis('off')
    for i, (label, value) in enumerate(_key_numbers(metrics, limit_label)):
        y = 1 - i * 0.2
        key_ax.text(0, y, label, fontsize=6.8, color=INK_2, va='top')
        key_ax.text(1, y, value, fontsize=7.5, color=INK, fontweight='bold', va='top', ha='right')

    # Summary table + secondary numbers
    table_ax = fig.add_subplot(gs[1, :2])
    _style_axis(table_ax, 'Quality across the field' + (f'  (mean ± std of {len(frames)} frames)'
                                                        if len(frames) > 1 else ''))
    _draw_table(table_ax, metrics, limit_label)
    side = fig.add_subplot(gs[1, 2])
    side.axis('off')
    _style_axis(side, 'Calibration & geometry')
    numbers = [(label, value) for label, value in _secondary_numbers(metrics) if value != '–']
    for i, (label, value) in enumerate(numbers):
        y = 0.86 - i * 0.13
        side.text(0, y, label, fontsize=6.5, color=INK_2, va='top', transform=side.transAxes)
        side.text(1, y, value, fontsize=7.2, color=INK, fontweight='bold', va='top', ha='right',
                  transform=side.transAxes)
        side.plot([0, 1], [y - 0.1] * 2, color=GRID, lw=0.5, transform=side.transAxes)

    # Field maps
    nbx, nby = _grid_shape(len(stars), nx / ny)
    ax = fig.add_subplot(gs[2, 0])
    _draw_map(fig, ax, _binned_map(stars, 'fwhm', nbx, nby)[0], nx, ny, 'PSF size (FWHM)',
              'FWHM (px), cell median', '{:.2f}', 'No FWHM measurements')
    ax = fig.add_subplot(gs[2, 1])
    _draw_whiskers(ax, stars, nbx, nby, nx, ny)
    ax = fig.add_subplot(gs[2, 2])
    _draw_map(fig, ax, _binned_map(stars, limit_column, nbx, nby)[0], nx, ny, 'Limiting magnitude (5σ)',
              limit_label + ', cell median', '{:.1f}', 'Not enough stars')

    # Vignetting
    matched = stars[stars['matched']]
    mbx, mby = _grid_shape(len(matched), nx / ny)
    mbx, mby = max(3, mbx - 1), max(2, mby - 1)
    ax = fig.add_subplot(gs[3, 0])
    _draw_map(fig, ax, _binned_map(matched, 'rel_illum', mbx, mby)[0] * 100, nx, ny,
              'Relative illumination (stars)', '% of center, cell median', '{:.0f}',
              'Needs catalog-matched stars\n(--astrometry --absolute)')
    ax = fig.add_subplot(gs[3, 1])
    sky_grids = [g for g in (_sky_relative(m) for m in metas) if np.isfinite(g).all()]
    has_sky = bool(sky_grids)
    if has_sky:
        sky = gaussian_filter(np.mean(sky_grids, axis=0) * 100, 1.0, mode='nearest')
    else:
        sky = np.full((len(metas[0]['sky_grid']),) * 2, np.nan)
    _style_axis(ax, 'Relative illumination (sky background)')
    ax.set_xticks([])
    ax.set_yticks([])
    if has_sky:
        im = ax.imshow(sky, origin='lower', extent=(0, nx, 0, ny), cmap=SEQUENTIAL, aspect='equal',
                       interpolation='bilinear')
        levels = np.arange(np.floor(sky.min() / 5) * 5, sky.max() + 5, 5)
        if len(levels) > 1:
            gx = (np.arange(sky.shape[1]) + 0.5) / sky.shape[1] * nx
            gy = (np.arange(sky.shape[0]) + 0.5) / sky.shape[0] * ny
            cs = ax.contour(gx, gy, sky, levels=levels, colors='white', linewidths=0.6)
            ax.clabel(cs, fontsize=5, fmt='%d')
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.06, pad=0.03, aspect=30)
        cbar.set_label('% of center sky level', fontsize=6, color=INK_2)
        cbar.ax.tick_params(labelsize=5.5, colors=INK_2, length=2)
        cbar.outline.set_edgecolor(GRID)
    else:
        ax.set_facecolor('#f6f6f4')
        ax.text(0.5, 0.5, 'Sky background is zero at the center\n(black-clipped or bias-subtracted frames)',
                ha='center', va='center', fontsize=6.5, color=INK_MUTED, transform=ax.transAxes)
        ax.set_aspect(ny / nx)

    ax = fig.add_subplot(gs[3, 2])
    _style_axis(ax, 'Light falloff vs field radius')
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    n = sky.shape[0]
    g = (np.arange(n) + 0.5) / n
    gu, gv = np.meshgrid(g, g)
    grho = np.hypot((gu - 0.5) * nx, (gv - 0.5) * ny) / np.hypot(nx / 2, ny / 2)
    c, med, lo, hi = _radial_profile(grho.ravel(), sky.ravel(), 10)
    if has_sky:
        ax.fill_between(c, lo, hi, color=SERIES_2, alpha=0.15, lw=0)
        ax.plot(c, med, color=SERIES_2, lw=2, label='Sky background')
    if not matched.empty:
        ax.scatter(matched['rho'], matched['rel_illum'] * 100, s=3, color=INK_MUTED, alpha=0.35, lw=0)
        c, med, lo, hi = _radial_profile(matched['rho'].values, matched['rel_illum'].values * 100,
                                         int(np.clip(len(matched) // 15, 3, 10)))
        if len(c):
            ax.plot(c, med, color=SERIES_1, lw=2, marker='o', ms=3, label='Stars (catalog)')
    ax.axhline(100, color=INK_MUTED, lw=0.6, ls='--')
    shown = [sky.ravel()] if has_sky else [np.array([100.0])]
    if not matched.empty:
        shown.append(np.nanpercentile(matched['rel_illum'] * 100, [5, 95]))
    shown = np.concatenate(shown)
    ax.set_ylim(min(np.nanmin(shown), 90) - 5, max(np.nanmax(shown), 100) + 5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Field radius (0 = center, 1 = corner)', fontsize=6, color=INK_2)
    ax.set_ylabel('% of center', fontsize=6, color=INK_2)
    ax.legend(fontsize=6, frameon=False, loc='lower left')

    # Radial profiles
    _draw_radial(fig.add_subplot(gs[4, 0]), stars, 'fwhm', 'FWHM vs field radius', 'FWHM (px)')
    _draw_radial(fig.add_subplot(gs[4, 1]), stars, 'elongation', 'Elongation vs field radius',
                 'Elongation (a/b)')
    ax = fig.add_subplot(gs[4, 2])
    _draw_radial(ax, stars, limit_column, 'Limiting magnitude vs field radius', limit_label)

    notes = [
        limit_note,
        'Zones by distance from center as a fraction of the center-to-corner distance: center < 1/3, '
        'middle 1/3-2/3, edge > 2/3; corners are the four corner cells of a 3 × 3 grid.',
        'Table values are per-frame medians averaged over frames; maps and profiles pool all frames. '
        'Map cells with fewer than 3 stars are left blank. Sky illumination includes any sky gradient '
        'and camera bias offset; use evenly lit (twilight/flat) frames for a clean vignetting curve.',
    ]
    wrapped = '\n'.join(textwrap.fill('• ' + note, 175) for note in notes)
    fig.text(0.07, 0.012, wrapped, fontsize=5.3, color=INK_MUTED, va='bottom', linespacing=1.4)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Lens quality report saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build the single-page lens quality PDF from a CSVs/ directory.")
    parser.add_argument("csvs_dir", help="Directory holding *_results.csv and *_quality.json files.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PDF path (default: lens_quality_report.pdf next to the CSVs directory).")
    parser.add_argument("--title", default=None, help="Report title (e.g. the lens name).")
    args = parser.parse_args()
    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csvs_dir)),
                                         'lens_quality_report.pdf')
    build_report(args.csvs_dir, output, title=args.title)


if __name__ == "__main__":
    main()
