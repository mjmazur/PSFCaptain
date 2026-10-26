import json
import os

import numpy as np
import pandas as pd

from psfcaptain.quality import measure_field_quality, write_quality_meta
from psfcaptain.report import build_report


def _synthetic_field(nx=600, ny=400, sky=100.0, noise=5.0, n_stars=150, seed=1):
    """Gaussian stars on a vignetted sky (sky falls to ~70% in the corners)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rho2 = ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) / ((nx / 2) ** 2 + (ny / 2) ** 2)
    data = sky * (1 - 0.3 * rho2) + rng.normal(0, noise, (ny, nx))
    xs = rng.uniform(20, nx - 20, n_stars)
    ys = rng.uniform(20, ny - 20, n_stars)
    fluxes = 10 ** rng.uniform(2.5, 4.0, n_stars)
    sigma = 1.5
    for x, y, f in zip(xs, ys, fluxes):
        data += f / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    df = pd.DataFrame({
        'xcentroid': xs, 'ycentroid': ys, 'flux': fluxes,
        'mag_instr': -2.5 * np.log10(fluxes), 'mag_abs': np.nan,
        'fwhm': np.full(n_stars, 2.355 * sigma), 'fwhm_arc': np.nan,
        'elongation': np.full(n_stars, 1.1), 'theta': np.full(n_stars, 30.0),
    })
    return data, df


def test_snr_matches_injected_flux():
    data, df = _synthetic_field()
    df, meta = measure_field_quality(data, None, df, None, 600, 400)
    # Aperture of r = FWHM holds ~94% of a Gaussian's flux.
    ratio = np.median(df['ap_flux'] / df['flux'])
    assert 0.85 < ratio < 1.0
    expected_snr = df['ap_flux'] / (5.0 * np.sqrt(np.pi * meta['aperture_radius'] ** 2))
    assert np.allclose(np.median(df['snr'] / expected_snr), 1.0, atol=0.15)
    # The injected vignetting shows up in the sky grid: corners are darker than the center.
    sky = np.asarray(meta['sky_grid'])
    assert sky[0, 0] < 0.85 * sky[16, 16]


def test_build_report_writes_pdf_and_metrics(tmp_path):
    csvs = tmp_path / 'CSVs'
    csvs.mkdir()
    for i in range(2):
        data, df = _synthetic_field(seed=i)
        df, meta = measure_field_quality(data, None, df, None, 600, 400)
        meta.update({'image': f'frame{i}.fits', 'finder': 'dao', 'catalog': None, 'zero_point': None,
                     'n_detected': len(df), 'n_valid': len(df)})
        df.to_csv(csvs / f'frame{i}.fits_results.csv', index=False)
        write_quality_meta(meta, str(csvs), f'frame{i}.fits')

    out = tmp_path / 'report.pdf'
    assert build_report(str(csvs), str(out)) == str(out)
    assert out.stat().st_size > 10_000
    metrics = pd.read_csv(tmp_path / 'report_metrics.csv')
    assert len(metrics) == 2
    assert np.allclose(metrics['fwhm|Field'], 2.355 * 1.5)
    assert (metrics['sky_illum|Corners'] < 85).all()
