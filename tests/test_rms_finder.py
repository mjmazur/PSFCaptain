import numpy as np

from psfcaptain.finders import RMSStarFinder


def _star_field(sigma_x, sigma_y, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = 200, 300
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(10, 1, (ny, nx))
    for x, y in [(60, 50), (150, 100), (240, 150)]:
        data += 500 * np.exp(-((xx - x) ** 2 / (2 * sigma_x ** 2) + (yy - y) ** 2 / (2 * sigma_y ** 2)))
    return data


def test_round_star_fwhm():
    sources = RMSStarFinder(fwhm=4.7, threshold=20)(_star_field(2.0, 2.0))
    assert len(sources) == 3
    assert np.allclose(sources['rms_fwhm'], 2.355 * 2.0, rtol=0.02)


def test_elongated_star_fwhm_and_orientation():
    sources = RMSStarFinder(fwhm=4.7, threshold=20)(_star_field(3.0, 1.5))
    assert len(sources) == 3
    # FWHM of the equal-area circular Gaussian, elongation a/b, major axis along +x.
    assert np.allclose(sources['rms_fwhm'], 2.355 * np.sqrt(3.0 * 1.5), rtol=0.02)
    assert np.allclose(sources['rms_elongation'], 2.0, rtol=0.03)
    assert np.all(np.abs(sources['rms_theta']) < 2)
