"""Star detection algorithms.

Wraps the photutils finders and provides RMSStarFinder, a Gaussian-fitting
finder ported from the Pi in the Sky / RMS methodology.
"""

import numpy as np


def twoDGaussian(params, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """ Defines a 2D Gaussian distribution. """
    x, y, saturation = params

    if isinstance(saturation, np.ndarray):
        saturation = saturation[0, 0]
    
    xo = float(xo)
    yo = float(yo)

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2)))

    # Limit values to saturation level
    g[g > saturation] = saturation

    return g.ravel()

@np.vectorize
def gamma_correction(intensity, gamma, bp=0, wp=255):
    """ Correct the given intensity for gamma. """
    if intensity < 0:
        intensity = 0
    x = (intensity - bp) / (wp - bp)
    if x > 0:
        return bp + (wp - bp) * (x**(1.0/gamma))
    else:
        return bp

class RMSStarFinder:
    def __init__(self, fwhm, threshold, gamma=1.0, neighborhood_size=10, segment_radius=15, roundness_threshold=0.1, max_feature_ratio=2.0, exclude_border=False, xycoords=None):
        self.fwhm = fwhm
        self.threshold = threshold
        self.gamma = gamma
        self.neighborhood_size = neighborhood_size
        self.segment_radius = segment_radius
        self.roundness_threshold = roundness_threshold
        self.max_feature_ratio = max_feature_ratio
        self.exclude_border = exclude_border
        self.xycoords = xycoords

    def __call__(self, data):
        import scipy.ndimage as ndimage
        import scipy.optimize as opt
        from astropy.table import Table
        
        # We need a positive data for RMS finder. In star_measure.py, data is background subtracted
        # calculate image mean
        avepixel_mean = 0.0 # because it's already background subtracted
        
        # Apply a mean filter to the image to reduce noise
        data_filtered = ndimage.convolve(data.astype(np.float32), weights=np.full((2, 2), 1.0/4))
        
        if self.xycoords is not None:
            if len(self.xycoords) == 0:
                return None
            x = self.xycoords[:, 0]
            y = self.xycoords[:, 1]
        else:
            # Locate local maxima on the image
            data_max = ndimage.maximum_filter(data_filtered, self.neighborhood_size)
            maxima = (data_filtered == data_max)
            data_min = ndimage.minimum_filter(data_filtered, self.neighborhood_size)
            diff = ((data_max - data_min) > self.threshold)
            maxima[diff == 0] = 0
            
            if self.exclude_border:
                border = int(self.fwhm * 2)
                border_mask = np.ones_like(maxima)*255
                border_mask[:border,:] = 0
                border_mask[-border:,:] = 0
                border_mask[:,:border] = 0
                border_mask[:,-border:] = 0
                maxima[border_mask == 0] = 0
                
            # Find and label the maxima
            labeled, num_objects = ndimage.label(maxima)
            if num_objects == 0:
                return None
                
            # Find centres of mass of each labeled objects
            xy = np.array(ndimage.center_of_mass(data_filtered, labeled, range(1, num_objects+1)))
            
            if len(xy) == 0:
                return None
                
            # Unpack star coordinates
            y, x = np.hsplit(xy, 2)
            y = y.flatten()
            x = x.flatten()
        
        x_fitted = []
        y_fitted = []
        amplitude_fitted = []
        intensity_fitted = []
        fwhm_fitted = []
        elongation_fitted = []
        theta_fitted = []
        
        segment_radius = self.segment_radius
        ny, nx = data.shape
        
        initial_guess = (30.0, segment_radius, segment_radius, 1.0, 1.0, 0.0, avepixel_mean)
        
        for yi, xi in zip(y, x):
            y_min = int(yi - segment_radius)
            y_max = int(yi + segment_radius)
            x_min = int(xi - segment_radius)
            x_max = int(xi + segment_radius)
            
            if y_min < 0 or x_min < 0 or y_max > ny or x_max > nx:
                continue
                
            star_seg = data[y_min:y_max, x_min:x_max]
            y_ind, x_ind = np.indices(star_seg.shape)
            saturation = np.max(data) * 2 * np.ones_like(y_ind) # Prevent clamping
            
            try:
                import warnings
                from scipy.optimize import OptimizeWarning
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', OptimizeWarning)
                    popt, pcov = opt.curve_fit(twoDGaussian, (y_ind, x_ind, saturation), star_seg.ravel(), 
                                               p0=initial_guess, maxfev=200)
            except RuntimeError:
                continue
                
            amplitude, yo, xo, sigma_y, sigma_x, theta, offset = popt
            
            if min(sigma_y/sigma_x, sigma_x/sigma_y) < self.roundness_threshold:
                continue
                
            if (4*sigma_x*sigma_y/segment_radius**2 > self.max_feature_ratio):
                continue
                
            crop_y_min = int(yo - 3*sigma_y) + 1
            if crop_y_min < 0: crop_y_min = 0
            crop_y_max = int(yo + 3*sigma_y) + 1
            if crop_y_max >= star_seg.shape[0]: crop_y_max = star_seg.shape[0] - 1
            crop_x_min = int(xo - 3*sigma_x) + 1
            if crop_x_min < 0: crop_x_min = 0
            crop_x_max = int(xo + 3*sigma_x) + 1
            if crop_x_max >= star_seg.shape[1]: crop_x_max = star_seg.shape[1] - 1
            
            if (y_max - y_min) < 3:
                crop_y_min = int(yo - 2)
                crop_y_max = int(yo + 2)
            if (x_max - x_min) < 3:
                crop_x_min = int(xo - 2)
                crop_x_max = int(xo + 2)
                
            star_seg_crop = star_seg[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            if star_seg_crop.shape[0] == 0 or star_seg_crop.shape[1] == 0:
                continue
                
            # Gamma correct the star segment and background offset
            star_seg_crop_corr = gamma_correction(star_seg_crop.astype(np.float32), self.gamma)
            offset_corr = gamma_correction(offset, self.gamma)
            
            intensity = np.sum(star_seg_crop_corr - offset_corr)
            if intensity <= 0:
                continue
                
            sigma_fitted = np.sqrt(sigma_x**2 + sigma_y**2)
            fwhm_val = 2.355 * sigma_fitted
            elongation_val = max(abs(sigma_x), abs(sigma_y)) / max(min(abs(sigma_x), abs(sigma_y)), 1e-6)
            theta_deg = np.degrees(theta) % 360
            
            x_fitted.append(x_min + xo)
            y_fitted.append(y_min + yo)
            amplitude_fitted.append(amplitude)
            intensity_fitted.append(intensity)
            fwhm_fitted.append(fwhm_val)
            elongation_fitted.append(elongation_val)
            theta_fitted.append(theta_deg)
            
        if not x_fitted:
            return None
            
        t = Table()
        t['id'] = np.arange(1, len(x_fitted) + 1)
        t['xcentroid'] = x_fitted
        t['ycentroid'] = y_fitted
        t['flux'] = intensity_fitted
        t['peak'] = amplitude_fitted
        t['rms_fwhm'] = fwhm_fitted
        t['rms_elongation'] = elongation_fitted
        t['rms_theta'] = theta_fitted
        return t
