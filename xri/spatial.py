"""Continuous, moving-PSF calibration: foil positions are observations, not voxels.

Detector pitch and sample depth are distinct coordinates. A fitted 2-D Gaussian
PSF is a model assumption that must be checked, not extra measured information.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def gaussian_image(coords, amplitude, row, column, sigma_row, sigma_col, corr, offset):
    rr, cc = coords
    u, v = (rr-row)/sigma_row, (cc-column)/sigma_col
    return amplitude*np.exp(-(u*u-2*corr*u*v+v*v)/(2*(1-corr*corr)))+offset


def fit_moving_psf(response_px, pixels_rc, calibration_x, *, detector_pitch_mm,
                   min_peak_fraction=.3, min_r_squared=.8, edge_margin_pixels=1.2,
                   min_observations=3, max_trajectory_rms_pixels=.75,
                   sigma_bounds_pixels=(.3, 12.), max_fit_evaluations=4000):
    """Fit spot location/width at bright, uncropped foil positions, then trajectory.

    Constant spatial background is fitted separately. Intermediate depths move
    a SINGLE spot along the trajectory; they are not mixtures of distant spots.
    """
    response, pixels, x = np.asarray(response_px, float), np.asarray(pixels_rc, float), np.asarray(calibration_x, float)
    if response.shape != (len(pixels), len(x)) or pixels.shape[1] != 2:
        raise ValueError("Expected response[p,x] and global detector pixels[p,2]")
    if not np.isfinite(response).all() or np.any(response < 0):
        raise ValueError("PSF observations must be finite and nonnegative")
    minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
    peak_height = np.percentile(response, 99.5, axis=0)-np.percentile(response, 20, axis=0)
    candidates = np.flatnonzero(peak_height > min_peak_fraction*peak_height.max())
    fits = []
    for i in candidates:
        image = response[:, i]
        center = pixels[image.argmax()]
        background = max(float(np.percentile(image, 20)), 0)
        amp = max(float(image.max()-background), 1e-10)
        initial = [amp, *center, *([float(np.clip(2., *sigma_bounds_pixels))]*2), 0., background]
        lower = [0., *minimum, *([sigma_bounds_pixels[0]]*2), -.9, 0.]
        upper = [np.inf, *maximum, *([sigma_bounds_pixels[1]]*2), .9, np.inf]
        try:
            fitted, covariance = curve_fit(gaussian_image, pixels.T, image,
                                           p0=initial, bounds=(lower, upper), maxfev=max_fit_evaluations)
        except (RuntimeError, ValueError):
            continue
        predicted = gaussian_image(pixels.T, *fitted)
        ss_res = float(np.sum((predicted-image)**2))
        ss_total = float(np.sum((image-image.mean())**2))
        r2 = 1-ss_res/ss_total if ss_total > 0 else 0
        # Edge peaks cannot establish an unbiased centroid and width.
        # Reject edge-centred calibration spots, without changing any ROI or
        # reconstruction voxel. The Gaussian still estimates truncated tails.
        centroid = np.asarray(fitted[1:3])
        inside = np.all((centroid-minimum > edge_margin_pixels) & (maximum-centroid > edge_margin_pixels))
        if r2 >= min_r_squared and inside:
            fits.append(dict(x=float(x[i]), parameters=fitted, r_squared=r2, frame_index=int(i)))
    if len(fits) < min_observations:
        raise ValueError(f"Only {len(fits)} usable PSF observations; cannot fit a reliable trajectory")
    valid_x = np.asarray([f['x'] for f in fits])
    parameters = np.asarray([f['parameters'] for f in fits])
    weights = np.sqrt(parameters[:, 0])
    row_line = np.polyfit(valid_x, parameters[:, 1], 1, w=weights)
    col_line = np.polyfit(valid_x, parameters[:, 2], 1, w=weights)
    slopes = np.array([row_line[0], col_line[0]])
    speed = float(np.linalg.norm(slopes))
    if speed <= 0:
        raise ValueError("No depth-dependent detector displacement")
    location_error = np.sqrt((np.polyval(row_line, valid_x)-parameters[:, 1])**2+
                             (np.polyval(col_line, valid_x)-parameters[:, 2])**2)
    location_rms = float(np.sqrt(np.mean(location_error**2)))
    if location_rms > max_trajectory_rms_pixels:
        raise ValueError(f"Centroid trajectory residual {location_rms:.3g} pixels exceeds {max_trajectory_rms_pixels}; possible mixed spots")
    shape = np.median(parameters[:, 3:6], axis=0)
    amplitude = float(np.median(parameters[:, 0]))
    sr, sc, corr = shape
    covariance = np.array([[sr*sr, corr*sr*sc], [corr*sr*sc, sc*sc]])
    projected_sigma = float(np.sqrt(slopes@covariance@slopes)/speed)
    return dict(row_line=row_line, column_line=col_line, sigma_row=float(sr), sigma_col=float(sc),
                correlation=float(corr), amplitude=amplitude,
                fitted_x_min=float(valid_x.min()), fitted_x_max=float(valid_x.max()),
                detector_pixels_per_sample_mm=speed,
                detector_pitch_mm=float(detector_pitch_mm),
                sample_mm_per_detector_pixel=1/speed,
                detector_mm_per_sample_mm=detector_pitch_mm*speed,
                projected_psf_fwhm_sample_mm=2.354820045*projected_sigma/speed,
                centroid_fit_rms_detector_pixels=location_rms,
                valid_observations=len(fits), fit_r2_min=float(min(f['r_squared'] for f in fits)),
                observations=[dict(x=f['x'], row=float(f['parameters'][1]), column=float(f['parameters'][2]),
                                   r_squared=f['r_squared']) for f in fits])


def evaluate_moving_psf(model, pixels_rc, sample_x):
    pixels, x = np.asarray(pixels_rc), np.asarray(sample_x)
    rr, cc = pixels[:, 0, None], pixels[:, 1, None]
    return gaussian_image((rr, cc), model['amplitude'], np.polyval(model['row_line'], x)[None],
                          np.polyval(model['column_line'], x)[None], model['sigma_row'],
                          model['sigma_col'], model['correlation'], 0.)
