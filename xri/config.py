"""Strict JSON configuration; paths are relative to the configuration file."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import math

DEFAULTS = {
    "input": {"calibration_nxs": None, "sample_nxs": None, "roi_table": None},
    "acquisition": {
        "calibration_exposure_s": None, "sample_exposure_s": None,
        "monitor": "D_SiC_Current1", "detector": "D_LAMBDA",
        "foil_thickness_mm": 0.05, "detector_pixel_pitch_mm": 0.055,
    },
    "roi": {"excluded_ids": [], "label_overrides": {}},
    "grid": {"x_step_mm": None},
    "psf": {
        "min_peak_fraction": 0.3, "min_r_squared": 0.8,
        "edge_margin_pixels": 1.2, "min_observations": 3,
        "max_trajectory_rms_pixels": 0.75, "sigma_bounds_pixels": [0.3, 12.0],
        "max_fit_evaluations": 4000,
    },
    "reconstruction": {
        "fit_gains": True, "anchor_roi": "36_VU-C1", "gain_bounds": [0.05, 20.0],
        "max_gain_iterations": 500, "gain_tolerance": 1e-5,
        "nnls_max_iterations": 20000, "nnls_tolerance": 1e-7,
    },
    "tv": {
        "enabled": True, "lambda": None, "relative_fraction": 0.005,
        "max_iterations": 40000, "tolerance": 1e-6,
    },
    "runtime": {"chunk_frames": 16, "output_dir": None},
    "output": {"save_diagnostics": True, "plots": True},
}


def _merge(defaults, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    unknown = set(supplied)-set(defaults)
    if unknown:
        raise ValueError(f"Unknown fields at {prefix}: {sorted(unknown)}")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        result[key] = _merge(defaults[key], value, prefix+key+".") if isinstance(defaults[key], dict) and defaults[key] else value
    return result


def _positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def load_config(file):
    file = Path(file).resolve()
    config = _merge(DEFAULTS, json.loads(file.read_text(encoding="utf-8-sig")))
    for section, fields in (("input", DEFAULTS["input"]), ("runtime", ["output_dir"])):
        for key in fields:
            value = config[section][key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing path {section}.{key}")
            path = Path(value).expanduser()
            config[section][key] = str((path if path.is_absolute() else file.parent/path).resolve())
    for key in ("calibration_exposure_s", "sample_exposure_s", "foil_thickness_mm", "detector_pixel_pitch_mm"):
        _positive(config["acquisition"][key], "acquisition."+key)
    for key in ("monitor", "detector"):
        value = config["acquisition"][key]
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(f"Invalid acquisition.{key}")
    if config["grid"]["x_step_mm"] is not None:
        _positive(config["grid"]["x_step_mm"], "grid.x_step_mm")
    p = config["psf"]
    for key in ("min_peak_fraction", "min_r_squared"):
        _positive(p[key], "psf."+key)
        if p[key] > 1:
            raise ValueError(f"psf.{key} must be at most 1")
    for key in ("edge_margin_pixels", "max_trajectory_rms_pixels"):
        _positive(p[key], "psf."+key)
    for key in ("min_observations", "max_fit_evaluations"):
        _positive(p[key], "psf."+key, integer=True)
    if p["min_observations"] < 3:
        raise ValueError("psf.min_observations must be at least 3")
    for section, key in (("psf", "sigma_bounds_pixels"), ("reconstruction", "gain_bounds")):
        bounds = config[section][key]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{section}.{key} must be [lower,upper]")
        for value in bounds:
            _positive(value, section+"."+key)
        if bounds[1] <= bounds[0]:
            raise ValueError(f"{section}.{key} must increase")
    r = config["reconstruction"]
    if not r["gain_bounds"][0] <= 1 <= r["gain_bounds"][1]:
        raise ValueError("Gain bounds must contain the fixed anchor 1")
    if not isinstance(r["anchor_roi"], str) or not r["anchor_roi"]:
        raise ValueError("reconstruction.anchor_roi must be a nonempty ROI key")
    _positive(r["max_gain_iterations"], "reconstruction.max_gain_iterations", integer=True)
    _positive(r["gain_tolerance"], "reconstruction.gain_tolerance")
    _positive(r["nnls_max_iterations"], "reconstruction.nnls_max_iterations", integer=True)
    _positive(r["nnls_tolerance"], "reconstruction.nnls_tolerance")
    _positive(config["runtime"]["chunk_frames"], "runtime.chunk_frames", integer=True)
    tv = config["tv"]
    for key in ("lambda", "relative_fraction"):
        value = tv[key]
        if key == "lambda" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"tv.{key} must be finite and nonnegative")
    _positive(tv["max_iterations"], "tv.max_iterations", integer=True)
    _positive(tv["tolerance"], "tv.tolerance")
    for section, key in (("tv", "enabled"), ("reconstruction", "fit_gains"), ("output", "save_diagnostics"), ("output", "plots")):
        if not isinstance(config[section][key], bool):
            raise ValueError(f"{section}.{key} must be true or false")
    roi = config["roi"]
    if not isinstance(roi["excluded_ids"], list) or not isinstance(roi["label_overrides"], dict):
        raise ValueError("ROI exclusions must be a list; label overrides must be an object")
    roi["excluded_ids"] = [str(v) for v in roi["excluded_ids"]]
    if any(not isinstance(v, str) or not v or "/" in v for v in roi["label_overrides"].values()):
        raise ValueError("ROI labels must be nonempty strings without /")
    return config
