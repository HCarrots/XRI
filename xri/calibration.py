"""Pixel-preserving reference extraction; lengths mm, energy keV, pixels row/column."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np

@dataclass(frozen=True)
class PixelROI:
    key: str
    label: str
    detector: str
    bbox: tuple[int, int, int, int]
    source_id: str
    placeholder: bool = False
    source_label: str = ""

    @property
    def pixels_rc(self):
        r0, r1, c0, c1 = self.bbox
        r, c = np.mgrid[r0:r1, c0:c1]
        return np.column_stack((r.ravel(), c.ravel()))


def read_roi_table(file, detector="D_LAMBDA", image_shape=(516, 516), label_overrides=None, exclude_ids=()):
    """Read DT_Be_ROI TSV: x1/x2 are columns, y1/y2 rows; stops exclusive.

    Source IDs disambiguate duplicate labels without guessing physical names.
    Explicit exclude_ids are flagged and retained in calibration provenance.
    No ROI geometry is auto-adjusted or classified by box size.
    """
    records = []
    with Path(file).open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            source_id = row.get("", str(index)) or str(index)
            c0, c1, r0, r1 = [int(row[k]) for k in ("x1", "x2", "y1", "y2")]
            dx, dy, ex, ey = [int(row.get(k, 0) or 0) for k in
                                ("x_shift", "y_shift", "x_expand", "y_expand")]
            c0, c1, r0, r1 = c0+dx-ex, c1+dx+ex, r0+dy-ey, r1+dy+ey
            if not (0 <= r0 < r1 <= image_shape[0] and 0 <= c0 < c1 <= image_shape[1]):
                raise ValueError(f"ROI {source_id} is outside {detector}")
            source_label = row["roi_label"]
            label = (label_overrides or {}).get(source_id, source_label)
            records.append(PixelROI(f"{source_id}_{label}", label, detector,
                                    (r0, r1, c0, c1), source_id,
                                    source_id in {str(v) for v in exclude_ids}, source_label))
    if not records or len({r.key for r in records}) != len(records) or len({r.source_id for r in records}) != len(records):
        raise ValueError("Empty ROI table or duplicate source IDs/keys")
    return records


def map_grid(x_setpoint, energy_setpoint, x_axis, energy_axis,
             x_tolerance=1e-6, energy_tolerance=1e-8):
    """Map each actual frame using its setpoints, never by blind reshape.

    Floating-point near-duplicates, serpentine order and repeats are supported.
    Missing grid cells fail explicitly. Readbacks are recorded separately.
    """
    x, e = np.asarray(x_setpoint), np.asarray(energy_setpoint)
    xa, ea = np.asarray(x_axis, float), np.asarray(energy_axis, float)
    if x.ndim != 1 or e.shape != x.shape or not x.size:
        raise ValueError("Frame coordinates must be nonempty matching vectors")
    if any(a.ndim != 1 or not a.size or not np.isfinite(a).all() or
           np.any(np.diff(a) <= 0) for a in (xa, ea)):
        raise ValueError("Axes must be finite and strictly increasing")
    if not np.isfinite(x).all() or not np.isfinite(e).all():
        raise ValueError("Nonfinite frame coordinates")
    ix = np.abs(x[:, None]-xa).argmin(axis=1)
    ie = np.abs(e[:, None]-ea).argmin(axis=1)
    if np.any(np.abs(x-xa[ix]) > x_tolerance) or np.any(np.abs(e-ea[ie]) > energy_tolerance):
        raise ValueError("A frame setpoint is outside the specified grid tolerance")
    occupancy = np.zeros((len(xa), len(ea)), dtype=int)
    np.add.at(occupancy, (ix, ie), 1)
    if np.any(occupancy == 0):
        raise ValueError(f"Missing grid cells: {np.argwhere(occupancy == 0).tolist()}")
    return ix, ie, occupancy


def energy_weights(energy_axis, bounds=None):
    """Trapezoidal keV weights; for one selected energy use weight 1.

    Bounds select actual grid points, with no extrapolation or interpolation.
    """
    axis = np.asarray(energy_axis, float)
    if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError("Invalid energy axis")
    keep = np.ones(len(axis), bool) if bounds is None else (axis >= bounds[0]) & (axis <= bounds[1])
    selected = np.flatnonzero(keep)
    if not selected.size:
        raise ValueError("No energies in the requested band")
    weights = np.zeros(len(axis))
    if len(selected) == 1:
        weights[selected] = 1
    else:
        steps = np.diff(axis[selected])
        weights[selected] = np.r_[steps[0]/2, (steps[:-1]+steps[1:])/2, steps[-1]/2]
    return weights


def integrate_response(counts_epx, dose_ex, weights):
    """Return band response and counting variance, both shaped (pixel, x).

    Repeated measurements are pooled as sum(counts)/sum(I0*t) per (E,x).
    Variance assumes independent Poisson counts and exact monitor/dwell values;
    it excludes calibration noise correlations and background uncertainty.
    """
    counts, dose, w = (np.asarray(a, float) for a in (counts_epx, dose_ex, weights))
    if counts.ndim != 3 or dose.shape != (counts.shape[0], counts.shape[2]) or w.shape != (counts.shape[0],):
        raise ValueError("Expected counts[E,p,x], dose[E,x], weights[E]")
    if not all(np.isfinite(a).all() for a in (counts, dose, w)) or np.any(dose <= 0) or np.any(counts < 0):
        raise ValueError("Counts/weights must be finite and dose strictly positive")
    factors = w[:, None, None] / dose[:, None, :]
    return (np.sum(counts*factors, axis=0), np.sum(counts*factors**2, axis=0))


def export_calibration(source, destination, roi_file, *, exposure_s=0.5,
                       monitor="D_SiC_Current1", foil_thickness_mm=0.05,
                       x_axis=None, energy_axis=None, chunk_frames=16,
                       label_overrides=None, exclude_ids=(), detector="D_LAMBDA"):
    """Stream a grid NeXus scan to a NEW HDF5 calibration without changing source.

    Includes raw ROI counts at every energy, monitor/exposure, fixed pixel
    coordinates, whole-detector band images and response matrices. Background
    is retained. Only the configured detector is read; other detectors require
    their own ROI definitions and are not implicitly combined.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if chunk_frames < 1 or not np.isfinite([exposure_s, foil_thickness_mm]).all() or exposure_s <= 0 or foil_thickness_mm <= 0:
        raise ValueError("Chunk size, dwell and foil thickness must be positive")
    if x_axis is None or energy_axis is None:
        raise ValueError("Provide planned calibration X and energy axes")
    xa, ea = np.asarray(x_axis, float), np.asarray(energy_axis, float)
    with h5py.File(source, "r") as src:
        frame_x = src["entry/instrument/M_Stage_X_setpoint"][:]
        frame_e = src["entry/instrument/M_DCM_B5Energy_setpoint"][:]
        ix, ie, occupancy = map_grid(frame_x, frame_e, xa, ea)
        i0 = src[f"entry/data/{monitor}"][:]
        if i0.shape != ix.shape or not np.isfinite(i0).all() or np.any(i0 <= 0):
            raise ValueError("I0 must be a positive finite current per frame")
        rois = read_roi_table(roi_file, detector=detector, image_shape=src["entry/data/"+detector].shape[1:],
                              label_overrides=label_overrides, exclude_ids=exclude_ids)
        coverage = np.zeros(src["entry/data/"+detector].shape[1:], dtype=int)
        for roi in rois:
            if not roi.placeholder:
                r0, r1, c0, c1 = roi.bbox
                coverage[r0:r1, c0:c1] += 1
        if np.any(coverage > 1):
            raise ValueError("Usable ROIs overlap: avoid counting shared pixels as independent measurements")
        dose = np.zeros((len(ea), len(xa)))
        np.add.at(dose, (ie, ix), i0*exposure_s)
        w = energy_weights(ea)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(destination, "x") as out:
            out.attrs.update(schema="XRI.calibration.v1", source_file=str(source.resolve()),
                             roi_file=str(Path(roi_file).resolve()), monitor=monitor,
                             exposure_s=exposure_s, foil_thickness_mm=foil_thickness_mm,
                             response_units="counts / (monitor_current * s) * keV",
                             background_subtracted=False, absolute_density_calibrated=False,
                             roi_stop_convention="exclusive", calibration_status="provisional: inspect ROI tails and background")
            for name, values in dict(x_mm=xa, energy_kev=ea, energy_weights_kev=w,
                                     occupancy_xe=occupancy, dose_ex=dose).items():
                out.create_dataset(name, data=values)
            frames = out.create_group("frames")
            for name, values in dict(x_index=ix, energy_index=ie, x_setpoint_mm=frame_x,
                                     energy_setpoint_kev=frame_e, i0=i0,
                                     exposure_s=np.full(len(ix), exposure_s),
                                     x_readback_mm=src["entry/instrument/M_Stage_X"][:] if "entry/instrument/M_Stage_X" in src else frame_x,
                                     energy_readback_kev=src["entry/instrument/M_DCM_B5Energy"][:] if "entry/instrument/M_DCM_B5Energy" in src else frame_e).items():
                frames.create_dataset(name, data=values)
            groups = {}
            for roi in rois:
                group = out.create_group("rois/"+roi.key)
                group.attrs.update(label=roi.label, source_id=roi.source_id,
                                   source_label=roi.source_label,
                                   detector=roi.detector, placeholder=roi.placeholder)
                group.create_dataset("bbox_rc", data=roi.bbox)
                group.create_dataset("pixels_rc", data=roi.pixels_rc)
                groups[roi.key] = group.create_dataset("counts_epx", shape=(len(ea), len(roi.pixels_rc), len(xa)),
                    dtype="uint64", chunks=(1, len(roi.pixels_rc), 1), compression="gzip", shuffle=True)
            for detector in (detector,):
                if f"entry/data/{detector}" not in src:
                    continue
                data = src[f"entry/data/{detector}"]
                if data.ndim != 3 or len(data) != len(ix):
                    raise ValueError(f"{detector}: expected frame,row,column")
                image_raw = np.zeros(data.shape[1:], float)
                band = np.zeros((len(xa), *data.shape[1:]), float)
                selected_rois = [r for r in rois if r.detector == detector]
                # Work one x at a time: memory does not scale with total raw frame count.
                for x in range(len(xa)):
                    buffers = {r.key: np.zeros((len(ea), len(r.pixels_rc)), np.uint64) for r in selected_rois}
                    indices = np.flatnonzero(ix == x)
                    for start in range(0, len(indices), chunk_frames):
                        ids = indices[start:start+chunk_frames]
                        batch = data[ids].astype(np.uint64)
                        image_raw += batch.sum(axis=0)
                        factors = w[ie[ids]]/dose[ie[ids], x]
                        band[x] += np.einsum("f,frc->rc", factors, batch)
                        for roi in selected_rois:
                            r0, r1, c0, c1 = roi.bbox
                            values = batch[:, r0:r1, c0:c1].reshape(len(ids), -1)
                            np.add.at(buffers[roi.key], ie[ids], values)
                    for roi in selected_rois:
                        groups[roi.key][:, :, x] = buffers[roi.key]
                dg = out.create_group("detectors/"+detector)
                dg.create_dataset("raw_sum_rc", data=image_raw, compression="gzip")
                dg.create_dataset("band_response_xrc", data=band, compression="gzip", shuffle=True)
            for roi in rois:
                group = out["rois/"+roi.key]
                response, variance = integrate_response(group["counts_epx"][:], dose, w)
                group.create_dataset("R_band_px", data=response, compression="gzip")
                group.create_dataset("R_band_variance_px", data=variance, compression="gzip")
    return destination
