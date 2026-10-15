"""NeXus frame mapping, dose normalization and matched energy integration."""
from __future__ import annotations
import ast
from pathlib import Path
import h5py
import numpy as np
from . import calibration as psf

def planned_axes(handle, motor_names):
    """Parse inclusive planned grids as literal data; never eval scan commands."""
    value = handle["entry/gui/plan_cmd"][()]
    command = ast.literal_eval(value.decode() if isinstance(value, bytes) else str(value))
    if not isinstance(command, tuple) or command[0] != "P.grid_scan":
        raise ValueError("Expected P.grid_scan metadata")
    args = command[1][1:]
    if len(args) != 4*len(motor_names):
        raise ValueError("Unexpected number of planned motors")
    axes, names = [], []
    for start in range(0, len(args), 4):
        motor, low, high, count = args[start:start+4]
        if not np.isfinite([low, high, count]).all() or int(count) != count or count < 2 or high <= low:
            raise ValueError("Axes must increase and have at least two points")
        names.append(motor.replace(".", "_"))
        axes.append(np.linspace(low, high, int(count)))
    if tuple(names) != tuple(motor_names):
        raise ValueError(f"Expected planned motor order {motor_names}, got {names}")
    return axes


def frame_grid(coordinates, axes, tolerances=(1e-6, 1e-6, 1e-8)):
    """Map recorded setpoints even for incomplete, repeated or snake scans."""
    indices = []
    for coords, axis, tolerance in zip(coordinates, axes, tolerances):
        coords, axis = np.asarray(coords, float), np.asarray(axis, float)
        if coords.ndim != 1 or not np.isfinite(coords).all():
            raise ValueError("Invalid frame coordinates")
        if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("Invalid planned axis")
        idx = np.abs(coords[:, None]-axis).argmin(axis=1)
        if np.any(np.abs(coords-axis[idx]) > tolerance):
            raise ValueError("Frame coordinate is outside the planned grid")
        indices.append(idx)
    if len(indices) != 3 or len({len(i) for i in indices}) != 1:
        raise ValueError("Expected matching Y,Z,E vectors")
    occupancy = np.zeros(tuple(len(a) for a in axes), dtype=int)
    np.add.at(occupancy, tuple(indices), 1)
    return indices, occupancy


def interpolation_weights(source_energy, target_energy):
    """Linear spectral interpolation followed by target-grid quadrature.

    Return effective weights on ORIGINAL reference energies, preserving the
    covariance induced by interpolation when calculating counting variance.
    No energy extrapolation and no interpolation of the spatial x axis.
    """
    source = np.asarray(source_energy, float)
    target = np.asarray(target_energy, float)
    if source.ndim != 1 or len(source) < 2 or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
        raise ValueError("Invalid calibration energy grid")
    target_weights = psf.energy_weights(target)
    if target[0] < source[0]-1e-10 or target[-1] > source[-1]+1e-10:
        raise ValueError("Sample energy range extends outside calibration")
    target = np.clip(target, source[0], source[-1])
    left = np.clip(np.searchsorted(source, target, side="right")-1, 0, len(source)-2)
    fraction = (target-source[left])/(source[left+1]-source[left])
    effective = np.zeros(len(source))
    np.add.at(effective, left, target_weights*(1-fraction))
    np.add.at(effective, left+1, target_weights*fraction)
    return effective, target_weights


def prepare_inputs(source, calibration, destination, *, exposure_s=0.5,
                   monitor="D_SiC_Current1", chunk_frames=16):
    """Stream a saved sample scan using the calibration's EXACT ROI pixels.

    All incomplete transverse locations become NaN, accompanied by occupancy
    and complete_spectrum_yz. Exposure is supplied by the user, not inferred.
    ROI matrices contain measured background, consistent with calibration v1.
    """
    source, calibration, destination = map(Path, (source, calibration, destination))
    if destination.exists():
        raise FileExistsError(destination)
    if not np.isfinite(exposure_s) or exposure_s <= 0 or chunk_frames < 1:
        raise ValueError("Invalid dwell or chunk size")
    with h5py.File(source, "r") as raw, h5py.File(calibration, "r") as ref:
        axes = planned_axes(raw, ("M_Stage_Y", "M_Stage_Z", "M_DCM_B5Energy"))
        coord_names = ("M_Stage_Y", "M_Stage_Z", "M_DCM_B5Energy")
        coordinates = [raw["entry/instrument/"+n+"_setpoint"][:] for n in coord_names]
        indices, occupancy = frame_grid(coordinates, axes)
        iy, iz, ie = indices
        complete = (occupancy > 0).all(axis=2)
        i0 = raw["entry/data/"+monitor][:]
        if i0.shape != iy.shape or not np.isfinite(i0).all() or np.any(i0 <= 0):
            raise ValueError("Sample monitor must be positive and match frame count")
        expected = int(np.prod([len(a) for a in axes]))
        canonical = (iy*len(axes[1])+iz)*len(axes[2])+ie
        dose = np.zeros(occupancy.shape)
        np.add.at(dose, (iy, iz, ie), i0*exposure_s)
        effective, w = interpolation_weights(ref["energy_kev"][:], axes[2])
        factors = w[ie]/dose[iy, iz, ie]
        keys = sorted((k for k in ref["rois"] if not ref["rois/"+k].attrs["placeholder"]),
                      key=lambda k: int(ref["rois/"+k].attrs["source_id"]))
        responses, reference_variance, measured, variance, roi_defs = [], [], [], [], []
        for key in keys:
            g = ref["rois/"+key]
            response, response_variance = psf.integrate_response(g["counts_epx"][:], ref["dose_ex"][:], effective)
            responses.append(response)
            reference_variance.append(response_variance)
            shape = (len(g["pixels_rc"]), len(axes[0]), len(axes[1]))
            measured.append(np.zeros(shape))
            variance.append(np.zeros(shape))
            roi_defs.append((g.attrs["detector"], tuple(g["bbox_rc"][:]), g["pixels_rc"][:]))
        for detector in sorted({d[0] for d in roi_defs}):
            dataset = raw["entry/data/"+detector]
            if dataset.ndim != 3 or len(dataset) != len(iy):
                raise ValueError("Frame count differs between metadata and detector")
            chosen = [j for j, d in enumerate(roi_defs) if d[0] == detector]
            for start in range(0, len(dataset), chunk_frames):
                stop = min(start+chunk_frames, len(dataset))
                batch = dataset[start:stop].astype(float)
                for j in chosen:
                    _, (r0, r1, c0, c1), pixels = roi_defs[j]
                    patch = batch[:, r0:r1, c0:c1]
                    values = patch[:, pixels[:, 0]-r0, pixels[:, 1]-c0]
                    # Explicit per-frame accumulation handles duplicate coordinates.
                    for local, frame in enumerate(range(start, stop)):
                        measured[j][:, iy[frame], iz[frame]] += factors[frame]*values[local]
                        variance[j][:, iy[frame], iz[frame]] += factors[frame]**2*values[local]
        for array in measured+variance:
            array[:, ~complete] = np.nan
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(destination, "x") as out:
            out.attrs.update(schema="XRI.inputs.v1", source_file=str(source.resolve()),
                             calibration_file=str(calibration.resolve()), exposure_s=exposure_s,
                             monitor=monitor, expected_frames=expected, actual_frames=len(iy),
                             canonical_order_verified=bool(np.array_equal(canonical, np.arange(len(iy)))),
                             background_subtracted=False,
                             integration="linear calibration spectral interpolation then sample-grid trapezoidal integration")
            for name, value in dict(x_mm=ref["x_mm"][:], y_mm=axes[0], z_mm=axes[1],
                                    energy_kev=axes[2], energy_weights_kev=w,
                                    reference_effective_energy_weights_kev=effective,
                                    occupancy_yze=occupancy, complete_spectrum_yz=complete,
                                    dose_yze=dose, frame_y_index=iy, frame_z_index=iz,
                                    frame_energy_index=ie, i0=i0).items():
                out.create_dataset(name, data=value)
            out.create_dataset("roi_keys", data=np.asarray(keys, dtype=h5py.string_dtype()))
            for j, key in enumerate(keys):
                g = out.create_group("rois/"+key)
                g.attrs.update(ref["rois/"+key].attrs)
                for name, value in dict(R_matched_px=responses[j], R_matched_variance_px=reference_variance[j], D_band_pyz=measured[j],
                                        D_variance_pyz=variance[j],
                                        bbox_rc=roi_defs[j][1], pixels_rc=roi_defs[j][2]).items():
                    g.create_dataset(name, data=value, compression="gzip", shuffle=True)
    return destination


def load_inputs(file):
    with h5py.File(file, "r") as f:
        keys = f["roi_keys"].asstr()[:].tolist()
        matrices = [f["rois/"+k+"/R_matched_px"][:] for k in keys]
        data = [f["rois/"+k+"/D_band_pyz"][:] for k in keys]
        axes = [f[n+"_mm"][:] for n in ("x", "y", "z")]
        complete = f["complete_spectrum_yz"][:]
    return matrices, data, axes, keys, complete


def export_result(destination, density, axes, *, input_file, solver, metadata=None):
    """Save requested 3-D float32 intensity under the reference's dataset name.

    The reference lacks coordinates, so scientific metadata/axes are saved
    explicitly here. No resizing to reference dimensions or invented 200-axis.
    """
    density = np.asarray(density)
    if density.shape != tuple(len(a) for a in axes):
        raise ValueError("Result shape must match x,y,z coordinates")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "x") as out:
        dataset = out.create_dataset("reinterp_insect_ck", data=density.astype(np.float32),
                                     compression="gzip", shuffle=True)
        dataset.attrs.update(axes="X,Y,Z", quantity="relative Kapton-equivalent elastic scattering density",
                             units="relative", background_subtracted=False,
                             energy_integrated=True, absolute_density_calibrated=False)
        for name, axis in zip(("x_mm", "y_mm", "z_mm"), axes):
            scale = out.create_dataset(name, data=axis)
            scale.make_scale(name)
            index = ("x_mm", "y_mm", "z_mm").index(name)
            dataset.dims[index].attach_scale(scale)
            dataset.dims[index].label = name
        out.attrs.update(source_inputs=str(Path(input_file).resolve()), solver=solver,
                         schema="XRI.result.v1")
        for key, value in (metadata or {}).items():
            out.attrs[key] = value
    return destination
