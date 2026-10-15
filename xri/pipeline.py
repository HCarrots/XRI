"""Self-contained raw NeXus -> calibrated, elastic-integrated I(X,Y,Z)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np
import scipy

from . import __version__, calibration, nexus, solver, spatial


def json_write(path, value):
    def convert(item):
        return item.tolist() if isinstance(item, np.ndarray) else item.item()
    Path(path).write_text(json.dumps(value, default=convert, ensure_ascii=False,
                                    indent=2, allow_nan=False)+"\n", encoding="utf-8")


def validate_inputs(config):
    """Read metadata and ROI definitions without creating output files."""
    for name, path in config["input"].items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"input.{name}: {path}")
    acq = config["acquisition"]
    report = {}
    for name, motors in (
        ("calibration", ("M_Stage_X", "M_DCM_B5Energy")),
        ("sample", ("M_Stage_Y", "M_Stage_Z", "M_DCM_B5Energy")),
    ):
        path = config["input"][name+"_nxs"]
        with h5py.File(path, "r") as f:
            axes = nexus.planned_axes(f, motors)
            coords = [f["entry/instrument/"+m+"_setpoint"][:] for m in motors]
            if name == "calibration":
                *_, occupancy = calibration.map_grid(*coords, *axes)
            else:
                _, occupancy = nexus.frame_grid(coords, axes)
                if np.any(occupancy == 0):
                    raise ValueError(f"Sample scan incomplete: {np.count_nonzero(occupancy == 0)} missing YZE cells; no missing-as-zero reconstruction")
            images = f["entry/data/"+acq["detector"]]
            if images.ndim != 3 or len(images) != len(coords[0]):
                raise ValueError(f"{name}: detector frame count or dimensions do not match motor records")
            if not np.issubdtype(images.dtype, np.unsignedinteger):
                raise ValueError("Expected unsigned raw photon counts")
            monitor = f["entry/data/"+acq["monitor"]][:]
            if monitor.shape != coords[0].shape or not np.isfinite(monitor).all() or np.any(monitor <= 0):
                raise ValueError(f"{name}: I0 must be finite, positive and match frames")
            report[name] = dict(
                source_file=str(Path(path).resolve()),
                axes={motor: axis.tolist() for motor, axis in zip(motors, axes)},
                grid_shape=list(occupancy.shape), actual_frames=len(images),
                expected_grid_cells=int(occupancy.size),
                repeated_frames=int(occupancy.sum()-np.count_nonzero(occupancy)),
                image_shape=list(images.shape[1:]),
                exposure_s=acq[name+"_exposure_s"],
            )
    if report["calibration"]["image_shape"] != report["sample"]["image_shape"]:
        raise ValueError("Calibration and sample detector shapes differ")
    nexus.interpolation_weights(
        report["calibration"]["axes"]["M_DCM_B5Energy"],
        report["sample"]["axes"]["M_DCM_B5Energy"],
    )
    rois = calibration.read_roi_table(
        config["input"]["roi_table"], detector=acq["detector"],
        image_shape=report["calibration"]["image_shape"],
        label_overrides=config["roi"]["label_overrides"], exclude_ids=config["roi"]["excluded_ids"],
    )
    ids = {r.source_id for r in rois}
    if not set(config["roi"]["excluded_ids"]) <= ids or not set(config["roi"]["label_overrides"]) <= ids:
        raise ValueError("ROI exclusions/overrides refer to nonexistent source IDs")
    coverage = np.zeros(report["calibration"]["image_shape"], int)
    active = [r for r in rois if not r.placeholder]
    if not active:
        raise ValueError("No usable ROI remains")
    for roi in active:
        r0, r1, c0, c1 = roi.bbox
        coverage[r0:r1, c0:c1] += 1
    if np.any(coverage > 1):
        raise ValueError("Usable detector ROIs overlap; shared counts cannot be treated as independent")
    anchor = config["reconstruction"]["anchor_roi"]
    if config["reconstruction"]["fit_gains"] and anchor not in {r.key for r in active}:
        raise ValueError(f"Gain anchor {anchor} is not an included ROI")
    report["roi_keys"] = [r.key for r in active]
    report["excluded_roi_ids"] = [r.source_id for r in rois if r.placeholder]
    report["roi_pixels"] = int(sum(len(r.pixels_rc) for r in active))
    return report


def make_x_grid(calibration_x, models, requested_step=None):
    """Full calibrated interval, centered uniform grid; no sensitivity cropping."""
    original = np.asarray(calibration_x, float)
    step = float(np.median([m["sample_mm_per_detector_pixel"] for m in models.values()])) if requested_step is None else float(requested_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Invalid sample-space X step")
    count = int(np.floor((original[-1]-original[0])/step+1e-9))+1
    if count < 2:
        raise ValueError("Sample X step exceeds the calibration span")
    x = (original[0]+original[-1])/2 + step*(np.arange(count)-(count-1)/2)
    return x, step


def run(config):
    report = validate_inputs(config)
    output = Path(config["runtime"]["output_dir"])
    # No overwrite/resume of potentially stale scientific intermediates.
    output.mkdir(parents=True, exist_ok=False)

    def log(message):
        line = datetime.now().strftime("%H:%M:%S")+" "+message
        print(line, flush=True)
        with (output/"run.log").open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")

    json_write(output/"config.resolved.json", config)
    json_write(output/"input_validation.json", report)
    versions = dict(xri=__version__, python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, h5py=h5py.__version__)
    sources = {}
    for key, path in config["input"].items():
        stat = Path(path).stat()
        sources[key] = dict(path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    sources["roi_table"]["sha256"] = hashlib.sha256(Path(config["input"]["roi_table"]).read_bytes()).hexdigest()
    json_write(output/"provenance.json", dict(versions=versions, inputs=sources,
               started_utc=datetime.now(timezone.utc).isoformat()))
    acq, rec = config["acquisition"], config["reconstruction"]
    chunk = config["runtime"]["chunk_frames"]
    reference = output/"calibration.h5"
    inputs = output/"tomography_inputs.h5"
    log("1/6 Extracting calibration pixels and pooled monitor doses")
    calibration.export_calibration(
        config["input"]["calibration_nxs"], reference, config["input"]["roi_table"],
        exposure_s=acq["calibration_exposure_s"], monitor=acq["monitor"],
        detector=acq["detector"], foil_thickness_mm=acq["foil_thickness_mm"],
        x_axis=report["calibration"]["axes"]["M_Stage_X"],
        energy_axis=report["calibration"]["axes"]["M_DCM_B5Energy"],
        chunk_frames=chunk, label_overrides=config["roi"]["label_overrides"],
        exclude_ids=config["roi"]["excluded_ids"],
    )
    log("2/6 Mapping sample YZE frames and matching reference/sample energy integrals")
    nexus.prepare_inputs(config["input"]["sample_nxs"], reference, inputs,
                         exposure_s=acq["sample_exposure_s"], monitor=acq["monitor"], chunk_frames=chunk)
    responses0, measurements0, axes, all_keys, complete = nexus.load_inputs(inputs)
    if not complete.all():
        raise ValueError("Incomplete sample spectra")
    log("3/6 Fitting moving elliptical PSFs and sample-to-detector trajectories")
    models, rejected, pixels = {}, {}, {}
    with h5py.File(inputs, "r") as f:
        for key, observed in zip(all_keys, responses0):
            pixels[key] = f["rois/"+key+"/pixels_rc"][:]
            try:
                models[key] = spatial.fit_moving_psf(observed, pixels[key], axes[0],
                    detector_pitch_mm=acq["detector_pixel_pitch_mm"], **config["psf"])
            except ValueError as error:
                rejected[key] = str(error)
                log(f"PSF rejected {key}: {error}")
    json_write(output/"psf_models.json", dict(models=models, rejected=rejected))
    if not models:
        raise ValueError("All PSF models failed quality checks")
    keys = list(models)
    if rec["fit_gains"] and rec["anchor_roi"] not in models:
        raise ValueError(f"Gain anchor {rec['anchor_roi']} failed PSF checks; choose a verified anchor explicitly")
    x, dx = make_x_grid(axes[0], models, config["grid"]["x_step_mm"])
    axes[0] = x
    responses = [spatial.evaluate_moving_psf(models[k], pixels[k], x) for k in keys]
    measurements = [measurements0[all_keys.index(k)] for k in keys]
    widths = dict(voxel_width_mm=dx, foil_thickness_mm=acq["foil_thickness_mm"])
    log(f"Full grid: {len(x)} X x {len(axes[1])} Y x {len(axes[2])} Z; dx={dx:.9g} mm; {len(keys)} accepted ROIs")
    log("4/6 Solving joint NNLS and relative analyzer gains")

    def gain_progress(iteration, change):
        if iteration % 20 == 0:
            log(f"Gain iteration {iteration}: relative change {change:.4g}")

    if rec["fit_gains"]:
        fitted = solver.fit_crystal_gains(responses, measurements,
            anchor_index=keys.index(rec["anchor_roi"]), max_iterations=rec["max_gain_iterations"],
            tolerance=rec["gain_tolerance"], gain_bounds=rec["gain_bounds"],
            nnls_max_iterations=rec["nnls_max_iterations"], nnls_tolerance=rec["nnls_tolerance"],
            progress=gain_progress, **widths)
    else:
        ops, _, h, back = solver._joint_terms(responses, measurements, **widths)
        rho, inner = solver.solve_gram_nnls(h, back, return_info=True,
            max_iterations=rec["nnls_max_iterations"], tolerance=rec["nnls_tolerance"])
        predicted = [(a@rho.reshape(len(x), -1)).reshape(d.shape) for a, d in zip(ops, measurements)]
        residuals = [p-d for p, d in zip(predicted, measurements)]
        fitted = dict(density=rho, predicted=predicted, residuals=residuals, gains=np.ones(len(keys)),
                      converged=inner["converged"], iterations=0, gain_change_history=[], relative_gain_change=0., nnls_info=inner)
    gains = fitted["gains"]
    operators, _, gram, rhs = solver._joint_terms(responses, measurements, gains=gains, **widths)
    spacing = [float(np.diff(a).mean()) for a in axes]
    tv = None
    beta = None
    primary = fitted["density"]
    method = "joint NNLS"
    log("5/6 TV regularization with fixed fitted gains" if config["tv"]["enabled"] else "5/6 TV disabled")
    if config["tv"]["enabled"]:
        settings = config["tv"]
        beta = settings["lambda"]
        if beta is None:
            beta = float(settings["relative_fraction"]*np.percentile(np.abs(rhs), 95)*min(spacing))
        log(f"TV lambda = {beta:.9g}")
        tv = solver.reconstruct_tv(responses, measurements, gains=gains, spacing_mm=spacing,
            lambda_tv=beta, max_iterations=settings["max_iterations"], tolerance=settings["tolerance"],
            initial_density=fitted["density"], **widths,
            progress=lambda i, p, d: log(f"TV iteration {i}: primal change {p:.4g}, dual change {d:.4g}"))
        if tv["converged"]:
            primary, method = tv["density"], "joint NNLS gains + fixed-gain TV"
        else:
            log("TV did not converge: primary output uses NNLS; the TV iterate is saved separately")
    if not fitted["converged"]:
        log("Gain fit did not converge: outputs are provisional; inspect iteration diagnostics")
    sensitivity = np.sqrt(np.diag(gram))
    eigenvalues = np.linalg.eigvalsh(gram)
    rank = int(np.count_nonzero(eigenvalues > max(eigenvalues[-1], 0)*1e-14))
    data_norm = float(np.sqrt(sum(np.sum(d*d) for d in measurements)))

    def relative_residual(errors):
        norm = float(np.sqrt(sum(np.sum(e*e) for e in errors)))
        return norm/data_norm if data_norm > 0 else None

    summary = dict(
        status="provisional", solver=method, shape_xyz=list(primary.shape), spacing_xyz_mm=spacing,
        ranges_xyz_mm=[[float(a[0]), float(a[-1])] for a in axes],
        x_grid_policy="full calibrated span; centered uniform centers; no sensitivity cropping",
        x_step_source="median calibrated sample mm per detector pixel" if config["grid"]["x_step_mm"] is None else "explicit sample-space step",
        detector_pixel_pitch_mm=acq["detector_pixel_pitch_mm"], foil_thickness_mm=acq["foil_thickness_mm"],
        calibration_x_positions=len(report["calibration"]["axes"]["M_Stage_X"]),
        energy_range_kev=[report["sample"]["axes"]["M_DCM_B5Energy"][i] for i in (0, -1)],
        energy_points=len(report["sample"]["axes"]["M_DCM_B5Energy"]),
        input_roi_count=len(all_keys), accepted_roi_count=len(keys), roi_keys=keys, rejected_psf_models=rejected,
        gains_by_roi=dict(zip(keys, gains.tolist())), gain_anchor=rec["anchor_roi"] if rec["fit_gains"] else None,
        gain_fit_enabled=rec["fit_gains"], gain_fit_converged=fitted["converged"],
        gain_fit_iterations=fitted["iterations"], gain_relative_change=fitted["relative_gain_change"],
        nnls_converged=fitted["nnls_info"]["converged"], nnls_final_iterations=fitted["nnls_info"]["iterations"],
        nnls_projected_kkt_relative=fitted["nnls_info"]["projected_kkt_relative"],
        gains_at_bounds=[k for k, g in zip(keys, gains) if np.isclose(g, rec["gain_bounds"][0]) or np.isclose(g, rec["gain_bounds"][1])],
        tv_enabled=config["tv"]["enabled"], tv_lambda=beta,
        tv_converged=tv["converged"] if tv else None, tv_iterations=tv["iterations"] if tv else 0,
        tv_primal_relative_change=tv["primal_relative_change"] if tv else None,
        tv_dual_relative_change=tv["dual_relative_change"] if tv else None,
        nnls_relative_l2_residual=relative_residual(fitted["residuals"]),
        tv_relative_l2_residual=relative_residual(tv["residuals"]) if tv else None,
        numerical_response_rank=rank, x_cells=len(x), gram_eigenvalue_relative_cutoff=1e-14,
        roi_relative_l2_residual={k: float(np.linalg.norm(e)/np.linalg.norm(d)) if np.linalg.norm(d)>0 else None
            for k, e, d in zip(keys, (tv if tv and tv["converged"] else fitted)["residuals"], measurements)},
        sample_background_subtracted=False, fitted_reference_offset_in_forward_model=False,
        postprocessing_crop=False, postprocessing_smooth=False, symmetry_constraint=False,
        limitations=[
            "Sample X grid spacing is not measured spatial resolution; 21 foil positions are observations, not the X cell count.",
            "Gaussian moving PSF, affine centroid, constant amplitude/width; finite foil thickness is not deconvolved.",
            "Full X grid includes weak or unobservable depths: consult sensitivity_x and gram eigenvalues; zeros are not proof of empty material.",
            "TV reduces variation within the inverse problem; it does not add measurements or improve measured resolution.",
            "Relative Kapton-equivalent elastic scattering density only; no sample background, absorption or composition correction.",
            "Unweighted least squares; counting variances exclude monitor, model, gain and background uncertainty.",
        ],
    )
    log("6/6 Saving density, axes, responses, residuals and provenance")
    metadata = dict(detector_pixel_pitch_mm=acq["detector_pixel_pitch_mm"],
        sample_x_spacing_mm=dx, foil_thickness_mm=acq["foil_thickness_mm"],
        crystal_gains=gains, crystal_keys=",".join(keys), provisional=True,
        gain_fit_converged=fitted["converged"], nnls_converged=fitted["nnls_info"]["converged"],
        tv_requested=config["tv"]["enabled"], no_geometric_postprocessing=True)
    if tv:
        metadata.update(tv_converged=tv["converged"], tv_lambda=beta)
    for filename, density, label in (
        ("reconstruction.h5", primary, method),
        ("reconstruction_nnls.h5", fitted["density"], "joint NNLS"),
    ):
        path = nexus.export_result(output/filename, density, axes, input_file=inputs, solver=label, metadata=metadata)
        with h5py.File(path, "a") as f:
            f["sensitivity_x"] = sensitivity
            f["sensitivity_x"].attrs["definition"] = "sqrt(sum_a,p A_a[p,x]^2); includes fitted gains and dx/foil thickness"
            f["gram_eigenvalues"] = eigenvalues
    if tv:
        nexus.export_result(output/"reconstruction_tv.h5", tv["density"], axes,
            input_file=inputs, solver="fixed-gain TV", metadata=metadata)
    with h5py.File(output/"solver_diagnostics.h5", "x") as f:
        f["gain_relative_change_history"] = fitted["gain_change_history"]
        f["x_mm"], f["sensitivity_x"], f["gram_eigenvalues"] = x, sensitivity, eigenvalues
        f["roi_keys"] = np.asarray(keys, dtype=h5py.string_dtype())
        f["gains"] = gains
        if config["output"]["save_diagnostics"]:
            for i, key in enumerate(keys):
                g = f.create_group("rois/"+key)
                arrays = dict(response_px=responses[i], operator_px=operators[i],
                              measured_pyz=measurements[i], predicted_nnls_pyz=fitted["predicted"][i],
                              residual_nnls_pyz=fitted["residuals"][i], pixels_rc=pixels[key])
                if tv:
                    arrays.update(predicted_tv_pyz=tv["predicted"][i], residual_tv_pyz=tv["residuals"][i])
                for name, value in arrays.items():
                    g.create_dataset(name, data=value, compression="gzip", shuffle=True)
    json_write(output/"summary.json", summary)
    if config["output"]["plots"]:
        from .plots import diagnostics_plot
        diagnostics_plot(primary, axes, sensitivity, models, output/"diagnostics.png")
    log(f"Completed: {output/'reconstruction.h5'}; NNLS relative residual={summary['nnls_relative_l2_residual']}")
    return summary
