"""Physical regression checks and complete raw-file pipeline smoke test."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xri import calibration, nexus, solver, spatial
from xri.config import load_config
from xri.pipeline import make_x_grid, run, validate_inputs


class PhysicsTests(unittest.TestCase):
    def test_frame_mapping_with_repeats_roundoff_and_missing(self):
        ix, ie, occ = calibration.map_grid([-1, -1, 0, 0, 0], [2, 3, 3+1e-15, 2, 2], [-1, 0], [2, 3])
        np.testing.assert_array_equal(occ, [[1, 1], [2, 1]])
        np.testing.assert_array_equal(ie, [0, 1, 1, 0, 0])
        with self.assertRaises(ValueError):
            calibration.map_grid([-1, -1, 0], [2, 3, 3], [-1, 0], [2, 3])
        _, occ3 = nexus.frame_grid([[0, 0, 0], [2, 2, 3], [4, 5, 5+1e-15]], [[0, 1], [2, 3], [4, 5]])
        np.testing.assert_array_equal(occ3.sum(axis=2), [[2, 1], [0, 0]])

    def test_dose_integral_and_covariance_aware_variance(self):
        dose = np.array([[1, 2], [2, 1], [1, 3]], float)
        image = np.array([[2, 4], [5, 7]], float)
        counts = dose[:, None, :]*image[None]
        weights = calibration.energy_weights([1, 2, 4])
        response, variance = calibration.integrate_response(counts, dose, weights)
        np.testing.assert_allclose(response, 3*image)
        np.testing.assert_allclose(variance, np.sum(weights[:, None, None]**2*counts/dose[:, None, :]**2, axis=0))
        effective, target_weights = nexus.interpolation_weights([1, 2, 4], [1, 1.5, 4])
        # The same source-energy count contributes to multiple interpolated energies.
        # Effective weights must be combined before squaring for variance.
        r, var = calibration.integrate_response(counts, dose, effective)
        np.testing.assert_allclose(r, 3*image)
        np.testing.assert_allclose(var, np.sum(counts*(effective[:, None, None]/dose[:, None, :])**2, axis=0))
        self.assertAlmostEqual(effective.sum(), target_weights.sum())
        with self.assertRaises(ValueError):
            nexus.interpolation_weights([1, 2], [.9, 2])

    def test_moving_spot_mapping_and_full_x_grid(self):
        rr, cc = np.mgrid[0:30, 0:12]
        pixels = np.column_stack([rr.ravel(), cc.ravel()])
        x = np.linspace(-1, 1, 9)
        observations = np.column_stack([spatial.gaussian_image(pixels.T, 100, 15+5*t, 5+.3*t, 1.1, 1.3, .1, 2) for t in x])
        model = spatial.fit_moving_psf(observations, pixels, x, detector_pitch_mm=.055)
        np.testing.assert_allclose(model["row_line"], [5, 15], atol=1e-4)
        self.assertAlmostEqual(model["sample_mm_per_detector_pixel"], 1/np.hypot(5, .3), places=5)
        grid, step = make_x_grid([-5, 5], {"one": model})
        self.assertGreater(len(grid), 21)
        np.testing.assert_allclose(step, 1/np.hypot(5, .3), atol=1e-5)
        self.assertLess(grid[0], -4.8)
        self.assertGreater(grid[-1], 4.8)
        finer = np.linspace(-1, 1, 51)
        response = spatial.evaluate_moving_psf(model, pixels, finer)
        self.assertGreater(np.linalg.matrix_rank(response), len(x))
        middle = spatial.evaluate_moving_psf(model, pixels, [.1])[:, 0].reshape(rr.shape)
        self.assertIn(np.unravel_index(middle.argmax(), middle.shape)[0], [15, 16])

    def test_nnls_physical_scale_and_anchored_gains(self):
        responses = [np.eye(3)]*3
        truth = np.array([[1, 2], [3, 1], [.5, 1]])
        gains = [1, .3, 2]
        data = [g*r@truth*.1/.05 for g, r in zip(gains, responses)]
        result = solver.fit_crystal_gains(responses, data, anchor_index=0, max_iterations=400,
            tolerance=1e-8, voxel_width_mm=.1, foil_thickness_mm=.05)
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(result["gains"], gains, atol=1e-6)
        np.testing.assert_allclose(result["density"], truth, atol=1e-6)

    def test_near_null_modes_do_not_create_exploding_density(self):
        # The numerical zero mode must be projected out of BOTH H and rhs.
        rho, status = solver.solve_gram_nnls(np.diag([1., 1e-30]),
            np.array([[1., 2.], [1e-10, 2e-10]]), return_info=True)
        self.assertTrue(status["converged"])
        np.testing.assert_allclose(rho, [[1., 2.], [0., 0.]], atol=1e-12)

    def test_nnls_matches_reference_on_resolved_problem(self):
        from scipy.optimize import nnls
        rng = np.random.default_rng(8)
        operator = rng.uniform(size=(40, 7))
        measured = rng.uniform(size=(40, 4))
        rho, status = solver.solve_gram_nnls(operator.T@operator, operator.T@measured,
            tolerance=1e-10, return_info=True)
        expected = np.column_stack([nnls(operator, col)[0] for col in measured.T])
        self.assertTrue(status["converged"])
        np.testing.assert_allclose(rho, expected, atol=1e-7)

    def test_tv_gradient_adjoint_and_objective(self):
        rng = np.random.default_rng(3)
        shape, spacing = (4, 2, 3), [.07, .055, .054]
        volume = rng.normal(size=shape)
        dual = rng.normal(size=(3, *shape))
        self.assertAlmostEqual(float(np.sum(solver.spatial_gradient(volume, spacing)*dual)),
            float(np.sum(volume*solver.gradient_adjoint(dual, spacing))), places=10)
        truth = np.array([.2, 1, .4])[:, None, None]*np.ones((3, 2, 2))
        r = [np.eye(3)]
        widths = dict(voxel_width_mm=.05, foil_thickness_mm=.05)
        tv = solver.reconstruct_tv(r, [truth], spacing_mm=[.5]*3, lambda_tv=.01,
            initial_density=truth, max_iterations=10000, tolerance=1e-8, **widths)
        self.assertTrue(tv["converged"])
        initial_objective = .01*np.linalg.norm(solver.spatial_gradient(truth, [.5]*3), axis=0).sum()
        self.assertLess(tv["objective"], initial_objective)
        self.assertTrue((tv["density"] >= 0).all())

    def test_export_xyz_schema_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.h5"
            rho = np.ones((3, 2, 2))
            axes = [[-1, 0, 1], [0, 1], [-1, 0]]
            nexus.export_result(path, rho, axes, input_file="inputs.h5", solver="NNLS")
            with h5py.File(path) as f:
                self.assertEqual(f["reinterp_insect_ck"].dtype, np.dtype("float32"))
                self.assertEqual(f["reinterp_insect_ck"].attrs["axes"], "X,Y,Z")
                self.assertEqual(len(f["reinterp_insect_ck"].dims[0]), 1)
            with self.assertRaises(FileExistsError):
                nexus.export_result(path, rho, axes, input_file="inputs.h5", solver="NNLS")


def synthetic_inputs(root):
    roi = root/"roi.tsv"
    roi.write_text("\troi_label\tx1\tx2\ty1\ty2\n1\told\t0\t19\t0\t47\n", encoding="utf-8")
    rr, cc = np.mgrid[0:47, 0:19]
    pixels = np.column_stack([rr.ravel(), cc.ravel()])
    calibration_x = np.linspace(-1, 1, 9)
    def image(x):
        return spatial.gaussian_image(pixels.T, 2000, 23+10*x, 9+.3*x, 1.1, 1.3, .1, 0).reshape(rr.shape)
    with h5py.File(root/"foil.nxs", "w") as f:
        f["entry/gui/plan_cmd"] = str(("P.grid_scan", (["D.LAMBDA"], "M.Stage_X", -1, 1, 9,
            "M.DCM_B5Energy", 1, 2, 3), {}))
        xx, ee = np.repeat(calibration_x, 3), np.tile([1., 1.5, 2.], 9)
        f["entry/instrument/M_Stage_X_setpoint"] = xx
        f["entry/instrument/M_DCM_B5Energy_setpoint"] = ee
        f["entry/data/D_SiC_Current1"] = np.full(27, 2.)
        f["entry/data/D_LAMBDA"] = np.array([np.rint(image(x)+2) for x in xx], dtype=np.uint16)
    # Put a point at X=0. All Y/Z locations and high-Z boundary must survive.
    sample = np.rint(image(0)*.1/.05*.2).astype(np.uint16)
    with h5py.File(root/"sample.nxs", "w") as f:
        f["entry/gui/plan_cmd"] = str(("P.grid_scan", (["D.LAMBDA"], "M.Stage_Y", 0, 1, 2,
            "M.Stage_Z", -1, 0, 2, "M.DCM_B5Energy", 1, 2, 2), {}))
        f["entry/instrument/M_Stage_Y_setpoint"] = np.repeat([0., 1.], 4)
        f["entry/instrument/M_Stage_Z_setpoint"] = np.tile(np.repeat([-1., 0.], 2), 2)
        f["entry/instrument/M_DCM_B5Energy_setpoint"] = np.tile([1., 2.], 4)
        f["entry/data/D_SiC_Current1"] = np.tile([2., 4.], 4)
        f["entry/data/D_LAMBDA"] = np.array([sample*d for d in np.tile([1, 2], 4)], dtype=np.uint16)
    config = dict(
        input=dict(calibration_nxs="foil.nxs", sample_nxs="sample.nxs", roi_table="roi.tsv"),
        acquisition=dict(calibration_exposure_s=.5, sample_exposure_s=.5),
        roi=dict(label_overrides={"1": "test"}), grid=dict(x_step_mm=.1),
        reconstruction=dict(anchor_roi="1_test"),
        tv={"lambda": 0., "max_iterations": 1000},
        runtime=dict(output_dir="output", chunk_frames=4), output=dict(plots=False),
    )
    path = root/"config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


class PipelineTests(unittest.TestCase):
    def test_complete_raw_pipeline_without_crop_or_surface_smoothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = synthetic_inputs(root)
            config = load_config(path)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(config)
            self.assertEqual(summary["shape_xyz"], [21, 2, 2])
            self.assertTrue(summary["tv_converged"])
            self.assertFalse(summary["postprocessing_crop"])
            self.assertFalse(summary["postprocessing_smooth"])
            with h5py.File(root/"output/reconstruction.h5") as f:
                x = f["x_mm"][:]
                rho = f["reinterp_insect_ck"][:]
                self.assertAlmostEqual(x[np.argmax(rho[:, 0, 0])], 0.)
                self.assertAlmostEqual(float(rho[:, 0, 0].sum()), .2, delta=.01)
                np.testing.assert_allclose(f["z_mm"][:], [-1, 0])
                np.testing.assert_allclose(rho[:, 0, 0], rho[:, 1, 1], atol=1e-6)
            self.assertTrue((root/"output/reconstruction_nnls.h5").exists())
            self.assertTrue((root/"output/reconstruction_tv.h5").exists())
            self.assertTrue((root/"output/solver_diagnostics.h5").exists())
            with self.assertRaises(FileExistsError):
                run(config)

    def test_incomplete_scan_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = synthetic_inputs(root)
            with h5py.File(root/"sample.nxs", "a") as f:
                # Duplicate the last energy; leave a planned grid cell unmeasured.
                f["entry/instrument/M_DCM_B5Energy_setpoint"][-1] = 1.
            config = load_config(path)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                run(config)
            self.assertFalse((root/"output").exists())

    def test_tv_nonconvergence_falls_back_and_disabled_tv_is_supported(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, supplied = synthetic_inputs(root)
                supplied["tv"].update(enabled=enabled, max_iterations=1)
                supplied["reconstruction"]["fit_gains"] = False
                path.write_text(json.dumps(supplied), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = run(load_config(path))
                self.assertEqual(summary["solver"], "joint NNLS")
                with h5py.File(root/"output/reconstruction.h5") as primary, h5py.File(root/"output/reconstruction_nnls.h5") as baseline:
                    np.testing.assert_array_equal(primary["reinterp_insect_ck"][:], baseline["reinterp_insect_ck"][:])
                self.assertEqual((root/"output/reconstruction_tv.h5").exists(), enabled)
                if enabled:
                    self.assertFalse(summary["tv_converged"])
                    with h5py.File(root/"output/reconstruction_tv.h5") as f:
                        self.assertFalse(f.attrs["tv_converged"])

    def test_explicit_exclusion_and_strict_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = synthetic_inputs(root)
            report = validate_inputs(load_config(path))
            self.assertEqual(report["roi_keys"], ["1_test"])
            for section, key, value in (("acquisition", "sample_exposure_s", 0),
                ("tv", "max_iterations", 0), ("grid", "x_step_mm", -.1),
                ("output", "plots", "false"), ("runtime", "chunk_frames", 1.5),
                ("tv", "smooth", True)):
                supplied = deepcopy(original)
                supplied.setdefault(section, {})[key] = value
                path.write_text(json.dumps(supplied), encoding="utf-8")
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    load_config(path)
            roi = root/"small_roi.tsv"
            roi.write_text("\troi_label\tx1\tx2\ty1\ty2\n31\tsmall\t0\t10\t0\t10\n", encoding="utf-8")
            self.assertFalse(calibration.read_roi_table(roi)[0].placeholder)
            self.assertTrue(calibration.read_roi_table(roi, exclude_ids=["31"])[0].placeholder)


if __name__ == "__main__":
    unittest.main()
