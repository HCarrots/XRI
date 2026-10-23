"""Exercise the Bash entry point against small, complete raw HDF5 scans."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py

from test_xri import synthetic_inputs


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Requires POSIX Bash")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="xri launcher ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "XRI with spaces"
        self.project.mkdir()
        source = Path(__file__).resolve().parents[1]
        for name in ("run.sh", "run_xri.py"):
            shutil.copy2(source / name, self.project / name)
        shutil.copytree(source / "xri", self.project / "xri", ignore=shutil.ignore_patterns("__pycache__"))
        self.caller = self.root / "unrelated working directory"
        self.caller.mkdir()
        self.config_path = self.make_config(self.project)

    def make_config(self, directory):
        path, config = synthetic_inputs(directory)
        for key, original, renamed in (
            ("calibration_nxs", "foil.nxs", "foil scan.hdf5"),
            ("sample_nxs", "sample.nxs", "sample scan.h5"),
        ):
            destination = directory / renamed
            (directory / original).rename(destination)
            config["input"][key] = str(destination.resolve())
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def launch(self, *args):
        return subprocess.run(
            [shutil.which("bash"), str(self.project / "run.sh"), *map(str, args)],
            cwd=self.caller, env=dict(os.environ, PYTHON_BIN=sys.executable),
            capture_output=True, text=True, timeout=60,
        )

    def test_default_config_full_run_and_existing_results_protected(self):
        checked = self.launch("--validate-only")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        output = self.project / "output"
        self.assertFalse(output.exists())

        completed = self.launch()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((output / "reconstruction.h5").is_file())
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["shape_xyz"], [21, 2, 2])
        self.assertTrue(summary["tv_converged"])
        before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in output.iterdir()}

        repeated = self.launch()
        self.assertEqual(repeated.returncode, 1, repeated.stderr)
        self.assertIn("XRI ERROR:", repeated.stderr)
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).digest() for p in output.iterdir()})

    def test_config_override_and_invalid_inputs(self):
        alternate = self.root / "other data"
        alternate.mkdir()
        config_path = self.make_config(alternate)
        config_path = config_path.rename(alternate / "other settings.json")
        # The default config is deliberately unusable: --config must override it.
        self.config_path.write_text("{}", encoding="utf-8")
        for supplied in (str(config_path), os.path.relpath(config_path, self.caller)):
            checked = self.launch("--config", supplied, "--validate-only")
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertFalse((alternate / "output").exists())

        sample = alternate / "sample scan.h5"
        sample.unlink()
        missing = self.launch("--config", config_path)
        self.assertEqual(missing.returncode, 1, missing.stderr)
        self.assertIn("input.sample_nxs", missing.stderr)
        self.assertFalse((alternate / "output").exists())

        with h5py.File(sample, "w") as handle:
            handle["unrelated_data"] = [1, 2, 3]
        invalid = self.launch("--config", config_path)
        self.assertEqual(invalid.returncode, 1, invalid.stderr)
        self.assertIn("XRI ERROR:", invalid.stderr)
        self.assertFalse((alternate / "output").exists())


if __name__ == "__main__":
    unittest.main()
