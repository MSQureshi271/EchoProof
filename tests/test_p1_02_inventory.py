"""Local checks for the portable P1-02 collector, using temporary fake projects."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/p1_02_colab_inventory.py"
spec = importlib.util.spec_from_file_location("inventory", SOURCE)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def make_project(self, root):
        project = root / "project"
        project.mkdir()
        (project / "IMPLEMENTATION_PLAN.md").write_text("fixture only")
        for relative, extension in [("genuine/wav", ".wav"), ("genuine/transcripts", ".txt"),
                                    ("tts/XTTSv2", ".wav")]:
            folder = project / "dataset" / relative
            folder.mkdir(parents=True)
            (folder / ("pair-001" + extension)).write_bytes(b"not actual audio")
        return project

    def test_pairing_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(Path(tmp))
            report = inventory.dataset_inventory(project)
            self.assertTrue(report["structural_check_passed"])
            self.assertEqual(report["complete_stem_triples"], 1)
            (project / "dataset/genuine/transcripts/pair-001.txt").unlink()
            report = inventory.dataset_inventory(project)
            self.assertFalse(report["structural_check_passed"])
            self.assertEqual(report["missing_by_folder"]["transcripts"]["count"], 1)

    def test_null_and_invalid_budget(self):
        self.assertIsNone(inventory.positive_optional(None, "budget"))
        for value in [-1, 0, True, float("nan"), float("inf"), "10"]:
            with self.assertRaises(ValueError):
                inventory.positive_optional(value, "budget")

    def test_reports_preserved_and_checksums_detect_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self.make_project(root)
            hardware = {"packages": {k: "fixture" for k in ["numpy", "torch", "pandas", "pyarrow",
                        "scipy", "scikit-learn", "librosa", "soundfile", "PyYAML", "pytest"]},
                        "pip_check": {"returncode": 0}}
            probe = {"numpy": {"ok": True}, "torch": {"ok": True, "cuda_available": False, "device": "cpu"}}
            with patch.object(inventory, "hardware_inventory", return_value=hardware), patch.object(
                    inventory, "tensor_probe", return_value=probe):
                report, output = inventory.run_inventory(project, root / "scratch", request_gpu=True)
                self.assertEqual(report["status"], "needs_attention")
                self.assertTrue(report["storage"]["fresh_process_read_verified"])
                self.assertFalse(report["storage"]["restart_verified"])
                canonical = (project / "docs/environment.md").read_bytes()
                _, output2 = inventory.run_inventory(project, root / "scratch")
            self.assertNotEqual(output, output2)
            self.assertEqual(canonical, (project / "docs/environment.md").read_bytes())
            self.assertTrue(inventory.verify_saved_run(project, output.relative_to(project))["all_saved_hashes_match"])
            notebook = json.loads((SOURCE.parents[1] / "notebooks/P1_02_Colab_Runtime_Setup.ipynb").read_text(encoding="utf-8"))
            recovery_cell = next(c["source"] for c in notebook["cells"]
                                 if c["cell_type"] == "code" and c["source"].startswith("SAVED_RUN_RELATIVE_PATH"))
            recovery_cell = recovery_cell.replace("SAVED_RUN_RELATIVE_PATH = ''", "SAVED_RUN_RELATIVE_PATH = " + repr(output.relative_to(project).as_posix()))
            recovery_cell = recovery_cell.replace("CONFIRM_FRESH_RUNTIME = False", "CONFIRM_FRESH_RUNTIME = True")
            # Simulate only the notebook branch, not a real cloud runtime restart.
            namespace = dict(vars(inventory), PROJECT_ROOT=project)
            exec(compile(recovery_cell, "recovery-cell", "exec"), namespace)
            saved_recovery = json.loads(next(output.glob("restart_verification-*.json")).read_text())
            self.assertTrue(saved_recovery["all_saved_hashes_match"])
            (output / "persistence_probe.txt").write_text("altered")
            with self.assertRaises(ValueError):
                inventory.verify_saved_run(project, output.relative_to(project))
            with self.assertRaises(ValueError):
                inventory.verify_saved_run(project, "../elsewhere")

    def test_wrong_project_or_scratch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                inventory.run_inventory(root, root / "scratch")
            project = self.make_project(root)
            with self.assertRaises(ValueError):
                inventory.run_inventory(project, project / "scratch")

    def test_notebook_code_is_valid_and_collector_matches(self):
        notebook = json.loads((SOURCE.parents[1] / "notebooks/P1_02_Colab_Runtime_Setup.ipynb").read_text(encoding="utf-8"))
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        for cell in code_cells:
            compile(cell["source"], "notebook-cell", "exec")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
        self.assertIn(SOURCE.read_text(encoding="utf-8").strip(), [c["source"].strip() for c in code_cells])


if __name__ == "__main__":
    unittest.main()
