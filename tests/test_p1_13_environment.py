"""Lock integrity, dependency graph, runtime drift and historical artifact recovery."""
import copy
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from echoproof.artifacts import make_manifest, write_json
from echoproof.environment_lock import (dependency_closure, export_environment, inspect_export,
                                        parse_lock, verify_environment, verify_files)


class EnvironmentTests(unittest.TestCase):
    def test_lock_rejects_install_options_urls_duplicates_and_empty(self):
        self.assertEqual(parse_lock("# comment\nPyYAML==6.0.3\ntorch==2.11.0+cu130\n"),
                         {"pyyaml": "6.0.3", "torch": "2.11.0+cu130"})
        for text in ("", "--extra-index-url https://example.com", "x @ file:///tmp/x", "x>=1", "X==1\nx==2", "x==1; python_version>'3'"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_lock(text)

    def test_dependency_closure_respects_extras_markers_and_constraints(self):
        def dist(name):
            graph = {"root": ("1", ['child[fast]>=2', 'ignored; python_version < "2"']),
                     "child": ("2", ['leaf==3; extra == "fast"']), "leaf": ("3", [])}
            version, requires = graph[name]
            return SimpleNamespace(version=version, requires=requires, read_text=lambda _: None)
        with patch("echoproof.environment_lock.metadata.distribution", side_effect=dist), patch(
                "echoproof.environment_lock.metadata.version", side_effect=lambda n: dist(n).version):
            self.assertEqual(dependency_closure(("root",)), {"child": "2", "leaf": "3", "root": "1"})
        with patch("echoproof.environment_lock.metadata.distribution", side_effect=dist), patch(
                "echoproof.environment_lock.metadata.version", return_value="0"):
            with self.assertRaises(ValueError):
                dependency_closure(("root",))

    def test_direct_local_dependency_is_not_silently_pinned(self):
        dist = SimpleNamespace(read_text=lambda _: '{"dir_info":{"editable":true}}')
        with patch("echoproof.environment_lock.metadata.distribution", return_value=dist):
            with self.assertRaises(ValueError):
                dependency_closure(("local",))

    def test_recovery_rejects_escape_missing_and_modified_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "one").write_text("original")
            from echoproof.artifacts import file_digest
            snapshot = {"one": file_digest(root / "one")}
            self.assertEqual(verify_files(root, snapshot)["mismatches"], [])
            (root / "one").write_text("changed")
            self.assertEqual(verify_files(root, snapshot)["mismatches"], ["one"])
            (root / "one").unlink()
            self.assertEqual(verify_files(root, snapshot)["mismatches"], ["one"])
            with self.assertRaises(ValueError):
                verify_files(root, {"../outside": "abc"})

    def _export(self, root):
        (root / "pyproject.toml").write_text("fixture")
        history = root / "runs/phase1/old"
        history.mkdir(parents=True)
        (history / "checkpoint.pt").write_bytes(b"fixture only")
        write_json(history / "artifact_manifest.json", make_manifest(history, ["checkpoint.pt"]))
        environment = {"gpu": "Tesla T4", "cuda_available": True, "torch_cuda_build": "13.0",
                       "ffmpeg_version": "fixture ffmpeg", "nvidia_driver": "fixture driver"}
        with patch("echoproof.environment_lock.dependency_closure", return_value={"torch": "2.11.0+cu130"}), patch(
                "echoproof.environment.capture_environment", return_value=copy.deepcopy(environment)), patch(
                "echoproof.environment_lock.metadata.distributions", return_value=[]), patch(
                "echoproof.environment_lock.torch_distribution_provenance", return_value={"installer": "fixture", "metadata_sha256": {}}), patch(
                "echoproof.environment_lock.subprocess.run", return_value=SimpleNamespace(returncode=0)), patch(
                "echoproof.environment_lock.boot_id", return_value="old-boot"):
            export_environment(root, root / "export")
        return environment

    def test_export_immutable_and_manifest_detects_lock_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._export(root)
            self.assertEqual(inspect_export(root / "export")[1]["torch"], "2.11.0+cu130")
            with self.assertRaises(FileExistsError):
                export_environment(root, root / "export")
            (root / "export/requirements-lock.txt").write_text("torch==0\n")
            with self.assertRaises(ValueError):
                inspect_export(root / "export")

    def test_fresh_runtime_package_and_source_drift_control_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = self._export(root)
            with patch("echoproof.environment.capture_environment", return_value=environment), patch(
                    "echoproof.environment_lock.torch_distribution_provenance", return_value={"installer": "fixture", "metadata_sha256": {}}), patch(
                    "echoproof.environment_lock.metadata.version", return_value="2.11.0+cu130"), patch(
                    "echoproof.environment_lock.boot_id", return_value="new-boot"):
                result = verify_environment(root, root / "export")
                self.assertEqual(result["status"], "ready_for_checks")
                self.assertEqual(result["drive_recovery"]["checked_files"], 2)
                with patch("echoproof.environment_lock.boot_id", return_value="old-boot"):
                    self.assertEqual(verify_environment(root, root / "export")["status"], "needs_attention")
                with patch("echoproof.environment_lock.metadata.version", return_value="0"):
                    self.assertIn("torch", verify_environment(root, root / "export")["package_mismatches"])
                (root / "pyproject.toml").write_text("edited after export")
                self.assertEqual(verify_environment(root, root / "export")["source_verification"]["mismatches"], ["pyproject.toml"])

    def test_runner_stops_before_installation_in_same_vm(self):
        import importlib.util
        import platform
        script = Path(__file__).resolve().parents[1] / "scripts/p1_13_colab_verify.py"
        spec = importlib.util.spec_from_file_location("p113_runner_test", script)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exported = root / "export"
            exported.mkdir()
            for name in ("source_files.json", "recovery_files.json"):
                (exported / name).write_text("{}")
            expected = {"boot_id": "same-boot", "python_version": platform.python_version(),
                        "system": platform.system(), "machine": platform.machine(), "dependency_lock": "fixture"}
            helpers = SimpleNamespace(inspect_export=lambda _: (exported, {}, expected), boot_id=lambda: "same-boot",
                                      verify_files=lambda *_: {"checked_files": 1, "mismatches": []})
            with patch.object(runner, "_module", return_value=helpers), patch.object(runner, "_run") as run:
                report, folder = runner.reproduce(root, exported)
                run.assert_not_called()
                self.assertEqual(report["status"], "needs_attention")
                self.assertFalse(report["checks"]["preflight"]["fresh_runtime"])
                self.assertTrue((folder / "verification.json").is_file())

    def test_stage_a_repairs_missing_jedi_and_keeps_failed_checks_blocking(self):
        import importlib.util
        script = Path(__file__).resolve().parents[1] / "scripts/p1_13_colab_verify.py"
        spec = importlib.util.spec_from_file_location("p113_export_test", script)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for missing, install_code, check_code in ((True, 0, 0), (True, 1, 0), (False, 0, 1), (False, 0, 0)):
            with self.subTest(missing=missing, install=install_code, check=check_code), tempfile.TemporaryDirectory() as temp:
                def version(name):
                    if name == "jedi" and missing:
                        raise importlib.metadata.PackageNotFoundError(name)
                    return "fixture-version"
                def run(root, argv, **kwargs):
                    code = install_code if "install" in argv else check_code if "check" in argv else 0
                    return {"returncode": code, "argv": list(map(str, argv)), "stdout": "", "stderr": ""}
                with patch.object(runner.metadata, "version", side_effect=version), patch.object(runner, "_run", side_effect=run) as command:
                    report, folder = runner.export(temp)
                successful = (not missing or install_code == 0) and check_code == 0
                self.assertEqual(report["status"], "exported_for_reproduction" if successful else "needs_attention")
                self.assertEqual("repair_jedi" in report["checks"], missing)
                self.assertEqual("export" in report["checks"], successful)
                if missing:
                    self.assertEqual(command.call_args_list[0].args[1][-1], "jedi")
                self.assertTrue((folder / "verification.json").is_file())


if __name__ == "__main__":
    unittest.main()
