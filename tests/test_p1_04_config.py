"""P1-04 boundary and CLI behavior checks; no research files are read."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml
from echoproof.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        (self.root / "pyproject.toml").write_text("", encoding="utf-8")
        self.file = self.root / "config.yaml"
        self.data = yaml.safe_load((ROOT / "configs/phase1.yaml").read_text(encoding="utf-8"))
        self.data["paths"]["staging_root"] = str(Path(self.temp.name) / "scratch")
        self.write()

    def write(self):
        self.file.write_text(yaml.safe_dump(self.data), encoding="utf-8")

    def test_deterministic_export_and_cli(self):
        config = load_config(self.file, self.root)
        self.assertEqual(config.to_json(), load_config(self.file, self.root).to_json())
        self.assertEqual(json.loads(config.to_json())["metric_scope"], "software_probe")
        result = subprocess.run([sys.executable, "-m", "echoproof.cli", "validate-config", "--project-root", str(self.root), "--config", "config.yaml"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), json.loads(config.to_json()))

    def test_rejects_wrong_labels_unknown_keys_and_dimensions(self):
        for modify in (lambda d: d["labels"].update(tts=0), lambda d: d.update(extra=True), lambda d: d["audio"].update(crop_samples=0), lambda d: d["probe"].update(timing_limit_seconds=float("nan"))):
            with self.subTest(modify=modify):
                self.data = yaml.safe_load((ROOT / "configs/phase1.yaml").read_text(encoding="utf-8"))
                self.data["paths"]["staging_root"] = str(Path(self.temp.name) / "scratch")
                modify(self.data)
                self.write()
                with self.assertRaises(ConfigError):
                    load_config(self.file, self.root)

    def test_rejects_output_escape_and_research_fixture(self):
        for name, value in (("run_root", "runs/../../escape"), ("fixtures", "dataset/genuine/wav"), ("staging_root", "dataset")):
            with self.subTest(name=name):
                original = self.data["paths"][name]
                self.data["paths"][name] = value
                self.write()
                with self.assertRaises(ConfigError):
                    load_config(self.file, self.root)
                self.data["paths"][name] = original

    def test_duplicate_keys_and_unsafe_yaml_rejected(self):
        for value in ("seed: 1\nseed: 2\n", "!!python/object/apply:os.system ['echo bad']\n"):
            self.file.write_text(value, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(self.file, self.root)

    def test_cli_invalid_config_is_structured(self):
        self.data["labels"]["tts"] = 0
        self.write()
        result = subprocess.run([sys.executable, "-m", "echoproof.cli", "validate-config", "--project-root", str(self.root), "--config", "config.yaml"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["type"], "invalid_config")


if __name__ == "__main__":
    unittest.main()
