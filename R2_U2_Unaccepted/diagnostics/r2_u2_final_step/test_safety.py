"""Standard-library safety/decision tests; no torch, audio, or GPU execution."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("diagnostic", HERE / "diagnose.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class SafetyTests(unittest.TestCase):
    def checkpoint(self):
        return {"epoch": 2, "finished": True, "skipped_steps": 2,
                "scaler": {"growth_interval": 2000, "growth_factor": 2.,
                           "backoff_factor": .5, "scale": 16384., "_growth_tracker": 0}}

    def test_final_skip_premise(self):
        result = diagnostic.premise(self.checkpoint(), [566.] * 4, 4530, 4, 4)
        self.assertEqual(result["pre_attempt_scale"], 32768.)
        self.assertEqual(result["expected_attempts"], 568)

    def test_premise_fails_closed(self):
        for key, value in (("scale", 32768.), ("_growth_tracker", 1),
                           ("growth_interval", 100), ("backoff_factor", .25)):
            latest = self.checkpoint()
            latest["scaler"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnostic.premise(latest, [566.] * 4, 4530, 4, 4)

    def test_wrong_counters_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic.premise(self.checkpoint(), [567.] * 4, 4530, 4, 4)

    def test_unfinished_rejected(self):
        latest = self.checkpoint()
        latest["finished"] = False
        with self.assertRaises(ValueError):
            diagnostic.premise(latest, [566.] * 4, 4530, 4, 4)

    def cases(self):
        return [{"name": n, "scaled_nonfinite": n not in ("fp16_scale1", "fp32_scale1"),
                 "matches_original": True, "rng_unchanged": True}
                for n in ("original", "original_repeat", "instrumented", "fp16_scale1", "fp32_scale1")]

    def test_no_reproduction_not_success(self):
        cases = self.cases()
        cases[0]["scaled_nonfinite"] = False
        self.assertIn("did not reproduce", diagnostic.conclusion(cases))

    def test_instrumentation_mismatch_limits_conclusion(self):
        cases = self.cases()
        cases[2]["matches_original"] = False
        self.assertIn("parity failed", diagnostic.conclusion(cases))

    def test_fp32_failure_not_blamed_on_scaling(self):
        cases = self.cases()
        cases[-1]["scaled_nonfinite"] = True
        self.assertIn("beyond FP16", diagnostic.conclusion(cases))

    def test_scaling_conclusion_is_qualified(self):
        self.assertIn("first historical event remains unresolved", diagnostic.conclusion(self.cases()))

    def test_rng_consumption_prevents_exact_reconstruction_claim(self):
        cases = self.cases()
        cases[0]["rng_unchanged"] = False
        self.assertIn("not an exact historical reconstruction", diagnostic.conclusion(cases))

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic.inside(HERE, "../escape")

    def test_script_contains_no_training_or_publication_calls(self):
        tree = ast.parse((HERE / "diagnose.py").read_text(encoding="utf-8"))
        forbidden = {"fit", "step", "run_warmup", "save", "append_ledger", "write_json",
                     "_optimizer_step", "_load_resume", "_write_predictions", "_ledger"}
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(forbidden.intersection(calls))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update":
                self.assertIn(ast.unparse(n.func.value), ("report", "os.environ"))
        constructors = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn("Trainer", constructors)

    def test_only_new_report_is_opened_for_writing(self):
        tree = ast.parse((HERE / "diagnose.py").read_text(encoding="utf-8"))
        modes = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "open":
                modes.append(n.args[0].value)
        self.assertEqual(sorted(modes), ["rb", "x"])

    def test_notebook_schema_and_code_parse(self):
        notebook = json.loads((HERE / "R2_U2_Final_Step_Diagnostic.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                # Colab may autosave outputs into this new notebook while running
                # these checks. Only delivery-time checks require empty outputs.
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
