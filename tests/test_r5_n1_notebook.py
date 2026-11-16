"""Guard the clean notebook's sealed-run boundary without mounting Drive."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_sealed_execution_is_opt_in():
    notebook = json.loads((ROOT / "notebooks/R5_N1_Clean_Colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    setup = next(x for x in code if "RUN_SEALED =" in x)
    assignments = {node.targets[0].id: node.value for node in ast.parse(setup).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    for name in ("RUN_SEALED", "RESUME", "RECOVER_STALE_CLAIM", "RUN_ANALYSIS", "GR4_VERIFIED", "GR4_EXCEPTION_APPROVED",
                 "SCOPE_CHANGE_CONTROL_RESOLVED", "PRIOR_TEST_ACCESS_RECORDED"):
        assert name in assignments and isinstance(assignments[name], ast.Constant) and type(assignments[name].value) is bool
    run_cell = next(x for x in code if "--confirm-sealed-run" in x)
    tree = ast.parse(run_cell)
    assert isinstance(tree.body[0], ast.If)
    assert ast.unparse(tree.body[0].test) == "not RUN_SEALED"
    for required in ("GR4_VERIFIED", "GR4_EXCEPTION_APPROVED", "SCOPE_CHANGE_CONTROL_RESOLVED", "PRIOR_TEST_ACCESS_RECORDED",
                     "TESTS_OK", "PREFLIGHT_OK", "PREFLIGHT_SNAPSHOT", "--resume-reason"):
        assert required in run_cell
    assert any("--load-check" in c and "Ready to run on this machine: YES" in c for c in code)
    assert any("transformers==5.22.0" in c for c in code)
    assert any("analysis_clean_v1" in c for c in code)
