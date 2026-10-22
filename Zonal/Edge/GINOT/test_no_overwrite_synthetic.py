from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Optional


def _load_guard(script_path: Path):
    source = script_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(script_path))
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_prohibit_resume_or_overwrite":
            isolated = ast.Module(body=[node], type_ignores=[])
            code = compile(isolated, filename=str(script_path), mode="exec")
            ns = {"Path": Path, "Optional": Optional}
            exec(code, ns)
            return ns["_prohibit_resume_or_overwrite"]
    raise RuntimeError(f"No _prohibit_resume_or_overwrite guard found in {script_path}")


def test_no_overwrite_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    training_scripts = [
        repo_root / "Zonal/Edge/GINOT/Training_script.py",
        repo_root / "Uniform/Edge/GINOT/Training_script.py",
        repo_root / "Zonal/Edge_no_stress/GINOT/Training_script.py",
        repo_root / "Zonal/Edge_arc_feat/GINOT/Training_script.py",
    ]
    if not training_scripts:
        raise RuntimeError("No GINOT training scripts found for synthetic guard test.")

    for script_path in training_scripts:
        guard = _load_guard(script_path)

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "dummy.pt"
            ckpt_path.touch()

            try:
                guard("MATCH_250K", ckpt_path)
                raise AssertionError(f"Expected RuntimeError for existing checkpoint in {script_path}")
            except RuntimeError as exc:
                msg = str(exc)
                assert "GINOT-A" in msg
                assert "MATCH_250K" in msg
                assert str(ckpt_path) in msg
                assert "overwrite and resume are prohibited" in msg

        with tempfile.TemporaryDirectory() as td:
            resume_target = Path(td) / "resume_candidate.pt"
            try:
                guard("MATCH_250K", resume_target, resume_request="--resume")
                raise AssertionError(f"Expected RuntimeError for resume request in {script_path}")
            except RuntimeError as exc:
                msg = str(exc)
                assert "GINOT-A" in msg
                assert "MATCH_250K" in msg
                assert "overwrite and resume are prohibited" in msg


if __name__ == "__main__":
    test_no_overwrite_smoke()
    print("No-overwrite synthetic checks passed for all GINOT training scripts.")
