"""Guard the project layout: old module paths, dead commands and broken doc links.

Text/AST checks only; no project module is executed (find_spec imports only
the docstring-only parent packages).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ROOT / ".git",
    ROOT / "results",
    ROOT / "docs" / "history",
    ROOT / "scripts" / "legacy",
    ROOT / "web_safod" / "site",
    ROOT / "web_safod" / "cache",
}
EXCLUDED_NAMES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
THIS_FILE = Path(__file__).resolve()

OLD_NAME_PATTERNS = [
    r"\bfrom\s+src\.(grid|model|signal_processing|solver_numpy|solver_numba_fused"
    r"|solver_numpy_pointforce|simulator|source|source_spreading|source_injection"
    r"|sampling|receivers|das|misfit|analytical_2d|analytical_moment_tensor|plotting)"
    r"\s+import\b",
    r"\bweb_safod\.src\b",
    r"web_safod/src/",
    r"\bscripts\.(validate_pointforce_2d|benchmark_backends|run_performance_study)\b",
    r"\bscripts\.validation\.sbp_sat_",
    r"\bsbp_sat_(operator_qc|time_energy_qc|adjoint_dot_product|taylor)\b",
    r"\bsbp_sat\.forward_qc\b",
    r"\bscripts\.safod\.(models|find_catalog_events|check_catalog_coverage"
    r"|screen_events|diagnose_geometry)\b",
    r"scripts/safod/models/",
    r"^#SBATCH\s+--(output|error)=(logs/|[^/\s]+$)",
    r"results/(real_event_|forward_real_event_|compare_real_synthetic_|scaling\b"
    r"|final_figures|model_qc|safod/initial_model|examples/numpy_homogeneous_baseline"
    r"|examples/homogeneous_bilinear_free_surface|pointforce_validation)",
    r"results/validation/elastic/pointforce(?!_2d)\b",
    r"'elastic'\s*/\s*'pointforce'(?!_2d)",
    r"(^|[\s\"'(`])(outputs|reports)/",
    r"\.\./\.\./\.\./\.\./project_archives",
    r"--output-dir\s+results/validation/sbp_sat/(reference_v2|sherlock_20261016)/",
]

TEXT_SUFFIXES = {".py", ".md", ".sbatch", ".sh", ".ini", ".cfg", ".toml"}


def _excluded(path: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in path.parts):
        return True
    return any(path == d or d in path.parents for d in EXCLUDED_DIRS)


def _project_files():
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not _excluded(path):
            yield path


def _notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _scanned_texts():
    for path in _project_files():
        if path.resolve() == THIS_FILE:
            continue
        rel = path.relative_to(ROOT)
        if path.suffix in TEXT_SUFFIXES or path.name == "requirements.txt":
            yield rel, path.read_text(encoding="utf-8", errors="replace")
        elif path.suffix == ".json" and rel.parts[0] in {"config"}:
            yield rel, path.read_text(encoding="utf-8")
        elif path.suffix == ".json" and rel.parts[:2] == ("web_safod", "config"):
            yield rel, path.read_text(encoding="utf-8")
        elif path.suffix == ".ipynb":
            yield rel, _notebook_code(path)


def test_no_old_names():
    compiled = [re.compile(p, re.MULTILINE) for p in OLD_NAME_PATTERNS]
    hits = []
    for rel, text in _scanned_texts():
        for pattern in compiled:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{rel}:{line}: {match.group(0).strip()!r}")
    assert not hits, "Old layout names found:\n" + "\n".join(hits)


_PYTHON_M = re.compile(r"python[0-9.]*\s+(?:-B\s+)?-m\s+([A-Za-z_][\w.]*)")
_NOTEBOOK_M = re.compile(r"""['"]-m['"]\s*,\s*['"]([A-Za-z_][\w.]*)['"]""")


def test_python_m_targets_resolve():
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    for name in ("scripts", "src", "web_safod"):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin and Path(spec.origin).resolve().is_relative_to(ROOT), (
            f"package {name!r} resolves outside the project: {spec and spec.origin}"
        )
    missing = []
    for rel, text in _scanned_texts():
        if rel.suffix not in {".md", ".sbatch", ".sh", ".ipynb"}:
            continue
        pattern = _NOTEBOOK_M if rel.suffix == ".ipynb" else _PYTHON_M
        for module in pattern.findall(text):
            module = module.rstrip(".")
            if not module.startswith(("scripts.", "src.", "web_safod.")):
                continue
            if importlib.util.find_spec(module) is None:
                missing.append(f"{rel}: python -m {module}")
    assert not missing, "Unresolvable python -m targets:\n" + "\n".join(missing)


_MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def test_markdown_links():
    broken = []
    for path in _project_files():
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for target in _MD_LINK.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve()
            if not target_path.is_relative_to(ROOT):
                broken.append(f"{path.relative_to(ROOT)}: {target} (leaves the project)")
                continue
            if target_path.is_relative_to(ROOT / "results") and not target_path.parent.exists():
                continue  # results/ is local; skip runs absent from this checkout
            if not target_path.exists():
                broken.append(f"{path.relative_to(ROOT)}: {target}")
    assert not broken, "Broken markdown links:\n" + "\n".join(broken)
