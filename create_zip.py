"""One-command packager:  python create_zip.py
Creates student_performance_prediction.zip next to this folder, skipping caches."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "student_performance_prediction.zip"
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".zip"}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(ROOT.rglob("*")):
        rel = f.relative_to(ROOT)
        if f.is_dir() or SKIP_DIRS & set(rel.parts) or f.suffix in SKIP_SUFFIX:
            continue
        z.write(f, Path(ROOT.name) / rel)
print(f"Created {OUT}")
