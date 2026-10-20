"""Run the normal test target with a selected RCF backend in a project copy."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("term", "atomic", "types")
BACKENDS = ("sage", "flint")


def main(backend: str) -> int:
    with tempfile.TemporaryDirectory(prefix=f"logic1-test-{backend}-") as temporary_dir:
        project = Path(temporary_dir) / "logic1"
        shutil.copytree(
            ROOT,
            project,
            ignore=shutil.ignore_patterns(".mypy_cache", ".pytest_cache", "__pycache__"),
        )

        wrappers = [project / "logic1" / "theories" / "RCF" / f"{name}.py"
                    for name in MODULES]
        for name, wrapper in zip(MODULES, wrappers):
            contents = wrapper.read_text().strip()
            expected = {f"from logic1.theories.RCF.{choice}.{name} import *"
                        for choice in BACKENDS}
            if contents not in expected:
                raise RuntimeError(f"Unexpected contents in {ROOT / wrapper.relative_to(project)}")

        for name, wrapper in zip(MODULES, wrappers):
            wrapper.write_text(f"from logic1.theories.RCF.{backend}.{name} import *\n")

        env = os.environ.copy()
        env["DOT_SAGE"] = str(Path(temporary_dir) / "sage")
        print(f"Running make test with {backend.upper()} in {project}", flush=True)
        return subprocess.run(["make", "test"], cwd=project, env=env, check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=BACKENDS)
    sys.exit(main(parser.parse_args().backend))
