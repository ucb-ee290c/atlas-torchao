"""Access to the reference author's quantization code, pinned as git submodules.

The reference repositories cannot be pip-installed next to atlas-torchao: their pins conflict with
ours (numpy<2.0 and transformers==4.53.2 against numpy==2.2.6 and transformers==5.5.4), their
package list misses `funct_models_vector` (imported by pi0_inout/hw_float_ops.py:24), and
`pi0_inout/__init__.py:31-54` imports the whole model patcher. So the submodule roots are put on
sys.path (appended, never ahead of our own modules) and the reference modules are imported from
there, or loaded as single files by path with `load_file`.

Every use goes through `require()`, which checks that each submodule's checked-out commit is the
pinned one: a number attributed to "the reference" must come from the reference revision we cite.
`provenance()` returns {repo: {commit, files: {path: sha256}}} for every reference file actually
loaded in this process, so a result JSON records exactly which bytes produced it.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"

# Pinned revisions (full shas). pi0-quant comes first on sys.path: it is the only one that carries
# dependencies/fpex (needed by funct_models_vector/lane_boxes/exp.py:146-151); the funct_models_*
# and func_models_sa trees are byte-identical between the two repositories at these pins.
PINS = {
    "pi0-quant": "8bc0f6a6cb3faa237eb20b2354a0c8bc7afa524e",
    "smolVLA-quant": "337363277c347927c45121a40c8ed049d72cff82",
}
ORDER = ("pi0-quant", "smolVLA-quant")

_checked = False


class ReferenceUnavailable(RuntimeError):
    """A submodule is missing or not at its pinned commit."""


def repo_path(repo: str) -> Path:
    return THIRD_PARTY / repo


def head(repo: str) -> str | None:
    path = repo_path(repo)
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_pins() -> dict:
    """{repo: (expected, actual)} for every submodule that is missing or off its pin."""
    bad = {}
    for repo, pin in PINS.items():
        actual = head(repo)
        if actual != pin:
            bad[repo] = (pin, actual)
    return bad


def require() -> None:
    """Check the pins once and put the submodule roots on sys.path.

    Set ATLAS_REF_ALLOW_UNPINNED=1 to use a different checkout deliberately (the provenance block
    then records the commit that was actually used).
    """
    global _checked
    if _checked:
        return
    bad = check_pins()
    if bad and os.environ.get("ATLAS_REF_ALLOW_UNPINNED") != "1":
        lines = [f"  {r}: expected {e}, found {a or 'missing'}" for r, (e, a) in bad.items()]
        raise ReferenceUnavailable(
            "reference submodules are not at their pinned commits:\n" + "\n".join(lines) +
            "\nrun: git submodule update --init third_party/pi0-quant third_party/smolVLA-quant"
            "\n(do not use --recursive: smolVLA-quant's nested npu_model is not needed)")
    for repo in ORDER:
        root = str(repo_path(repo))
        if root not in sys.path:
            sys.path.append(root)
    _checked = True


def import_ref(name: str):
    """Import a reference module (e.g. 'funct_models_ipt.ipt_numba.ipt_rtl_linear')."""
    require()
    return importlib.import_module(name)


def load_file(repo: str, relpath: str, name: str | None = None):
    """Load one reference source file as a standalone module (plan F3).

    sys.path is restored afterwards, so a file that inserts its own repository at the front of
    sys.path (as smolVLA-quant experiments/run_eval.py:74-75 does) cannot reorder our imports.
    """
    require()
    path = repo_path(repo) / relpath
    name = name or f"_ref_{repo.replace('-', '_')}_{relpath.replace('/', '_').removesuffix('.py')}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.path[:] = saved
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    """{repo: {commit, pinned, files: {relpath: sha256}}} for the reference files loaded so far.

    C headers next to a loaded module are included: the C functional models compile them at first
    use (ipt_rtl_linear_c.py:366-391, systolic_array_rtl_linear.py:107-132).
    """
    out = {}
    for repo in ORDER:
        root = repo_path(repo).resolve()
        files: dict[str, str] = {}
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            p = Path(f).resolve()
            if root not in p.parents:
                continue
            files[str(p.relative_to(root))] = _sha256(p)
            for h in p.parent.glob("*.h"):
                files[str(h.relative_to(root))] = _sha256(h)
        if files:
            out[repo] = {"commit": head(repo), "pinned": PINS[repo], "files": dict(sorted(files.items()))}
    return out
