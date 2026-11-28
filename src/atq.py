"""atq: one command for the main jobs of atlas-torchao.

    atq export MODEL EXPORT_DIR [options]      quantize and write model.safetensors, policy.json, manifest.json
    atq eval   MODEL [options]                 evaluate per --modes (re-evaluate an export with --from-export DIR)
    atq fx     EXPORT_DIR FX_DIR [options]     FX graph bundle of an export (native FP8 by default)
    atq verify EXPORT_DIR FX_DIR [OUT_JSON]    independent checks of an export + FX bundle

`python -m atq ...` is the same command without the console script (PYTHONPATH=src).

Each subcommand is a thin translation into the existing command, which then runs unchanged in this process, as if it
had been started from the shell (same parser, same code, same outputs):

    export / eval, family llm      python src/eval.py --model MODEL ...
    export / eval, family policy   python src/eval_policy.py --policy MODEL ...      (SmolVLA, Multi-Task DiT)
    export / eval, family dronet   python src/eval_dronet.py --weights MODEL ...
    fx                             python -m exporter.fx_export --export-dir EXPORT_DIR --out-dir FX_DIR ...
    verify                         python src/scripts/tile_flow/verify_bundle.py EXPORT_DIR FX_DIR OUT_JSON ...

Every option after the positional arguments is passed through verbatim, so all options of the underlying command work
(`atq export MODEL EXPORT_DIR --help` lists them). The equivalent command is printed before it runs; --dry-run prints
it and stops.

Defaults the subcommands add (an explicit option given after them overrides each one):
    export   --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --out <EXPORT_DIR>_cli.json
    fx       --native-fp8 (--no-native-fp8 leaves it out: the decode-to-aten graph)
    verify   OUT_JSON = <FX_DIR>_verify_bundle.json

The model family is --family, or else, without guessing and without the network: MODEL ending in .npz is dronet;
--dataset given is policy (only eval_policy.py takes one); otherwise the model's config.json (a local directory, or
the Hugging Face cache) decides: `model_type` is llm, a LeRobot `type` is policy (then --dataset is required).
Pi0 is not covered: it runs in openpi's separate environment, from --checkpoint-dir and --openpi-dir; use
src/eval_pi0.py (docs/pi0.md).
"""
from __future__ import annotations

import json
import os
import runpy
import shlex
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
DEFAULT_ALGO = "atlas_v1_awq_tile_static"     # explicit recipe name, never the moving `atlas_v1` alias
FAMILIES = {                                  # family -> (script under src/, its model flag)
    "llm": ("eval.py", "--model"),
    "policy": ("eval_policy.py", "--policy"),
    "dronet": ("eval_dronet.py", "--weights"),
}
VERIFY_SCRIPT = "scripts/tile_flow/verify_bundle.py"
FX_MODULE = "exporter.fx_export"
SUBCOMMANDS = {               # name -> (positional names, required count, one-line summary)
    "export": (("MODEL", "EXPORT_DIR"), 2, "quantize MODEL and write the export to EXPORT_DIR (must not exist)"),
    "eval": (("MODEL",), 1, "evaluate MODEL per --modes; --from-export DIR re-evaluates an export"),
    "fx": (("EXPORT_DIR", "FX_DIR"), 2, "FX graph bundle of an export in FX_DIR (must not exist)"),
    "verify": (("EXPORT_DIR", "FX_DIR", "OUT_JSON"), 2, "check an export and its FX bundle; writes OUT_JSON"),
}


class UsageError(Exception):
    pass


def _has(args, flag):
    """True when `flag` appears in args, as `--flag` or `--flag=value`."""
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _take_option(args, flag, takes_value):
    """Remove every occurrence of `flag` from args; returns (value or True, or None when absent; the rest)."""
    found, rest, i = None, [], 0
    while i < len(args):
        a = args[i]
        if a == flag:
            if takes_value:
                if i + 1 >= len(args):
                    raise UsageError(f"{flag} needs a value")
                found, i = args[i + 1], i + 2
                continue
            found = True
        elif takes_value and a.startswith(flag + "="):
            found = a.split("=", 1)[1]
        else:
            rest.append(a)
        i += 1
    return found, rest


def _positionals(args, names, required):
    """The leading arguments that are not options (up to len(names)); options must follow them."""
    pos = []
    while args and len(pos) < len(names) and not args[0].startswith("-"):
        pos.append(args.pop(0))
    if len(pos) < required:
        missing = " ".join(names[len(pos):required])
        raise UsageError(f"missing {missing} (the positional arguments come first, before any option)")
    return pos, args


def _model_config(model):
    """config.json of a local model directory or of an HF id already in the cache (never the network); None if unknown."""
    path = Path(model) / "config.json"
    if not path.is_file():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model, "config.json")
        except Exception:     # not a repo id, or no hub: the family stays unknown
            cached = None
        if not isinstance(cached, str):
            return None
        path = Path(cached)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def detect_family(model, rest):
    """The model family from unambiguous, offline evidence; UsageError when there is none."""
    if model.endswith(".npz"):
        return "dronet"
    if _has(rest, "--dataset"):
        return "policy"
    cfg = _model_config(model)
    if isinstance(cfg, dict):
        if "model_type" in cfg:
            return "llm"
        if "type" in cfg:
            raise UsageError(f"{model} is a LeRobot policy ({cfg['type']}): add --dataset REPO_ID")
    raise UsageError(f"cannot tell the model family of {model} offline (no config.json in the directory or the "
                     f"Hugging Face cache); pass --family {'|'.join(FAMILIES)}")


def translate(argv):
    """atq argv (without the program name) -> (target, legacy argv, dry_run).
    target is ("path", script path) or ("module", module name); legacy argv excludes the program name."""
    if not argv or argv[0] in ("-h", "--help"):
        raise UsageError(None)
    cmd, args = argv[0], list(argv[1:])
    if cmd not in SUBCOMMANDS:
        raise UsageError(f"unknown subcommand {cmd!r}; one of: {', '.join(SUBCOMMANDS)}")
    names, required, _ = SUBCOMMANDS[cmd]
    dry_run, args = _take_option(args, "--dry-run", False)
    family = None
    if cmd in ("export", "eval"):
        family, args = _take_option(args, "--family", True)
        if family is not None and family not in FAMILIES:
            raise UsageError(f"--family must be one of {', '.join(FAMILIES)}, not {family!r}")
    no_native = None
    if cmd == "fx":
        no_native, args = _take_option(args, "--no-native-fp8", False)
        if _has(args, "--verify-dir"):
            required = 0                       # replay of an existing bundle: atq fx --verify-dir FX_DIR
    if args and args[0] in ("-h", "--help"):
        raise UsageError(cmd)                 # help before the positionals: this subcommand's own help
    pos, rest = _positionals(args, names, required)

    if cmd in ("export", "eval"):
        model = pos[0]
        family = family or detect_family(model, rest)
        script, model_flag = FAMILIES[family]
        defaults = []
        if cmd == "export":
            export_dir = pos[1].rstrip("/") or pos[1]
            if not _has(rest, "--modes"):
                defaults += ["--modes", "ref_conv"]
            if not _has(rest, "--export-only"):
                defaults += ["--export-only"]
            if not _has(rest, "--algo"):
                defaults += ["--algo", DEFAULT_ALGO]
            if not _has(rest, "--out"):
                defaults += ["--out", export_dir + "_cli.json"]
            defaults += ["--export-dir", pos[1]]
        return ("path", str(SRC / script)), [model_flag, model, *defaults, *rest], bool(dry_run)
    if cmd == "fx":
        head = []
        if len(pos) == 1:
            raise UsageError("give EXPORT_DIR and FX_DIR, or only --verify-dir FX_DIR")
        if pos:
            head = ["--export-dir", pos[0], "--out-dir", pos[1]]
        if not no_native and not _has(rest, "--native-fp8") and not _has(rest, "--verify-dir"):
            head.append("--native-fp8")
        return ("module", FX_MODULE), [*head, *rest], bool(dry_run)
    # verify
    export_dir, fx_dir = pos[0], pos[1]
    out = pos[2] if len(pos) > 2 else fx_dir.rstrip("/") + "_verify_bundle.json"
    return ("path", str(SRC / VERIFY_SCRIPT)), [export_dir, fx_dir, out, *rest], bool(dry_run)


def legacy_command(target, legacy_argv):
    """The equivalent shell command (run from the repository root with PYTHONPATH=src)."""
    kind, name = target
    if kind == "module":
        head = ["python", "-m", name]
    else:
        try:
            shown = os.path.relpath(name)
        except ValueError:
            shown = name
        head = ["python", shown if not shown.startswith("..") else name]
    return shlex.join([*head, *legacy_argv])


def run(target, legacy_argv):
    """Run the existing command in this process exactly as the shell would: as __main__, with its own sys.argv."""
    kind, name = target
    saved_argv, saved_path = sys.argv[:], sys.path[:]
    try:
        if kind == "module":
            sys.argv = [name, *legacy_argv]       # runpy replaces argv[0] with the module's file
            runpy.run_module(name, run_name="__main__", alter_sys=True)
        else:
            sys.argv = [name, *legacy_argv]
            sys.path.insert(0, str(Path(name).parent))   # `python script.py` puts the script's directory first
            runpy.run_path(name, run_name="__main__")
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path
    return 0


def _usage(cmd=None):
    if cmd is None:
        lines = ["usage: atq {export,eval,fx,verify} ...", ""]
        for name, (names, required, summary) in SUBCOMMANDS.items():
            shown = " ".join(n if i < required else f"[{n}]" for i, n in enumerate(names))
            lines.append(f"  atq {name} {shown} [options]".ljust(56) + summary)
        lines += ["", "Every option after the positional arguments goes to the underlying command unchanged; "
                  "--dry-run prints that command.", "Details: python -m atq --help, docs/user-guide.md."]
        return "\n".join(lines)
    names, required, summary = SUBCOMMANDS[cmd]
    shown = " ".join(n if i < required else f"[{n}]" for i, n in enumerate(names))
    extra = {
        "export": "  --family {llm,policy,dronet}  default: detected (see python -m atq --help)\n"
                  f"  adds unless given: --modes ref_conv --export-only --algo {DEFAULT_ALGO} --out EXPORT_DIR_cli.json\n"
                  "  all options of the family's eval script: atq export MODEL EXPORT_DIR --help",
        "eval": "  --family {llm,policy,dronet}  default: detected (see python -m atq --help)\n"
                "  all options of the family's eval script: atq eval MODEL --help",
        "fx": "  --no-native-fp8               leave out --native-fp8 (the decode-to-aten graph)\n"
              "  atq fx --verify-dir FX_DIR  fresh-process replay of an existing bundle\n"
              "  all options of exporter.fx_export: atq fx EXPORT_DIR FX_DIR --help",
        "verify": "  OUT_JSON defaults to FX_DIR_verify_bundle.json\n"
                  "  all options of verify_bundle.py: atq verify EXPORT_DIR FX_DIR --help",
    }[cmd]
    return f"usage: atq {cmd} {shown} [options]\n\n{summary}\n\n{extra}\n  --dry-run                     print the equivalent command and stop"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    try:
        target, legacy_argv, dry_run = translate(argv)
    except UsageError as e:
        msg = e.args[0] if e.args else None
        if msg is None or msg in SUBCOMMANDS:
            print(_usage(msg))
            return 0 if argv else 2
        usage = _usage(argv[0] if argv[0] in SUBCOMMANDS else None).splitlines()[0]
        print(f"{usage}\natlas: error: {msg}", file=sys.stderr)
        return 2
    command = legacy_command(target, legacy_argv)
    if dry_run:
        print(command)
        return 0
    print(f"atq: {command}", file=sys.stderr, flush=True)
    return run(target, legacy_argv)


if __name__ == "__main__":
    raise SystemExit(main())
