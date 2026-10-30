#!/usr/bin/env python3
"""Check the public notebook, recorded evidence, links, and payload."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Callable
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Nemotron_T4_Colab.ipynb"
RESULTS = ROOT / "evidence" / "2026-10-30" / "small-gpu-results.json"
VERIFICATION = ROOT / "evidence" / "2026-10-30" / "verification.json"
HISTORY = ROOT / "evidence" / "2026-10-30" / "executed-history.ipynb"
SOURCE_MAP = ROOT / "evidence" / "2026-10-30" / "source-map.json"
SYNC_MARKER = re.compile(r"^# sync-script: ([A-Za-z0-9_./-]+)\s*$")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
HTML_REFERENCE = re.compile(r"\b(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
TEXT_SUFFIXES = {
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
BLOCKED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".zip",
}
SECRET_PATTERNS = (
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer credential", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{24,}")),
    ("local user path", re.compile(r"(?i)[A-Z]:\\(?:Users|Prj)\\")),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    errors: tuple[str, ...] = ()
    skipped: bool = False


def _read_notebook(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        import nbformat
    except ImportError:
        return None, ["Install nbformat to validate notebooks."]

    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:
        return None, [f"{path.relative_to(ROOT)} is not a valid notebook: {exc}"]
    return notebook, []


def _synced_code_errors(notebook: dict[str, object]) -> list[str]:
    errors: list[str] = []
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        return ["Notebook cells must be a list."]

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str):
            errors.append(f"Notebook cell {index} has invalid source text.")
            continue
        lines = source.splitlines(keepends=True)
        if not lines:
            continue
        marker = SYNC_MARKER.match(lines[0].rstrip("\r\n"))
        if marker is None:
            continue
        relative = Path(marker.group(1))
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            errors.append(f"Notebook cell {index} points to a missing or unsafe script: {relative}.")
            continue
        embedded = "".join(lines[1:])
        expected = target.read_text(encoding="utf-8").replace("\r\n", "\n")
        if embedded.replace("\r\n", "\n") != expected:
            errors.append(f"Notebook cell {index} does not match {relative}.")
    return errors


def check_notebook(scaffold: bool) -> CheckResult:
    missing = [path for path in (NOTEBOOK, ROOT / "scripts" / "lab.py") if not path.is_file()]
    if missing and scaffold:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        return CheckResult("Public notebook", (f"Scaffold mode skipped missing integration files: {names}.",), skipped=True)
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        return CheckResult("Public notebook", (f"Required integration files are missing: {names}.",))

    notebook, errors = _read_notebook(NOTEBOOK)
    if errors or notebook is None:
        return CheckResult("Public notebook", tuple(errors))

    cells = notebook.get("cells", [])
    code = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"]
    code_text: list[str] = []
    for index, cell in enumerate(code):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        code_text.append(source if isinstance(source, str) else "")
        if cell.get("outputs"):
            errors.append(f"Public notebook code cell {index} contains saved outputs.")
        if cell.get("execution_count") is not None:
            errors.append(f"Public notebook code cell {index} has an execution count.")

    source_text = "\n".join(code_text)
    if not code:
        errors.append("Public notebook has no code cells.")
    direct_import = re.search(
        r"(?m)^\s*(?:from\s+scripts\s+import\s+lab|import\s+scripts\.lab(?:\s+as\s+lab)?)\s*$",
        source_text,
    )
    scripts_path_import = (
        "sys.path.insert" in source_text
        and re.search(r"(?m)^\s*import\s+lab(?:\s+as\s+lab)?\s*$", source_text)
        and re.search(r"REPO_ROOT\s*/\s*[\"']scripts[\"']", source_text)
    )
    if not direct_import and not scripts_path_import:
        errors.append("Public notebook must import lab from the repository's scripts directory.")
    for call in ("lab.check_runtime", "lab.prepare", "lab.compare_samples"):
        if call not in source_text:
            errors.append(f"Public notebook does not call {call}.")
    errors.extend(_synced_code_errors(notebook))
    return CheckResult("Public notebook", tuple(errors))


def check_python_sources(scaffold: bool) -> CheckResult:
    errors: list[str] = []
    roots = (ROOT / "scripts", ROOT / "tools")
    for source_root in roots:
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (OSError, SyntaxError, UnicodeError) as exc:
                errors.append(f"{path.relative_to(ROOT)} does not compile: {exc}")
    return CheckResult("Python syntax", tuple(errors))


def check_revision_pins(scaffold: bool) -> CheckResult:
    errors: list[str] = []
    expected = {
        ROOT / "scripts" / "lab.py": "c17667641642c4265ff2b7037dd101b7d4567634",
        ROOT / "scripts" / "prepare_cpp.py": "d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14",
    }
    for path, revision in expected.items():
        if not path.is_file():
            errors.append(f"Pinned source file is missing: {path.relative_to(ROOT)}.")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Could not read {path.relative_to(ROOT)}: {exc}")
            continue
        if revision not in source:
            errors.append(f"{path.relative_to(ROOT)} does not contain its required revision pin.")
    return CheckResult("Model and upstream revision pins", tuple(errors))


def check_evidence(scaffold: bool) -> CheckResult:
    errors: list[str] = []
    try:
        record = json.loads(RESULTS.read_text(encoding="utf-8"))
        verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return CheckResult("Recorded evidence", (f"Could not read the result record: {exc}",))

    runs = record.get("runs")
    if record.get("status") != "complete" or not isinstance(runs, list):
        return CheckResult("Recorded evidence", ("The historical result record is not complete.",))
    expected = {(1519, 0), (1519, 8), (1607, 0), (1607, 8)}
    observed: set[tuple[int, int]] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"Recorded run {index} is not an object.")
            continue
        try:
            observed.add((int(run["sample_id"]), int(run["gpu_layers"])))
        except (KeyError, TypeError, ValueError):
            errors.append(f"Recorded run {index} has no sample ID or GPU-layer count.")
        if run.get("returncode") != 0 or not isinstance(run.get("transcript"), str) or not run["transcript"].strip():
            errors.append(f"Recorded run {index} has no successful transcript.")
        budget = run.get("gpu_budget")
        if run.get("gpu_layers") == 0:
            if not isinstance(budget, dict):
                errors.append(f"Recorded run {index} is missing its T4 memory-reservation details.")
            elif budget.get("free_bytes_before_native") != 4294770688:
                errors.append(f"Recorded run {index} has an unexpected free-memory value.")
            elif "not a test on a physical 4GB GPU" not in budget.get("method", ""):
                errors.append(f"Recorded run {index} does not label the 4 GB GPU limitation.")
    if observed != expected or len(runs) != 4:
        errors.append(f"Expected four runs for two clips at 0 and 8 GPU layers; found {sorted(observed)}.")
    if verification.get("status") != "passed" or verification.get("run_count") != 4:
        errors.append("The saved verification summary does not match the four recorded runs.")
    return CheckResult("Recorded evidence", tuple(errors))


def check_source_map(scaffold: bool) -> CheckResult:
    errors: list[str] = []
    try:
        manifest = json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return CheckResult("Provenance map", (f"Could not read source-map.json: {exc}",))

    files = manifest.get("files")
    if not isinstance(files, list):
        return CheckResult("Provenance map", ("source-map.json must contain a files list.",))
    for item in files:
        if not isinstance(item, dict):
            errors.append("source-map.json contains a non-object entry.")
            continue
        relative = Path(str(item.get("destination", "")))
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            errors.append(f"Mapped file is missing or unsafe: {relative}.")
            continue
        if item.get("mutable_after_adaptation"):
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != item.get("source_sha256"):
            errors.append(f"Immutable evidence hash changed: {relative}.")
        if target.stat().st_size != item.get("source_bytes"):
            errors.append(f"Immutable evidence size changed: {relative}.")
    return CheckResult("Provenance map", tuple(errors))


def _markdown_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    links = [match.group(1) for match in MARKDOWN_LINK.finditer(text)]
    links.extend(match.group(1) for match in HTML_REFERENCE.finditer(text))
    return links


def check_local_links(scaffold: bool) -> CheckResult:
    errors: list[str] = []
    markdown_files = sorted(ROOT.rglob("*.md"))
    for markdown in markdown_files:
        if any(part in {".git", ".venv", "node_modules"} for part in markdown.parts):
            continue
        try:
            links = _markdown_targets(markdown)
        except (OSError, UnicodeError) as exc:
            errors.append(f"Could not read {markdown.relative_to(ROOT)}: {exc}")
            continue
        for value in links:
            target = value.strip().split("#", 1)[0].split("?", 1)[0]
            if not target or target.startswith("#"):
                continue
            if urlsplit(target).scheme or target.startswith("//"):
                continue
            decoded = unquote(target)
            candidate = (markdown.parent / decoded).resolve()
            if not candidate.is_relative_to(ROOT):
                errors.append(f"Local link leaves the repository: {markdown.relative_to(ROOT)} -> {value}.")
            elif not candidate.exists():
                errors.append(f"Broken local link: {markdown.relative_to(ROOT)} -> {value}.")
    return CheckResult("Local links", tuple(errors))


def check_public_payload(scaffold: bool) -> CheckResult:
    errors: list[str] = []
    ignored_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".yomiyasu", ".yomiyasu-final", "node_modules"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_dirs for part in path.parts):
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"Model weight or archive file is present: {path.relative_to(ROOT)}.")
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"File exceeds 50 MiB: {path.relative_to(ROOT)}.")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Could not scan {path.relative_to(ROOT)} for credentials: {exc}")
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"Possible {label} found in {path.relative_to(ROOT)}.")
    return CheckResult("Public payload", tuple(errors))


def check_history_notebook(scaffold: bool) -> CheckResult:
    if not HISTORY.is_file():
        return CheckResult("Historical CLI capture", ("The captured CLI notebook is missing.",))
    notebook, errors = _read_notebook(HISTORY)
    if errors or notebook is None:
        return CheckResult("Historical CLI capture", tuple(errors))
    cells = notebook.get("cells", [])
    outputs = sum(
        len(cell.get("outputs", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code" and isinstance(cell.get("outputs", []), list)
    )
    if outputs == 0:
        errors.append("Historical CLI capture has no stored command outputs.")
    return CheckResult("Historical CLI capture", tuple(errors))


def _run_checks(scaffold: bool) -> int:
    checks: tuple[Callable[[bool], CheckResult], ...] = (
        check_notebook,
        check_python_sources,
        check_revision_pins,
        check_evidence,
        check_source_map,
        check_history_notebook,
        check_local_links,
        check_public_payload,
    )
    failed = False
    for check in checks:
        result = check(scaffold)
        if result.skipped:
            print(f"SKIP {result.name}: {result.errors[0]}")
        elif result.errors:
            failed = True
            for error in result.errors:
                print(f"FAIL {result.name}: {error}")
        else:
            print(f"PASS {result.name}")
    if failed:
        return 1
    print("Repository checks passed. These checks do not run Colab or model inference.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Allow the notebook and scripts/lab.py to be absent while preparing the repository.",
    )
    args = parser.parse_args()
    return _run_checks(args.scaffold)


if __name__ == "__main__":
    raise SystemExit(main())
