#!/usr/bin/env python3
"""Verify that the dated Colab notebook run has reproducible delivery proof."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("evidence/notebook-validation-2026-10-30")
PUBLIC_NOTEBOOK = Path("notebooks/Nemotron_T4_Colab.ipynb")
RESULT_NOTEBOOK = Path("notebooks/results/T4_2026-10-30.ipynb")
RAW_NOTEBOOK = EVIDENCE / "raw-executed.ipynb"
RAW_SOURCE_NOTEBOOK = EVIDENCE / "raw-source.ipynb"
EXECUTION_LOG = EVIDENCE / "execution.log"
REMOTE_FACTS = EVIDENCE / "notebook-validation.json"
PROVENANCE = EVIDENCE / "execution-provenance.json"

SOURCE_COMMIT = "33a38c8faaed84eb8a4c781f35327d2c21a9e2f2"
NOTEBOOK_SOURCE_REVISION = "d16e3035b9fe5f278cfee8291fe717666e611f37"
NOTEBOOK_SOURCE_SHA256 = "1378f1c6c21de274e8b45cab1950c8d5339275461316964efd4295ad19087947"
MODEL_REVISION = "c17667641642c4265ff2b7037dd101b7d4567634"
NEMO_REVISION = "d8a1abf2ea9d11e8f98ad0e10cc2c12ef1cb3a14"
EXPECTED_CODE_CELLS = 9
EXPECTED_DECODER_BYTES = 20_352_806_848
EXPECTED_ENCODER_BYTES = 1_670_410_080
EXPECTED_TENSOR_TYPES = {"Q4_0": 48}
SOURCE_FILES = {
    "lab.py": Path("scripts/lab.py"),
    "bootstrap.py": Path("scripts/bootstrap.py"),
    "prepare_cpp.py": Path("scripts/prepare_cpp.py"),
    "convert_q4_profile.py": Path("scripts/convert_q4_profile.py"),
}
SAMPLES = {
    "fleurs-ja-01.wav": Path("audio/fleurs-ja-01.wav"),
    "fleurs-ja-02.wav": Path("audio/fleurs-ja-02.wav"),
}
OUTPUT_PATHS = {
    "source_notebook": RAW_SOURCE_NOTEBOOK,
    "notebook": RESULT_NOTEBOOK,
    "raw_notebook": RAW_NOTEBOOK,
    "execution_log": EXECUTION_LOG,
}
ANSI_ESCAPE = re.compile(
    r"\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))"
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class NotebookArtifact:
    path: Path
    code_sources: tuple[str, ...]
    code_outputs: tuple[tuple[dict[str, Any], ...], ...]


def _inside_root(root: Path, relative: str | Path) -> Path:
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_source_sha256(path: Path) -> str:
    source = path.read_bytes().decode("utf-8")
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return "".join(value)
    return None


def _load_json(root: Path, relative: Path, label: str, errors: list[str]) -> object | None:
    path = _inside_root(root, relative)
    if not path.is_file():
        errors.append(f"Missing {label}: {relative.as_posix()}.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read {label} {relative.as_posix()}: {exc}.")
        return None


def _schema_errors(document: object, relative: Path) -> list[str]:
    label = relative.as_posix()
    if not isinstance(document, dict):
        return [f"{label}: notebook root must be an object."]

    errors: list[str] = []
    if document.get("nbformat") != 4:
        errors.append(f"{label}: nbformat must be 4.")
    minor = document.get("nbformat_minor")
    if not isinstance(minor, int) or isinstance(minor, bool) or minor < 0:
        errors.append(f"{label}: nbformat_minor must be a non-negative integer.")
    if not isinstance(document.get("metadata"), dict):
        errors.append(f"{label}: notebook metadata must be an object.")
    cells = document.get("cells")
    if not isinstance(cells, list):
        errors.append(f"{label}: cells must be a list.")
        return errors

    cell_ids: set[str] = set()
    for index, cell in enumerate(cells):
        prefix = f"{label}: cell {index}"
        if not isinstance(cell, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            errors.append(f"{prefix} has an invalid cell_type.")
        if not isinstance(cell.get("metadata"), dict):
            errors.append(f"{prefix} metadata must be an object.")
        if _text(cell.get("source")) is None:
            errors.append(f"{prefix} source must be a string or a list of strings.")
        if isinstance(minor, int) and minor >= 5:
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not cell_id or len(cell_id) > 64:
                errors.append(f"{prefix} needs a non-empty id of at most 64 characters.")
            elif cell_id in cell_ids:
                errors.append(f"{prefix} repeats cell id {cell_id!r}.")
            else:
                cell_ids.add(cell_id)

        if cell_type != "code":
            continue
        execution_count = cell.get("execution_count")
        if execution_count is not None and (
            not isinstance(execution_count, int) or isinstance(execution_count, bool)
        ):
            errors.append(f"{prefix} execution_count must be an integer or null.")
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            errors.append(f"{prefix} outputs must be a list.")
            continue
        for output_index, output in enumerate(outputs):
            output_prefix = f"{prefix} output {output_index}"
            if not isinstance(output, dict):
                errors.append(f"{output_prefix} must be an object.")
                continue
            output_type = output.get("output_type")
            if output_type == "stream":
                if output.get("name") not in {"stdout", "stderr"}:
                    errors.append(f"{output_prefix} has an invalid stream name.")
                if _text(output.get("text")) is None:
                    errors.append(f"{output_prefix} stream text must be a string or string list.")
            elif output_type in {"display_data", "execute_result"}:
                if not isinstance(output.get("data"), dict):
                    errors.append(f"{output_prefix} data must be an object.")
                if not isinstance(output.get("metadata"), dict):
                    errors.append(f"{output_prefix} metadata must be an object.")
                if output_type == "execute_result":
                    count = output.get("execution_count")
                    if count is not None and (
                        not isinstance(count, int) or isinstance(count, bool)
                    ):
                        errors.append(f"{output_prefix} execution_count must be an integer or null.")
            elif output_type == "error":
                if not all(isinstance(output.get(key), str) for key in ("ename", "evalue")):
                    errors.append(f"{output_prefix} needs string ename and evalue fields.")
                if not isinstance(output.get("traceback"), list) or not all(
                    isinstance(line, str) for line in output.get("traceback", [])
                ):
                    errors.append(f"{output_prefix} traceback must be a list of strings.")
            else:
                errors.append(f"{output_prefix} has an invalid output_type.")
    try:
        import nbformat
    except ImportError:
        errors.append(f"{label}: install nbformat to validate the notebook schema.")
    else:
        try:
            nbformat.validate(nbformat.from_dict(document))
        except Exception as exc:
            errors.append(f"{label}: nbformat schema validation failed: {exc}.")
    return errors


def _load_notebook(
    root: Path, relative: Path, label: str, errors: list[str]
) -> NotebookArtifact | None:
    document = _load_json(root, relative, label, errors)
    if document is None:
        return None
    errors.extend(_schema_errors(document, relative))
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        return None

    code_sources: list[str] = []
    code_outputs: list[tuple[dict[str, Any], ...]] = []
    for index, cell in enumerate(document["cells"]):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = _text(cell.get("source"))
        if source is None:
            source = ""
        code_sources.append(source.replace("\r\n", "\n").replace("\r", "\n"))
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            outputs = []
        well_formed_outputs = tuple(output for output in outputs if isinstance(output, dict))
        code_outputs.append(well_formed_outputs)

    return NotebookArtifact(relative, tuple(code_sources), tuple(code_outputs))


def _verify_notebook_execution(
    artifacts: dict[str, NotebookArtifact | None], errors: list[str]
) -> None:
    names = ("source_notebook", "raw_notebook", "notebook", "public_notebook")
    present = [artifacts[name] for name in names if artifacts.get(name) is not None]
    for artifact in present:
        if len(artifact.code_sources) != EXPECTED_CODE_CELLS:
            errors.append(
                f"{artifact.path.as_posix()}: expected {EXPECTED_CODE_CELLS} code cells, "
                f"found {len(artifact.code_sources)}."
            )
    if len(present) == len(names):
        expected = present[0].code_sources
        for artifact in present[1:]:
            if artifact.code_sources != expected:
                errors.append(
                    f"Code-cell source sequence differs between {present[0].path.as_posix()} "
                    f"and {artifact.path.as_posix()}."
                )

    for name in ("raw_notebook", "notebook"):
        artifact = artifacts.get(name)
        if artifact is None:
            continue
        for index, outputs in enumerate(artifact.code_outputs, start=1):
            if not outputs:
                errors.append(
                    f"{artifact.path.as_posix()}: code cell {index}/{EXPECTED_CODE_CELLS} "
                    "has no saved output."
                )
            for output in outputs:
                if output.get("output_type") == "error":
                    errors.append(
                        f"{artifact.path.as_posix()}: code cell {index}/{EXPECTED_CODE_CELLS} "
                        f"saved an execution error ({output.get('ename', 'unknown')})."
                    )


def _verify_provenance(root: Path, errors: list[str]) -> None:
    provenance = _load_json(root, PROVENANCE, "execution provenance", errors)
    if not isinstance(provenance, dict):
        return
    if provenance.get("status") != "complete":
        errors.append("execution-provenance.json status must be 'complete'.")
    expected_fields = {
        "source_commit": SOURCE_COMMIT,
        "notebook_source_revision": NOTEBOOK_SOURCE_REVISION,
        "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
        "code_cells": EXPECTED_CODE_CELLS,
    }
    for key, expected in expected_fields.items():
        if provenance.get(key) != expected:
            errors.append(f"execution-provenance.json {key} must equal {expected!r}.")

    actual_outputs = provenance.get("actual_outputs")
    if not isinstance(actual_outputs, dict):
        errors.append("execution-provenance.json actual_outputs must be an object.")
        return
    for name, expected_path in OUTPUT_PATHS.items():
        entry = actual_outputs.get(name)
        if not isinstance(entry, dict):
            errors.append(f"execution-provenance.json actual_outputs.{name} must be an object.")
            continue
        recorded_path = entry.get("path")
        if recorded_path != expected_path.as_posix():
            errors.append(
                f"execution-provenance.json actual_outputs.{name}.path must be "
                f"{expected_path.as_posix()!r}."
            )
        recorded_hash = entry.get("sha256")
        if not isinstance(recorded_hash, str) or not HEX_SHA256.fullmatch(recorded_hash):
            errors.append(f"execution-provenance.json actual_outputs.{name}.sha256 is invalid.")
            continue
        try:
            path = _inside_root(root, expected_path)
            if not path.is_file():
                errors.append(f"Missing recorded output: {expected_path.as_posix()}.")
            elif _sha256(path) != recorded_hash:
                errors.append(f"SHA256 mismatch for {expected_path.as_posix()}.")
            if name == "source_notebook" and recorded_hash != NOTEBOOK_SOURCE_SHA256:
                errors.append("The raw source notebook hash is not the pinned original hash.")
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"Cannot verify {expected_path.as_posix()}: {exc}.")


def _verify_remote_facts(root: Path, errors: list[str]) -> None:
    facts = _load_json(root, REMOTE_FACTS, "Colab notebook validation facts", errors)
    if not isinstance(facts, dict):
        return
    if facts.get("status") != "complete":
        errors.append("notebook-validation.json status must be 'complete'.")
    if facts.get("source_commit") != SOURCE_COMMIT:
        errors.append(f"notebook-validation.json source_commit must be {SOURCE_COMMIT}.")
    if facts.get("model_revision") != MODEL_REVISION:
        errors.append(f"notebook-validation.json model_revision must be {MODEL_REVISION}.")
    if facts.get("nemo_revision") != NEMO_REVISION:
        errors.append(f"notebook-validation.json nemo_revision must be {NEMO_REVISION}.")
    if facts.get("gguf_expert_tensor_types") != EXPECTED_TENSOR_TYPES:
        errors.append("notebook-validation.json must report exactly 48 Q4_0 expert tensors.")
    if facts.get("decoder_bytes") != EXPECTED_DECODER_BYTES:
        errors.append(f"notebook-validation.json decoder_bytes must be {EXPECTED_DECODER_BYTES}.")

    encoder_files = facts.get("encoder_files")
    if not isinstance(encoder_files, dict) or not encoder_files:
        errors.append("notebook-validation.json encoder_files must be a non-empty object.")
    else:
        sizes = list(encoder_files.values())
        if not all(isinstance(size, int) and not isinstance(size, bool) for size in sizes):
            errors.append("notebook-validation.json encoder_files values must be byte counts.")
        elif sum(sizes) != EXPECTED_ENCODER_BYTES:
            errors.append(
                f"notebook-validation.json encoder file bytes must sum to {EXPECTED_ENCODER_BYTES}."
            )

    packages = facts.get("packages")
    if not isinstance(packages, (dict, list)) or not packages:
        errors.append("notebook-validation.json packages must be a non-empty object or list.")

    source_hashes = facts.get("source_files_sha256")
    if not isinstance(source_hashes, dict):
        errors.append("notebook-validation.json source_files_sha256 must be an object.")
    else:
        by_basename = {Path(str(name)).name: digest for name, digest in source_hashes.items()}
        for basename, relative in SOURCE_FILES.items():
            remote_hash = by_basename.get(basename)
            if not isinstance(remote_hash, str) or not HEX_SHA256.fullmatch(remote_hash):
                errors.append(f"Missing valid source hash for {basename} in remote facts.")
                continue
            try:
                path = _inside_root(root, relative)
                if not path.is_file():
                    errors.append(f"Missing public source file: {relative.as_posix()}.")
                elif _normalized_source_sha256(path) != remote_hash:
                    errors.append(
                        f"Remote source hash for {basename} does not match the public script."
                    )
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"Cannot verify public source {relative.as_posix()}: {exc}.")

    runs = facts.get("runs")
    if not isinstance(runs, list) or len(runs) != len(SAMPLES):
        errors.append("notebook-validation.json must contain exactly two completed runs.")
        return
    seen_samples: set[str] = set()
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            errors.append(f"notebook-validation.json run {index} must be an object.")
            continue
        if run.get("status") != "complete":
            errors.append(f"notebook-validation.json run {index} status must be 'complete'.")
        audio_name = Path(str(run.get("audio", ""))).name
        if audio_name not in SAMPLES:
            errors.append(f"notebook-validation.json run {index} has an unexpected audio path.")
            continue
        if audio_name in seen_samples:
            errors.append(f"notebook-validation.json repeats {audio_name}.")
        seen_samples.add(audio_name)
        if run.get("gpu_layers") != 0:
            errors.append(f"notebook-validation.json run {index} must use gpu_layers=0.")
        transcript = run.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            errors.append(f"notebook-validation.json run {index} needs a non-empty transcript.")
        elapsed = run.get("elapsed_seconds_including_model_load")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            errors.append(
                f"notebook-validation.json run {index} needs a positive model-load-inclusive duration."
            )
        recorded_audio_hash = run.get("audio_sha256")
        if not isinstance(recorded_audio_hash, str) or not HEX_SHA256.fullmatch(recorded_audio_hash):
            errors.append(f"notebook-validation.json run {index} has an invalid audio_sha256.")
            continue
        audio_path = _inside_root(root, SAMPLES[audio_name])
        if not audio_path.is_file():
            errors.append(f"Missing repository audio: {SAMPLES[audio_name].as_posix()}.")
        elif _sha256(audio_path) != recorded_audio_hash:
            errors.append(f"Audio SHA256 mismatch for {audio_name}.")
    if seen_samples != set(SAMPLES):
        missing = ", ".join(sorted(set(SAMPLES) - seen_samples))
        errors.append(f"Remote facts are missing required sample run(s): {missing}.")


def _verify_execution_log(root: Path, errors: list[str]) -> None:
    path = _inside_root(root, EXECUTION_LOG)
    if not path.is_file():
        errors.append(f"Missing execution log: {EXECUTION_LOG.as_posix()}.")
        return
    try:
        contents = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        errors.append(f"Cannot read execution log: {exc}.")
        return
    markers = [
        re.search(rf"Executing\s+cell\s+{index}/{EXPECTED_CODE_CELLS}\b", contents)
        for index in range(1, EXPECTED_CODE_CELLS + 1)
    ]
    if any(marker is None for marker in markers):
        errors.append("execution.log must show Colab CLI execution of cells 1/9 through 9/9.")
    elif [marker.start() for marker in markers if marker is not None] != sorted(
        marker.start() for marker in markers if marker is not None
    ):
        errors.append("execution.log cell markers are out of order.")


def verify(root: Path) -> list[str]:
    errors: list[str] = []
    artifacts = {
        "source_notebook": _load_notebook(root, RAW_SOURCE_NOTEBOOK, "raw source notebook", errors),
        "raw_notebook": _load_notebook(root, RAW_NOTEBOOK, "raw executed notebook", errors),
        "notebook": _load_notebook(root, RESULT_NOTEBOOK, "readable result notebook", errors),
        "public_notebook": _load_notebook(root, PUBLIC_NOTEBOOK, "public source notebook", errors),
    }
    _verify_notebook_execution(artifacts, errors)

    raw_source = _inside_root(root, RAW_SOURCE_NOTEBOOK)
    if raw_source.is_file():
        try:
            if _sha256(raw_source) != NOTEBOOK_SOURCE_SHA256:
                errors.append("raw-source.ipynb bytes differ from the pinned original public notebook.")
        except OSError as exc:
            errors.append(f"Cannot hash raw-source.ipynb: {exc}.")

    _verify_remote_facts(root, errors)
    _verify_provenance(root, errors)
    _verify_execution_log(root, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (defaults to the parent of this script's tools directory)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        errors = verify(root)
    except (OSError, UnicodeError, ValueError) as exc:
        errors = [f"Verification could not complete: {exc}."]
    if errors:
        print("Notebook execution proof: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Notebook execution proof: PASS")
    print(f"- {EXPECTED_CODE_CELLS} code cells in the raw-source, raw-executed, result, and public notebooks")
    print("- original source bytes, execution artifacts, log markers, and remote hashes verified")
    print("- two complete T4 CPU-decoder runs match the pinned audio and model facts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
