#!/usr/bin/env python3
"""Reproducible, opt-in llama-bench runner for llama-Gears.

Dry-run by default. Does not download models, access network, use a shell,
modify an installed engine or replace the Qwen Chat runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")


def parse_profile(raw: str) -> dict[str, int | str]:
    """LABEL:N_GPU_LAYERS:N_CPU_MOE_LAYERS."""
    parts = raw.split(":")
    if len(parts) != 3 or not LABEL_PATTERN.fullmatch(parts[0]):
        raise argparse.ArgumentTypeError("profile must be LABEL:NGL:N_CPU_MOE")
    try:
        ngl, cpu_moe = map(int, parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("offload layer counts must be integers") from exc
    if not 0 <= ngl <= 999 or not 0 <= cpu_moe <= 999:
        raise argparse.ArgumentTypeError("offload layers must be between 0 and 999")
    return {"label": parts[0], "n_gpu_layers": ngl, "n_cpu_moe": cpu_moe}


def file_manifest(paths: list[Path], hash_files: bool) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"model GGUF shard does not exist: {path}")
        item: dict[str, Any] = {
            "path": str(path.resolve()), "size_bytes": path.stat().st_size
        }
        if hash_files:
            digest = hashlib.sha256()
            with path.open("rb") as reader:
                for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            item["sha256"] = digest.hexdigest()
        result.append(item)
    return result


def build_command(bench: Path, model: Path, profile: dict[str, Any],
                  prompt: int, generation: int, reps: int,
                  threads: int | None, context_depth: int) -> list[str]:
    args = [str(bench), "-m", str(model), "-ngl", str(profile["n_gpu_layers"]),
            "-ncmoe", str(profile["n_cpu_moe"]), "-p", str(prompt), "-n",
            str(generation), "-r", str(reps), "-d", str(context_depth),
            "-o", "json"]
    if threads is not None:
        args.extend(["-t", str(threads)])
    return args


def validate_results(rows: Any, expected: dict[str, Any],
                     prompt: int, generation: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("llama-bench returned no result rows")
    seen_pp = False
    seen_tg = False
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("llama-bench output has a non-object row")
        if row.get("n_gpu_layers") != expected["n_gpu_layers"]:
            raise ValueError("benchmark output does not match requested GPU offload")
        if row.get("n_cpu_moe") != expected["n_cpu_moe"]:
            raise ValueError("benchmark output does not match requested CPU expert offload")
        rate = row.get("avg_ts")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError("benchmark output is missing valid avg_ts")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("benchmark avg_ts must be finite and positive")
        seen_pp |= row.get("n_prompt") == prompt and row.get("n_gen") == 0
        seen_tg |= row.get("n_prompt") == 0 and row.get("n_gen") == generation
        parsed.append(row)
    if not seen_pp or not seen_tg:
        raise ValueError("benchmark must include requested prefill and decode tests")
    return parsed


def run_one(command: list[str], profile: dict[str, Any], timeout: int) -> dict[str, Any]:
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                             check=False, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"profile": profile, "status": "error", "error": str(exc)}
    if run.returncode:
        return {"profile": profile, "status": "error",
                "returncode": run.returncode, "stderr_tail": run.stderr[-1500:]}
    try:
        prompt = int(command[command.index("-p") + 1])
        generation = int(command[command.index("-n") + 1])
        rows = validate_results(json.loads(run.stdout), profile, prompt, generation)
    except ValueError as exc:
        return {"profile": profile, "status": "invalid-output", "error": str(exc),
                "stdout_tail": run.stdout[-1200:]}
    return {"profile": profile, "status": "ok", "rows": rows,
            "stderr_tail": run.stderr[-900:] if run.stderr else ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True,
                        help="Existing llama-bench executable")
    parser.add_argument("--model", type=Path, required=True, help="First GGUF shard")
    parser.add_argument("--shard", type=Path, action="append", default=[],
                        help="Extra model shard for provenance; repeat as needed")
    parser.add_argument("--profile", type=parse_profile, action="append", required=True,
                        help="LABEL:N_GPU_LAYERS:N_CPU_MOE_LAYERS, repeat per profile")
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generation", type=int, default=128)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--hash", action="store_true", help="SHA256 explicitly listed shards")
    parser.add_argument("--execute", action="store_true", help="Run actual inference tests")
    parser.add_argument("--report", type=Path, help="Write JSON (fail if file already exists)")
    args = parser.parse_args(argv)
    if min(args.prompt, args.generation, args.reps, args.timeout) <= 0:
        parser.error("prompt, generation, repetitions and timeout must be positive")
    if args.depth < 0 or args.threads is not None and args.threads <= 0:
        parser.error("depth must be nonnegative; threads must be positive")
    if len({p["label"] for p in args.profile}) != len(args.profile):
        parser.error("profile labels must be distinct")
    if not args.model.is_file():
        parser.error("model file does not exist")
    if args.execute and not args.bench.is_file():
        parser.error("benchmark executable does not exist")
    try:
        manifest = file_manifest([args.model, *args.shard], args.hash)
    except ValueError as exc:
        parser.error(str(exc))
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "executed" if args.execute else "dry-run",
        "platform": platform.platform(),
        "model_shards": manifest,
        "benchmark_executable": str(args.bench),
        "commands": [], "results": [],
        "caution": "llama-bench excludes TTFT, peak VRAM and semantic correctness."
    }
    failures = 0
    for profile in args.profile:
        command = build_command(args.bench, args.model, profile, args.prompt,
                                args.generation, args.reps, args.threads, args.depth)
        report["commands"].append({"profile": profile, "argv": command})
        if args.execute:
            result = run_one(command, profile, args.timeout)
            report["results"].append(result)
            failures += result["status"] != "ok"
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        try:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        except FileExistsError:
            parser.error("report exists; refusing to overwrite")
    else:
        print(rendered)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
