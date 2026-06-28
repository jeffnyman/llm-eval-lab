"""
logger.py — Phase 2: Structured observability for every model call.

Appends one JSON record per ModelResult to a JSONL file in logs/.
All calls from the same run share a run_id so you can group them.

Log location defaults to logs/runs.jsonl — override via LOG_PATH in .env.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_eval_lab.router import ModelResult

DEFAULT_LOG_PATH = Path("logs/runs.jsonl")


def _get_log_path() -> Path:
  raw = os.getenv("LOG_PATH", str(DEFAULT_LOG_PATH))
  return Path(raw)


def make_run_id() -> str:
  """Short, readable run identifier — first 8 chars of a UUID4."""
  return str(uuid.uuid4())[:8]


def result_to_record(result: ModelResult, run_id: str) -> dict:
  """
  Serialise a ModelResult into a flat dict suitable for JSONL.

  Keeping it flat (rather than nested) makes it easier to load into
  pandas or query with jq without extra unpacking.
  """
  return {
    "run_id": run_id,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_id": result.model_id,
    "model_name": result.model_name,
    "provider": result.provider,
    "prompt": result.prompt,
    "output": result.output,
    "latency_ms": result.latency_ms,
    "input_tokens": result.input_tokens,
    "output_tokens": result.output_tokens,
    "error": result.error,
  }


def log_results(
  results: list[ModelResult],
  run_id: str,
  log_path: Path | None = None,
) -> Path:
  """
  Append all results from a single run to the JSONL log.

  Creates the logs/ directory if it doesn't exist.
  Returns the path written to, so callers can report it.
  """
  path = log_path or _get_log_path()
  path.parent.mkdir(parents=True, exist_ok=True)

  with path.open("a", encoding="utf-8") as f:
    for result in results:
      record = result_to_record(result, run_id)
      f.write(json.dumps(record) + "\n")

  return path


def read_log(log_path: Path | None = None) -> list[dict]:
  """
  Read all records from the JSONL log into a list of dicts.
  Returns an empty list if the log doesn't exist yet.
  """
  path = log_path or _get_log_path()

  if not path.exists():
    return []

  records = []

  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()

      if line:
        records.append(json.loads(line))

  return records


def read_run(run_id: str, log_path: Path | None = None) -> list[dict]:
  """Return all records for a specific run_id."""
  return [r for r in read_log(log_path) if r["run_id"] == run_id]
