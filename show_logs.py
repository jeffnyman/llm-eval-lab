"""
show_logs.py — Browse and inspect past runs from the JSONL log.

Usage:
    uv run python show_logs.py               # list all runs
    uv run python show_logs.py --run abc123  # show full output for one run
    uv run python show_logs.py --tail 5      # show last N runs
    uv run python show_logs.py --stats       # latency summary per model
"""

import argparse
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from llm_eval_lab.logger import read_log, read_run

console = Console()


def group_by_run(records: list[dict]) -> dict[str, list[dict]]:
  """Group records by run_id, preserving insertion order."""
  groups: dict[str, list[dict]] = {}
  for r in records:
    groups.setdefault(r["run_id"], []).append(r)
  return groups


def render_run_summary_table(runs: dict[str, list[dict]]) -> Table:
  table = Table(
    box=box.SIMPLE_HEAD,
    show_header=True,
    header_style="bold dim",
    title="[bold]Logged runs[/]",
    title_justify="left",
  )
  table.add_column("Run ID", style="bold cyan")
  table.add_column("Timestamp", style="dim")
  table.add_column("Models", style="dim")
  table.add_column("Prompt (truncated)")
  table.add_column("Errors", justify="center")

  for run_id, records in runs.items():
    timestamp = records[0]["timestamp"][:19].replace("T", " ")
    models = ", ".join(r["model_id"] for r in records)
    prompt = records[0]["prompt"]
    truncated = prompt[:60] + "…" if len(prompt) > 60 else prompt
    errors = sum(1 for r in records if r.get("error"))
    error_cell = f"[red]{errors}[/]" if errors else "[dim]—[/]"
    table.add_row(run_id, timestamp, models, truncated, error_cell)

  return table


def render_run_detail(records: list[dict]) -> None:
  run_id = records[0]["run_id"]
  prompt = records[0]["prompt"]

  console.print()
  console.rule(f"[bold]Run {run_id}[/]")
  console.print(
    Panel(prompt, title="[dim]Prompt[/]", border_style="dim", padding=(0, 2))
  )
  console.print()

  color_map = {"anthropic": "orange3", "openai": "green3", "ollama": "cyan"}

  for r in records:
    color = color_map.get(r["provider"], "white")

    if r.get("error"):
      body = Text(f"ERROR: {r['error']}", style="bold red")
    else:
      body = Text(r["output"] or "[no output]")

    token_info = ""
    if r.get("input_tokens") is not None:
      token_info = f"  •  {r['input_tokens']} in / {r['output_tokens']} out tokens"

    subtitle = f"{r['latency_ms']:.0f} ms{token_info}  •  {r['timestamp'][:19].replace('T', ' ')} UTC"

    console.print(
      Panel(
        body,
        title=f"[bold {color}]{r['model_id']}[/]  [dim]{r['model_name']}[/]",
        subtitle=f"[dim]{subtitle}[/]",
        border_style=color,
        padding=(1, 2),
      )
    )
    console.print()


def render_stats(records: list[dict]) -> None:
  """Per-model latency and token stats across all runs."""
  model_data: dict[str, list[dict]] = defaultdict(list)

  for r in records:
    model_data[r["model_id"]].append(r)

  table = Table(
    box=box.SIMPLE_HEAD,
    show_header=True,
    header_style="bold dim",
    title="[bold]Per-model stats (all runs)[/]",
    title_justify="left",
  )
  table.add_column("Model", style="bold")
  table.add_column("Provider", style="dim")
  table.add_column("Calls", justify="right")
  table.add_column("Errors", justify="right")
  table.add_column("Avg latency", justify="right")
  table.add_column("Min latency", justify="right")
  table.add_column("Max latency", justify="right")
  table.add_column("Avg out tokens", justify="right", style="dim")

  for model_id, rows in sorted(model_data.items()):
    provider = rows[0]["provider"]
    calls = len(rows)
    errors = sum(1 for r in rows if r.get("error"))
    latencies = [r["latency_ms"] for r in rows if not r.get("error")]
    out_tokens = [r["output_tokens"] for r in rows if r.get("output_tokens")]

    avg_lat = f"{sum(latencies) / len(latencies):.0f} ms" if latencies else "—"
    min_lat = f"{min(latencies):.0f} ms" if latencies else "—"
    max_lat = f"{max(latencies):.0f} ms" if latencies else "—"
    avg_tok = f"{sum(out_tokens) / len(out_tokens):.0f}" if out_tokens else "—"

    error_cell = f"[red]{errors}[/]" if errors else "[dim]0[/]"

    table.add_row(
      model_id, provider, str(calls), error_cell, avg_lat, min_lat, max_lat, avg_tok
    )

  console.print()
  console.print(table)
  console.print()


def main():
  parser = argparse.ArgumentParser(description="Inspect the LLM Eval Lab run log.")

  parser.add_argument(
    "--run", metavar="RUN_ID", help="Show full output for a specific run"
  )

  parser.add_argument("--tail", metavar="N", type=int, help="Show last N runs only")

  parser.add_argument(
    "--stats", action="store_true", help="Show per-model latency stats"
  )

  parser.add_argument("--log", default=None, help="Path to JSONL log file")

  args = parser.parse_args()

  log_path = Path(args.log) if args.log else None
  records = read_log(log_path)

  if not records:
    console.print(
      "\n[dim]No log entries found. Run [bold]uv run python run.py[/] first.[/]\n"
    )
    return

  if args.run:
    run_records = read_run(args.run, log_path)

    if not run_records:
      console.print(f"\n[red]No records found for run_id: {args.run}[/]\n")
      return

    render_run_detail(run_records)
    return

  if args.stats:
    render_stats(records)
    return

  # Default: list all (or tail N) runs
  runs = group_by_run(records)

  if args.tail:
    run_items = list(runs.items())
    runs = dict(run_items[-args.tail :])

  console.print()
  console.print(render_run_summary_table(runs))
  console.print(
    f"[dim]  Total records: {len(records)}  •  "
    f"Log: {log_path or 'logs/runs.jsonl'}[/]\n"
  )


if __name__ == "__main__":
  main()
