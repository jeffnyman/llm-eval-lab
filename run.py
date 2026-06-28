"""
run.py — Entry point for Phase 1 + 2.

Routes a prompt to all configured backends, prints a comparison,
and logs every result to logs/runs.jsonl.

Usage:
    uv run python run.py
    uv run python run.py --prompt "Your custom prompt here"
    uv run python run.py --config path/to/config.yaml
    uv run python run.py --no-log
"""

import argparse

from llm_eval_lab.router import route, ModelResult
from llm_eval_lab.logger import log_results, make_run_id

DEFAULT_PROMPT = (
  "Explain the difference between a unit test and an integration test. "
  "Give one concrete example of each."
)

COLORS = {
  "anthropic": "\033[33m",  # orange
  "openai": "\033[32m",  # green
  "ollama": "\033[36m",  # cyan
  "reset": "\033[0m",
  "dim": "\033[2m",
  "bold": "\033[1m",
  "red": "\033[31m",
}


def c(key: str) -> str:
  return COLORS.get(key, "")


def hr(char: str = "─", width: int = 72) -> str:
  return char * width


def print_result(result: ModelResult) -> None:
  col = c(result.provider)
  reset = c("reset")
  dim = c("dim")
  bold = c("bold")
  red = c("red")

  print(hr())
  print(f"{bold}{col}{result.model_id}{reset}  {dim}{result.model_name}{reset}")
  print()

  if result.error:
    print(f"{red}ERROR: {result.error}{reset}")
  else:
    print(result.output)

  print()
  token_info = ""

  if result.input_tokens is not None:
    token_info = f"  •  {result.input_tokens} in / {result.output_tokens} out tokens"

  print(f"{dim}{result.latency_ms:.0f} ms{token_info}{reset}")


def print_summary(results: list[ModelResult]) -> None:
  print()
  print(hr("═"))
  print(f"{c('bold')}Summary{c('reset')}")
  print(hr("═"))

  fmt = "{:<12} {:<12} {:>10} {:>10} {:>10} {:>8}"

  print(fmt.format("Model", "Provider", "Latency", "In tok", "Out tok", "Status"))
  print(hr())

  for r in results:
    status = "✓ ok" if not r.error else "✗ error"

    print(
      fmt.format(
        r.model_id,
        r.provider,
        f"{r.latency_ms:.0f} ms",
        str(r.input_tokens or "—"),
        str(r.output_tokens or "—"),
        status,
      )
    )
  print()


def main():
  parser = argparse.ArgumentParser(
    description="Route a prompt to multiple LLM backends."
  )

  parser.add_argument(
    "--prompt", default=DEFAULT_PROMPT, help="Prompt to send to all models"
  )

  parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

  parser.add_argument(
    "--no-log", action="store_true", help="Skip writing to the run log"
  )

  args = parser.parse_args()

  run_id = make_run_id()

  print()
  print(hr("═"))
  print(f"{c('bold')}LLM Eval Lab — Phase 2: Routing + Observability{c('reset')}")
  print(hr("═"))
  print(f"\n{c('dim')}Prompt:{c('reset')} {args.prompt}\n")
  print(f"{c('dim')}Calling models...{c('reset')}\n")

  results = route(args.prompt, config_path=args.config)

  for result in results:
    print_result(result)

  print_summary(results)

  if not args.no_log:
    log_path = log_results(results, run_id)

    print(
      f"{c('dim')}Run {c('bold')}{run_id}{c('reset')} {c('dim')}logged → {log_path}{c('reset')}"
    )
    print(f"{c('dim')}Inspect: uv run python show_logs.py --run {run_id}{c('reset')}\n")


if __name__ == "__main__":
  main()
