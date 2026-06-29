"""
eval_run.py — Standalone Phase 3 runner.

Reads the most recent run from the log (or a specific run via --run),
scores each output with DeepEval, and prints results.

Usage:
    uv run python eval_run.py                    # score the latest run
    uv run python eval_run.py --run <run_id>     # score a specific run
    uv run python eval_run.py --prompt "..."     # run a fresh prompt and score it
"""

import argparse
# from collections import defaultdict

from llm_eval_lab.logger import read_log, read_run, log_results, make_run_id
from llm_eval_lab.router import route
from llm_eval_lab.eval import evaluate_many, EvalResult, THRESHOLDS

PASS = "\033[32m✓ pass\033[0m"
FAIL = "\033[31m✗ fail\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
RST = "\033[0m"
HR = "─" * 72


def fmt_score(value: float | None, metric: str) -> str:
  """Format a score with pass/fail indicator."""
  if value is None:
    return f"{DIM}—{RST}"

  # Bias and toxicity: lower is better
  if metric in ("bias", "toxicity"):
    ok = value <= THRESHOLDS[metric]
  else:
    ok = value >= THRESHOLDS[metric]

  color = "\033[32m" if ok else "\033[31m"

  return f"{color}{value:.2f}{RST}"


def print_eval_result(r: EvalResult) -> None:
  status = PASS if r.passed else FAIL

  print(f"\n{HR}")
  print(f"{BOLD}{r.model_id}{RST}  {status}")
  print()

  if r.error:
    print(f"  {DIM}Skipped: {r.error}{RST}")
    return

  print(f"  Answer relevancy : {fmt_score(r.answer_relevancy, 'answer_relevancy')}")

  if r.answer_relevancy_reason:
    print(f"  {DIM}→ {r.answer_relevancy_reason}{RST}")

  print()

  print(f"  Bias             : {fmt_score(r.bias, 'bias')}")

  if r.bias_reason:
    print(f"  {DIM}→ {r.bias_reason}{RST}")

  print()

  print(f"  Toxicity         : {fmt_score(r.toxicity, 'toxicity')}")

  if r.toxicity_reason:
    print(f"  {DIM}→ {r.toxicity_reason}{RST}")


def print_summary(results: list[EvalResult]) -> None:
  print(f"\n{'═' * 72}")
  print(f"{BOLD}Eval Summary{RST}")
  print(f"{'═' * 72}")

  fmt = "{:<14} {:>18} {:>8} {:>10} {:>10}"

  print(fmt.format("Model", "Answer Relevancy", "Bias", "Toxicity", "Result"))
  print(HR)

  for r in results:
    print(
      fmt.format(
        r.model_id,
        fmt_score(r.answer_relevancy, "answer_relevancy")
        if not r.error
        else f"{DIM}skipped{RST}",
        fmt_score(r.bias, "bias") if not r.error else "",
        fmt_score(r.toxicity, "toxicity") if not r.error else "",
        PASS if r.passed else FAIL,
      )
    )
  print()

  passed = sum(1 for r in results if r.passed)

  print(
    f"{DIM}  {passed}/{len(results)} passed  •  "
    f"thresholds: relevancy≥{THRESHOLDS['answer_relevancy']} | "
    f"bias≤{THRESHOLDS['bias']} | "
    f"toxicity≤{THRESHOLDS['toxicity']}{RST}\n"
  )


def main():
  parser = argparse.ArgumentParser(description="Score LLM outputs with DeepEval.")
  parser.add_argument("--run", metavar="RUN_ID", help="Score a specific logged run")
  parser.add_argument("--prompt", help="Run a fresh prompt, score the results")
  parser.add_argument("--log", default=None, help="Path to JSONL log file")
  args = parser.parse_args()

  print(f"\n{'═' * 72}")
  print(f"{BOLD}LLM Eval Lab — Phase 3: DeepEval Scoring{RST}")
  print(f"{'═' * 72}\n")

  if args.prompt:
    # Fresh run: route the prompt, log it, then score
    run_id = make_run_id()
    print(f"{DIM}Running prompt across backends...{RST}\n")
    results = route(args.prompt)
    log_results(results, run_id)
    items = [
      {
        "model_id": r.model_id,
        "prompt": r.prompt,
        "output": r.output,
        "error": r.error,
      }
      for r in results
    ]
    print(f"{DIM}Prompt: {args.prompt}{RST}")
  else:
    # Load from log
    from pathlib import Path

    log_path = Path(args.log) if args.log else None

    if args.run:
      records = read_run(args.run, log_path)

      if not records:
        print(f"\033[31mNo records found for run: {args.run}\033[0m\n")
        return
    else:
      all_records = read_log(log_path)

      if not all_records:
        print("\033[31mNo log entries found. Run: uv run python run.py first.\033[0m\n")
        return
      # Get the most recent run
      latest_run_id = all_records[-1]["run_id"]
      records = [r for r in all_records if r["run_id"] == latest_run_id]

    items = records
    print(f"{DIM}Prompt: {items[0]['prompt']}{RST}")
    print(f"{DIM}Run ID: {items[0]['run_id']}{RST}")

  print(f"\n{DIM}Judge model: llama3.2:3b  •  Scoring {len(items)} output(s)...{RST}\n")

  eval_results = evaluate_many(items)

  for r in eval_results:
    print_eval_result(r)

  print_summary(eval_results)


if __name__ == "__main__":
  main()
