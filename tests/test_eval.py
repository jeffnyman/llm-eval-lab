"""
tests/test_eval.py — DeepEval pytest integration for LLM Eval Lab.

Runs the same metrics as eval_run.py but via pytest, using DeepEval's
native assert_test() integration. This means:

  - Each model output is a separate test case
  - Failures show up as pytest failures with reasons
  - You can run the full suite with: uv run deepeval test run tests/test_eval.py
  - Or with standard pytest: uv run pytest tests/test_eval.py -v

The test cases are generated dynamically from the most recent logged run,
so you can run run.py first to populate the log, then run the tests against
whatever outputs were produced.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

from llm_eval_lab.logger import read_log
from llm_eval_lab.eval import JUDGE_MODEL, JUDGE_BASE_URL, THRESHOLDS, OllamaJudge


def get_latest_run_records() -> list[dict]:
  """Load the most recent run's records from the log."""
  records = read_log()

  if not records:
    return []

  latest_run_id = records[-1]["run_id"]

  return [r for r in records if r["run_id"] == latest_run_id]


def make_judge() -> OllamaJudge:
  return OllamaJudge(
    model=JUDGE_MODEL,
    base_url=JUDGE_BASE_URL,
    temperature=0,
  )


# Build test cases at collection time from the latest logged run.
# Records with errors or empty output are skipped.
_records = get_latest_run_records()
_scoreable = [r for r in _records if not r.get("error") and r.get("output", "").strip()]


@pytest.mark.parametrize("record", _scoreable, ids=[r["model_id"] for r in _scoreable])
def test_answer_relevancy(record: dict):
  """Output should be relevant to the prompt."""
  judge = make_judge()

  metric = AnswerRelevancyMetric(
    threshold=THRESHOLDS["answer_relevancy"],
    model=judge,
    include_reason=True,
  )

  test_case = LLMTestCase(
    input=record["prompt"],
    actual_output=record["output"],
  )

  assert_test(test_case, [metric])


@pytest.mark.parametrize("record", _scoreable, ids=[r["model_id"] for r in _scoreable])
def test_bias(record: dict):
  """Output should not exhibit demographic or ideological bias."""
  judge = make_judge()

  metric = BiasMetric(
    threshold=THRESHOLDS["bias"],
    model=judge,
    include_reason=True,
  )

  test_case = LLMTestCase(
    input=record["prompt"],
    actual_output=record["output"],
  )

  assert_test(test_case, [metric])


@pytest.mark.parametrize("record", _scoreable, ids=[r["model_id"] for r in _scoreable])
def test_toxicity(record: dict):
  """Output should not contain toxic or harmful content."""
  judge = make_judge()

  metric = ToxicityMetric(
    threshold=THRESHOLDS["toxicity"],
    model=judge,
    include_reason=True,
  )

  test_case = LLMTestCase(
    input=record["prompt"],
    actual_output=record["output"],
  )

  assert_test(test_case, [metric])
