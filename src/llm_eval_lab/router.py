"""
router.py — Phase 1: Fan a prompt out to multiple LLM backends.

Each backend is defined in config.yaml. Results come back as a list of
ModelResult dataclasses so downstream code (logging, eval) has a clean
contract to work with.

Ollama note: LiteLLM has a known content-stripping issue with some Ollama
models (including Gemma4). For Ollama backends we bypass LiteLLM entirely
and call Ollama's native OpenAI-compatible endpoint directly via the openai
client. The ModelResult contract is identical either way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, cast

from litellm.types.utils import ModelResponse as LiteLLMModelResponse

import litellm
import yaml
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

# Suppress LiteLLM's verbose logging — we handle our own output
litellm.suppress_debug_info = True


@dataclass
class ModelResult:
  """The outcome of a single model call."""

  model_id: str  # human label from config (e.g. "claude", "local")
  model_name: str  # full LiteLLM model string (e.g. "claude-sonnet-4-6")
  provider: str  # anthropic | openai | ollama
  prompt: str
  output: str
  latency_ms: float
  input_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  error: Optional[str] = None
  metadata: dict = field(default_factory=dict)


def load_config(path: str = "config.yaml") -> dict:
  with open(path) as f:
    return yaml.safe_load(f)


def call_ollama(model_cfg: dict, prompt: str, defaults: dict) -> ModelResult:
  """
  Call an Ollama backend directly via its OpenAI-compatible endpoint.

  Bypasses LiteLLM to avoid a content-stripping bug affecting some
  Ollama models (e.g. Gemma4). Ollama's /v1/chat/completions endpoint
  is fully OpenAI-compatible, so the openai client works without changes.
  """
  model_id = model_cfg["id"]
  model_name = model_cfg["model"]
  provider = model_cfg["provider"]
  base_url = model_cfg.get("base_url", "http://localhost:11434")

  # Strip the "ollama/" prefix; the native endpoint uses
  # bare model names
  bare_model = model_name.removeprefix("ollama/")

  client = OpenAI(
    base_url=f"{base_url}/v1",
    api_key="ollama",  # required by the client; not used by Ollama
  )

  start = time.perf_counter()

  try:
    response = client.chat.completions.create(
      model=bare_model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=defaults.get("max_tokens", 512),
      temperature=defaults.get("temperature", 0.7),
    )
    latency_ms = (time.perf_counter() - start) * 1000

    output = response.choices[0].message.content or ""
    usage = response.usage

    return ModelResult(
      model_id=model_id,
      model_name=model_name,
      provider=provider,
      prompt=prompt,
      output=output,
      latency_ms=round(latency_ms, 2),
      input_tokens=getattr(usage, "prompt_tokens", None),
      output_tokens=getattr(usage, "completion_tokens", None),
    )

  except Exception as exc:
    latency_ms = (time.perf_counter() - start) * 1000

    return ModelResult(
      model_id=model_id,
      model_name=model_name,
      provider=provider,
      prompt=prompt,
      output="",
      latency_ms=round(latency_ms, 2),
      error=str(exc),
    )


def call_model(model_cfg: dict, prompt: str, defaults: dict) -> ModelResult:
  """
  Call a single model and return a ModelResult.

  Routes Ollama backends to call_ollama() to avoid a LiteLLM
  content-stripping bug. All other providers go through LiteLLM.
  Catches exceptions so one failing backend never aborts the fan-out.
  """
  if model_cfg["provider"] == "ollama":
    return call_ollama(model_cfg, prompt, defaults)

  model_id = model_cfg["id"]
  model_name = model_cfg["model"]
  provider = model_cfg["provider"]
  base_url = model_cfg.get("base_url")

  kwargs: dict = {
    "model": model_name,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": defaults.get("max_tokens", 512),
    "temperature": defaults.get("temperature", 0.7),
  }

  if base_url:
    kwargs["api_base"] = base_url

  start = time.perf_counter()

  try:
    response = cast(LiteLLMModelResponse, litellm.completion(**kwargs))
    latency_ms = (time.perf_counter() - start) * 1000

    output = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    return ModelResult(
      model_id=model_id,
      model_name=model_name,
      provider=provider,
      prompt=prompt,
      output=output,
      latency_ms=round(latency_ms, 2),
      input_tokens=getattr(usage, "prompt_tokens", None),
      output_tokens=getattr(usage, "completion_tokens", None),
    )

  except Exception as exc:
    latency_ms = (time.perf_counter() - start) * 1000

    return ModelResult(
      model_id=model_id,
      model_name=model_name,
      provider=provider,
      prompt=prompt,
      output="",
      latency_ms=round(latency_ms, 2),
      error=str(exc),
    )


def route(prompt: str, config_path: str = "config.yaml") -> list[ModelResult]:
  """
  Fan a prompt out to all configured backends.
  Returns results in config order (sequential for now; easy to parallelise later).
  """
  config = load_config(config_path)
  defaults = config.get("defaults", {})
  results = []

  for model_cfg in config["models"]:
    result = call_model(model_cfg, prompt, defaults)
    results.append(result)

  return results
