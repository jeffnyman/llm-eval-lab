"""
debug_ollama.py — Inspect the raw LiteLLM response from your local Ollama model.

Run this if your local model shows [no output] to see exactly what
the response object looks like. Useful for diagnosing model-specific
response structure differences.

Usage:
    uv run python debug_ollama.py
"""

import json
from typing import cast

import yaml
import litellm
from litellm.types.utils import ModelResponse as LiteLLMModelResponse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
litellm.suppress_debug_info = True

config = yaml.safe_load(open("config.yaml"))
local_cfg = next(m for m in config["models"] if m["provider"] == "ollama")

print(f"Model: {local_cfg['model']}")
print(f"Base URL: {local_cfg.get('base_url', 'not set')}\n")

response = cast(LiteLLMModelResponse, litellm.completion(
  model=local_cfg["model"],
  messages=[{"role": "user", "content": "Say hello in one sentence."}],
  max_tokens=64,
  api_base=local_cfg.get("base_url"),
))

print("=== response.choices[0].message ===")
msg = response.choices[0].message
print(f"  .content      : {repr(msg.content)}")
print(f"  .role         : {repr(msg.role)}")
print(f"  attrs         : {[a for a in dir(msg) if not a.startswith('_')]}")
print()
print("=== response.usage ===")
print(f"  {getattr(response, 'usage', None)}")
print()
print("=== raw model_response dict ===")

print(json.dumps(response.model_dump(), indent=2, default=str))
