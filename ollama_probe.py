"""
probe_ollama.py — Bare minimum test of the direct OpenAI client against Ollama.
Run this independently of the rest of the project to isolate the connection.

Usage:
    uv run python probe_ollama.py
"""

import yaml
from openai import OpenAI

config = yaml.safe_load(open("config.yaml"))
local_cfg = next(m for m in config["models"] if m["provider"] == "ollama")

base_url = local_cfg.get("base_url", "http://localhost:11434")
bare_model = local_cfg["model"].removeprefix("ollama/")

print(f"Connecting to : {base_url}/v1")
print(f"Model         : {bare_model}")
print()

client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")

response = client.chat.completions.create(
  model=bare_model,
  messages=[{"role": "user", "content": "In one sentence, what is a unit test?"}],
  max_tokens=80,
)

msg = response.choices[0].message

print(f"finish_reason : {response.choices[0].finish_reason}")
print(f"content       : {repr(msg.content)}")
print(f"usage         : {response.usage}")
