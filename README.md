# LLM Eval Lab

A pedagogical project for routing, observing, and evaluating LLM outputs across multiple backends, built with a tester's mindset.

**Phases:**
- Phase 1 — Multi-backend routing (Claude, GPT-4o, Ollama)
- Phase 2 — Observability: structured JSONL logging per call
- Phase 3 — Evaluation: scoring outputs with DeepEval

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [Ollama](https://ollama.com) — for local model inference
- API keys for Anthropic and OpenAI

---

## Setup

### 1. Install uv

For any POSIX system:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows Powershell:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install the project

```bash
git clone <repo-url>
cd llm-eval-lab
uv sync
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

Note that if you are solely using Ollama and local models, you don't need API keys.

### 4. Install and start Ollama

Download from [ollama.com](https://ollama.com), then pull a model:

```bash
# Pick any model; smaller is faster for experimentation
ollama pull llama3.2        # ~2GB, good balance
ollama pull phi3:mini       # ~2.3GB, Microsoft's compact model
ollama pull mistral         # ~4GB, popular baseline

# Verify it's running
curl http://localhost:11434/api/tags
```

### 5. Update config.yaml

Edit `config.yaml` and set the `model` field under `id: local` to match what you pulled:

```yaml
- id: local
  model: ollama/llama3.2   # change this to match your pulled model
  provider: ollama
  base_url: http://localhost:11434
```

---

## Running Phase 1 + 2

```bash
# Default prompt: routes to all backends and logs the run
uv run python run.py

# Custom prompt
uv run python run.py --prompt "What makes a good test assertion?"

# Skip logging (useful while iterating on config)
uv run python run.py --no-log
```

## Inspecting the log (Phase 2)

Every run appends to `logs/runs.jsonl`. Use `show_logs.py` to browse it:

```bash
# List all runs
uv run python show_logs.py

# Show full outputs for a specific run
uv run python show_logs.py --run <run_id>

# Show last 5 runs only
uv run python show_logs.py --tail 5

# Per-model latency and token stats across all runs
uv run python show_logs.py --stats
```

You can also query the raw JSONL directly with `jq`:

```bash
# All run IDs
jq -r '.run_id' logs/runs.jsonl | sort -u

# Average latency per model
jq -r '[.model_id, .latency_ms] | @tsv' logs/runs.jsonl | \
  awk '{sum[$1]+=$2; count[$1]++} END {for (m in sum) print m, sum[m]/count[m] "ms"}'
```

To run those on Windows in Powershell:

```bash
Get-Content logs/runs.jsonl | ConvertFrom-Json | Select-Object -ExpandProperty run_id | Sort-Object -Unique

Get-Content logs/runs.jsonl | ConvertFrom-Json |
    Group-Object model_id |
    ForEach-Object {
        $avg = ($_.Group.latency_ms | Measure-Object -Average).Average
        [PSCustomObject]@{
            Model = $_.Name
            AvgLatency = "$([Math]::Round($avg, 2))ms"
        }
    } | Format-Table -HideTableHeaders
```

---

## Troubleshooting Ollama

If a local model shows `[no output]` or you can't tell whether Ollama is reachable:

```bash
# Test raw connectivity via the OpenAI-compatible endpoint
uv run python ollama_probe.py

# Inspect the full LiteLLM response object to diagnose output issues
uv run python ollama_debug.py
```

---
