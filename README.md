# BitQwen Data Pipeline & Factory

The BitQwen data pipeline turns raw code (HuggingFace, GitHub, Kaggle, Usenet,
and LLM-generated examples from this factory) into a clean, decontaminated,
balanced, curriculum-ordered SFT dataset. The factory in this folder is one
*source* in that pipeline; the rest of the pipeline lives in `../scripts/` and
is documented here so operators have one complete reference.

This folder is a **self-contained copy** of the factory from
`scripts/background_data_factory/` plus accurate, current documentation for the
entire pipeline.

> **Read this first if you are resuming work.** For the project-level wiki see
> [../help-start-here/README.md](../help-start-here/README.md). For the live
> session log see [../todo/BQdoing.md](../todo/BQdoing.md). For lessons learned
> see [../research/MISTAKES.md](../research/MISTAKES.md).

---

## 1. The Full Pipeline

```
Raw Data Sources (HF, GitHub, Kaggle, Usenet, Factory)
  │
  ▼
Assembler (merge + dedup + decontaminate)        FREE
  │   scripts/assemble_master_dataset.py
  ▼
NASA Tier 0 (regex + AST + REPL + decontam)      FREE
  │   scripts/dual_filter.py — 11/11 self-test
  ▼
AST Auto-Fix (recover ~25% of rejections)        FREE
  │   scripts/ast_autofix.py — 29/29 self-test
  ▼
Docker Sandbox (recover ~20% more, install pkgs) FREE
  │   scripts/code_sandbox/sandbox.py
  ▼
NASA Tier 1 (ling-2.6-flash semantic check)      $0.40 / 200K
  │   scripts/dual_filter.py --tier 1
  ▼
NASA Tier 2 (gpt-5.6-luna spot-check, ~2.5%)     $0.24 / 200K
  │   scripts/dual_filter.py --tier 2
  ▼
Complexity Rating (AST metrics + curriculum)     FREE
  │   scripts/complexity_rater.py — 31/31 self-test
  ▼
Rebalancer (honest targets, NO padding)          FREE
  │   scripts/rebalance_dataset.py
  ▼
Final Dataset (train + eval)                     → Kaggle / HF
```

### Why this order

- **NASA Tier 0 is free and rejects ~90% of raw code.** This is normal for
  downloaded dumps (StarCoder, CodeSearchNet, GitHub scrapes) — the code
  snippets are often truncated, mixed with HTML/JS, or contain license headers.
  Running the free tiers first means the paid LLM tiers only ever see code that
  already parses and runs.
- **AST Auto-Fix + Docker Sandbox recover ~45% of Tier 0 rejections for $0.**
  See §4 below. Rejected code is saved (`rejections_tier0.jsonl`) so it can be
  fed to the auto-fixer and (later) used for bug-injection training.
- **Tier 1 + Tier 2 cost ~$0.64 per 200K examples.** Tier 1 (`ling-2.6-flash`,
  $0.01/$0.03 per M tokens) answers "does the code answer the question?" for
  every survivor. Tier 2 (`gpt-5.6-luna`) spot-checks ~3K survivors to catch
  Tier 1 mistakes; if disagreement >5% the whole batch is flagged.
- **Complexity rating + rebalancer run after validation** so the difficulty
  distribution and category targets are computed on clean code only.

---

## 2. Factory CLI (full reference)

The factory (`generate.py`) is config-driven: switch modes, prompt files,
providers, and budgets by editing `factory_config.json` (or via CLI) — no code
changes needed.

```bash
# Help
python generate.py --help         # or:  python generate.py /?

# Generation
python generate.py --loop                       # run forever with config
python generate.py --config cfg.json --loop     # use a custom config
python generate.py --max-examples 1000          # generate N then exit
python generate.py --output ./data              # override output dir

# Prompt files (6 shipped: default, pocketflow, dspy, litellm, ast_ir, bug_injection)
python generate.py --prompts prompts/dspy.jsonl --loop   # switch prompt file
python generate.py --list-prompts                        # list available files
python generate.py --add-prompts prompts/custom.jsonl    # register + validate

# Mode + budget
python generate.py --mode free                  # free mode (default, $0 spend)
python generate.py --mode paid --budget 5.00    # paid mode, $5/day cap
python generate.py --budget 0                   # back to unlimited (free)

# Category balance (func_gen, cot_debug, syllogism)
python generate.py --balance 0.5,0.3,0.2

# Status + self-test
python generate.py --status                     # mode, providers, prompts, budget, progress
python generate.py --self-test                  # offline self-test (no API calls)
```

All mutating CLI commands (`--mode`, `--budget`, `--balance`, `--add-prompts`,
`--prompts`) persist their changes back to `factory_config.json`, so they
survive restarts. In Docker, `docker-compose.yml` bind-mounts the config file
read-write so CLI commands run inside the container persist back to the host.

### Config file: `factory_config.json`

Read on startup AND every loop iteration, so edits take effect on the next
cycle (no rebuild needed).

```json
{
  "mode": "free",
  "providers": {
    "aihubmix":   { "enabled": true, "models": [...], "daily_cap": 7500, "weight": 50 },
    "litellm":    { "enabled": true, "models": [...], "daily_cap": 0,    "weight": 30 },
    "openrouter": { "enabled": true, "models": [...], "daily_cap": 100,  "weight": 10 },
    "hf":         { "enabled": true, "models": [...], "daily_cap": 500,  "weight": 10 }
  },
  "prompts": {
    "prompt_file": "prompts/default.jsonl",
    "categories": { "func_gen": 0.6, "cot_debug": 0.25, "syllogism": 0.15 }
  },
  "budget": { "daily_limit": 0.0, "currency": "USD", "spent_today": 0.0 },
  "output": { "dir": "/data", "format": "jsonl", "split_by_date": true },
  "rate_limits": { "default_per_min": 5, "default_per_day": 500 }
}
```

- `mode`: `"free"` (default, $0 spend) or `"paid"` (uses budget cap).
- `providers.*.enabled`: turn a provider on/off without removing its config.
- `providers.*.weight`: weighted random selection among available providers.
- `providers.*.daily_cap`: 0 means unlimited (used for local LiteLLM).
- `providers.*.key_env`: the env var name holding the API key.
- `prompts.prompt_file`: which prompt file to draw seeds from.
- `prompts.categories`: the target category split (normalized to sum 1.0).

### Prompt files (`prompts/`)

Each file is JSONL: one JSON object per line with `prompt`, `category`, and
`difficulty` fields. The factory ships with 6 prompt files (20+ prompts each):

| File | Category focus | Purpose |
|------|---------------|---------|
| `prompts/default.jsonl` | func_gen / cot_debug / syllogism | General Python SFT |
| `prompts/pocketflow.jsonl` | func_gen | PocketFlow agent patterns (flows, nodes, state) |
| `prompts/dspy.jsonl` | func_gen | DSPy declarative pipeline patterns (signatures, modules, teleprompters) |
| `prompts/litellm.jsonl` | func_gen | LiteLLM routing patterns (fallback, retry, rate limits) |
| `prompts/bug_injection.jsonl` | cot_debug | Bug injection with bytecode IR (Level 2 scope + Level 3 async) |
| `prompts/ast_ir.jsonl` | func_gen | AST/IR reasoning tasks (structural + execution thinking) |

---

## 3. All Pipeline Tools (reference)

Every tool below has a `--self-test` (offline, no API calls) unless noted.
Run the self-test after any code change.

| Tool | Purpose | Self-Test | Cost |
|------|---------|-----------|------|
| `scripts/dual_filter.py` | NASA 3-tier validation (regex+AST+REPL+decontam → ling → luna) | 11/11 | $0.64 / 200K |
| `scripts/ast_autofix.py` | LLM-free code repair (10+ fix types: colons, imports, indent, license, HTML…) | 29/29 | FREE |
| `scripts/complexity_rater.py` | AST difficulty metrics + curriculum interleaving | 31/31 | FREE |
| `scripts/rebalance_dataset.py` | Honest balancing (no padding, stratified eval holdout) | pass | FREE |
| `scripts/model_router.py` | Provider/model/routing matrix CLI (export litellm/factory/gen) | working | FREE |
| `scripts/bandit_router.py` | Thompson-sampling cost-aware router (FrugalGPT-style) | pass | FREE |
| `scripts/mine_logs.py` | Mine ZCode sqlite + rollout logs → ChatML | pass | FREE |
| `scripts/update_prices.py` | Auto-update price matrix from 4 providers | working | FREE |
| `scripts/generate_ast_ir_data.py` | AST+IR pair generator (code↔AST, code↔bytecode) | 51/51 | FREE |
| `scripts/dataset_validator.py` | 11-check pre-upload validator | 11/11 | FREE |
| `scripts/dataset_extractor.py` | 7 subcommands (scan/sample/filter/convert/stats/dedup/merge) | 25/25 | FREE |
| `scripts/validate_code_data.py` | AST + sandbox execution validation | pass | FREE |
| `scripts/code_sandbox/sandbox.py` | Docker-isolated code execution (install missing pkgs) | pass | FREE |
| `scripts/assemble_master_dataset.py` | Merge + dedup + categorize + decontaminate | — | FREE |
| `scripts/decontaminate.py` | 4-layer HumanEval/MBPP contamination check | — | FREE |

### Recovery pipeline (the cheap path)

```bash
# 1. Run NASA Tier 0 (FREE) and SAVE rejections
python scripts/dual_filter.py --input raw.jsonl --output clean.jsonl \
    --local-only --save-rejections rejections_tier0.jsonl --report t0.json

# 2. Auto-fix the rejections (FREE, no LLM)
python scripts/ast_autofix.py --input rejections_tier0.jsonl \
    --output recovered.jsonl --validate --report recovery.json

# 3. Send recovered + clean through Tier 1 + Tier 2 (the only paid step)
python scripts/dual_filter.py --input recovered.jsonl --output validated.jsonl \
    --report t12.json

# 4. Rate complexity + reorder (FREE)
python scripts/complexity_rater.py --input validated.jsonl \
    --output curriculum.jsonl --report complexity.json

# 5. Rebalance to honest targets (FREE)
python scripts/rebalance_dataset.py --input curriculum.jsonl \
    --output-train train_final.jsonl --output-eval eval_final.jsonl \
    --report rebalance.json
```

Typical recovery: AST auto-fix recovers ~25% of Tier 0 rejections, the Docker
sandbox (which can `pip install` missing packages and re-test imports) recovers
another ~20%. That means ~45% of the 90% Tier 0 rejects can be salvaged for $0.

---

## 4. Providers (6 providers, 1001+ models)

Managed by `scripts/model_router.py` + `scripts/routing_config.json` +
`scripts/prices.json` (auto-built by `scripts/update_prices.py`).

| Provider | Models | Balance / Tier | Notes |
|----------|--------|----------------|-------|
| OpenRouter | 400 | $4.76 balance | Rich pricing; free + paid models |
| SiliconFlow | 73 | $25.91 balance | `api.siliconflow.com` (NOT `.cn`) |
| AIHubMix | 380 | 15 free, needs recharge | 15 free coding models, $0/1M tokens |
| Ollama Cloud | 18 | 5 free, free tier | 1 concurrent, 5h session / 7d weekly |
| HuggingFace router | Qwen-7B/32B, Llama-3.1-8B | free, rate-limited | WORKS |
| LiteLLM local | 143 | unlimited | `http://host.docker.internal:4000/v1` — FREE |

**Estimated daily factory output (free mode):** ~3,000+ examples/day (~90K+/month
before filtering, ~50–70K clean after `ast.parse` + `dual_filter`).

> Together AI is INTENTIONALLY DISABLED in free-only mode (it would spend the
> $5 budget). SiliconFlow non-free is also disabled.

### Routing rules (model_router.py)

| Rule | Cascade |
|------|---------|
| `generation_agent` | free → (stop) |
| `syllogism_agent` | free → cheap → (stop) |
| `validation_agent` | ultra_cheap → medium → (stop) |

### Bandit router (scripts/bandit_router.py)

A contextual multi-armed bandit (Thompson sampling, Beta posterior per
model+context bucket, cost-aware reward `r = quality − λ·cost`) that learns
which model to route each query to. See
[../research/MODEL-ROUTING-MATH-RESEARCH.md](../research/MODEL-ROUTING-MATH-RESEARCH.md).

```bash
python scripts/bandit_router.py init --config scripts/routing_config.json
python scripts/bandit_router.py route --query "Write a Python function" --budget 0.01
python scripts/bandit_router.py feedback --model "ling-2.6-flash" --success True --cost 0.002 --latency 1.5
python scripts/bandit_router.py status
python scripts/bandit_router.py train --from-logs data/personalized/zcode_mined.jsonl
python scripts/bandit_router.py export-litellm
```

---

## 5. Cost Analysis

| Task | Cost | Notes |
|------|------|-------|
| NASA Tier 0 (regex+AST+REPL+decontam) | $0 | per any size |
| AST Auto-Fix recovery | $0 | ~25% of rejects recovered |
| Docker sandbox recovery | $0 | ~20% more recovered |
| NASA Tier 1 (ling-2.6-flash, 200K) | $0.40 | semantic "does code answer question?" |
| NASA Tier 2 (gpt-5.6-luna, 3K sample) | $0.24 | spot-check ~2.5% of survivors |
| **200K validation (Tier 0+1+2)** | **$0.64** | recommended for current goal |
| **1M dataset validation** | **$4.14** | 5× the 200K cost |
| Factory generation (free mode) | $0 | free providers only |
| Factory generation (paid mode) | ~$2 | bounded by `--budget` |
| Download 900GB raw + clean | $0 | 5–7 days, 1.1TB disk, 90% rejected |
| **Recommended: factory + recovery for 200K** | **$2–4** | best quality/time tradeoff |

### Disk space (2026-08-08)

- E: drive: 4.6TB total, 2.5TB available (47% used) — plenty for 900GB.
- C: drive: 1.9TB total, 18GB available (100% full) — FULL, avoid writing here.
- Current data: 12GB total (5.8GB master, 2.7GB AST/IR, 654MB HF, 533MB scratch).

---

## 6. Self-Play Loop (Phase 12)

The pipeline is designed to co-evolve the model and the harness. Each
generation feeds the next:

```
Gen 1: raw code (HF/GitHub/Kaggle/Usenet/Factory)
   → NASA Tier 0 rejects 90%  →  SAVE rejections (don't discard)
   → AST autofix recovers ~25%  →  Docker sandbox recovers ~20% more
   → Ling + GPT (Tier 1+2) validate  →  Gen 2 training set
   → Train BitQwen on Gen 2
Gen 3: BitQwen generates new code  →  harness validates  →  Gen 3 set
   → still-failing code  →  Markov debugging + bug-injection training
   → repeat
```

Key properties:
- **The harness gets better at catching errors** (rejection reasons are saved
  and turned into bug-injection examples + Markov shell debugging data).
- **The model gets better at avoiding errors** (it trains on its own clean
  outputs plus counter-examples built from its failures).
- **10–20% of rollouts use real REPL** to avoid model collapse (the rest use
  the trained validator, Bitty/CodeBERTa, as a pre-filter).

See [../research/bitnet-training/SELF-PLAY-RECURSIVE-LLM.md](../research/bitnet-training/SELF-PLAY-RECURSIVE-LLM.md)
and the Phase 12 section of [../help-start-here/ROADMAP.md](../help-start-here/ROADMAP.md).

### Log mining for personalized data (Phase 10)

`scripts/mine_logs.py` mines (prompt, response, outcome) triples from local
agent logs (ZCode `db.sqlite` 216MB, 11K messages + rollout JSONL) and formats
them as ChatML. Recurring errors are clustered and turned into counter-examples
(a sibling successful turn, or a templated fix). Output:
`data/personalized/zcode_mined.jsonl`. This feeds both the self-play loop and
the bandit router (`bandit_router.py train --from-logs`).

```bash
python scripts/mine_logs.py --source zcode --output data/personalized/zcode_mined.jsonl
python scripts/mine_logs.py --source zcode --stats
python scripts/mine_logs.py --source zcode --rollouts --output data/personalized/rollouts.jsonl
```

---

## 7. Quick Start

### With Docker (recommended for the factory)

```bash
cd factory
cp .env.example .env          # fill in your own keys (placeholders only)
docker compose up -d --build  # starts the factory in the background
docker compose logs -f        # tail live progress
```

### Run without Docker (local Python)

```bash
pip install "openai>=1.51.0,<2.0.0" requests==2.32.3 tenacity==8.5.0

export AIHUBMIX_API_KEY=...      # primary external free provider
export LITELLM_MASTER_KEY=...    # optional, if your local proxy requires it
export HF_TOKEN=...              # optional fallback
export OPENROUTER_API_KEY=...    # optional fallback

python generate.py --loop --output ./data
python generate.py --output ./data --max-examples 100   # bounded test
python generate.py --self-test                          # offline self-test
```

### Run the full validation pipeline on a month of factory output

```bash
cat data/generated_*.jsonl > raw_all.jsonl
python ../scripts/dual_filter.py --input raw_all.jsonl --output clean.jsonl \
    --save-rejections rejections.jsonl --report filter_report.json
python ../scripts/ast_autofix.py --input rejections.jsonl --output recovered.jsonl --validate
python ../scripts/complexity_rater.py --input clean.jsonl --output curriculum.jsonl --report c.json
python ../scripts/rebalance_dataset.py --input curriculum.jsonl \
    --output-train train_final.jsonl --output-eval eval_final.jsonl --report r.json
```

The resulting `train_final.jsonl` is ready for SFT.

---

## 8. Output

- `/data/generated_{YYYY-MM-DD}.jsonl` — one file per day, ChatML format:
  ```json
  {"text": "<|im_start|>user\n{seed}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>", "provider": "aihubmix", "category": "func_gen", "ts": "2026-08-07T14:22:01.123456"}
  ```
- `/data/.progress.json` — accepted-example counters + total (resume on restart).
- `/data/usage.json` — per-provider daily request usage (rate limiter state).

---

## 9. Validation (factory-level)

Every factory response is validated before being written:

1. Extract the last ` ```python ` fenced block (or fall back to raw text).
2. Must contain `def ` or `class ` AND pass `ast.parse`, OR be >200 chars of
   substantive reasoning.

Rejected responses are logged and not written (but the API call still counts
against the rate limit, since the provider was still charged).

### The error-body invariant (critical — MISTAKES.md D2)

Every `provider.generate()` catches ALL exceptions (429, 500, timeout,
protobuf-JSON error bodies, etc.) and returns `None`. A `None` response is
NEVER passed to the validator — the factory just moves on to the next provider.
This fixes the bug where 913 API calls were made and 0 accepted (429 error
bodies were being validated as if they were model responses).

---

## 10. Rate Limiting

Every provider is gated by a thread-safe `RateLimiter` that persists per-day
usage to `/data/usage.json`. A provider is **skipped** the moment it hits its
daily cap — no wasted API calls. Usage resets at local midnight automatically.

- **Thread-safe:** a `threading.Lock` guards every read/increment/write.
- **Persistent:** `usage.json` survives container restarts.
- **Per-provider caps:** see the providers table. `0` means unlimited (LiteLLM).
- When ALL providers are exhausted, the factory sleeps until the next-day
  reset (not a fixed 10 min) so it doesn't burn cycles.

---

## 11. Resume / Graceful Shutdown

- On `SIGTERM`/`SIGINT` the factory finishes the current request, saves
  `.progress.json` and `usage.json`, prints the daily summary, and exits cleanly.
- `restart: always` in `docker-compose.yml` brings it back automatically.
- On startup it reads `.progress.json` (accepted counts) and `usage.json`
  (rate-limit usage); both restore only the current day's state.

---

## 12. Lark Notification Setup (from research)

The user wants phone notifications for long-running pipeline jobs (NASA
validation, training). Discord requires face ID on mobile; Lark does not and
is compatible with ZCode hooks.

Recommended approach (best-effort webhook, no SDK needed):

1. Create a Lark custom bot in a group chat → copy the webhook URL
   (`https://open.larksuite.com/open-apis/bot/v2/hook/<id>`).
2. Store the URL in `scripts/.env` as `LARK_WEBHOOK_URL`.
3. Post a JSON payload via `urllib` (stdlib, no deps):
   ```json
   {"msg_type": "text", "content": {"text": "NASA Tier 0 done: 31.7K/274K survived"}}
   ```
4. Hook into pipeline scripts via a small `notify(msg)` helper, or via a ZCode
   `Stop` hook that fires when a background task finishes.

Keep it best-effort (wrap in try/except, swallow network errors) so a flaky
webhook never breaks the pipeline.

---

## 13. Files

| File | Purpose |
|---|---|
| `generate.py` | Main factory script (config-driven CLI, 4 providers, rate limiter, validation) |
| `factory_config.json` | Config (mode, providers, prompts, budget, output) |
| `Dockerfile` | Slim Python 3.11 image (openai, requests, tenacity) |
| `docker-compose.yml` | Compose: restart always, mounts config + prompts, 512M limit |
| `.env.example` | Credentials template (placeholders only — never commit real keys) |
| `prompts/*.jsonl` | 6 prompt files (default, pocketflow, dspy, litellm, bug_injection, ast_ir) |
| `seeds/` | Legacy categorized seed module (`python_seed_prompts.py` + `seeds_for_generation.jsonl`) — used as a fallback when no prompt file is configured, and required for the self-test |
| `AGENTS.md` | Factory + pipeline agent instructions |
| `.zcode/skills/bitqwen-factory/SKILL.md` | ZCode skill for the factory |
| `.opencode/skills/bitqwen-factory/SKILL.md` | OpenCode skill mirror |

> **Note:** `seeds/` is included so the self-test passes and the categorized
> seed loader is available as a fallback. The factory prefers the prompt file
> configured in `factory_config.json` (`prompts/default.jsonl`) but falls back
> to the legacy categorized seeds if no prompt file is set.

---

## 14. Security

`.env.example` contains **placeholders only**. Never commit real API keys to
version control. The factory reads every credential from environment variables
— there are no hardcoded keys anywhere in the codebase.

> **Historical note:** an earlier `.env.example` in
> `scripts/background_data_factory/` accidentally contained real API keys.
> Those keys were rotated. Always verify `.env.example` contains only
> placeholders before committing.

---

## 15. See Also

- [../help-start-here/README.md](../help-start-here/README.md) — project wiki entry point
- [../help-start-here/ROADMAP.md](../help-start-here/ROADMAP.md) — Phase A–H + 9–12 roadmap
- [../help-start-here/RECIPE-DRAFT.md](../help-start-here/RECIPE-DRAFT.md) — BitNet training recipe
- [../help-start-here/FILE-INVENTORY.md](../help-start-here/FILE-INVENTORY.md) — workspace file map
- [AGENTS.md](./AGENTS.md) — factory + pipeline agent instructions
- [../research/MISTAKES.md](../research/MISTAKES.md) — lessons learned (don't repeat)
- [../research/MODEL-ROUTING-MATH-RESEARCH.md](../research/MODEL-ROUTING-MATH-RESEARCH.md) — bandit routing math
- [../research/LOG-MINING-PERSONALIZED-DATA.md](../research/LOG-MINING-PERSONALIZED-DATA.md) — log mining design
- [../research/AST-IR-TRAINING-DATA.md](../research/AST-IR-TRAINING-DATA.md) — AST/IR data design
- [../scripts/background_data_factory/](../scripts/background_data_factory/) — the original factory source
