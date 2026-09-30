> 🧭 NOTE 2026-08-13: The build-models workspace now runs the **wayward multi-plan structure**.
> For workspace-wide orientation see [../masterplan.md](../masterplan.md) + [../AGENTS.md](../AGENTS.md).
> This factory feeds `plans/plan-01-bitqwen/` (training) and `plans/plan-03-data-miner/` (orchestration).
> The file below remains the factory-specific operating guide.

# AGENTS.md — Factory & Pipeline Agent Instructions

> Instructions for any agent (ZCode, OpenCode, or human) operating the BitQwen
> data pipeline and factory. Read this before starting, stopping, configuring,
> validating, or debugging the pipeline.
>
> For full pipeline docs see [README.md](./README.md). For the project wiki see
> [../help-start-here/README.md](../help-start-here/README.md). For lessons
> learned see [../research/MISTAKES.md](../research/MISTAKES.md).

---

## Pipeline Overview

The full pipeline is documented in [README.md §1](./README.md#1-the-full-pipeline).
In short:

```
Raw sources → Assembler → NASA Tier 0 (FREE)
  → AST Auto-Fix (FREE) → Docker Sandbox (FREE)
  → NASA Tier 1 ($0.40) → NASA Tier 2 ($0.24)
  → Complexity Rating (FREE) → Rebalancer (FREE) → Final Dataset
```

The factory in this folder is one *source*. The validation/recovery/balancing
tools live in `../scripts/`. This file covers both.

---

## Factory Location

This standalone factory lives at `factory/`. The original source lives at
`scripts/background_data_factory/`. Both are functionally identical — this
standalone copy includes `seeds/` (legacy categorized seed module) so the
self-test passes and the categorized seed loader is available as a fallback.
The preferred, config-driven approach is to use prompt files
(`prompts/*.jsonl`), which the factory reads based on `factory_config.json`.

---

## How to Start the Factory

### Docker (preferred)
```bash
cd factory
cp .env.example .env          # fill in keys (placeholders only)
docker compose up -d --build  # starts in background, restart: always
docker compose logs -f        # tail live progress
```

### Local Python (no Docker)
```bash
cd factory
pip install "openai>=1.51.0,<2.0.0" requests==2.32.3 tenacity==8.5.0
export AIHUBMIX_API_KEY=...
python generate.py --loop --output ./data
```

### Verify it started
```bash
python generate.py --status
# or inside Docker:
docker compose exec bitqwen-data-factory python generate.py --status
```
You should see `mode: 'free'`, the 4 providers listed, and the active prompt
file. If `[fatal] no providers available` appears, no API keys are set — check
`.env` / environment variables.

---

## How to Stop the Factory

```bash
docker compose down            # graceful (saves progress + usage)
# or, for local Python: Ctrl+C (SIGINT — finishes current request, saves, exits)
```
`docker compose down` sends SIGTERM; the factory finishes the current request,
saves `.progress.json` and `usage.json`, prints the daily summary, and exits.

---

## How to Run NASA Validation

The 3-tier pipeline is in `../scripts/dual_filter.py`. Tier 0 is free; Tier 1
+ 2 cost ~$0.64 per 200K examples.

```bash
# Tier 0 only (FREE — regex + AST + REPL + decontam). Saves rejections.
python ../scripts/dual_filter.py --input raw.jsonl --output clean.jsonl \
    --local-only --save-rejections rejections_tier0.jsonl --report t0.json

# Full 3-tier pipeline (Tier 0 + 1 + 2). ~$0.64 / 200K.
python ../scripts/dual_filter.py --input raw.jsonl --output validated.jsonl \
    --save-rejections rejections_tier0.jsonl --report full.json

# Skip LLM layers (quick local test)
python ../scripts/dual_filter.py --input data.jsonl --output clean.jsonl --skip-llm

# Custom Tier 2 spot-check sample size
python ../scripts/dual_filter.py --input data.jsonl --output clean.jsonl --spot-check 5000

# Self-test (11/11 must pass)
python ../scripts/dual_filter.py --self-test
```

**Expected rejection rate:** ~90% of raw downloaded code fails Tier 0. This is
normal (truncated snippets, HTML/JS mixed in, license headers). Don't panic —
the recovery pipeline below salvages ~45% of rejections for free.

---

## How to Run AST Auto-Fix Recovery

`../scripts/ast_autofix.py` repairs code with NO LLM (10+ deterministic fix
types: license headers, HTML/JS extraction, line endings, indentation, control
chars, missing colons, unbalanced brackets, missing imports, optional
autopep8/black/ruff). 29/29 self-test.

```bash
# Fix rejections + validate (re-run REPL validator after fixing)
python ../scripts/ast_autofix.py --input rejections_tier0.jsonl \
    --output recovered.jsonl --validate --report recovery.json

# Self-test
python ../scripts/ast_autofix.py --self-test
```

Typical recovery: ~25% of Tier 0 rejections are fixed and pass REPL.

### Docker sandbox recovery (next stage)

`../scripts/code_sandbox/sandbox.py` runs fixed code in a Docker-isolated
container and can `pip install` missing packages before testing imports. This
recovers another ~20% (code that failed only because of a missing import).

```bash
# Run sandbox validation on the recovered set
python ../scripts/validate_code_data.py --input recovered.jsonl \
    --output sandbox_validated.jsonl --sandbox
```

Then feed `sandbox_validated.jsonl` back into NASA Tier 1 + 2 for the semantic
check (the only paid step).

---

## How to Rate Complexity + Rebalance

```bash
# Complexity rating + curriculum interleaving (FREE, 31/31 self-test)
python ../scripts/complexity_rater.py --input validated.jsonl \
    --output curriculum.jsonl --report complexity.json

# Rebalance to honest targets (FREE, NO padding)
python ../scripts/rebalance_dataset.py --input curriculum.jsonl \
    --output-train train_final.jsonl --output-eval eval_final.jsonl \
    --report rebalance.json

# Stats only (no write)
python ../scripts/rebalance_dataset.py --input data.jsonl --stats-only

# Self-tests
python ../scripts/complexity_rater.py --self-test
python ../scripts/rebalance_dataset.py --self-test
```

Rebalancer targets: func_gen 50% / cot_debug 20% / syllogism 10% / ast_ir 10%
/ bug_injection 5% / framework 3% / other 2%. If a category is under target,
the report says so honestly — it does NOT pad with surplus func_gen (the
PewDiePie failure, MISTAKES.md M1).

---

## How to Use the Bandit Router

`../scripts/bandit_router.py` — Thompson-sampling, cost-aware router that
learns which model to route each query to (FrugalGPT-style). Beta posterior per
(model, context bucket), reward `r = quality − λ·cost`.

```bash
# Initialize from routing config
python ../scripts/bandit_router.py init --config ../scripts/routing_config.json

# Route a query
python ../scripts/bandit_router.py route --query "Write a Python function" --budget 0.01

# Feed back an outcome (quality defaults to 1.0 on success, 0.0 on failure)
python ../scripts/bandit_router.py feedback --model "ling-2.6-flash" \
    --success True --cost 0.002 --latency 1.5

# Status (posterior per model)
python ../scripts/bandit_router.py status

# Train from mined logs
python ../scripts/bandit_router.py train --from-logs data/personalized/zcode_mined.jsonl

# Export to LiteLLM config
python ../scripts/bandit_router.py export-litellm
```

---

## How to Mine Logs

`../scripts/mine_logs.py` — mines (prompt, response, outcome) triples from
local agent logs (ZCode `db.sqlite` + rollout JSONL) and formats them as
ChatML. Recurring errors are clustered and turned into counter-examples.

```bash
# Mine ZCode sqlite → ChatML
python ../scripts/mine_logs.py --source zcode --output ../data/personalized/zcode_mined.jsonl

# Stats only (no write)
python ../scripts/mine_logs.py --source zcode --stats

# Mine rollout JSONL files (verbatim request/response per LLM call)
python ../scripts/mine_logs.py --source zcode --rollouts --output ../data/personalized/rollouts.jsonl

# Filter to specific error types
python ../scripts/mine_logs.py --source zcode \
    --errors tool_timeout,tool_execution_failed --output ../data/personalized/errors.jsonl

# Self-test
python ../scripts/mine_logs.py --self-test
```

Output goes to `data/personalized/`. This feeds both the self-play loop and the
bandit router (`train --from-logs`).

---

## How to Manage Providers + Prices

```bash
# Model router CLI (6 providers, 1001+ models)
python ../scripts/model_router.py providers list
python ../scripts/model_router.py models list
python ../scripts/model_router.py routing show
python ../scripts/model_router.py estimate --agent validation_agent --examples 200000
python ../scripts/model_router.py export-litellm
python ../scripts/model_router.py export-factory
python ../scripts/model_router.py status

# Auto-update the price matrix from OpenRouter + SiliconFlow + AIHubMix + LiteLLM
python ../scripts/update_prices.py
python ../scripts/update_prices.py --cheapest
python ../scripts/update_prices.py --tier free
```

Config: `../scripts/routing_config.json`. Price matrix:
`../scripts/prices.json`. Generated LiteLLM config:
`../scripts/litellm_config.yaml`.

---

## How to Add a New Prompt File

1. Create a `.jsonl` file in `prompts/`. Each line:
   ```json
   {"prompt": "Write a Python function that...", "category": "func_gen", "difficulty": "intermediate"}
   ```
   - `prompt` (required): the seed text.
   - `category` (optional, default `func_gen`): one of `func_gen`, `cot_debug`,
     `syllogism` (or a custom category — the rotator supports any set).
   - `difficulty` (optional, default `intermediate`).
2. Register it (validates it loads, persists to config):
   ```bash
   python generate.py --add-prompts prompts/my_custom.jsonl
   ```
3. The factory now uses it. If running in Docker, the bind-mount means the host
   file is read live — no rebuild needed.

To switch back to a different file:
```bash
python generate.py --prompts prompts/dspy.jsonl --loop
```

---

## How to Switch Providers

Providers are controlled by `factory_config.json` under `providers.*`. To toggle
a provider without removing its config, set `enabled: false`:
```json
{ "providers": { "hf": { "enabled": false } } }
```

To change a provider's weight (selection probability relative to others):
```json
{ "providers": { "aihubmix": { "weight": 70 } } }
```

To change the daily cap:
```json
{ "providers": { "openrouter": { "daily_cap": 200 } } }
```

To add API keys, edit `.env` (or set env vars). The `key_env` field in config
names the env var the factory reads:
- `AIHUBMIX_API_KEY` — primary external free provider
- `LITELLM_MASTER_KEY` — local proxy (optional if no master key set)
- `HF_TOKEN` — HuggingFace fallback
- `OPENROUTER_API_KEY` — OpenRouter fallback

Edits to `factory_config.json` take effect on the **next loop iteration** (the
factory re-reads it every cycle). No restart needed.

---

## How to Switch Modes (free / paid)

```bash
python generate.py --mode free                 # $0 spend (default)
python generate.py --mode paid --budget 5.00   # paid mode, $5/day cap
python generate.py --budget 0                  # back to unlimited (free)
```
These persist to `factory_config.json`. In `paid` mode, once `spent_today >=
daily_limit` the factory stops generating. The budget is re-read each loop, so
editing the JSON mid-run takes effect on the next iteration.

---

## How to Adjust the Category Balance

```bash
python generate.py --balance 0.5,0.3,0.2   # func_gen,cot_debug,syllogism
```
Fractions are normalized to sum 1.0. The `ConfigCategoryRotator` builds a
100-step pattern that converges to the requested split (within 1% over each
cycle). Persists to config.

---

## How to Check Output

```bash
# Status (mode, providers, prompts, budget, progress, usage today)
python generate.py --status

# See the generated files
ls -la data/
# generated_2026-08-07.jsonl  .progress.json  usage.json

# Count today's examples
wc -l data/generated_$(date +%F).jsonl

# Inspect a few examples
head -n 3 data/generated_$(date +%F).jsonl

# Tail live progress (Docker)
docker compose logs -f
```

Each record: `{"text": "<ChatML>", "provider": "aihubmix", "category": "func_gen", "ts": "..."}`.

---

## Common Issues + Fixes (updated from MISTAKES.md)

### 1. `[fatal] no providers available`
**Cause:** No API keys set, or all providers disabled in config.
**Fix:** Check `.env` has real keys (not placeholders). Run `python generate.py
--status` — the `key=` column shows `yes`/`no` per provider. Set at least
`AIHUBMIX_API_KEY` (primary free provider).

### 2. All 429s / rate-limited immediately
**Cause:** Daily caps exhausted, or per-model rate limit too aggressive.
**Fix:** Run `python generate.py --status` to see `Usage today`. If caps are
hit, wait for midnight reset (or raise `daily_cap` in config). For AIHubMix
429s, the factory auto-rotates to the next of 15 models — if all 15 are in
cooldown, it backs off for 60s.

### 3. 0 examples accepted despite many API calls (MISTAKES.md D2)
**Cause (historical):** Error bodies (429 protobuf JSON) were passed to the
validator as if they were model responses. 913 API calls, 0 accepted.
**Fix (already applied):** Every `provider.generate()` catches ALL exceptions
and returns `None`. `None` is never validated. If this recurs, check that no
provider is returning an error body instead of raising. Run `python generate.py
--self-test` — the "provider returns None on exception" test confirms the
contract.

### 4. Factory not picking up config changes
**Cause:** Running in Docker without the bind-mount, or editing a different
config file.
**Fix:** `docker-compose.yml` bind-mounts `./factory_config.json` read-write.
Confirm with `docker compose exec bitqwen-data-factory cat /app/factory_config.json`.
The factory re-reads config every loop iteration, so edits take effect within
one cycle.

### 5. Disk full from generated JSONL
**Cause:** Months of `generated_*.jsonl` accumulating.
**Fix:** Archive or delete old files in `data/`. The factory writes one file
per day and never deletes them automatically. Pipe through `dual_filter.py`
to produce a cleaned, deduped dataset. Note: C: drive is FULL (18GB free) —
write to E: drive only.

### 6. `openai` not installed
**Cause:** Running locally without installing deps.
**Fix:** `pip install "openai>=1.51.0,<2.0.0" requests==2.32.3 tenacity==8.5.0`.
In Docker, the Dockerfile installs these.

### 7. LiteLLM proxy unreachable (`Connection refused`)
**Cause:** The local LiteLLM proxy at `http://host.docker.internal:4000/v1`
isn't running on the host.
**Fix:** Start the LiteLLM proxy on the host, or disable LiteLLM in config
(`"litellm": {"enabled": false}`) and rely on AIHubMix + HF + OpenRouter. The
factory falls back automatically — LiteLLM returning `None` just means the next
provider is tried.

### 8. Self-test fails
**Cause:** Usually a corrupted `factory_config.json` or missing prompt files.
**Fix:** Run `python generate.py --self-test` and read the assertion message.
The self-test is offline and exercises all logic; a failure means the factory
code/config/prompt files are inconsistent. Verify all 6 prompt files are
present in `prompts/` (each 20+ lines).

### 9. NASA Tier 0 rejects 90% of code
**Cause:** Normal for raw downloaded code (StarCoder, CodeSearchNet, GitHub
dumps — truncated snippets, HTML/JS mixed in, license headers).
**Fix:** Don't discard rejections — run them through `ast_autofix.py`
(recovers ~25%) and the Docker sandbox (recovers ~20% more) for free. See
§"How to Run AST Auto-Fix Recovery" above.

### 10. Wrong SiliconFlow domain (MISTAKES.md D3)
**Cause:** Tested `api.siliconflow.cn` (401).
**Fix:** Correct domain is `api.siliconflow.com`. Check `SILICONFLOW_BASE_URL`
in env, not the account page URL.

### 11. CastError from mixed JSONL schema (MISTAKES.md D1)
**Cause:** 2 rows had extra keys (`messages`, `meta`) alongside `text`.
**Fix:** Use a manual JSONL parser, not `load_dataset("json")`. Keep only the
`text` field.

---

## Operating Rules

1. **Never commit real API keys.** `.env.example` is placeholders only. `.env`
   is gitignored. If you accidentally write keys to `.env.example`, rotate them
   immediately.
2. **Free mode is the default.** Do not switch to `paid` mode without explicit
   user instruction — it spends money.
3. **Don't disable the error-body invariant.** Every `provider.generate()` MUST
   catch all exceptions and return `None`. Never pass an error body to the
   validator. (See MISTAKES.md D2 / issue #3 above.)
4. **Don't remove the rate limiter.** It's the only thing keeping us under free
   quotas. Caps are conservative on purpose.
5. **Run `--self-test` after any code change.** It's offline and catches
   regressions in validation, rate limiting, provider selection, and config
   handling. Every pipeline tool has one.
6. **The factory is idempotent and resumable.** Killing and restarting it is
   safe — `.progress.json` and `usage.json` restore the current day's state.
7. **Save NASA rejections, don't discard them.** They feed the AST auto-fix
   recovery loop AND bug-injection / Markov-debugging training data.
8. **Never pad the dataset to hit category targets.** Report honest gaps
   (MISTAKES.md M1 — the PewDiePie failure).
9. **Write to E: drive, not C:.** C: is full (18GB free). E: has 2.5TB free.

---

## Self-Test Checklist

Before declaring the pipeline healthy, run and confirm each tool's self-test:

```bash
python generate.py --self-test                    # factory
python ../scripts/dual_filter.py --self-test      # NASA 3-tier (11/11)
python ../scripts/ast_autofix.py --self-test      # auto-fix (29/29)
python ../scripts/complexity_rater.py --self-test # complexity (31/31)
python ../scripts/rebalance_dataset.py --self-test
python ../scripts/bandit_router.py --self-test
python ../scripts/mine_logs.py --self-test
python ../scripts/generate_ast_ir_data.py --self-test   # 51/51
python ../scripts/dataset_validator.py --self-test      # 11/11
python ../scripts/dataset_extractor.py --self-test      # 25/25
```

Expected factory output ends with `=== self-test PASSED ===`. Key sub-tests:
- validation OK
- RateLimiter initial / unlimited / cap enforcement / remaining / persistence / daily_summary / thread-safety
- `_pick_provider` blocks exhausted / multiple exhausted / falls back to unlimited / skips broken
- provider returns None on exception
- factory catches uncaught provider exception
- factory skips None responses (no validation)
- CategoryRotator 60/25/15 split (120/50/30 over 200 calls)
- categorized seeds OK
- AIHubMixProvider offline construction + rotation
- AIHubMix 15 free models OK
- free-only provider config OK
- config load/save round-trip OK
- `_deep_merge` fills defaults OK
- prompt files OK (6 files, all >=20 prompts)
- `load_prompts` OK
- `categorized_prompts` OK
- `ConfigCategoryRotator` fractions OK
- Factory config weights/sleep/budget OK
- Factory budget exhaustion stops picks OK
- Factory uses prompt file OK

If any sub-test fails, do not start the factory — fix the regression first.
