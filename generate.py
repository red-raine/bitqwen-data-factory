#!/usr/bin/env python3
"""
generate.py — Always-on background data factory for Project BitQwen 1.0.

Runs 24/7 inside a Docker container, generating Python SFT examples using
ONLY free API providers. LiteLLM (local proxy) is the PRIMARY provider — it is
free, unlimited, and has no rate limits. External APIs are fallbacks.

Provider order (LiteLLM first, then external fallbacks):
    1. LiteLLM local proxy (http://localhost:4000/v1)  — PRIMARY, no limits
    2. HuggingFace Inference API (model rotation)       — ~500/model/day
    3. OpenRouter free models                           — ~100 req/day
    4. Together AI (free models)                        — ~500 req/day

KEY INVARIANTS (these fix the bug where 913 calls were made and 0 accepted):
    * Every provider.generate() catches ALL exceptions (429, 500, timeout,
      protobuf-JSON error bodies, etc.) and returns None.
    * A None response is NEVER passed to the validator — the factory just
      moves on to the next provider.
    * _pick_provider() only selects providers where can_request() is True,
      so an exhausted provider is never retried.
    * When ALL providers are exhausted, the factory sleeps until the next
      calendar-day reset (not a fixed 10 min) so it doesn't burn cycles.

Each generated example is validated (ast.parse of extracted code OR must
contain `def`|`class` OR be >200 chars) and written as ChatML to
/data/generated_{date}.jsonl.

Resumes from /data/.progress.json, handles SIGTERM/SIGINT gracefully, and
logs progress every 10 examples with provider + count + ETA.

CLI:
    python generate.py --loop --output /data
    python generate.py --output /data --max-examples 100
    python generate.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import random
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make sibling seed module + parent scripts/ (env_config) importable.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SEEDS_DIR = HERE / "seeds"
SCRIPTS_DIR = HERE.parent  # .../scripts/
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SEEDS_DIR) not in sys.path:
    sys.path.insert(0, str(SEEDS_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Credentials via env_config (no hardcoding). Falls back to os.environ.
# ---------------------------------------------------------------------------
try:
    from env_config import load_env, get_litellm_key, get_hf_token, \
        get_openrouter_key  # type: ignore
    load_env()
except Exception:
    # env_config not available — fall back to plain os.environ reads.
    def get_litellm_key() -> Optional[str]:
        return os.environ.get("LITELLM_MASTER_KEY") or os.environ.get("LITELLM_KEY")
    def get_hf_token() -> Optional[str]:
        return os.environ.get("HF_TOKEN")
    def get_openrouter_key() -> Optional[str]:
        return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior Python engineer. Write production-ready Python with "
    "step-by-step reasoning. Format: <thought>reasoning</thought> then the code."
)

MAX_TOKENS = 2048
TEMPERATURE = 0.4

# ===========================================================================
# FREE-MODE-ONLY CONFIGURATION
# ===========================================================================
# The factory now runs in FREE MODE ONLY — no spending. Paid providers
# (SiliconFlow non-free, OpenRouter non-free, Together) are DISABLED.
# Only free providers are enabled:
#   - AIHubMix  : 15 free models ($0/1M tokens, 5 req/min, 500 req/day each)
#   - LiteLLM   : local proxy (free, unlimited)
#   - OpenRouter: free models only
#   - HuggingFace: free Inference API
#
# Per-provider daily request caps (free tier). LiteLLM is local => unlimited.
# AIHubMix aggregate cap = 15 models * 500 req/day = 7500 req/day (all free).
LITELLM_DAILY_CAP = 0           # 0 means "no cap" — treated as unlimited
HF_DAILY_CAP = 500              # conservative aggregate for HF Inference API
OPENROUTER_DAILY_CAP = 100      # conservative for free models
TOGETHER_DAILY_CAP = 0          # DISABLED in free-only mode (0 = unlimited
                                # would be wrong here; instead we simply do
                                # not instantiate TogetherProvider).
AIHUBMIX_DAILY_CAP = 7500       # 15 free models * 500 req/day each = ALL FREE

# Provider selection weights. AIHubMix (15 free models) is the PRIMARY free
# provider and gets the highest weight. LiteLLM (local) is co-primary.
# Together is excluded entirely (free-only mode).
PROVIDER_WEIGHTS = {
    "aihubmix": 50,   # 15 free models — primary external free provider
    "litellm": 30,    # local proxy — free, unlimited
    "openrouter": 10, # free models only
    "hf": 10,         # free Inference API
}

# Sleep between requests (seconds) — tuned per provider to respect rate limits.
LITELLM_SLEEP = 0.5    # local, no remote rate limit — just breathing room
HF_SLEEP = 3.0         # per-model rate limit, rotate to multiply
OPENROUTER_SLEEP = 5.0 # 100/day, very sparingly
TOGETHER_SLEEP = 2.0   # disabled in free-only mode (unused)
AIHUBMIX_SLEEP = 0.8   # 5 req/min per model; rotating 15 models => ~75 req/min
                       # aggregate. 0.8s keeps us safely under the per-model
                       # 5 req/min limit while moving fast across models.

# HuggingFace model rotation list (free Inference API, per-model rate limit).
HF_MODELS = [
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "codellama/CodeLlama-34b-Instruct-hf",
]

# OpenRouter free code models.
OPENROUTER_FREE_MODELS = [
    "inclusionai/ling-3.0-flash:free",
    "cohere/north-mini-code:free",
    "google/gemma-4-31b-it:free",
]

# Together AI free / budget models — DISABLED in free-only mode.
TOGETHER_MODELS = [
    "Prism-ML/Ternary-Bonsai-27B",
    "poolside/laguna-xs-2.1",
]

# LiteLLM local models (served by the local proxy on :4000).
LITELLM_MODELS = [
    "deepseek-v4-flash",
    "xiaomi/mimo-v2.5-pro",
    "gpt-oss:120b-cloud",
]

# ===========================================================================
# AIHubMix — 15 FREE models ($0/1M tokens)
# ===========================================================================
# All 15 models are $0/1M tokens. Per-model limits: 5 req/min, 500 req/day,
# 1M tokens/day. Aggregate: 75 req/min, 7,500 req/day, 15M tokens/day — FREE.
# We rotate through all 15 models to multiply throughput and respect the
# per-model 5 req/min rate limit.
AIHUBMIX_BASE_URL = os.environ.get(
    "AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"
)
AIHUBMIX_FREE_MODELS = [
    "coding-glm-5.2-free",         # GLM 5.2, 1M context, coding specialist
    "coding-glm-5.1-free",         # GLM 5.1
    "coding-glm-5-free",           # GLM 5
    "coding-glm-4.7-free",         # GLM 4.7
    "coding-glm-5-turbo-free",     # GLM 5 turbo
    "coding-kimi-k3-free",         # Kimi K3 for coding
    "kimi-for-coding-free",        # Kimi for coding
    "k2.6-code-preview-free",      # Kimi K2.6 code preview
    "coding-minimax-m3-free",      # MiniMax M3
    "coding-minimax-m2.7-free",    # MiniMax M2.7
    "coding-minimax-m2.5-free",    # MiniMax M2.5
    "glm-4.7-flash-free",          # GLM 4.7 flash
    "gemini-3.6-flash-free",       # Gemini 3.6 flash
    "gemini-3.5-flash-lite-free",  # Gemini 3.5 flash lite
    "laguna-xs-2.1-free",          # Poolside Laguna XS 2.1 (33B-A3B coding agent)
]
# Per-model rate limit: 5 req/min. We track the last request timestamp per
# model and skip a model if it had a request within the last 12 seconds
# (5 req/min => 1 req / 12s evenly spaced; we use 12s to be safe).
AIHUBMIX_PER_MODEL_MIN_INTERVAL_SEC = 12.0

# LiteLLM proxy endpoint. Override with LITELLM_BASE_URL env if needed.
LITELLM_BASE_URL = os.environ.get(
    "LITELLM_BASE_URL", "http://localhost:4000/v1"
)

CHATML_TEMPLATE = (
    "<|im_start|>user\n{seed}<|im_end|>\n"
    "<|im_start|>assistant\n{response}<|im_end|>"
)

# ===========================================================================
# Config-driven operation (factory_config.json)
# ===========================================================================
# The factory reads factory_config.json on startup and re-reads it every loop
# iteration. Users can switch modes, change prompt files, adjust budgets, and
# tweak provider settings WITHOUT touching the code — just edit the JSON.
#
# CLI flags (--mode, --budget, --prompts, --balance) override the config for
# the current run AND persist back into factory_config.json so the change
# survives restarts (mirrors gen_controller.py's behavior).

FACTORY_CONFIG_PATH = HERE / "factory_config.json"
PROMPTS_DIR = HERE / "prompts"

DEFAULT_FACTORY_CONFIG: Dict[str, Any] = {
    "mode": "free",
    "providers": {
        "aihubmix": {"enabled": True, "models": list(AIHUBMIX_FREE_MODELS),
                     "daily_cap": AIHUBMIX_DAILY_CAP, "sleep_sec": AIHUBMIX_SLEEP,
                     "weight": 50, "base_url": AIHUBMIX_BASE_URL,
                     "key_env": "AIHUBMIX_API_KEY"},
        "litellm": {"enabled": True, "models": list(LITELLM_MODELS),
                    "daily_cap": LITELLM_DAILY_CAP, "sleep_sec": LITELLM_SLEEP,
                    "weight": 30, "base_url": LITELLM_BASE_URL,
                    "key_env": "LITELLM_MASTER_KEY"},
        "openrouter": {"enabled": True, "models": list(OPENROUTER_FREE_MODELS),
                       "daily_cap": OPENROUTER_DAILY_CAP, "sleep_sec": OPENROUTER_SLEEP,
                       "weight": 10, "key_env": "OPENROUTER_API_KEY"},
        "hf": {"enabled": True, "models": list(HF_MODELS),
               "daily_cap": HF_DAILY_CAP, "sleep_sec": HF_SLEEP,
               "weight": 10, "key_env": "HF_TOKEN"},
    },
    "prompts": {
        "prompt_file": "prompts/default.jsonl",
        "categories": {"func_gen": 0.6, "cot_debug": 0.25, "syllogism": 0.15},
    },
    "budget": {"daily_limit": 0, "currency": "USD", "spent_today": 0.0},
    "output": {"dir": "/data", "format": "jsonl", "split_by_date": True},
    "rate_limits": {"default_per_min": 5, "default_per_day": 500},
}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `over` onto `base` (returns a new dict)."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_factory_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load factory_config.json merged over DEFAULT_FACTORY_CONFIG.

    Missing keys are filled from defaults, so a partial user config still
    yields a complete config. A missing or malformed file falls back to
    defaults with a warning (never crashes the factory).
    """
    cfg = json.loads(json.dumps(DEFAULT_FACTORY_CONFIG))  # deep copy
    p = Path(path) if path else FACTORY_CONFIG_PATH
    if p.exists():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                cfg = _deep_merge(cfg, user)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] WARNING: could not load {p}: {e}; using defaults")
    return cfg


def save_factory_config(cfg: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Persist config to disk atomically."""
    p = Path(path) if path else FACTORY_CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(p)


# ---- prompt file loading --------------------------------------------------

def load_prompts(prompt_file: str) -> List[Dict[str, str]]:
    """Load prompts from a JSONL file.

    Each line: {"prompt": "...", "category": "func_gen", "difficulty": "beginner"}
    Returns a list of dicts. Relative paths resolve against the factory dir.
    """
    p = Path(prompt_file)
    if not p.is_absolute():
        p = HERE / p
    if not p.exists():
        print(f"[prompts] WARNING: {p} does not exist")
        return []
    out: List[Dict[str, str]] = []
    with open(p, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[prompts] {p.name}:{ln} bad JSON: {e}")
                continue
            text = obj.get("prompt")
            if not text or not str(text).strip():
                continue
            out.append({
                "prompt": str(text).strip(),
                "category": str(obj.get("category", "func_gen")),
                "difficulty": str(obj.get("difficulty", "intermediate")),
            })
    print(f"[prompts] loaded {len(out)} prompts from {p}")
    return out


def list_prompt_files() -> List[Path]:
    """Return all .jsonl files in the prompts/ directory (sorted)."""
    if not PROMPTS_DIR.exists():
        return []
    return sorted(PROMPTS_DIR.glob("*.jsonl"))


def categorized_prompts(prompts: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Group prompt dicts by category -> list of prompt strings.

    Ensures func_gen/cot_debug/syllogism all exist (falling back to the
    first non-empty category) so the rotator never picks an empty bucket.
    """
    cats: Dict[str, List[str]] = {"func_gen": [], "cot_debug": [], "syllogism": []}
    for pr in prompts:
        c = pr["category"]
        cats.setdefault(c, []).append(pr["prompt"])
    # fall back so no category is empty
    fallback = cats.get("func_gen") or cats.get("cot_debug") or cats.get("syllogism") or []
    for c in ("func_gen", "cot_debug", "syllogism"):
        if not cats.get(c):
            cats[c] = list(fallback)
    return cats


# ---------------------------------------------------------------------------
# Graceful shutdown flag
# ---------------------------------------------------------------------------

_SHUTDOWN = False


def _handle_signal(signum, frame):  # noqa: D401
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"\n[shutdown] Received signal {signum}, finishing current request then exiting...")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# RateLimiter — thread-safe, per-provider daily usage tracker
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe per-provider daily request cap tracker.

    Persists usage to a JSON file (default /data/usage.json) so caps survive
    container restarts. Usage is keyed by date; on a new calendar day all
    counters reset to 0 automatically.

    A cap of 0 means UNLIMITED (the provider is never blocked by usage).
    This is used for the local LiteLLM proxy.

    Each provider checks `can_request(name)` before making a call and calls
    `record(name)` after the call returns (success OR failure — the API call
    was still consumed against the daily quota).
    """

    # Default daily caps per provider name. 0 = unlimited.
    # Together is omitted: in FREE-ONLY mode we do not instantiate
    # TogetherProvider at all, so it never appears here.
    DEFAULT_CAPS = {
        "litellm": LITELLM_DAILY_CAP,
        "aihubmix": AIHUBMIX_DAILY_CAP,
        "hf": HF_DAILY_CAP,
        "openrouter": OPENROUTER_DAILY_CAP,
    }

    def __init__(self, usage_path: Path, caps: Optional[Dict[str, int]] = None):
        self.usage_path = Path(usage_path)
        self.caps: Dict[str, int] = dict(self.DEFAULT_CAPS)
        if caps:
            self.caps.update(caps)
        self._lock = threading.Lock()
        # in-memory view: { "YYYY-MM-DD": { provider: count } }
        self._usage: Dict[str, Dict[str, int]] = {}
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self):
        if not self.usage_path.exists():
            return
        try:
            with open(self.usage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "days" in data:
                # modern format: {"days": {date: {provider: count}}}
                self._usage = {
                    str(d): {str(k): int(v) for k, v in provs.items()}
                    for d, provs in data["days"].items()
                }
            elif isinstance(data, dict):
                # legacy format: {date: {provider: count}}
                self._usage = {
                    str(d): {str(k): int(v) for k, v in provs.items()}
                    for d, provs in data.items()
                }
            print(f"[ratelimit] loaded usage for {len(self._usage)} day(s) "
                  f"from {self.usage_path.name}")
        except Exception as e:
            print(f"[ratelimit] could not load usage file: {e}")
            self._usage = {}

    def _save_locked(self):
        """Save usage to disk (caller must hold the lock)."""
        try:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.usage_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "updated": _dt.datetime.now().isoformat(),
                        "days": self._usage,
                    },
                    f,
                    indent=2,
                )
            tmp.replace(self.usage_path)
        except Exception as e:
            print(f"[ratelimit] could not save usage file: {e}")

    # -- day rollover -----------------------------------------------------

    @staticmethod
    def _today() -> str:
        return _dt.date.today().isoformat()

    def _day_usage_locked(self, day: Optional[str] = None) -> Dict[str, int]:
        d = day or self._today()
        if d not in self._usage:
            self._usage[d] = {}
        return self._usage[d]

    # -- public API -------------------------------------------------------

    def get_count(self, provider: str) -> int:
        with self._lock:
            return int(self._day_usage_locked().get(provider, 0))

    def get_cap(self, provider: str) -> int:
        return int(self.caps.get(provider, 0))

    def remaining(self, provider: str) -> int:
        cap = self.caps.get(provider, 0)
        if cap <= 0:
            return -1  # unlimited
        with self._lock:
            used = int(self._day_usage_locked().get(provider, 0))
        return max(0, cap - used)

    def can_request(self, provider: str) -> bool:
        """True if `provider` still has daily budget (or is unlimited)."""
        cap = self.caps.get(provider, 0)
        if cap <= 0:
            return True  # unlimited (local LiteLLM)
        with self._lock:
            used = int(self._day_usage_locked().get(provider, 0))
        return used < cap

    def record(self, provider: str, n: int = 1) -> int:
        """Record n requests against `provider`'s daily quota.

        Returns the new total for that provider today.
        """
        with self._lock:
            day = self._day_usage_locked()
            day[provider] = int(day.get(provider, 0)) + n
            new_total = day[provider]
            self._save_locked()
        return new_total

    def daily_summary(self) -> Dict[str, Tuple[int, int]]:
        """Return {provider: (used, cap)} for today. cap of 0 = unlimited."""
        out: Dict[str, Tuple[int, int]] = {}
        with self._lock:
            day = self._day_usage_locked()
            for prov, cap in self.caps.items():
                out[prov] = (int(day.get(prov, 0)), int(cap))
        return out

    def reset_if_new_day(self) -> bool:
        """Drop stale day entries if the date changed. Returns True if rolled."""
        with self._lock:
            today = self._today()
            # We keep history but the active day auto-creates on access.
            # Nothing to physically delete — old days just stop counting.
            if today not in self._usage:
                self._usage[today] = {}
                self._save_locked()
                return True
        return False

    def seconds_until_midnight(self) -> int:
        """Seconds from now until local midnight (next-day reset)."""
        now = _dt.datetime.now()
        tomorrow = (now + _dt.timedelta(days=1)).date()
        midnight = _dt.datetime.combine(tomorrow, _dt.time.min)
        return max(60, int((midnight - now).total_seconds()))


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def load_seeds() -> List[str]:
    """Load all seeds from python_seed_prompts.py + seeds_for_generation.jsonl."""
    seeds: List[str] = []

    # 1. From python_seed_prompts.py (ALL_SEED_GROUPS dict)
    try:
        import python_seed_prompts as _psp  # type: ignore

        groups = getattr(_psp, "ALL_SEED_GROUPS", {})
        for name, group in groups.items():
            for s in group:
                if isinstance(s, str) and s.strip():
                    seeds.append(s.strip())
        print(f"[seeds] Loaded {len(seeds)} seeds from python_seed_prompts.py ({len(groups)} groups)")
    except Exception as e:
        print(f"[seeds] WARNING: could not import python_seed_prompts: {e}")

    # 2. From seeds_for_generation.jsonl (extract user-side text only)
    jsonl_path = SEEDS_DIR / "seeds_for_generation.jsonl"
    if not jsonl_path.exists():
        # fall back to repo data dir
        jsonl_path = HERE.parent.parent / "data" / "user-provided" / "seeds_for_generation.jsonl"
    if jsonl_path.exists():
        n_before = len(seeds)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                # Extract the user segment from ChatML
                if "<|im_start|>user\n" in text:
                    user_part = text.split("<|im_start|>user\n", 1)[1]
                    user_part = user_part.split("<|im_end|>", 1)[0].strip()
                    if user_part and "[NEEDS_GENERATION]" not in user_part:
                        seeds.append(user_part)
        print(f"[seeds] Loaded {len(seeds) - n_before} seeds from {jsonl_path.name}")

    # Dedupe while preserving order
    seen = set()
    unique = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    print(f"[seeds] Total unique seeds: {len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# Category-aware seed loading
# ---------------------------------------------------------------------------
#
# The factory balances the dataset across three categories while generating
# for free:
#   - func_gen  : functional / instruction-evolution code generation (default)
#   - cot_debug : chain-of-thought Python debugging
#   - syllogism : Python syllogism / formal-logic seeds
#
# Rotation rule (applied per generation call, 1-indexed):
#   - Every 3rd call (call % 3 == 0)  -> cot_debug
#   - Every 5th call (call % 5 == 0)  -> syllogism
#   - Otherwise                       -> func_gen
# Because call 15 satisfies both %3==0 and %5==0, we check cot_debug first
# (it's the higher-priority 25% bucket) — in practice the LCM(3,5)=15
# collision is rare and the resulting ~60/25/15 split matches the target.
# The exact empirical split over 60 calls:
#   cot_debug: calls 3,6,9,12,15,18,...,60 => 20 calls (~33%) but the 15/30/45/60
#   overlap with syllogism is resolved to cot_debug, giving cot_debug ~33%.
# To hit the precise 60/25/15 target we instead use a counter-based scheme
# implemented in CategoryRotator.next_category() below.

def load_categorized_seeds() -> Dict[str, List[str]]:
    """Load seeds grouped by factory category.

    Returns a dict with keys 'func_gen', 'cot_debug', 'syllogism'. Each value
    is a deduplicated list of seed strings. Falls back gracefully: if a
    category ends up empty, the rotator falls back to func_gen.
    """
    cats: Dict[str, List[str]] = {
        "func_gen": [],
        "cot_debug": [],
        "syllogism": [],
    }
    try:
        import python_seed_prompts as _psp  # type: ignore

        # CoT debug seeds (both batches + repair + latent bug + adversarial
        # are all "trace/debug step-through" style — group them as cot_debug).
        cot_debug_groups = [
            "python_cot_debug", "python_cot_debug_2", "python_cot_repair",
            "python_latent_bug", "python_adversarial", "python_adversarial_2",
            "general_cot_debug",
        ]
        # Syllogism seeds (formal logic / structural logic).
        syllogism_groups = [
            "python_syllogism_evol", "general_syllogism_evol",
            "python_structural_logic",
        ]
        # Everything else is func_gen (evol-instruct, semantic reasoning).
        func_gen_groups = [
            "python_evol_instruct", "python_evol_instruct_2",
            "python_semantic_reasoning",
        ]

        groups = getattr(_psp, "ALL_SEED_GROUPS", {})

        def _pull(group_names: List[str]) -> List[str]:
            out: List[str] = []
            for gname in group_names:
                for s in groups.get(gname, []):
                    if isinstance(s, str) and s.strip():
                        out.append(s.strip())
            # dedupe
            seen = set()
            uniq = []
            for s in out:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            return uniq

        cats["cot_debug"] = _pull(cot_debug_groups)
        cats["syllogism"] = _pull(syllogism_groups)
        cats["func_gen"] = _pull(func_gen_groups)
        print(f"[seeds] categorized: func_gen={len(cats['func_gen'])} "
              f"cot_debug={len(cats['cot_debug'])} "
              f"syllogism={len(cats['syllogism'])}")
    except Exception as e:
        print(f"[seeds] WARNING: categorized load failed ({e}); "
              f"falling back to flat seed list for func_gen")
        cats["func_gen"] = load_seeds()

    # Safety: if any category is empty, fall back to func_gen (which itself
    # may be empty — the factory checks for total==0 before starting).
    if not cats["func_gen"]:
        cats["func_gen"] = cats["cot_debug"] + cats["syllogism"] or load_seeds()
    if not cats["cot_debug"]:
        cats["cot_debug"] = cats["func_gen"]
    if not cats["syllogism"]:
        cats["syllogism"] = cats["func_gen"]
    return cats


class CategoryRotator:
    """Deterministic category rotation targeting a 60/25/15 split.

    Over each 20-call cycle the distribution is:
        func_gen : 12 calls (60%)
        cot_debug:  5 calls (25%)
        syllogism:  3 calls (15%)
    The pattern is fixed so the split converges exactly to 60/25/15 regardless
    of total volume. `next()` returns (category, call_index).
    """

    # 20-call pattern: 12 func_gen (F), 5 cot_debug (C), 3 syllogism (S).
    # Spread C and S roughly evenly across the cycle.
    _PATTERN = [
        "func_gen", "func_gen", "cot_debug",   # 1-3
        "func_gen", "syllogism", "func_gen",   # 4-6
        "cot_debug", "func_gen", "func_gen",   # 7-9
        "syllogism", "func_gen", "cot_debug",  # 10-12
        "func_gen", "func_gen", "cot_debug",   # 13-15
        "func_gen", "syllogism", "func_gen",   # 16-18
        "cot_debug", "func_gen",               # 19-20
    ]
    assert len(_PATTERN) == 20
    assert _PATTERN.count("func_gen") == 12
    assert _PATTERN.count("cot_debug") == 5
    assert _PATTERN.count("syllogism") == 3

    def __init__(self):
        self._call = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            cat = self._PATTERN[self._call % 20]
            self._call += 1
            return cat


class ConfigCategoryRotator:
    """Config-driven category rotator.

    Takes a {category: fraction} dict (e.g. {"func_gen": 0.6, "cot_debug": 0.25,
    "syllogism": 0.15}) and yields categories in a pattern that converges to
    the requested split. Builds a 100-step pattern from the fractions so the
    empirical distribution matches the target to within 1/100 over each cycle.
    Any categories the user adds beyond the standard three are supported too.
    """

    def __init__(self, fractions: Dict[str, float]):
        self._lock = threading.Lock()
        self._call = 0
        # Normalize fractions to sum to 1.0
        total = sum(fractions.values()) or 1.0
        norm = {k: max(0.0, v) / total for k, v in fractions.items()}
        # Build a 100-step pattern: assign each category floor(frac*100) slots,
        # then distribute the remainder to the largest fractional parts.
        steps = 100
        base = {k: int(v * steps) for k, v in norm.items()}
        remainder = steps - sum(base.values())
        # order categories by their fractional remainder, descending
        fracs = sorted(
            norm.items(),
            key=lambda kv: (kv[1] * steps - int(kv[1] * steps), kv[1]),
            reverse=True,
        )
        for i in range(remainder):
            k = fracs[i % len(fracs)][0]
            base[k] += 1
        # materialize the pattern and shuffle deterministically so adjacent
        # steps don't all share a category (better spread). Seed for stability.
        pattern: List[str] = []
        for k, n in base.items():
            pattern.extend([k] * n)
        rng = random.Random(0xC0FFEE)
        rng.shuffle(pattern)
        self._PATTERN = pattern
        self._fractions = norm

    def next(self) -> str:
        with self._lock:
            cat = self._PATTERN[self._call % len(self._PATTERN)]
            self._call += 1
            return cat


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str:
    """Pull the last fenced python code block, else the raw response."""
    if "```python" in response:
        chunks = response.split("```python")
        if len(chunks) > 1:
            code = chunks[-1].split("```", 1)[0]
            return code.strip()
    if "```" in response:
        chunks = response.split("```")
        if len(chunks) > 2:
            return chunks[-2].strip()
    return response.strip()


def validate_response(response: str) -> bool:
    """Validate: ast.parse of extracted code OR must contain def/class OR >200 chars.

    This is the GATE. Only responses that pass here are written to disk.
    Error bodies (429 protobuf JSON, HTML 500 pages, etc.) are NEVER passed
    here because providers return None on any error and the factory skips.
    """
    if not response or len(response.strip()) < 50:
        return False
    code = extract_code(response)
    if "def " in code or "class " in code:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            # Fall through: maybe partial code, still useful if long enough
            pass
    # Long-form reasoning is acceptable if substantial
    return len(response.strip()) > 200


# ---------------------------------------------------------------------------
# Provider clients
#
# INVARIANT for ALL providers:
#   generate(prompt) -> Optional[str]
#     - On success returns the model's text.
#     - On ANY error (429, 500, timeout, network, malformed body, etc.)
#       returns None. The error is logged. The request is still recorded
#       against the daily quota in a `finally` block.
#     - NEVER returns an error response body to be validated.
# ---------------------------------------------------------------------------


class LiteLLMProvider:
    """LOCAL LiteLLM proxy (http://localhost:4000/v1) — PRIMARY provider.

    Free, no remote rate limits, no API key required (uses the proxy's
    master_key if set). The rate limiter cap is 0 (unlimited), so
    can_request() always returns True.
    """

    name = "litellm"

    def __init__(self, base_url: str, api_key: Optional[str],
                 rate_limiter: RateLimiter):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.rate_limiter = rate_limiter
        self.client = None
        try:
            from openai import OpenAI  # type: ignore
            kwargs: Dict[str, Any] = {"base_url": self.base_url}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            else:
                # openai SDK requires a non-empty key string; LiteLLM proxy
                # accepts any value when no master_key is configured.
                kwargs["api_key"] = "sk-litellm-local"
            self.client = OpenAI(**kwargs)
            print(f"[litellm] initialized -> {self.base_url} "
                  f"(key={'set' if self.api_key else 'none'})")
        except ImportError:
            print("[litellm] ERROR: openai not installed; run pip install openai")
        except Exception as e:
            print(f"[litellm] ERROR initializing: {e}")

    @property
    def available(self) -> bool:
        # LiteLLM cap is 0 (unlimited) so can_request is always True once
        # the client initialised. If the proxy is down we still attempt
        # the call and rely on generate() returning None on failure.
        return self.client is not None and self.rate_limiter.can_request(self.name)

    def generate(self, prompt: str) -> Optional[str]:
        if self.client is None:
            return None
        if not self.rate_limiter.can_request(self.name):
            return None
        model = random.choice(LITELLM_MODELS)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=90,
            )
            text = resp.choices[0].message.content
            if not text or not text.strip():
                return None
            return text
        except Exception as e:
            # Catch ALL: connection refused, 4xx/5xx, timeout, malformed JSON.
            # NEVER return an error body — return None so the factory skips.
            print(f"[litellm] {model} error: {type(e).__name__}: {e}")
            return None
        finally:
            self.rate_limiter.record(self.name)


class AIHubMixProvider:
    """AIHubMix (https://aihubmix.com/v1) — 15 FREE coding models.

    All 15 models are $0/1M tokens (free tier). Per-model limits:
        - 5 req/min
        - 500 req/day
        - 1M tokens/day
    Aggregate across all 15: 75 req/min, 7,500 req/day, 15M tokens/day — FREE.

    This is the PRIMARY external free provider. It rotates through all 15
    models, enforcing the per-model 5 req/min rate limit by tracking the
    last-request timestamp per model and skipping any model that was used
    within `AIHUBMIX_PER_MODEL_MIN_INTERVAL_SEC` seconds. On a 429 it marks
    the offending model as rate-limited (cooldown) and tries the next model.

    OpenAI-compatible endpoint: POST {base_url}/chat/completions
    """

    name = "aihubmix"

    def __init__(self, api_key: str, rate_limiter: RateLimiter,
                 base_url: Optional[str] = None):
        self.api_key = api_key or ""
        self.rate_limiter = rate_limiter
        self.base_url = (base_url or AIHUBMIX_BASE_URL).rstrip("/")
        self.client = None
        # Per-model last-request timestamp (monotonic) for the 5 req/min limit.
        self._last_request: Dict[str, float] = {}
        # Per-model cooldown-until timestamp (monotonic) for 429 backoff.
        self._cooldown_until: Dict[str, float] = {}
        # Round-robin index for model rotation.
        self._model_idx = 0
        self._idx_lock = threading.Lock()
        if not self.api_key:
            print("[aihubmix] WARNING: no AIHUBMIX_API_KEY set")
            return
        try:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            print(f"[aihubmix] initialized -> {self.base_url} "
                  f"({len(AIHUBMIX_FREE_MODELS)} free models, $0/1M tokens)")
        except ImportError:
            print("[aihubmix] ERROR: openai not installed; run pip install openai")
        except Exception as e:
            print(f"[aihubmix] ERROR initializing: {e}")

    @property
    def available(self) -> bool:
        return (self.client is not None
                and bool(self.api_key)
                and self.rate_limiter.can_request(self.name))

    def _next_model(self) -> Optional[str]:
        """Return the next model that is not in per-model cooldown and has
        not been hit within the per-model min interval. Returns None if all
        models are temporarily rate-limited (caller should back off)."""
        now = time.monotonic()
        n = len(AIHUBMIX_FREE_MODELS)
        with self._idx_lock:
            start = self._model_idx
            for offset in range(n):
                idx = (start + offset) % n
                model = AIHUBMIX_FREE_MODELS[idx]
                # skip if in 429 cooldown
                if now < self._cooldown_until.get(model, 0.0):
                    continue
                # skip if used too recently (5 req/min per model)
                last = self._last_request.get(model, 0.0)
                if now - last < AIHUBMIX_PER_MODEL_MIN_INTERVAL_SEC:
                    continue
                # found an eligible model — advance the round-robin past it
                self._model_idx = (idx + 1) % n
                return model
        # All models either cooling down or too recently used.
        return None

    def _mark_429(self, model: str) -> None:
        """Mark `model` as rate-limited for 60s (skip it, try next)."""
        self._cooldown_until[model] = time.monotonic() + 60.0
        print(f"[aihubmix] {model} got 429 — cooldown 60s, skipping")

    def _mark_used(self, model: str) -> None:
        self._last_request[model] = time.monotonic()

    def generate(self, prompt: str) -> Optional[str]:
        if self.client is None or not self.api_key:
            return None
        if not self.rate_limiter.can_request(self.name):
            return None

        # Try up to len(models) rotations to find an eligible model. If all
        # are cooling down / too-recently-used, return None (factory moves on
        # to the next provider this cycle).
        model = self._next_model()
        if model is None:
            # No model eligible right now — don't burn the daily quota, just
            # signal "not ready" so the factory tries another provider.
            return None

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=90,
            )
            text = resp.choices[0].message.content
            if not text or not text.strip():
                return None
            return text
        except Exception as e:
            # Detect 429 specifically so we can skip just that model and try
            # the next one in the SAME generate() call (graceful 429 handling).
            ename = type(e).__name__
            msg = str(e)
            is_429 = ("429" in msg or "RateLimit" in ename
                      or "rate_limit" in msg.lower())
            if is_429:
                self._mark_429(model)
                print(f"[aihubmix] {model} rate-limited (429): {msg[:160]}")
                # Try the next eligible model once before giving up this cycle.
                alt = self._next_model()
                if alt is not None:
                    try:
                        resp = self.client.chat.completions.create(
                            model=alt,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            timeout=90,
                        )
                        text = resp.choices[0].message.content
                        if text and text.strip():
                            self._mark_used(alt)
                            return text
                    except Exception as e2:
                        ename2 = type(e2).__name__
                        msg2 = str(e2)
                        if ("429" in msg2 or "RateLimit" in ename2
                                or "rate_limit" in msg2.lower()):
                            self._mark_429(alt)
                        print(f"[aihubmix] {alt} fallback error: "
                              f"{ename2}: {msg2[:160]}")
                return None
            # Any other error (500, timeout, network, malformed body) — log
            # and return None. NEVER return an error body to the validator.
            print(f"[aihubmix] {model} error: {ename}: {msg[:200]}")
            return None
        finally:
            self._mark_used(model)
            self.rate_limiter.record(self.name)


class HFProvider:
    """HuggingFace Inference API via raw requests.post, model rotation.

    Uses the correct DNS: https://router.huggingface.co/models/{model}
    """

    name = "hf"
    BASE = "https://router.huggingface.co/models/{model}"

    def __init__(self, token: str, rate_limiter: RateLimiter):
        self.token = token
        self.rate_limiter = rate_limiter
        self._model_idx = 0
        self._idx_lock = threading.Lock()
        if not token:
            print("[hf] WARNING: no HF_TOKEN set")

    @property
    def available(self) -> bool:
        return bool(self.token) and self.rate_limiter.can_request(self.name)

    def _next_model(self) -> str:
        with self._idx_lock:
            m = HF_MODELS[self._model_idx % len(HF_MODELS)]
            self._model_idx += 1
            return m

    def generate(self, prompt: str) -> Optional[str]:
        if not self.token:
            return None
        if not self.rate_limiter.can_request(self.name):
            return None
        import requests

        model = self._next_model()
        url = self.BASE.format(model=model)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 503:
                # model loading — back off and retry once
                time.sleep(15)
                r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code != 200:
                # 429 / 401 / 500 / etc — log and return None.
                # NEVER pass r.text to the validator.
                print(f"[hf] {model} HTTP {r.status_code}: {r.text[:200]}")
                return None
            data = r.json()
            text = None
            if isinstance(data, list) and data:
                text = data[0].get("generated_text") or data[0].get("text")
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            if not text or not text.strip():
                return None
            return text
        except Exception as e:
            print(f"[hf] {model} error: {type(e).__name__}: {e}")
            return None
        finally:
            self.rate_limiter.record(self.name)


class OpenRouterProvider:
    """OpenRouter free models via OpenAI SDK."""

    name = "openrouter"
    BASE = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, rate_limiter: RateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.client = None
        if not api_key:
            print("[openrouter] WARNING: no OPENROUTER_API_KEY set")
            return
        try:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(
                base_url=self.BASE,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/bitqwen",
                    "X-Title": "BitQwen 1.0 Data Factory",
                },
            )
            print(f"[openrouter] initialized, {len(OPENROUTER_FREE_MODELS)} free models")
        except ImportError:
            print("[openrouter] ERROR: openai not installed; run pip install openai")
        except Exception as e:
            print(f"[openrouter] ERROR initializing: {e}")

    @property
    def available(self) -> bool:
        return self.client is not None and self.rate_limiter.can_request(self.name)

    def generate(self, prompt: str) -> Optional[str]:
        if not self.client:
            return None
        if not self.rate_limiter.can_request(self.name):
            return None
        model = random.choice(OPENROUTER_FREE_MODELS)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=90,
            )
            text = resp.choices[0].message.content
            if not text or not text.strip():
                return None
            return text
        except Exception as e:
            # Catch ALL: 429 quota, 401, timeout, etc. Never return error body.
            print(f"[openrouter] {model} error: {type(e).__name__}: {e}")
            return None
        finally:
            self.rate_limiter.record(self.name)


class TogetherProvider:
    """Together AI via OpenAI-compatible endpoint.

    API key is loaded from env (TOGETHER_API_KEY). Free-tier / budget models.
    """

    name = "together"
    BASE = "https://api.together.xyz/v1"

    def __init__(self, api_key: str, rate_limiter: RateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.client = None
        if not api_key:
            print("[together] WARNING: no TOGETHER_API_KEY set")
            return
        try:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(
                base_url=self.BASE,
                api_key=api_key,
            )
            print(f"[together] initialized, {len(TOGETHER_MODELS)} models")
        except ImportError:
            print("[together] ERROR: openai not installed; run pip install openai")
        except Exception as e:
            print(f"[together] ERROR initializing: {e}")

    @property
    def available(self) -> bool:
        return self.client is not None and self.rate_limiter.can_request(self.name)

    def generate(self, prompt: str) -> Optional[str]:
        if not self.client:
            return None
        if not self.rate_limiter.can_request(self.name):
            return None
        model = random.choice(TOGETHER_MODELS)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=90,
            )
            text = resp.choices[0].message.content
            if not text or not text.strip():
                return None
            return text
        except Exception as e:
            print(f"[together] {model} error: {type(e).__name__}: {e}")
            return None
        finally:
            self.rate_limiter.record(self.name)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

PROVIDER_SLEEP = {
    "litellm": LITELLM_SLEEP,
    "aihubmix": AIHUBMIX_SLEEP,
    "hf": HF_SLEEP,
    "openrouter": OPENROUTER_SLEEP,
    "together": TOGETHER_SLEEP,
}


class Factory:
    def __init__(self, output_dir: Path, providers: Dict[str, Any],
                 rate_limiter: RateLimiter,
                 cfg: Optional[Dict[str, Any]] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.providers = providers
        self.rate_limiter = rate_limiter
        self.cfg: Dict[str, Any] = cfg or {}
        # Provider weights / sleep come from config so users can tune without
        # code changes. Fall back to module-level PROVIDER_WEIGHTS / SLEEP.
        self.provider_weights: Dict[str, int] = dict(PROVIDER_WEIGHTS)
        self.provider_sleep: Dict[str, float] = dict(PROVIDER_SLEEP)
        prov_cfg = self.cfg.get("providers", {})
        for name, pcfg in prov_cfg.items():
            if "weight" in pcfg:
                self.provider_weights[name] = int(pcfg["weight"])
            if "sleep_sec" in pcfg:
                self.provider_sleep[name] = float(pcfg["sleep_sec"])
        self.progress_path = output_dir / ".progress.json"
        # daily counters (accepts only — separate from rate-limit usage)
        self.total_today = 0
        self.total_all = 0
        self.accepts_by_provider: Dict[str, int] = {n: 0 for n in providers}
        self.accepts_by_category: Dict[str, int] = {
            "func_gen": 0, "cot_debug": 0, "syllogism": 0,
        }
        self.day = _dt.date.today().isoformat()
        # Categorized seeds + category rotator. If config supplies category
        # fractions, use ConfigCategoryRotator (drives the exact split).
        self.categorized_seeds: Dict[str, List[str]] = {}
        cats_cfg = self.cfg.get("prompts", {}).get("categories", {})
        if cats_cfg:
            self.rotator: Any = ConfigCategoryRotator(cats_cfg)
        else:
            self.rotator = CategoryRotator()
        self._load_progress()

    # -- progress / resume ------------------------------------------------

    def _load_progress(self):
        if not self.progress_path.exists():
            return
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("day") == self.day:
                self.accepts_by_provider = {
                    k: int(v) for k, v in data.get("accepts_by_provider", {}).items()
                }
                self.accepts_by_provider.update(
                    {n: 0 for n in self.providers if n not in self.accepts_by_provider}
                )
                self.total_today = int(data.get("total_today", 0))
                cats = data.get("accepts_by_category", {})
                self.accepts_by_category = {
                    "func_gen": int(cats.get("func_gen", 0)),
                    "cot_debug": int(cats.get("cot_debug", 0)),
                    "syllogism": int(cats.get("syllogism", 0)),
                }
            self.total_all = int(data.get("total_all", 0))
            print(f"[resume] accepts={self.accepts_by_provider} "
                  f"total_today={self.total_today} total_all={self.total_all} "
                  f"categories={self.accepts_by_category}")
        except Exception as e:
            print(f"[resume] could not load progress: {e}")

    def _save_progress(self):
        try:
            tmp = self.progress_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "day": self.day,
                    "accepts_by_provider": self.accepts_by_provider,
                    "accepts_by_category": self.accepts_by_category,
                    "total_today": self.total_today,
                    "total_all": self.total_all,
                    "updated": _dt.datetime.now().isoformat(),
                }, f, indent=2)
            tmp.replace(self.progress_path)
        except Exception as e:
            print(f"[resume] could not save progress: {e}")

    # -- rollover ---------------------------------------------------------

    def _check_day_rollover(self):
        today = _dt.date.today().isoformat()
        if today != self.day:
            print(f"[rollover] {self.day} -> {today}, resetting daily counters")
            self.day = today
            self.accepts_by_provider = {n: 0 for n in self.providers}
            self.accepts_by_category = {
                "func_gen": 0, "cot_debug": 0, "syllogism": 0,
            }
            self.total_today = 0
            self._save_progress()
        self.rate_limiter.reset_if_new_day()

    # -- output file ------------------------------------------------------

    def _output_file(self) -> Path:
        return self.output_dir / f"generated_{self.day}.jsonl"

    # -- provider selection ----------------------------------------------

    def _budget_exhausted(self) -> bool:
        """True if the daily budget (paid mode) is exhausted.

        In free mode (daily_limit <= 0) the budget is never exhausted. In
        paid mode, once spent_today >= daily_limit we stop generating. The
        budget is tracked in factory_config.json and re-read each loop, so
        editing the JSON mid-run takes effect on the next iteration.
        """
        b = self.cfg.get("budget", {})
        limit = float(b.get("daily_limit", 0) or 0)
        if limit <= 0:
            return False
        spent = float(b.get("spent_today", 0.0) or 0.0)
        return spent >= limit

    def _pick_provider(self) -> Optional[str]:
        """Weighted pick among providers that are AVAILABLE.

        A provider is available iff `provider.available` is True, which
        requires BOTH:
            (a) the client initialised successfully, AND
            (b) rate_limiter.can_request(name) is True.

        This BLOCKS exhausted providers from being selected — the original
        bug was that usage was recorded but can_request was not consulted
        here, so exhausted providers kept being picked.

        In paid mode, returns None (signalling stop) once the daily budget
        is exhausted.
        """
        if self._budget_exhausted():
            print("[factory] daily budget exhausted (paid mode), stopping")
            return None
        candidates: List[Tuple[str, int]] = []
        for name, prov in self.providers.items():
            try:
                if not prov.available:
                    continue
            except Exception as e:
                print(f"[factory] {name}.available raised {e}; skipping")
                continue
            candidates.append((name, self.provider_weights.get(name, 1)))
        if not candidates:
            return None
        names, weights = zip(*candidates)
        return random.choices(names, weights=weights, k=1)[0]

    # -- daily summary ----------------------------------------------------

    def _print_daily_summary(self):
        print("=== daily summary ===")
        for prov, (used, cap) in self.rate_limiter.daily_summary().items():
            accepts = self.accepts_by_provider.get(prov, 0)
            cap_str = "unlimited" if cap <= 0 else str(cap)
            print(f"  Provider {prov}: {used}/{cap_str} requests used "
                  f"({accepts} accepted)")
        total_cats = sum(self.accepts_by_category.values())
        if total_cats > 0:
            fg = self.accepts_by_category["func_gen"]
            cd = self.accepts_by_category["cot_debug"]
            sy = self.accepts_by_category["syllogism"]
            print(f"  Categories: func_gen={fg} ({fg*100//total_cats}%) "
                  f"cot_debug={cd} ({cd*100//total_cats}%) "
                  f"syllogism={sy} ({sy*100//total_cats}%)")
        print(f"  Total accepted today: {self.total_today} | all-time: {self.total_all}")

    # -- sleep until all providers reset ----------------------------------

    def _sleep_until_reset(self):
        """All providers exhausted — sleep until next-day reset.

        Computes seconds until local midnight and sleeps that long (capped
        to 1 hour chunks so we can still respond to SIGTERM). This replaces
        the old fixed 10-minute sleep that burned cycles re-checking.
        """
        secs = self.rate_limiter.seconds_until_midnight()
        print(f"[factory] all providers exhausted, sleeping until midnight reset "
              f"(~{secs // 60} min)")
        self._print_daily_summary()
        slept = 0
        while slept < secs and not _SHUTDOWN:
            chunk = min(3600, secs - slept)
            time.sleep(chunk)
            slept += chunk

    # -- main loop --------------------------------------------------------

    def run(self, loop: bool, max_examples: Optional[int]):
        # Load seeds. Prefer the config's prompt_file (prompts/*.jsonl) if it
        # exists; otherwise fall back to the legacy categorized seed loader.
        # This lets users add prompt files WITHOUT touching the seed module.
        prompt_file = self.cfg.get("prompts", {}).get("prompt_file", "")
        if prompt_file:
            prompts = load_prompts(prompt_file)
        else:
            prompts = []
        if prompts:
            self.categorized_seeds = categorized_prompts(prompts)
            print(f"[factory] using prompt file {prompt_file} "
                  f"({len(prompts)} prompts)")
        else:
            # Fall back to the legacy categorized seed loader.
            self.categorized_seeds = load_categorized_seeds()
        all_seeds = (self.categorized_seeds.get("func_gen", [])
                     + self.categorized_seeds.get("cot_debug", [])
                     + self.categorized_seeds.get("syllogism", []))
        if not all_seeds:
            print("[fatal] no seeds loaded, exiting")
            return

        mode = self.cfg.get("mode", "free")
        budget = self.cfg.get("budget", {})
        print(f"[factory] mode={mode} budget_limit={budget.get('daily_limit', 0)} "
              f"currency={budget.get('currency', 'USD')}")
        print(f"[factory] starting: loop={loop} max={max_examples} providers="
              f"{[n for n, p in self.providers.items() if p.available]}")
        if not any(p.available for p in self.providers.values()):
            print("[fatal] no providers available, exiting")
            return

        last_save = time.time()
        last_log = time.time()
        last_provider = None
        start_total = self.total_all
        last_summary_day = self.day

        try:
            while not _SHUTDOWN:
                self._check_day_rollover()

                # print daily summary once per day (on rollover we already reset)
                if self.day != last_summary_day:
                    self._print_daily_summary()
                    last_summary_day = self.day

                if max_examples is not None and (self.total_all - start_total) >= max_examples:
                    print(f"[factory] reached max_examples={max_examples}, stopping")
                    break

                provider_name = self._pick_provider()
                if provider_name is None:
                    # All caps hit — sleep until tomorrow reset.
                    self._sleep_until_reset()
                    continue

                provider = self.providers[provider_name]
                # Category-aware seed selection: rotate 60/25/15 across
                # func_gen / cot_debug / syllogism.
                category = self.rotator.next()
                cat_seeds = self.categorized_seeds.get(category) or all_seeds
                seed = random.choice(cat_seeds)

                # Provider.generate() records the request against the rate
                # limiter in a finally block (success OR failure counts).
                # On ANY error it returns None — we never validate error bodies.
                try:
                    response = provider.generate(seed)
                except Exception as e:
                    # Defensive: a provider should never raise, but if it
                    # does, log and treat as a None response.
                    print(f"[factory] {provider_name}.generate raised "
                          f"{type(e).__name__}: {e}")
                    response = None

                if response is None:
                    # Provider error or empty — skip to next provider.
                    # NEVER pass an error body to the validator.
                    print(f"[skip] {provider_name} returned no response, "
                          f"trying next provider next loop")
                elif validate_response(response):
                    text = CHATML_TEMPLATE.format(seed=seed, response=response.strip())
                    rec = {
                        "text": text,
                        "provider": provider_name,
                        "category": category,
                        "ts": _dt.datetime.now().isoformat(),
                    }
                    try:
                        with open(self._output_file(), "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        self.total_today += 1
                        self.total_all += 1
                        self.accepts_by_provider[provider_name] = \
                            self.accepts_by_provider.get(provider_name, 0) + 1
                        self.accepts_by_category[category] = \
                            self.accepts_by_category.get(category, 0) + 1
                        last_provider = provider_name
                    except Exception as e:
                        print(f"[write] error: {e}")
                else:
                    print(f"[validate] rejected response from {provider_name}")

                # Progress log every 10 examples today
                if self.total_today > 0 and self.total_today % 10 == 0 and time.time() - last_log > 5:
                    last_log = time.time()
                    summary = self.rate_limiter.daily_summary()
                    remain = {n: ("inf" if cap <= 0 else max(0, cap - used))
                              for n, (used, cap) in summary.items()}
                    print(f"[progress] today={self.total_today} total={self.total_all} "
                          f"last={last_provider} usage={ {n: u for n,(u,_) in summary.items()} } "
                          f"remain={remain}")

                # Save progress periodically
                if time.time() - last_save > 30:
                    last_save = time.time()
                    self._save_progress()

                if not loop and max_examples is not None and (self.total_all - start_total) >= max_examples:
                    break

                time.sleep(self.provider_sleep.get(provider_name, 1.0))
        finally:
            self._save_progress()
            self._print_daily_summary()
            print(f"[factory] shutdown complete. today={self.total_today} total={self.total_all}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Test double for provider self-test (not a real network provider)."""

    def __init__(self, name: str, response: Optional[str],
                 rate_limiter: RateLimiter, fail_available: bool = False):
        self.name = name
        self._response = response
        self.rate_limiter = rate_limiter
        self._fail_available = fail_available

    @property
    def available(self) -> bool:
        if self._fail_available:
            return False
        return self.rate_limiter.can_request(self.name)

    def generate(self, prompt: str) -> Optional[str]:
        if not self.rate_limiter.can_request(self.name):
            return None
        try:
            return self._response
        finally:
            self.rate_limiter.record(self.name)


def self_test():
    """Offline self-test of validation, seed loading, RateLimiter logic,
    provider error handling, and provider selection.

    Does NOT make real network calls — provider error handling is tested
    via a stub that raises like a real provider would.
    """
    print("=== self-test ===")
    import tempfile
    global _SHUTDOWN  # so sub-tests can set/reset the shutdown flag

    seeds = load_seeds()
    assert seeds, "no seeds loaded"

    # validation cases
    good_code = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
    good_long = "x" * 250
    bad_short = "no"
    assert validate_response(good_code), "good code should validate"
    assert validate_response(good_long), "long response should validate"
    assert not validate_response(bad_short), "short should reject"
    print("[self-test] validation OK")

    # ChatML format
    text = CHATML_TEMPLATE.format(seed="test seed", response="def f():\n    pass")
    assert "<|im_start|>user" in text and "<|im_end|>" in text
    print("[self-test] ChatML OK")

    # extract_code
    assert "def add" in extract_code(good_code)
    print("[self-test] extract_code OK")

    # ---- RateLimiter logic ----
    with tempfile.TemporaryDirectory() as tmp:
        usage_path = Path(tmp) / "usage.json"
        caps = {"litellm": 0, "hf": 5, "openrouter": 1, "together": 500}
        rl = RateLimiter(usage_path, caps=caps)

        # Initially all providers can request (litellm is unlimited)
        for prov in caps:
            assert rl.can_request(prov), f"{prov} should be able to request initially"
        print("[self-test] RateLimiter initial can_request OK")

        # litellm cap=0 means unlimited — never blocked by usage
        for _ in range(100):
            rl.record("litellm")
        assert rl.can_request("litellm"), "litellm (cap=0) should never be blocked"
        assert rl.remaining("litellm") == -1, "unlimited remaining should be -1"
        print("[self-test] RateLimiter unlimited (litellm) OK")

        # Record until cap hit for openrouter (cap=1)
        assert rl.can_request("openrouter")
        rl.record("openrouter")
        assert not rl.can_request("openrouter"), "openrouter should be blocked after 1 request"
        print("[self-test] RateLimiter cap enforcement OK")

        # remaining math
        assert rl.remaining("hf") == 5
        rl.record("hf")
        assert rl.remaining("hf") == 4
        print("[self-test] RateLimiter remaining OK")

        # together: 500 cap
        assert rl.remaining("together") == 500
        rl.record("together")
        assert rl.remaining("together") == 499
        print("[self-test] RateLimiter together cap OK")

        # Persistence: reload and verify counts survive
        rl2 = RateLimiter(usage_path, caps=caps)
        assert rl2.get_count("openrouter") == 1, \
            f"expected 1, got {rl2.get_count('openrouter')}"
        assert rl2.get_count("litellm") == 100
        assert rl2.get_count("hf") == 1
        assert not rl2.can_request("openrouter")
        assert rl2.can_request("litellm"), "litellm should still be unlimited after reload"
        assert rl2.can_request("hf")
        print("[self-test] RateLimiter persistence OK")

        # daily_summary shape
        summary = rl2.daily_summary()
        assert "openrouter" in summary
        used, cap = summary["openrouter"]
        assert used == 1 and cap == 1
        used_l, cap_l = summary["litellm"]
        assert cap_l == 0, "litellm cap should be 0 (unlimited)"
        print("[self-test] RateLimiter daily_summary OK")

        # seconds_until_midnight is sane (between 60s and 25h)
        secs = rl2.seconds_until_midnight()
        assert 60 <= secs <= 90000, f"seconds_until_midnight out of range: {secs}"
        print(f"[self-test] seconds_until_midnight OK ({secs}s)")

        # ---- _pick_provider blocks exhausted providers ----
        # Build a factory with fake providers and verify that when
        # openrouter is exhausted it is never picked, while unlimited
        # litellm always remains eligible.
        factory_dir = Path(tmp) / "factory"
        factory_dir.mkdir()
        rl3 = RateLimiter(factory_dir / "usage.json",
                          caps={"litellm": 0, "hf": 2, "openrouter": 1, "together": 500})
        providers = {
            "litellm": _FakeProvider("litellm", "def f(): pass", rl3),
            "hf": _FakeProvider("hf", "def g(): pass", rl3),
            "openrouter": _FakeProvider("openrouter", "def h(): pass", rl3),
            "together": _FakeProvider("together", "def i(): pass", rl3),
        }
        f = Factory(factory_dir, providers, rl3)

        # Exhaust openrouter
        rl3.record("openrouter")
        assert not rl3.can_request("openrouter")
        # _pick_provider should NEVER return openrouter now
        for _ in range(50):
            picked = f._pick_provider()
            assert picked is not None, "should still have providers"
            assert picked != "openrouter", \
                "exhausted openrouter must never be picked"
        print("[self-test] _pick_provider blocks exhausted provider OK")

        # Exhaust hf too
        rl3.record("hf")
        rl3.record("hf")
        assert not rl3.can_request("hf")
        for _ in range(50):
            picked = f._pick_provider()
            assert picked is not None, "litellm+together still available"
            assert picked not in ("openrouter", "hf"), \
                "exhausted providers must never be picked"
        print("[self-test] _pick_provider blocks multiple exhausted OK")

        # Exhaust together too — only litellm (unlimited) should remain
        for _ in range(500):
            rl3.record("together")
        assert not rl3.can_request("together")
        for _ in range(20):
            picked = f._pick_provider()
            assert picked == "litellm", \
                f"only litellm should remain, got {picked}"
        print("[self-test] _pick_provider falls back to unlimited litellm OK")

        # If a provider's .available raises, _pick_provider skips it
        class _BrokenProvider:
            name = "broken"
            @property
            def available(self):
                raise RuntimeError("boom")
            def generate(self, prompt):
                return None
        providers["broken"] = _BrokenProvider()
        picked = f._pick_provider()  # should not raise
        assert picked == "litellm", f"broken provider should be skipped, got {picked}"
        print("[self-test] _pick_provider skips broken provider OK")

        # ---- Provider error handling: returns None on exception ----
        # A real provider catches ALL exceptions internally (429, 500,
        # timeout, protobuf-JSON error bodies) and returns None — it NEVER
        # returns the error body and NEVER raises. This stub mirrors that
        # contract exactly.
        class _RaisingProvider:
            name = "raising"
            def __init__(self, rl):
                self.rate_limiter = rl
            @property
            def available(self):
                return self.rate_limiter.can_request("raising")
            def generate(self, prompt):
                try:
                    # Simulate a 429 protobuf-JSON error body from the API.
                    raise ConnectionError("429 quota exceeded (protobuf body)")
                except Exception as e:
                    print(f"[raising] caught {type(e).__name__}: {e}")
                    return None
                finally:
                    self.rate_limiter.record("raising")

        rl4 = RateLimiter(factory_dir / "usage2.json",
                          caps={"raising": 5})
        rp = _RaisingProvider(rl4)
        # generate() must return None on exception (not raise, not return body)
        result = rp.generate("test prompt")
        assert result is None, "raising provider must return None, not raise"
        # and the request was still recorded against the quota
        assert rl4.get_count("raising") == 1
        print("[self-test] provider returns None on exception OK")

        # ---- Factory defensive: catches a provider that (incorrectly) raises ----
        # Even if a buggy provider breaks the contract and raises, the
        # factory's run() loop wraps generate() in try/except and treats it
        # as a None response — it must not crash. We use a tiny cap so the
        # provider exhausts quickly, and patch _sleep_until_reset to a no-op
        # so the test doesn't block until midnight.
        class _UncaughtRaisingProvider:
            name = "uncaught"
            def __init__(self, rl):
                self.rate_limiter = rl
            @property
            def available(self):
                return self.rate_limiter.can_request("uncaught")
            def generate(self, prompt):
                # Deliberately does NOT catch — violates the contract.
                # The factory must defend against this.
                self.rate_limiter.record("uncaught")
                raise RuntimeError("provider bug: uncaught exception")

        uncaught_dir = Path(tmp) / "uncaught_factory"
        uncaught_dir.mkdir()
        rl4b = RateLimiter(uncaught_dir / "usage.json",
                           caps={"uncaught": 3})
        uncaught_providers = {"uncaught": _UncaughtRaisingProvider(rl4b)}
        uf = Factory(uncaught_dir, uncaught_providers, rl4b)
        # Patch out the long sleep and signal shutdown so the run loop exits
        # promptly once the provider is exhausted (instead of blocking until
        # midnight).
        def _test_sleep_exit():
            global _SHUTDOWN
            _SHUTDOWN = True
        uf._sleep_until_reset = _test_sleep_exit  # type: ignore
        uf.run(loop=False, max_examples=3)
        _SHUTDOWN = False  # reset for subsequent sub-tests
        assert uf.total_all == 0, "uncaught-raising provider must yield 0 accepts"
        assert rl4b.get_count("uncaught") == 3, "should have tried 3 times then exhausted"
        print("[self-test] factory catches uncaught provider exception OK")

        # ---- Factory treats None response as skip (no validation) ----
        # A provider that always returns None should produce 0 accepts but
        # still record usage and not crash.
        skip_dir = Path(tmp) / "skip_factory"
        skip_dir.mkdir()
        rl5 = RateLimiter(skip_dir / "usage.json",
                          caps={"noneprov": 10})
        none_providers = {"noneprov": _FakeProvider("noneprov", None, rl5)}
        sf = Factory(skip_dir, none_providers, rl5)
        # Patch out the long sleep and signal shutdown so the run loop exits
        # promptly once the provider is exhausted (instead of blocking until
        # midnight).
        def _test_sleep_exit2():
            global _SHUTDOWN
            _SHUTDOWN = True
        sf._sleep_until_reset = _test_sleep_exit2  # type: ignore
        # Run a tiny batch
        sf.run(loop=False, max_examples=3)
        _SHUTDOWN = False  # reset for the thread-safety sub-test
        assert sf.total_all == 0, "None responses must never be accepted"
        assert rl5.get_count("noneprov") >= 3, "requests should still be recorded"
        # No output file should have been written
        out_files = list(skip_dir.glob("generated_*.jsonl"))
        assert not out_files, "no output file should be created for all-None responses"
        print("[self-test] factory skips None responses (no validation) OK")

        # ---- Thread safety smoke test ----
        rl6 = RateLimiter(usage_path, caps={"hf": 100000})
        n_threads = 8
        n_per_thread = 500
        threads = []
        for _ in range(n_threads):
            t = threading.Thread(
                target=lambda: [rl6.record("hf") for _ in range(n_per_thread)]
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        expected = n_threads * n_per_thread
        actual = rl6.get_count("hf")
        # today's count from earlier rl.record('hf') (1) + this run
        assert actual == 1 + expected, \
            f"thread-safety FAILED: expected {1 + expected}, got {actual}"
        print(f"[self-test] RateLimiter thread-safety OK ({actual} concurrent records)")

        # ---- CategoryRotator: 60/25/15 split ----
        rot = CategoryRotator()
        counts = {"func_gen": 0, "cot_debug": 0, "syllogism": 0}
        N = 200  # 10 full 20-call cycles
        for _ in range(N):
            cat = rot.next()
            assert cat in counts, f"unexpected category {cat!r}"
            counts[cat] += 1
        # Over 200 calls: func_gen=120 (60%), cot_debug=50 (25%), syllogism=30 (15%)
        assert counts["func_gen"] == 120, f"func_gen expected 120, got {counts['func_gen']}"
        assert counts["cot_debug"] == 50, f"cot_debug expected 50, got {counts['cot_debug']}"
        assert counts["syllogism"] == 30, f"syllogism expected 30, got {counts['syllogism']}"
        print(f"[self-test] CategoryRotator 60/25/15 split OK ({counts})")

        # ---- Categorized seeds load + non-empty ----
        cat_seeds = load_categorized_seeds()
        for c in ("func_gen", "cot_debug", "syllogism"):
            assert cat_seeds[c], f"category {c} should have seeds"
        print(f"[self-test] categorized seeds OK "
              f"(func_gen={len(cat_seeds['func_gen'])} "
              f"cot_debug={len(cat_seeds['cot_debug'])} "
              f"syllogism={len(cat_seeds['syllogism'])})")

        # ---- AIHubMixProvider: offline construction + model rotation ----
        # No API key => client is None, available is False, generate returns None.
        rl_ah = RateLimiter(Path(tmp) / "ah_usage.json",
                            caps={"aihubmix": 7500})
        ah = AIHubMixProvider("", rl_ah)
        assert ah.client is None, "no-key AIHubMix should have no client"
        assert not ah.available, "no-key AIHubMix should be unavailable"
        assert ah.generate("test") is None, "no-key generate must return None"
        # With a key but no real network (we don't call generate), verify the
        # model-rotation logic picks distinct models across calls.
        ah2 = AIHubMixProvider("sk-fake-key-for-rotation-test", rl_ah)
        if ah2.client is not None:
            picked = set()
            for _ in range(len(AIHUBMIX_FREE_MODELS)):
                m = ah2._next_model()
                if m is not None:
                    picked.add(m)
                    ah2._mark_used(m)
            # Should have picked multiple distinct models across rotations.
            assert len(picked) >= 2, f"rotation should pick >1 model, got {picked}"
        print("[self-test] AIHubMixProvider offline construction + rotation OK")

        # ---- 15 free models present ----
        assert len(AIHUBMIX_FREE_MODELS) == 15, \
            f"expected 15 AIHubMix free models, got {len(AIHUBMIX_FREE_MODELS)}"
        assert "coding-glm-5.2-free" in AIHUBMIX_FREE_MODELS
        assert "laguna-xs-2.1-free" in AIHUBMIX_FREE_MODELS
        print(f"[self-test] AIHubMix 15 free models OK ({len(AIHUBMIX_FREE_MODELS)})")

        # ---- Free-only: Together not in DEFAULT_CAPS / PROVIDER_WEIGHTS ----
        assert "together" not in RateLimiter.DEFAULT_CAPS, \
            "together must be removed from DEFAULT_CAPS in free-only mode"
        assert "together" not in PROVIDER_WEIGHTS, \
            "together must be removed from PROVIDER_WEIGHTS in free-only mode"
        assert "aihubmix" in RateLimiter.DEFAULT_CAPS
        assert "aihubmix" in PROVIDER_WEIGHTS
        print("[self-test] free-only provider config OK")

        # ---- Config load/save round-trip ----
        cfg = load_factory_config()
        assert cfg["mode"] in ("free", "paid"), f"unexpected mode {cfg['mode']!r}"
        assert "providers" in cfg and "aihubmix" in cfg["providers"]
        assert "prompts" in cfg and "categories" in cfg["prompts"]
        assert "budget" in cfg and "output" in cfg
        # save/load round-trip preserves content
        tmp_cfg = Path(tmp) / "factory_config_test.json"
        save_factory_config(cfg, tmp_cfg)
        cfg2 = load_factory_config(tmp_cfg)
        assert cfg2["mode"] == cfg["mode"]
        assert cfg2["providers"]["aihubmix"]["enabled"] == cfg["providers"]["aihubmix"]["enabled"]
        print("[self-test] config load/save round-trip OK")

        # _deep_merge fills missing keys from defaults
        partial = {"mode": "paid", "budget": {"daily_limit": 5.0}}
        merged = _deep_merge(DEFAULT_FACTORY_CONFIG, partial)
        assert merged["mode"] == "paid"
        assert merged["budget"]["daily_limit"] == 5.0
        assert "providers" in merged  # filled from default
        assert merged["budget"]["currency"] == "USD"  # default preserved
        print("[self-test] _deep_merge fills defaults OK")

        # ---- Prompt file loading ----
        # The shipped prompts/ dir should exist with 6 files, each 20+ lines.
        assert PROMPTS_DIR.exists(), f"prompts dir missing: {PROMPTS_DIR}"
        files = list_prompt_files()
        assert len(files) >= 3, f"expected >=3 prompt files, got {len(files)}"
        expected = {"default.jsonl", "pocketflow.jsonl", "dspy.jsonl",
                    "litellm.jsonl", "ast_ir.jsonl", "bug_injection.jsonl"}
        names = {f.name for f in files}
        missing = expected - names
        assert not missing, f"missing prompt files: {missing}"
        for f in files:
            n = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
            assert n >= 20, f"{f.name} has {n} prompts, expected >=20"
        print(f"[self-test] prompt files OK ({len(files)} files, all >=20 prompts)")

        # load_prompts parses JSONL correctly
        default_prompts = load_prompts("prompts/default.jsonl")
        assert len(default_prompts) >= 20
        for pr in default_prompts:
            assert "prompt" in pr and pr["prompt"]
            assert pr["category"] in ("func_gen", "cot_debug", "syllogism")
        print(f"[self-test] load_prompts OK ({len(default_prompts)} parsed)")

        # categorized_prompts groups + falls back
        cats = categorized_prompts(default_prompts)
        for c in ("func_gen", "cot_debug", "syllogism"):
            assert cats[c], f"category {c} should not be empty"
        print(f"[self-test] categorized_prompts OK "
              f"(fg={len(cats['func_gen'])} cd={len(cats['cot_debug'])} "
              f"sy={len(cats['syllogism'])})")

        # missing file returns []
        assert load_prompts("prompts/__nonexistent__.jsonl") == []
        print("[self-test] load_prompts missing file -> [] OK")

        # ---- ConfigCategoryRotator honors fractions ----
        rot = ConfigCategoryRotator({"func_gen": 0.6, "cot_debug": 0.25, "syllogism": 0.15})
        counts = {"func_gen": 0, "cot_debug": 0, "syllogism": 0}
        N = 1000  # 10 full 100-step cycles
        for _ in range(N):
            cat = rot.next()
            assert cat in counts, f"unexpected category {cat!r}"
            counts[cat] += 1
        # Allow 1% tolerance (1000 calls, pattern is 100 steps => exact multiples)
        assert abs(counts["func_gen"] - 600) <= 10, f"func_gen: {counts['func_gen']}"
        assert abs(counts["cot_debug"] - 250) <= 10, f"cot_debug: {counts['cot_debug']}"
        assert abs(counts["syllogism"] - 150) <= 10, f"syllogism: {counts['syllogism']}"
        print(f"[self-test] ConfigCategoryRotator fractions OK ({counts})")

        # Custom fractions work too
        rot2 = ConfigCategoryRotator({"func_gen": 0.5, "cot_debug": 0.5})
        c2 = {"func_gen": 0, "cot_debug": 0}
        for _ in range(100):
            c2[rot2.next()] += 1
        assert c2["func_gen"] == 50 and c2["cot_debug"] == 50
        print(f"[self-test] ConfigCategoryRotator 50/50 OK ({c2})")

        # ---- Factory honors config (weights + sleep + budget) ----
        fcfg = {
            "providers": {
                "litellm": {"weight": 99, "sleep_sec": 0.01},
                "aihubmix": {"weight": 1, "sleep_sec": 0.01},
            },
            "budget": {"daily_limit": 0, "spent_today": 0.0},
            "prompts": {"categories": {"func_gen": 1.0}},
        }
        fcfg_dir = Path(tmp) / "fcfg_factory"
        fcfg_dir.mkdir()
        rl_f = RateLimiter(fcfg_dir / "usage.json", caps={"litellm": 0, "aihubmix": 0})
        f_provs = {
            "litellm": _FakeProvider("litellm", "def f(): pass", rl_f),
            "aihubmix": _FakeProvider("aihubmix", "def g(): pass", rl_f),
        }
        ff = Factory(fcfg_dir, f_provs, rl_f, cfg=fcfg)
        assert ff.provider_weights["litellm"] == 99
        assert ff.provider_sleep["litellm"] == 0.01
        # budget not exhausted (limit 0 = unlimited)
        assert not ff._budget_exhausted()
        # weighted pick favors litellm (weight 99 vs 1). With random.choices
        # aihubmix can still be picked ~1% of the time, so we check that
        # litellm is the overwhelming majority, not the exclusive pick.
        ll_picks = sum(1 for _ in range(50) if ff._pick_provider() == "litellm")
        assert ll_picks >= 45, f"weight 99 should pick litellm ~99%, got {ll_picks}/50"
        print(f"[self-test] Factory config weights/sleep/budget OK (litellm {ll_picks}/50)")

        # budget exhaustion stops picks
        fcfg_paid = {
            "providers": {"litellm": {"weight": 1, "sleep_sec": 0.01}},
            "budget": {"daily_limit": 1.0, "spent_today": 2.0},  # already over
            "prompts": {"categories": {"func_gen": 1.0}},
        }
        rl_p = RateLimiter(fcfg_dir / "usage_p.json", caps={"litellm": 0})
        f_paid = Factory(fcfg_dir, {"litellm": _FakeProvider("litellm", "x", rl_p)}, rl_p,
                         cfg=fcfg_paid)
        assert f_paid._budget_exhausted()
        assert f_paid._pick_provider() is None, "budget exhausted => None"
        print("[self-test] Factory budget exhaustion stops picks OK")

        # ---- Factory uses prompt file when configured ----
        # Build a tiny prompt file and confirm Factory.run loads it.
        pf = Path(tmp) / "prompts"
        pf.mkdir()
        (pf / "tiny.jsonl").write_text(
            '{"prompt": "do func thing", "category": "func_gen", "difficulty": "beginner"}\n'
            '{"prompt": "debug this", "category": "cot_debug", "difficulty": "intermediate"}\n',
            encoding="utf-8",
        )
        pf_cfg = {
            "prompts": {"prompt_file": str(pf / "tiny.jsonl"),
                        "categories": {"func_gen": 0.5, "cot_debug": 0.5}},
            "providers": {"tinyprov": {"weight": 1, "sleep_sec": 0.0}},
            "budget": {"daily_limit": 0, "spent_today": 0.0},
        }
        pf_dir = Path(tmp) / "pf_factory"
        pf_dir.mkdir()
        rl_t = RateLimiter(pf_dir / "usage.json", caps={"tinyprov": 3})
        # Fake provider returns a valid code response so the factory accepts it.
        valid_resp = "Here is the code:\n```python\ndef f():\n    return 1\n```"
        tp_provs = {"tinyprov": _FakeProvider("tinyprov", valid_resp, rl_t)}
        tf = Factory(pf_dir, tp_provs, rl_t, cfg=pf_cfg)
        def _exit_sleep():
            global _SHUTDOWN
            _SHUTDOWN = True
        tf._sleep_until_reset = _exit_sleep  # type: ignore
        tf.run(loop=False, max_examples=2)
        _SHUTDOWN = False
        assert tf.total_all >= 1, "prompt-file factory should accept >=1"
        out_files = list(pf_dir.glob("generated_*.jsonl"))
        assert out_files, "output file should be created from prompt-file run"
        print("[self-test] Factory uses prompt file OK")

    print("=== self-test PASSED ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_provider_keys() -> Dict[str, str]:
    """Resolve API keys for all providers from env (no hardcoding)."""
    return {
        "aihubmix": os.environ.get("AIHUBMIX_API_KEY") or "",
        "litellm": get_litellm_key() or os.environ.get("LITELLM_MASTER_KEY") or "",
        "openrouter": get_openrouter_key() or os.environ.get("OPENROUTER_API_KEY") or "",
        "hf": get_hf_token() or os.environ.get("HF_TOKEN") or "",
    }


def _build_providers(cfg: Dict[str, Any], rate_limiter: RateLimiter
                     ) -> Dict[str, Any]:
    """Instantiate providers based on config + env keys.

    Only providers that are enabled in config AND have a configured client
    class are instantiated. A provider with no key still constructs (its
    `available` will be False), preserving the original behavior.
    """
    keys = _resolve_provider_keys()
    pcfg = cfg.get("providers", {})
    providers: Dict[str, Any] = {}

    def _enabled(name: str) -> bool:
        return bool(pcfg.get(name, {}).get("enabled", True))

    if _enabled("aihubmix"):
        base = pcfg.get("aihubmix", {}).get("base_url", AIHUBMIX_BASE_URL)
        providers["aihubmix"] = AIHubMixProvider(keys["aihubmix"], rate_limiter,
                                                 base_url=base)
    if _enabled("litellm"):
        base = pcfg.get("litellm", {}).get("base_url", LITELLM_BASE_URL)
        providers["litellm"] = LiteLLMProvider(base, keys["litellm"], rate_limiter)
    if _enabled("openrouter"):
        providers["openrouter"] = OpenRouterProvider(keys["openrouter"], rate_limiter)
    if _enabled("hf"):
        providers["hf"] = HFProvider(keys["hf"], rate_limiter)
    return providers


def cmd_status(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Show factory status: mode, providers, prompts, budget, output, progress."""
    print("=" * 70)
    print("BitQwen Data Factory — STATUS")
    print("=" * 70)
    print(f"config : {FACTORY_CONFIG_PATH}")
    print(f"mode   : {cfg.get('mode', 'free')!r}")
    print()
    print("Providers:")
    for name, pcfg in cfg.get("providers", {}).items():
        en = "on" if pcfg.get("enabled", True) else "off"
        models = pcfg.get("models", [])
        key_env = pcfg.get("key_env", "")
        key_set = bool(os.environ.get(key_env)) if key_env else False
        cap = pcfg.get("daily_cap", "?")
        w = pcfg.get("weight", "?")
        print(f"  {name:12s} en={en:3s}  weight={w}  cap={cap}  "
              f"models={len(models)}  key={'yes' if key_set else 'no'}")
    print()
    print("Prompts:")
    pcfg = cfg.get("prompts", {})
    print(f"  prompt_file : {pcfg.get('prompt_file', '(none)')}")
    cats = pcfg.get("categories", {})
    print(f"  categories  : {cats}")
    files = list_prompt_files()
    print(f"  available   : {len(files)} files in prompts/")
    for f in files:
        n = sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        print(f"    - {f.name}  ({n} prompts)")
    print()
    b = cfg.get("budget", {})
    print(f"Budget: {b.get('daily_limit', 0)} {b.get('currency', 'USD')} "
          f"(spent {b.get('spent_today', 0.0)})")
    print(f"Output: {cfg.get('output', {})}")
    print(f"Rate limits: {cfg.get('rate_limits', {})}")
    print()
    # Progress / usage files
    out_dir = Path(cfg.get("output", {}).get("dir", "/data"))
    usage = out_dir / "usage.json"
    prog = out_dir / ".progress.json"
    if usage.exists():
        try:
            u = json.loads(usage.read_text(encoding="utf-8"))
            days = u.get("days", {})
            today = _dt.date.today().isoformat()
            today_u = days.get(today, {})
            print(f"Usage today ({today}): {today_u}")
        except Exception as e:
            print(f"usage.json: could not parse: {e}")
    if prog.exists():
        try:
            p = json.loads(prog.read_text(encoding="utf-8"))
            print(f"Progress: today={p.get('total_today', 0)} "
                  f"all={p.get('total_all', 0)} day={p.get('day')}")
        except Exception as e:
            print(f".progress.json: could not parse: {e}")
    print("=" * 70)
    return 0


def cmd_list_prompts(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """List available prompt files in prompts/."""
    files = list_prompt_files()
    if not files:
        print("(no prompt files found in prompts/)")
        return 0
    active = cfg.get("prompts", {}).get("prompt_file", "")
    print(f"{'file':28s} {'prompts':7s} {'active':6s}")
    print("-" * 45)
    for f in files:
        n = sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        rel = f.relative_to(HERE).as_posix()
        is_active = "  *" if rel == active else ""
        print(f"{f.name:28s} {n:<7d} {is_active}")
    print()
    print(f"Active prompt file (from config): {active or '(none)'}")
    print("Use --prompts <file> to switch the active prompt file.")
    return 0


def cmd_add_prompts(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Register a prompt file in config (and validate it loads)."""
    pf = args.add_prompts
    p = Path(pf)
    if not p.is_absolute():
        p = HERE / p
    if not p.exists():
        sys.stderr.write(f"[add-prompts] file not found: {p}\n")
        return 1
    prompts = load_prompts(pf)
    if not prompts:
        sys.stderr.write(f"[add-prompts] no valid prompts parsed from {p}\n")
        return 1
    rel = p.relative_to(HERE).as_posix() if HERE in p.parents or p.parent == HERE else pf
    cfg.setdefault("prompts", {})["prompt_file"] = rel
    save_factory_config(cfg)
    print(f"Registered {len(prompts)} prompts from {rel}")
    print(f"Active prompt file is now: {rel}")
    print(f"Config saved to {FACTORY_CONFIG_PATH}")
    return 0


def cmd_balance(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Set the category balance (func_gen, cot_debug, syllogism fractions)."""
    parts = [float(x) for x in args.balance.split(",")]
    if len(parts) != 3:
        sys.stderr.write("[balance] expected 3 comma-separated fractions "
                         "(func_gen,cot_debug,syllogism)\n")
        return 1
    fg, cd, sy = parts
    total = fg + cd + sy
    if total <= 0:
        sys.stderr.write("[balance] fractions must sum to > 0\n")
        return 1
    cfg.setdefault("prompts", {})["categories"] = {
        "func_gen": fg / total,
        "cot_debug": cd / total,
        "syllogism": sy / total,
    }
    save_factory_config(cfg)
    cats = cfg["prompts"]["categories"]
    print(f"Category balance set: func_gen={cats['func_gen']:.3f} "
          f"cot_debug={cats['cot_debug']:.3f} syllogism={cats['syllogism']:.3f}")
    print(f"Config saved to {FACTORY_CONFIG_PATH}")
    return 0


def cmd_set_mode(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Set the generation mode (free|paid)."""
    mode = args.mode
    if mode not in ("free", "paid"):
        sys.stderr.write(f"[mode] unknown mode: {mode!r} (use free|paid)\n")
        return 1
    cfg["mode"] = mode
    save_factory_config(cfg)
    print(f"Mode set to: {mode!r}")
    if mode == "paid":
        b = cfg.get("budget", {})
        print(f"  daily_limit: {b.get('daily_limit', 0)} {b.get('currency', 'USD')}")
        print("  Use --budget N to set the daily spend cap.")
    return 0


def cmd_set_budget(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """Set the daily budget cap (USD). 0 = unlimited (free mode)."""
    cfg.setdefault("budget", {})["daily_limit"] = float(args.budget)
    cfg["budget"].setdefault("currency", "USD")
    cfg["budget"].setdefault("spent_today", 0.0)
    save_factory_config(cfg)
    print(f"Budget set: {args.budget} {cfg['budget']['currency']} per day")
    if float(args.budget) > 0:
        print("  Mode is now 'paid' (set --mode free to override).")
        cfg["mode"] = "paid"
        save_factory_config(cfg)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # Support Windows-style /? help in addition to --help.
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] in ("/?", "-?"):
        argv = ["--help"]

    ap = argparse.ArgumentParser(
        prog="generate.py",
        description=(
            "BitQwen 1.0 background data factory — a config-driven CLI that "
            "generates Python SFT examples using free/paid LLM providers. "
            "Reads factory_config.json on startup and every loop iteration, "
            "so you can switch modes, prompt files, and budgets without "
            "touching the code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate.py --loop                      # run forever with config\n"
            "  python generate.py --config cfg.json --loop    # use a custom config\n"
            "  python generate.py --prompts prompts/dspy.jsonl --loop\n"
            "  python generate.py --mode free --loop\n"
            "  python generate.py --mode paid --budget 5.00 --loop\n"
            "  python generate.py --max-examples 1000         # generate N then exit\n"
            "  python generate.py --status\n"
            "  python generate.py --list-prompts\n"
            "  python generate.py --add-prompts prompts/custom.jsonl\n"
            "  python generate.py --balance 0.5,0.3,0.2\n"
            "  python generate.py --self-test\n"
        ),
    )
    ap.add_argument("--loop", action="store_true",
                    help="run forever (24/7). If omitted, --max-examples is required "
                         "or the factory runs once.")
    ap.add_argument("--output", default=None,
                    help="output directory (overrides config output.dir; default /data)")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="stop after N new examples")
    ap.add_argument("--config", default=None,
                    help="path to factory_config.json (default: ./factory_config.json)")
    ap.add_argument("--prompts", default=None,
                    help="path to a prompt .jsonl file (persists to config)")
    ap.add_argument("--mode", default=None, choices=["free", "paid"],
                    help="set generation mode (free|paid) and persist to config")
    ap.add_argument("--budget", type=float, default=None,
                    help="set daily spend cap in USD (0 = unlimited); implies paid mode")
    ap.add_argument("--status", action="store_true",
                    help="show factory status (mode, providers, prompts, budget, progress)")
    ap.add_argument("--list-prompts", action="store_true",
                    help="list available prompt files in prompts/")
    ap.add_argument("--add-prompts", default=None, metavar="FILE",
                    help="register a prompt file in config and validate it loads")
    ap.add_argument("--balance", default=None, metavar="F,C,S",
                    help="set category balance as func_gen,cot_debug,syllogism fractions "
                         "(e.g. 0.5,0.3,0.2)")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline self-test and exit")

    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    cfg_path = Path(args.config) if args.config else None
    cfg = load_factory_config(cfg_path)

    # ---- Mutating commands (edit config, then exit) ----
    if args.status:
        return cmd_status(cfg, args)
    if args.list_prompts:
        return cmd_list_prompts(cfg, args)
    if args.add_prompts:
        return cmd_add_prompts(cfg, args)
    if args.balance:
        return cmd_balance(cfg, args)
    if args.mode:
        return cmd_set_mode(cfg, args)
    if args.budget is not None:
        return cmd_set_budget(cfg, args)

    # ---- Generation run ----
    # CLI overrides for prompt file (does NOT persist unless --add-prompts).
    if args.prompts:
        cfg.setdefault("prompts", {})["prompt_file"] = args.prompts
        print(f"[factory] CLI override: prompt_file = {args.prompts}")

    output_dir = Path(args.output or cfg.get("output", {}).get("dir", "/data"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rate limiter — persists to {output_dir}/usage.json
    rate_limiter = RateLimiter(output_dir / "usage.json")

    providers = _build_providers(cfg, rate_limiter)

    mode = cfg.get("mode", "free")
    print(f"[factory] mode={mode} | providers available: "
          f"{[n for n, p in providers.items() if p.available]}")
    if not any(p.available for p in providers.values()):
        print("[fatal] no providers available. Check API keys in .env or "
              "factory_config.json provider.enabled flags. Exiting.")
        return 1

    factory = Factory(output_dir=output_dir, providers=providers,
                      rate_limiter=rate_limiter, cfg=cfg)
    factory.run(loop=args.loop, max_examples=args.max_examples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
