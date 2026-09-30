FROM python:3.11-slim

LABEL org.opencontainers.image.title="BitQwen 1.0 Background Data Factory"
LABEL org.opencontainers.image.source="https://github.com/bitqwen"

# Install Python deps.
#   - openai:     LiteLLM proxy + OpenRouter + Together (OpenAI-compatible)
#   - requests:   raw HTTP for HuggingFace Inference API
#   - tenacity:   retry/backoff for transient errors
#
# NOTE: google-generativeai has been REMOVED — the Gemini providers were the
# source of the validation bug (429 protobuf-JSON error bodies passed to the
# validator). The factory now uses LiteLLM (local) as primary, with HF /
# OpenRouter / Together as free fallbacks.
RUN pip install --no-cache-dir \
        "openai>=1.51.0,<2.0.0" \
        requests==2.32.3 \
        tenacity==8.5.0

WORKDIR /app

# Copy generator + config + prompts. env_config.py (parent scripts/ module)
# is NOT copied because it lives outside the build context; generate.py has a
# built-in fallback that reads os.environ directly, and docker-compose.yml
# passes all credentials in as environment variables.
#
# NOTE: seeds/ contains the legacy categorized seed module (python_seed_prompts.py
# + seeds_for_generation.jsonl). generate.py falls back to the prompt file
# configured in factory_config.json (prompts/default.jsonl) when seeds are
# absent, but seeds/ is included so the self-test passes and the categorized
# seed loader is available as a fallback.
#
# factory_config.json and prompts/ are copied into the image as defaults.
# docker-compose.yml bind-mounts over them so host edits take effect without
# a rebuild, but the image is also self-contained (useful for `docker run`
# without compose).
COPY generate.py /app/generate.py
COPY factory_config.json /app/factory_config.json
COPY seeds/ /app/seeds/
COPY prompts/ /app/prompts/

# Output is written to /data (mounted volume) so generated JSONL persists
# across container restarts.
VOLUME ["/data"]

# Run forever in loop mode, reading factory_config.json on startup and every
# loop iteration. To inspect/steer the running container:
#   docker compose exec bitqwen-data-factory python generate.py --status
#   docker compose exec bitqwen-data-factory python generate.py --mode paid --budget 5
CMD ["python", "/app/generate.py", "--loop", "--output", "/data"]
