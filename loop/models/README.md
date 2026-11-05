# loop/models: uniform model-call layer

One function, `call()`, for Claude tiers (Claude Code CLI, headless) and open-weight models (vLLM OpenAI-compatible
server). Every call returns a `CallResult` with validated JSON, token usage, cost, latency, and appends a JSONL audit line
per attempt. Stdlib only; `jsonschema` is used if importable (it is not installed in the active-audit venv, so the
built-in minimal validator is what runs there).

## Files

| file | what |
|---|---|
| `models.json` | registry (list of entries). Add an entry, no code change. |
| `call.py` | `call()`, `healthy()`, `CallResult`, `ContextTooLong`, `load_registry()`, `validate()` |
| `consistency.py` | `sample_k()` self-consistency sampling |
| `cost.py` | `summarize(log_path)` and `python cost.py [calls.jsonl]` table |
| `budget.py` | `tokens_used(log_path)`, `over_ceiling(log_path, ceiling_tokens)` for the orchestrator |
| `smoke_real.py` | one real haiku call, plus one qwen7b-awq call if cherry answers within 5 s |
| `tests/` | pytest, no network, no real CLI, no real sleeping (`_run_claude`, `_http_post`, `_http_get` are stubbed by an autouse fixture; breaker and lock files go to a tmp dir; `time.sleep` over 2 s fails the test) |

## Usage

```python
import sys; sys.path.insert(0, "/homes/rdubash/active-audit/loop/models")
from call import call, healthy, ContextTooLong
from consistency import sample_k
from cost import summarize
from budget import tokens_used, over_ceiling

schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
r = call("haiku", "Classify: ...", system="You are a strict labeler.", schema=schema, tag="judge:round3")
if r.ok:
    r.obj["label"]; r.usage; r.cost_usd_plan; r.latency_s
else:
    r.error, r.text            # raw text of the last attempt

s = sample_k("qwen7b-awq", "Classify: ...", 5, key="label", schema=schema, temperature=0.8, tag="judge:sc")
s["agreement"], s["modal_value"], s["n_ok"], s["values"], s["results"]

summarize("calls.jsonl")      # {"total", "by_served_model", "by_requested_model", "by_tag", "fallbacks", "bad_lines"}
healthy("qwen35-9b")          # GET {endpoint}/v1/models, served id present; call() never calls this, the caller decides
tokens_used("calls.jsonl")    # {"claude": {"input", "output", "cache_read", "cache_write"}, "vllm": {...}}
over_ceiling("calls.jsonl", 50_000_000)   # claude tokens (all four kinds) > ceiling
```

Run the tests: `/homes/rdubash/active-audit/.venv/bin/python -m pytest -q tests` (from this directory).
Run the smoke test from a scratch directory (it writes `smoke_calls.jsonl` in cwd):
`/homes/rdubash/active-audit/.venv/bin/python /homes/rdubash/active-audit/loop/models/smoke_real.py`.

## CallResult

`obj, text, ok, error, model_name, model_id, provider, usage{input,output,cache_read,cache_write}, cost_usd_plan,
cost_usd_nominal, latency_s, attempts, requested_model, served_model, error_class`.

- `usage`, both costs, `latency_s` and `attempts` are summed over all tries of the logical call, fallback hops included
  (retries cost money). The log has the per-try numbers. `attempts` is how many tries were made.
- `requested_model` is what the caller asked for, `served_model` the registry name that produced this result (they differ
  after a fallback hop). `model_name`, `model_id` and `provider` describe the served model, so `model_name == served_model`.
- `error_class`: `None` when ok; `"infra"` when the last failure was infrastructure (connection refused, timeout, HTTP
  5xx or 404, Claude CLI not emitting JSON or reporting a 5xx, Claude rate limit still present after its retries); `"model"` for
  anything the model did (schema violation, unparseable or empty reply, other CLI errors). `"context"` is what a
  `ContextTooLong` log line carries; `call()` still raises that exception, so a `CallResult` never has it.
- `ok=False` after the last failed attempt: `obj=None`, `text` is the raw text of the last attempt, `error` says why.
- `cost_usd_plan` is the CLI `total_cost_usd` for claude and `0.0` for vllm. `cost_usd_nominal` equals the plan cost for
  claude and `tokens x nominal_price_per_m / 1e6` for vllm.
- Retried (up to `attempts` tries per model): CLI error / `is_error`, non-JSON CLI output, timeouts and any exception
  (connection refused, missing binary), HTTP errors, `stop_reason == "tool_use"` with no `structured_output`, unparseable
  JSON, schema violations. Not retried: vllm context overflow, which raises `ContextTooLong` (the attempt is logged
  first). No backoff between these attempts.

## Fallback chains

Each registry entry has `fallback: <model_name or null>`. `call()` resolves the chain up front (requested, its fallback,
that one's fallback: at most 2 hops; a repeat of a model already in the chain ends it; an unknown fallback name raises
`KeyError` before anything is called). Every model in the chain gets its own `attempts` tries. The next model is tried only
when the last failure of the current one has `error_class == "infra"`. A schema or other model failure, and
`ContextTooLong`, never hop. Every log line carries `requested_model`, `served_model` (equal to `model_name`) and
`error_class`; failed infra lines are logged under the model that failed, the fallback's lines under the fallback.

| requested | chain |
|---|---|
| `qwen35-9b` | haiku, sonnet |
| `qwen35-4b`, `r1-qwen3-8b`, `gemma4-12b`, `gptoss-20b`, `qwen7b-awq` | haiku |
| `haiku` | sonnet |
| `opus` | sonnet |
| `sonnet` | none |

`disabled: true` (on `gptoss-20b`, which does not fit the 16 GB card, and on `qwen7b-awq`) is advisory: `healthy()` returns
False for such entries without any network call, `call()` does not enforce it.

`healthy(model_name, timeout=3)`: True for claude entries; for vllm `GET {endpoint}/v1/models` must answer 200 with
`model_id` among `data[].id`; any exception, other status or missing id is False. Unknown names raise `KeyError`.

## Claude rate limits, breaker, concurrency slots

Applies to claude entries only (vllm calls ignore all of it).

- **Detection**: `is_error` true and the result text, stderr or `api_error_status` match (case-insensitive) one of: rate
  limit (also `rate_limit`, `rate-limit`), usage limit, limit reached, overloaded, 429, 529 (whole numbers only), too many
  requests, quota. A non-JSON reply with a non-zero exit code is checked on stderr and stdout the same way. A successful
  answer that merely mentions these words is not a rate limit.
- **Breaker**: on detection the call writes `{"until_epoch", "reason"}` to `$AR_BREAKER_FILE` (default
  `/tmp/claude-22830/-homes-rdubash/2d8b70a3-2fb9-42a6-aac4-20bd12ab9c6d/scratchpad/claude_breaker.json`; atomic replace; an
  existing later `until_epoch` is never shortened; if the file cannot be written the call just sleeps in-process). Before
  every Claude try, and again right after a slot is acquired, a call sleeps while `until_epoch` is in the future, polling
  every `BREAKER_POLL_S` (10) s. A missing or unparseable file means no wait.
- **Backoff**: retry i (0-based) blocks for `min(1800, 60 * 2**i)` s scaled by a random factor in [0.5, 1.0]: at most 60,
  120, 240, 480, 960, 1800, 1800, 1800 s. At most `RL_RETRIES` (8) retries per `call()`, shared across fallback hops, and
  they do not use up `attempts`. After the 8th retry fails, the model ends with `error_class="infra"`; a hop to the
  next Claude model in the chain then gets one try (it still waits out the breaker first).
- **Slots**: `_claude_slot()` is a context manager over `claude_slot_0..N-1.lock` files in `$AR_LOCK_DIR` (default: the
  scratchpad directory above) using non-blocking `flock`, N = `$AR_CLAUDE_MAX_CONCURRENT` (default 4). It takes the first free
  lock, otherwise polls every `SLOT_POLL_S` (0.5) s with no timeout. It wraps only the breaker re-check and the `claude -p`
  run, so a call never holds a slot while it backs off. `latency_s` includes time spent waiting for a slot.
- Locks and the breaker are plain local files: they coordinate processes and threads on one host. That is enough here
  because all loop processes run on one host (spruce). Other lab hosts have their own `/tmp`; a second host would need
  `AR_BREAKER_FILE` and `AR_LOCK_DIR` on a shared filesystem (and a check that `flock` works there).

## Observed Claude CLI output (claude 2.1.301, 2026-10-21, real haiku calls)

Command: `claude -p --model haiku --no-session-persistence --tools "" --strict-mcp-config --setting-sources ""
--output-format json [--json-schema '<json>'] [--system-prompt '<text>']`, prompt on stdin.

Top-level keys: `type, subtype, is_error, api_error_status, result, stop_reason, terminal_reason, total_cost_usd,
usage, modelUsage, num_turns, session_id, uuid, duration_ms, duration_api_ms, ttft_ms, ttft_stream_ms,
time_to_request_ms, first_content_frame_ms, queued_turn_count, result_index, permission_denials, safety_stops,
fast_mode_state, fast_mode_disabled_reason, subagent_stats`.

- With `--json-schema` there is one extra key: **`structured_output`** (the parsed object, e.g. `{"answer": 4}`).
  `result` also holds the same JSON as a string. Without `--json-schema` there is no `structured_output`.
- `usage` keys used: `input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens` (also
  `output_tokens_details, server_tool_use, service_tier, cache_creation{ephemeral_1h/5m}, iterations, speed, ...`).
  Our `usage.input` is `input_tokens` only, which excludes cache tokens; cache tokens are reported separately.
- `modelUsage` is keyed by model id (`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`) with `inputTokens,
  outputTokens, cacheReadInputTokens, cacheCreationInputTokens, costUSD, contextWindow (1000000), maxOutputTokens (128000),
  ...`. That is where the `max_context_tokens` values in `models.json` come from.
- **`stop_reason` is `"tool_use"` on every successful `--json-schema` call** (num_turns 2: the structured answer is
  delivered through a tool call), and `"end_turn"` without a schema. So the "tool_use means failure" rule is applied only
  when `structured_output` is missing. A tool_use reply without it is retried.
- Observed cost: haiku with schema 0.0002 to 0.0009 (the default CLI system prompt is written to cache, 1100 to 4100
  tokens, on a first call; a custom `--system-prompt` replaces it), haiku/sonnet/opus plain trivial calls 0.0006 / 0.0090 /
  0.0179.

## vLLM

`call(..., extra={...})` merges a dict into the request body last (so it can add or override fields) for vllm models and is
ignored for claude. Use it for per-model switches, e.g. `extra={"chat_template_kwargs": {"enable_thinking": False}}` on
Qwen3.5 or `extra={"reasoning_effort": "low"}` on gpt-oss. It is not stored in the registry and not written to the log.

`POST {endpoint}/v1/chat/completions`. Structured output first as `response_format: {"type":"json_schema",
"json_schema":{"name": <schema title or "response">, "schema": ...}}`. A 4xx reply whose body mentions unknown / unrecognized /
extra / not permitted / unsupported / unexpected triggers one immediate retry with `structured_outputs: {"json": schema}`;
if that works, `structured_outputs` is remembered per endpoint for the rest of the process. Top-level `guided_json` is
never sent: a live test on cherry (vLLM 0.30, Qwen2.5-7B-AWQ, reported by the coordinator) showed `response_format`
json_schema and `structured_outputs` both constrain the output, while `guided_json` is accepted but ignored (the model
answered with prose plus a fenced block). A 400 mentioning "context length" or
"maximum context" raises `ContextTooLong`. Reply parsing: `choices[0].message.content`, `usage.prompt_tokens`,
`usage.completion_tokens`.

## Assumptions and limits

- **My own build never reached a live vLLM server** (cherry:8000 refused connections), so the vllm path is covered by
  stubbed tests only. The structured-output behaviour above comes from the coordinator's live test of cherry. The
  error-body shapes (OpenAI style `{"error": {"message": ...}}` and FastAPI style
  `{"detail": [...]}`) are matched by substring search over the whole body.
- Registry entries `qwen35-9b` (cherry:8000, which it shares with `qwen7b-awq`, so only one is served at a time),
  `gptoss-20b` (pine:8000, disabled), `gemma4-12b` (pine:8000), `qwen35-4b` (walnut:8000) and `r1-qwen3-8b` (willow:8000)
  were added with the endpoints and prices the coordinator gave; their endpoints are placeholders to be confirmed, and the
  model_id is the same as the name. Nominal prices per M tokens: 0.10/0.30 for the 7B to 9B class, 0.15/0.60 gpt-oss-20b and
  gemma4-12b, 0.05/0.20 qwen35-4b. `qwen7b-awq` is kept but disabled, with fallback haiku (the task did not list a chain for it).
- `max_context_tokens: 8192` for every vllm entry is a guess (`qwen7b-awq` included) (the deploy flags were not available). Check `max_model_len` in
  `GET /v1/models` and edit `models.json`. The field is informational; nothing in the code enforces it.
- Claude: `temperature` and `max_tokens` are ignored (the CLI exposes neither), so Claude samples cannot be made
  deterministic or length-capped from here. For vllm, `temperature=None` means the server default.
- `--json-schema` is assumed to want an object at the top level (it is delivered as tool input). Prompt goes on stdin;
  `system` and the schema travel in argv, so each must stay under about 128 KB (an `E2BIG` shows up as a failed attempt).
- Claude calls inherit the user's login and run with tools disabled, MCP and settings sources off. `--bare` breaks login
  and is not used. Each Claude call is a fresh subprocess (about 1 to 2 s overhead for haiku).
- `sample_k` runs the k calls sequentially. `agreement` is the share of the modal value among calls that were ok and
  contained `key` (`n_ok`), so failed calls shrink the denominator; check `n_ok`. Ties go to the first value seen. Wrap in
  threads if you need parallelism (log appends are single `write` calls, safe for threads and local processes; over NFS
  from several hosts, use one `log_path` per host or process to avoid interleaving).
- The minimal validator checks `type` (including type lists, bool is not an integer), `enum`, `required`, nested
  `properties` and `items`. Nothing else (`pattern`, `minimum`, `oneOf`, `$ref`, ...); install `jsonschema` to get those.
- `cost.py` counts one log line as one attempt, so its `attempts` column is attempts, not logical calls. It groups by
  `served_model` (also exposed as `by_model`) and by `requested_model` (the tokens and dollars of fallback attempts count
  under the requested model there), and `fallbacks` counts attempts whose served model differs from the requested one,
  in total and per `requested->served` route. Old log lines without the new fields count under `model_name` for both.
- `budget.over_ceiling` is strict (`>`), counts input + output + cache_read + cache_write of one provider (default
  `claude`, the shared login; vllm tokens are free), and treats a missing log as zero.
- vllm HTTP 404 (a different model is loaded on the shared `cherry:8000`) and any 5xx (503 included) are
  `error_class="infra"` and hop; other 4xx (400, 422, ...) are `"model"` and do not.
