# Open-model servers (lab pool, vLLM 0.30 in active-audit/.venv-rl, launched with FLASHINFER_DISABLE_VERSION_CHECK=1 and the venv bin on PATH)

| host | served name | checkpoint | ctx | notes |
|---|---|---|---|---|
| cherry | qwen35-9b | RedHatAI/Qwen3.5-9B-quantized.w4a16 | 16384 | 32k OOMs on 16 GB; 0.95 util, max-num-seqs 4; coder test: 8k-token prompt in 7.0 s, correct SEARCH/REPLACE |
| walnut | qwen35-4b | cyankiwi/Qwen3.5-4B-AWQ-4bit | 32768 | coder test: 4.3 s, correct |
| willow | r1-qwen3-8b | stelterlab/DeepSeek-R1-0528-Qwen3-8B-AWQ | 32768 | reasoning parser deepseek_r1; needs max_tokens above 1500 or the answer is eaten by thinking; voter or analyst only |
| pine | gemma4-12b | google/gemma-4-12B-it-qat-w4a16-ct | 16384 | up 01:25, 14.5 GB used, JSON constrained ok; gpt-oss-20b OOMed on this card and is excluded |

Structured output: response_format json_schema works and constrains; structured_outputs json also works; guided_json is ignored by this server version. Thinking off for Qwen3.5 via chat_template_kwargs enable_thinking false.
Logs: ~/vllm_logs/<host>_*.log. Restart recipe in this file's history and in loop/models/README.md.
