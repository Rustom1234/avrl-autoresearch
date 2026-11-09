# Model cost per loop version

Claude at list price from the CLI; open-model calls estimated per token at hosted prices (OpenRouter public model list, https://openrouter.ai/api/v1/models, read 2026-10-30).

| arm | runs | Claude USD/run | open-model USD/run (est.) | total USD/run | calls/run | open calls/run | accepts | total USD per accepted change |
|---|---|---|---|---|---|---|---|---|
| A | 8 | 0.779 | 0.000 | 0.779 | 13.9 | 0.0 | 17 | 0.367 |
| B | 3 | 1.063 | 0.000 | 1.063 | 18.0 | 0.0 | 6 | 0.531 |
| C | 3 | 1.442 | 0.000 | 1.442 | 176.0 | 0.0 | 5 | 0.865 |
| D | 8 | 0.216 | 0.027 | 0.243 | 160.6 | 145.9 | 18 | 0.108 |

Whole run: Claude 15.48 USD for 5,749,144 tokens; open models 2,043,105 tokens, 0.21 USD at hosted per-token prices.
Open-model GPUs busy 0.46 hours: 0.08 to 0.12 USD if rented for that time.
Renting the open-model GPUs for the whole 13.8-hour run window, idle time included: 9.38 to 13.80 USD.

Calibration tryouts: Claude 2.89 USD; open models 0.39 USD at hosted per-token prices (0.37 to 0.55 USD as GPU rental for busy time).
