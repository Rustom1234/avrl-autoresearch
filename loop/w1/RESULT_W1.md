# W1 result: a reader model's attention and hidden states predict experiment outcomes inside a task, not across tasks

Nov 14 2026. Code: `loop/w1/` (build_data.py, extract.py, analyze.py). Full tables: `loop/w1/report.txt`.
Not preregistered; predictions were written in a working note before the analysis ran.

## Question

Before a proposed edit is given a GPU run, can a small open model that reads the edit tell how the run will end,
from its attention pattern or its hidden state? If so, a loop could skip weak proposals or hand them to a stronger
coder.

## Data and method

2,978 proposed edits that reached a GPU in earlier loops: 2,763 on the GPT task (written by Sonnet, Haiku, Gemma
12B, Qwen 4B) and 215 on the CIFAR task (Sonnet, Haiku), each with its real outcome. Two outcomes: crash (4 percent
on the GPT task) and promising (first run at least 0.75 noise units better than the current best, 12 percent).

The reader, Qwen3-4B, reads the task, the current file, the idea and the edit as a diff. Signals:

- **free**: what a loop has without any model (edit type, idea family, which coder wrote it, size of the diff,
  whether the diff touches the batch-size line);
- **verdict**: the reader's own yes-or-no answer to "will this run succeed and score better";
- **attention**: for the edit's tokens, how each head splits its attention between task text, file, idea and the
  edit itself, plus attention entropy (the Lookback Lens recipe);
- **hidden**: the reader's hidden states.

A linear classifier per signal. Score: AUROC, the chance that a random positive case is ranked above a random
negative one (0.5 is guessing, 1.0 is perfect). In task: whole loops held out. Across tasks: trained on the GPT
task, tested on CIFAR.

## Result

| Signal | Crash, in task | Crash, GPT to CIFAR | Promising, in task | Promising, GPT to CIFAR |
|---|---|---|---|---|
| free | 0.85 | 0.81 (0.65 to 0.91) | 0.87 | 0.64 (0.54 to 0.75) |
| verdict | 0.75 | 0.51 (0.30 to 0.71) | 0.66 | 0.67 (0.58 to 0.77) |
| attention | 0.91 | 0.42 (0.32 to 0.54) | 0.91 | 0.73 (0.60 to 0.85) |
| hidden | 0.92 | 0.50 (0.38 to 0.68) | 0.94 | 0.50 (0.37 to 0.62) |

Replay on the GPT task, dropping the lowest-scored third of proposals before the GPU:

| Score used | GPU runs saved | Promising proposals lost | Crashes avoided |
|---|---|---|---|
| free | 33% | 2% | 48% |
| attention | 33% | 3% | 44% |
| hidden | 33% | 2% | 51% |
| random | 33% | 33% | 33% |

## Reading

- **Inside one task the internal signals are strong** (0.91 to 0.94) and beat the free signals by 0.05 to 0.07.
- **They do not carry over.** Hidden states fall to guessing on CIFAR for both outcomes. They learned which edits win
  on the GPT task (above all the batch-size cut), not a general sign of a good experiment.
- **Attention shares are the one signal that keeps something across tasks for "promising"** (0.73, interval 0.60 to
  0.85), but that rests on 17 positive CIFAR cases, and for crashes attention transfers worst of all. Not a result yet.
- **For routing, the internals add nothing over the free signals**: both let a loop skip a third of its GPU runs
  and lose 2 to 3 percent of the promising proposals. The useful finding is the filter itself, and it needs only
  logged outcomes from the same task.
- Predictions before the run: crash 0.70 to 0.80 from hidden states (actual 0.92), promising 0.60 to 0.70 in task
  (actual 0.94), about 0.55 across tasks (actual 0.50), attention no better than hidden in task (held), verdict near
  chance (it was somewhat better, 0.66 to 0.75).

## Limits

One reader model, one training task, one transfer task with 17 and 13 positive cases. The free signals include the
batch-size flag, which is knowledge of the answer on this task. Prompts were rebuilt from logs, not the exact ones
the coders saw. Diffs were cut at 1,500 tokens in four shards and 700 in one (after a GPU memory error), which
touches few rows. The replay is on held-out predictions but not a live loop.

## Shared machines

One shard stopped with a GPU out-of-memory error inside PyTorch on walnut and was restarted once by hand after the
extractor's memory use was cut. The first run of the analysis script reached 24 GB of host memory on spruce because
of a loading bug, and failed quietly at its own address-space limit; no kernel kill. Fixed, measured on one shard
(65 MB), rerun at 1.3 GB.
