# Cluster launch scripts
Shell and Slurm scripts that launched the runs whose results are in `avrl/runs/`. Each fixes the
model, data, arms, seeds and budget of one run or queue, and gates on a free GPU or a predecessor flag.

Grouped by filename prefix:
- `e*`: early experiments (E05 to E10: rollout studies, judge-reward task, code RL with OLMo and Qwen)
- `m*`: allocator comparisons (M06 to M15: threshold, greedy, learned and reference arms)
- `p0*`, `p1*`: rollout collection and GRPO training runs
- `s1*`, `launch_s1_*`, `run_s1_*`: the 7B study (training on hannah, cross-family judge checks)
- `reeval*`: re-scoring saved adapters on the full test sets
- `timing*`, `smoke*`: step-time, memory and smoke checks

Machine names in filenames (goldberg, cherry, spruce, hannah, 5090) are the lab hosts the script ran on.
Paths assume the repo root as working directory (`bash avrl/launch/<script>.sh`). The `*_hannah` scripts
run from `~/avrl` on the cluster, a copy of this repo's `avrl/` directory.
