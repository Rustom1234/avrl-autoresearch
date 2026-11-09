Incumbent mean_adjusted is 1.1146 over 16 runs. Nothing has clearly beaten it yet.

Helped: nothing clearly. Slot 19 scored 1.1118 (-0.0028), the best result. It was a fix-up of the crashed Layered Adam run (slot 18), so the likely cause is the optimizer group change. It was still rejected, so replicate it once. Near-neutral and within noise: GQA with half the KV heads (1.1142, -0.0004), batch layer norm on the input (1.1140, -0.0005), and the shared Q/K/V bottleneck (1.1145, -0.0001).

Hurt: SwiGLU (+0.0071), the value conv (+0.0296), per-block gains (+0.0018), the extra residual block (+0.0017), reciprocal grad scaling (+0.1813) and the TDT warmup (+1.1161).

Many proposals were aborted, and two patches failed to apply.

Next: replicate slot 19, then tune learning rates, the schedule, batch size and model size. Copy SEARCH blocks exactly from the file.
