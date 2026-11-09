Incumbent mean_adjusted is 1.1146 over 16 runs. None of the 5 slots so far beat it, so all were rejected.

Did not help:
- LR warmup hurt. A 5% warmup scored 1.1246 (+0.0100) and a 10% warmup scored 1.1234 (+0.0088). With a fixed 300 s budget, warmup wastes training time. Drop it.
- HEAD_DIM 160 was much worse at 1.1463 (+0.0317). Widening the model is probably too costly in steps. Avoid larger models.
- U-Net skip connections scored 1.1175 (+0.0030), slightly worse.

Closest to the incumbent: global grad-norm clipping at 0.5 scored 1.1120 (-0.0026). It was rejected, but it was the only change that scored better than the incumbent. That gap is small and may be noise, so a replicate or a looser clip threshold is worth testing.

Next ideas:
- Use a smaller or more efficient model to get more steps.
- Tune the batch size.
- Tune the LR, weight decay and warmdown schedule.
- Run a replicate to estimate noise before trusting small deltas.
