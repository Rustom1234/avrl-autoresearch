Incumbent is slot 1 at 1.0935. Its fix of the gradient-accumulation crash was the only large gain (-0.0211). Small tweaks did not clear the bar: slot 4 scored 1.0924 (-0.0011), slot 9 scored 1.0928 (-0.0007) and slot 16 scored 1.0930 (-0.0005). All three were rejected, but all three beat the incumbent slightly. That points to a small real gain or to noise, so replicate one before building on it.

Schedule changes consistently hurt (1 accepted, 6 rejected). Cosine scored +0.0742, polynomial decay +0.0107, symmetric LR +0.0052, early linear decay +0.0269 and warmup +0.0116. Stop changing schedules.

Depth 24 hit OOM twice, so don't retry it. Several optimizer and clipping ideas crashed or were aborted because their specs were vague or the edits failed to patch.

Next steps: replicate slot 4. Then change one thing per slot, with a concrete minimal edit. Try learning rate, weight decay or total batch size, then a moderate DEPTH of 10 to 12 with a smaller device batch.
