Helped: seeded multi-start repulsion plus SLSQP polish (slot 0, -2.6343), then better initialisers: Lloyd on a dense grid (slot 2, -2.6359) and jittered hex rows (slot 8, -2.6360, incumbent). That beats the 2.635 target.

Did not help: population evolution (slot 1, -2.6317), staged radii-power objectives, corner-anchored and edge-ring inits (-2.6353), 5x5 grid init (-2.6343), power-diagram Lloyd, symmetric parametrisation. Re-polish, Newton, barrier, basin-hopping, differential evolution, iterated local search, screening, swap moves and margin-shrink variants all stayed at -2.6360. Since slot 8 nearly every slot has been flat (about 140 slots), so the basin looks saturated. Slots 3 and 8 each aborted once.

Next: run one replicate to check repeatability. Then try only a genuinely different topology or search objective, not more polish, screening, init or margin variants. Propose real edits, not no-op changes.
