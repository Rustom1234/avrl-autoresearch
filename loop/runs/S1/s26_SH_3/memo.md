Incumbent is slot 38 at -2.6360, past the 2.635 target. Slot 136 tied it at -2.6360 and was accepted, but it did not improve the score. Roughly 100 slots since slot 38 have not beaten it, so this is a plateau.

Helped: seeded multi-start SLSQP (-2.6130), jittered hex-row init (-2.6261), SA over centres (-2.6304), and SA swap plus smallest-circle relocation moves (-2.6360).

Did not help: other inits (ring, radial, corner, big-centre); penalty pre-stages; slower or reheated SA; parallel tempering (-1.1282); differential evolution; more starts; polish steps (all unchanged at -2.6304 or -2.6360). Many -0.0000 results, e.g. basin hopping in slots 41 and 50, probably mean timeouts or invalid packings.

Next: stop proposing repeats, since many were aborted. Try only untested small tweaks, such as other relocation fractions, and keep runs under 4 minutes with the safe radius shrink.
