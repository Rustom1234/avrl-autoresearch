Incumbent is slot 18 at -2.6360, measured once, so noise is unchecked. That is 0.001 below the best known 2.635.

Helped: seeded multi-start joint SLSQP (slot 0, -2.6057). Radius-scaled repulsion relaxation (slot 2, -0.0160). Mirror-symmetric search about x=0.5 (slot 7, -0.0081). Basin hopping (slot 17, -0.0045). Fixed count of 12 starts (slot 18, -0.0017).

Did not help: from slot 19 on, every try was unchanged or worse (best case +0.0000). That covered new start families, alternative polishes (LP, penalty, augmented-Lagrangian, trust-region), global searches (genetic, DE, ES) and basin-hopping variants. Many others were aborted as near-duplicates.

Next: replicate slot 18 once to check noise. Avoid new phases unless clearly different from those tried. The result is probably near the limit, so consider stopping and keeping slot 18.
