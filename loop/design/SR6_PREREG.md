# SR6: can a "check one part" tool undo the hiding? Rules written before any session

Written 2026-11-20 16:05, before any model session of this stage. Earlier stage: `loop/RESULT_SR5.md`.

## Why

SR5 showed that Haiku finds each term of a level 3 law when it is the only unknown one (S 1.00, CS 0.92, U 0.46),
but almost never when all three are unknown (0.04; 0.39 to 0.46 predicted from the parts). The oracle hints of SR5
remove the hiding by telling terms. This stage asks whether a tool can remove some of it with no hint about the
law and no strong model: a tool that lets Haiku confirm one term before it knows the others.

## The tool

`evaluate(code, free)` in `loop/sr/iso/tools_server.py`, scored by `loop/sr/iso/fit_part.py`. With `free` empty
it is the old `evaluate`. With `free` = a list of inputs, the candidate is fitted together with one free smooth
curve (cubic spline) of each named input. The curves are projected out, so only the candidate's constants are
searched. Returned: share = squares left by candidate plus curves, divided by squares left by the curves alone
(1 = the part adds nothing), and a verdict.

- Verdict rule: the fit is done with coarse curves (20 inner knots) and fine curves (40). The part is called
  right if the fine share is below 1e-9, or below 1e-4 and less than a third of the coarse share. Reason: a fixed
  cutoff cannot separate a right part (limited only by the curves, so it keeps falling as they get finer) from a
  near miss (it stalls). Offline, right parts reached 2.4e-6 and wrong ones 5.7e-7 at 40 knots.
- Constants: the 8 starting points of `fit.py` plus 16 more with magnitudes from 0.03 to 10; a fixed cap on model
  evaluations bounds the cost of a call without a clock. The complete-law check keeps its 8 points.
- A part is logged to `parts.jsonl`, never to `evals.jsonl`, so it can never be recorded as the attempt's law.
- What it can separate: the product term from all single-input terms (every input in `free`); then a single-input
  term from terms on other inputs. What it cannot: two single-input terms on the same input (S and U share an
  input in 4 of the 12 level 3 problems), and two product terms (level 4, not tested here).

Offline check with the true laws, no model call (`scratchpad/verify.py`): on level 3, 24 of 24 right parts
called right, 226 of 226 wrong parts (one factor or one unusual kind changed) called not right, 2 forms equal up
to a single-input term called right. The verdict rule and the extra starting points were set on these level 3
problems (and one level 6 failure showed the need for more starting points). Then on levels 2 and 6, which were
not used to set the cutoffs: 48 of 48 right parts called right; all wrong parts called not right except forms
that are the same part (saturation against hyperbola, equal up to a constant; one product with its factors
swapped). One part check costs 1 to 2 s of one core and 79 MB.

## Known work

Nothing here is new as a method. Free curves beside a parametric part: partially linear and semi-parametric
regression (Engle et al. 1986; Robinson 1988) and additive models (Hastie and Tibshirani 1986). Projecting out
the linear part: variable projection (Golub and Pereyra 1973). Splitting a law into separable parts before
searching: AI Feynman (Udrescu and Tegmark 2020), Luo et al. 2017. Residual-guided term search with language
models: RISR and L2G-SR (2026). These citations are from memory and were not rechecked today. This stage is an
application: the same known idea handed to a language-model solver as a tool.

## Design

- Tool arm: level 3, cells 000 (no hint) and 100 (S told), 12 problems, 2 fresh Haiku chains each, up to 4 steps,
  14 tool calls per step, as before. Problem folders: `loop/sr/problems_iso` (links to the SR4 and SR5 folders).
  Runner: `loop/sr/trie_iso.py`, which runs `trie.py` unchanged with the new tool server and system prompt.
- Without the tool: the stored chains of the same cells (000 from SR4, 100 from SR5, 24 chains each, run
  2026-11-19, same code, same problem text). They are not concurrent with the tool arm.
- Differences between the arms: the tool, and three edits of the system prompt that describe it
  (`trie_iso.py prompt` prints the result). The old method sentence is kept; the new text is added to it.
- Solved = validation NMSE at most 1e-10 for the best complete law, within 4 steps, as before.

## Claims

- **T1.** Mean over the 12 problems and the two cells of (share of chains solved with the tool minus share
  without) is above 0, with the 95% interval (resampling the 12 problems, 5,000 draws) above 0.
- Per cell, the same difference with its interval (descriptive).
- Use of the tool (descriptive, from `parts.jsonl`): share of chains with at least one part check; share with a
  "right" verdict on a part checked with every input free (the product term); share with a later "right" verdict
  with fewer free inputs; solved or not for each group.

Ceilings from SR5 (efforts add): cell 100 about 0.44, where full separation is always possible; cell 000 about
0.39 only in the 8 problems where S and U sit on different inputs, so about 0.3 overall. Without the tool:
0.12 and 0.04.

## Guesses, written before the run

Share solved with the tool: cell 000 0.12, cell 100 0.25. Chance that T1 passes: 0.35. Share of chains that use
the part check at least once: 0.9. Share that get a "right" verdict on a product part: 0.6.

Reading of a failure: if few chains use the part check, the run says only that Haiku did not take up the tool.
If many chains confirm the product term and still fail, separating the product term does not help and the
unusual term itself is the limit.

## Pilot

One chain of 2 steps on a level 6 problem (`bio_L6_s3`, not in the main set), about 0.02 USD, root
`loop/runs/SR6_pilot`. It checks mechanics only (the call format reaches the tool, files are written). Its
outcome is not counted. Wording may change after the pilot only to fix a mechanical fault, noted below.

## Cost and load

Haiku only, no Fable or Sonnet session. Cap 2.5 USD nominal, about 2.1 expected. 12 workers; a tool call is one
thread with 20 s of CPU and 4 GB of address space at most.

## Change of design before the main run (written 2026-11-20 16:15, before any main-run session)

Three pilot chains of 2 steps each on `bio_L6_s3` (not in the main set), 0.07 USD nominal together:

1. `loop/runs/SR6_pilot`: tool and wording as above. Mechanics worked. Haiku made 1 part check in 2 sessions
   (26 calls to its own analysis, 2 complete-law checks).
2. `loop/runs/SR6_pilot2`: the method text was made an ordered list of steps ("Work in this order, using the
   part check"). Haiku made 0 part checks in 2 sessions.
3. So an optional check is not taken up, and a main run with it would say nothing about hiding. The check was
   made automatic: `evaluate(code)` has no `free` argument any more; when a candidate is not exact, the reply
   also carries a **term check** (`loop/sr/iso/term_check.py`). It splits the candidate's return line into terms,
   scores the terms that mix inputs with free curves of every input, and, if they are right, scores for each
   input the mixing terms plus the terms of that input with free curves of the other inputs. The reply says for
   each group RIGHT or NOT right. The same check is added under each earlier candidate shown at the start of a
   step. `loop/runs/SR6_pilot3`: mechanics worked, every complete-law check carried the term check (5 of 5).

This replaces the tool description and the "use of the tool" lines above. The part scorer and its verdict rule
are unchanged. Offline check of the term check on complete candidates built from the true level 3 laws
(`scratchpad/verify_tc.py`): 36 of 36 verdict sets as expected (wrong unusual term: mixing part right, that
input's terms not right, the rest right; wrong mixing term: mixing part not right; simple term left out: that
input not right). The system prompt edits are those printed by `trie_iso.py prompt`.

What is unchanged: cells, problems, chains, budget, the stored chains as the comparison, claim T1 and the per
cell differences. No main-run session had been started when this was written.

Use of the check (descriptive, from `parts.jsonl`): share of chains that get "mixing part RIGHT" at least once;
share that get every group RIGHT at least once; solved or not for each group.

Guesses for the changed design: share solved with the term check, cell 000 0.10, cell 100 0.22. Chance that T1
passes: 0.3. Share of chains that get "mixing part RIGHT" at least once: 0.5.

Reading of a failure: if few chains ever get the mixing part right, the check came too rarely or Haiku did not
act on it, and the run says little about hiding. If many chains get it right and still fail, confirming the
product term does not help and the unusual term itself is the limit.

## Deviations

Design changed before the main run, see the section above. (During the offline check on 2026-11-20 about 15:30 one calibration script ran numpy without the
one-thread setting for about 10 minutes, about 14 threads of 32 on spruce, nobody else logged in. Stopped and
rerun with one thread per process.)

After the run (2026-11-20 17:35 and later): (1) `iso_table.py` stopped on a misprinted percent sign in one header
line; fixed that one character, no number depends on it. (2) Post hoc reading of the stored sessions, not in
these rules: `loop/sr/iso_why.py` (which chains proposed the unusual form, by kind of form; what happened after
the check narrowed the law to one wrong term). For it the form detector of `advice2.py` is applied per term of the
law, after constants and simple assignments are written into the return line; `advice2.py` itself is unchanged.
The split of the unusual forms into common and rare ones was made after seeing the data.

## Result (2026-11-20, main run 16:15 to 17:32)

189 sessions, 0 failed, 2.10 USD nominal (cap 2.5). Full write-up: `loop/RESULT_SR6.md`.

| cell | without check | with check | difference (95%) | guess |
|---|---|---|---|---|
| 000 | 1 of 24 | 0 of 24 | -0.042 (-0.125 to 0.000) | 0.10 |
| 100 | 3 of 24 | 4 of 24 | +0.042 (0.000 to +0.125) | 0.22 |

T1: mean difference 0.000 (95% -0.062 to +0.062): FAILED (guessed chance of a pass 0.3).
Chains with "mixing part RIGHT" at least once: 23 of 48 (guess 0.5).

Reading, by the rule written above ("if many chains get it right and still fail, confirming the product term does
not help and the unusual term itself is the limit"): about half the chains got the product term confirmed, kept it
(126 of 141 later checks) and still failed. Post hoc: 13 chains were narrowed to one wrong term, 11 with a session
left, 2 of those proposed the right form. Haiku proposes the three rare forms (squared log, sine of a root, exp of
a root) in 3 of 60 chains over all arms. The unusual term is the limit, and it is a limit of what Haiku proposes.
