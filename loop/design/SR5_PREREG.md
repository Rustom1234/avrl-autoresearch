# SR5: which parts of a hidden law does the cheap model need to be told?

Written 2026-11-19 15:51 by the coding agent, before any SR5 session was run. Follow-up to SR4 (`SR4_PREREG.md`,
`loop/RESULT_SR4.md`). Rustom asked to make the hint experiment solid while spending little. This stage uses
Haiku sessions only (cap 7 USD nominal) and stored sessions. No Sonnet or Fable session is run.

## Why

SR4 left three holes. (1) Its second claim failed on level 3: a hint about the ordinary terms also worked, so
"bits about the hard part" is not the whole explanation. (2) Only a few hint types were tried, so the effect of each
term cannot be separated. (3) The hints came from the answer key, and nothing was known about wrong hints or about
whether a strong model has the hint early in its own session.

The SR4 numbers already rule out two simple explanations on level 3 (exploratory, seen before this file):
- "More bits help more": 4 bits (hard term) gave 15 of 24, 10 bits (ordinary terms) gave 11 of 24.
- "Independent parts" (each unknown term is found with its own chance, chances multiply): finding the ordinary
  pair alone works 15 of 24 times and finding the hard term alone 11 of 24 times, so all three together should
  work about 0.29 of the time. Observed with no hint: 1 of 24. Unknown terms seem to hide each other.

## Design

A law is a sum of terms in a fixed order (`gen.LEVELS`): level 3 has S (one simple factor), CS (a common factor
times a simple one), U (one unusual factor); level 4 adds US (an unusual factor times a simple one). A hint cell is
the set of terms whose exact form is given, written as a mask in that order (level 3: `001` = only U given).
Wording is the same as in SR4. Each cell: the 12 problems of the level, 2 Haiku chains each, 4 steps, same solver
and success rule (validation error at most 1e-10). f(mask) = share of chains solved within 4 steps.

| Level | Cells run now | Cells reused from SR4 |
|---|---|---|
| 3 | 100, 010, 101, 011, 111, and W (a wrong form for the unusual term: the next kind in the generator's list) | 000 (G), 001 (U2), 110 (E) |
| 4 | 0111, 1011, 1101, 1110 (all but one term given) | 0000 (G, 4 problems), 0011 (U2), 1100 (E), 1111 (ALL, 4 problems), 0010 (U1, 4 problems) |

Level 3 then has the full 2x2x2 table. What is known: full factorial designs (Fisher), the Shapley value and
interaction indices for set functions (Shapley 1953, Grabisch 1997), advice complexity (Boeckenhauer et al. 2009),
short advice lifting weak models (Redwood Research; "Ask Your Advisor", arXiv 2610.13870). What is new here, if
anything, is the measurement: which terms of a hidden law a cheap model must be told, and whether unknown terms hide
each other. No literature search has been done yet for this exact design, so no novelty claim is made.

## Claims (95% intervals by resampling the 12 problems, 5,000 draws)

| Claim | Passes if |
|---|---|
| F1 hiding, level 3 | With p_S = f(011), p_CS = f(101), p_U = f(110), the independent-parts predictions are f(100) = p_CS p_U, f(010) = p_S p_U, f(001) = p_S p_CS, f(000) = p_S p_CS p_U. The mean of observed minus predicted over these four cells is below 0, interval below 0 |
| F2 bottleneck, level 4 | f(1110) is below f(1101): the unusual-times-simple term is harder to find than the single unusual term when everything else is given; interval of the difference below 0 |
| F3 hiding, level 4 | f(1100) is below f(1101) times f(1110); interval of the difference below 0 |

Descriptive, no claim: the Shapley value of each term on level 3; f(W) against f(000) and f(001).

My guesses, written before the run: f(100) 0.10, f(010) 0.20, f(101) 0.80, f(011) 0.95, f(111) 1.00, f(W) 0.03;
f(0111) 0.95, f(1011) 0.75, f(1101) 0.45, f(1110) 0.20. Chance that each claim passes: F1 0.6, F2 0.5, F3 0.4.

If F1 fails, the "terms hide each other" reading is dropped and the result is reported as plain independent parts.

## Stored sessions (no new session)

For the stored first attempts of Haiku, Sonnet and Fable on level 3 (`loop/runs/SR1/attempts`), find the first
candidate sent to the checker that contains the true unusual form on the right input (text patterns per kind; a
free exponent or phase counts). Check of the detector: it must fire on the final formula of every solved session.
Reported: share of sessions that ever have the form, and by which tool call. This says whether a short visit by a
strong model could supply the hint. It is descriptive; a real test needs new strong-model sessions and is not
part of this stage.

## Deviations

Written 2026-11-19 19:15, after the run.

1. The detector for the stored-session scan was changed after its first use. First version: it fired on the
   final formula of 4 of the 5 solved Fable sessions (miss: `phys_L3_s2`, a Gaussian written with a free exponent,
   `exp(-abs(c*t)**(2.0 + c))`, and constants written `p[k]` instead of `params[k]`). Changed: `p[k]` counts as a
   constant, and a Gaussian with a free exponent starting at 2 counts as a Gaussian. Second version fires on 5 of
   5. Numbers moved: Fable sessions that ever test the true form 5 of 12 to 7 of 12; among unsolved 1 of 7 to 2 of
   7. The reading did not change.
2. Two analyses were added after seeing the results and are not claims of this file: the "efforts add" check
   (`advice2.py seq`) and the wrong-hint scan (`advice2.py wrong`).
3. A literature search (five queries) was done after the run, see `loop/RESULT_SR5.md`.

## Result (2026-11-19)

240 chains, 584 Haiku sessions, 5.94 USD nominal, no failed session. F1 passed (-0.304, 95% -0.490 to -0.142).
F2 failed (-0.250, 95% -0.583 to +0.083). F3 passed (-0.283, 95% -0.514 to -0.104). Observed against the written
guesses: f(100) 0.12 (0.10), f(010) 0.21 (0.20), f(101) 0.92 (0.80), f(011) 1.00 (0.95), f(111) 1.00 (1.00),
f(W) 0.00 (0.03), f(0111) 1.00 (0.95), f(1011) 0.83 (0.75), f(1101) 0.71 (0.45), f(1110) 0.46 (0.20).
Write-up: `loop/RESULT_SR5.md`.
