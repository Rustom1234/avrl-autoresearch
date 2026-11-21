# SR4 result: the gap between the cheap and the strong model, measured in bits of advice

2026-11-19. Rules and guesses: `loop/design/SR4_PREREG.md` (written before each stage). Code: `loop/sr/advice.py`.
Runs: `loop/runs/SR4` (hinted problem versions in `loop/sr/problems_advice`). Haiku only: 5.86 USD nominal for the
pilot and stage 1 together, no failed session, no Fable or Sonnet session.

## Short version

A hint about the structure of the hidden law is added to the problem text, and its information is counted in bits
under the generator's own odds. Fresh Haiku chains of up to 4 steps then try the problem.

| Level | Hint | Bits | Chains solved within 4 steps | Problems solved |
|---|---|---|---|---|
| 3 | grammar only ("a sum of 3 terms, each a constant times one or two factors") | 0 | 1 of 24 | 1 of 12 |
| 3 | the one term with an unusual factor | 4.1 | 15 of 24 | 9 of 12 |
| 3 | the two ordinary terms | 10.3 | 11 of 24 | 7 of 12 |
| 4 | grammar only | 0 | 0 of 8 | 0 of 4 |
| 4 | the kinds of the two unusual factors, not where they go | 5.0 | 0 of 8 | 0 of 4 |
| 4 | one term with an unusual factor | 4.1 | 0 of 8 | 0 of 4 |
| 4 | the two ordinary terms | 10.3 | 1 of 24 | 1 of 12 |
| 4 | the two terms with an unusual factor | 11.1 | 10 of 24 | 6 of 12 |
| 4 | all four terms | 21.4 | 8 of 8 (all at step 1) | 4 of 4 |

For comparison, without hints: level 3 Haiku chains 0 of 48 within 4 steps and 10 of 48 within 12; one fresh Fable
attempt 5 of 12. Level 4 Haiku chains 0 of 48 within 12 steps and 4 of 48 within 24; Fable after a Haiku chain 3
of 4 (four problems only).

## Written claim

S1: at each level the share solved is higher with the hard-terms hint than with the ordinary-terms hint, interval
of the per-problem difference above 0.
- Level 4: +0.375 (95% +0.167 to +0.625), ahead on 6 problems, behind on 0. Pass.
- Level 3: +0.167 (95% -0.167 to +0.500), ahead on 6, behind on 3. Fail.

## Reading

1. The cheap model's shortfall can be stated in bits. At level 3 about 4 bits (one term) take Haiku from 1 of 24 to
   15 of 24, more than one fresh Fable attempt solves (5 of 12 problems against 9 of 12). At level 4 about 11
   bits take it from 0 to 10 of 24, on 6 of 12 problems.
2. Bits are not equal at level 4: 10 bits about the ordinary terms give 1 of 24, 11 bits about the hard terms give
   10 of 24. Below about 5 bits nothing happens there.
3. At level 3 the picture is different: the ordinary-terms hint also works (11 of 24). Once two of three terms are
   given, the one remaining term depends on one input and Haiku can read it off the residual, even when it is an
   unusual one. So what matters is what is left to find after the hint, not only how hard the revealed part is.
   Per bit the hard-term hint is still far more efficient (4.1 bits for 15 of 24 against 10.3 bits for 11 of 24),
   but that comparison was not the written claim.
4. The grammar sentence alone does nothing (1 of 24 and 0 of 8), so the effect is not the wording.

## Limits

- Hints come from the known answer, not from a model. This measures how much information is needed, not whether a
  strong model can supply it cheaply.
- 12 problems per level, 2 chains per cell, 4 steps. One task family made by us, one cheap model.
- Bits are counted under the generator's odds for the structure (factor kinds and inputs). The constants are not
  counted.
- Known work: short advice from a strong to a weak model (Redwood Research, "Untrusted advice"; "Ask Your
  Advisor", arXiv 2610.13870); graded hints from privileged information (arXiv 2502.10961). Not found in my
  searches: advice counted in exact bits against a known generator, or a curve of success against bits. That is
  an absence in five searches, not a proof.
