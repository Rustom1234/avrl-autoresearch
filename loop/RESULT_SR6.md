# SR6 result: telling the cheap model which term is wrong does not help, because it does not propose the missing form

2026-11-20. Rules file: `loop/design/SR6_PREREG.md` (written before the main run). Run folder: `loop/runs/SR6`.
Scoring: `loop/sr/iso_table.py`. Post hoc reading of the stored sessions: `loop/sr/iso_why.py`
(output saved in `loop/runs/SR6/iso_why.out`). Haiku only, no strong-model session.

## Words

- **Law**: the hidden formula of a problem. At level 3 it is a sum of three terms: S (a simple factor of one
  input), CS (a product of two factors on two different inputs, called the mixing or product term here) and U (an
  unusual factor of one input).
- **Chain**: up to 4 sessions of the cheap model (Haiku) on one problem; each session sees the best three laws of
  the earlier ones and may score 14 tool calls. A chain is solved when a law reaches validation error 1e-10.
- **Cell**: which terms the problem text tells the model. 000 = nothing, 100 = S told.
- **Hiding** (SR5): with several terms unknown, Haiku solves far fewer problems than the one-term-unknown results
  predict.

## Question

One explanation of hiding: when a law with several wrong terms scores badly, Haiku cannot tell which of its terms
is right, so it never knows what to keep. SR6 removes exactly that problem. Every time Haiku scores a law that is
not exact, the reply also says which groups of terms are right:

- the product term is checked with the single-input terms replaced by free smooth curves, so it can be confirmed
  before anything else is known;
- once it is right, the terms of each input are checked the same way.

Offline, the check gave the expected verdict on 36 of 36 test laws. It cannot separate two terms that depend on
the same input (4 of the 12 problems), and it cannot say which factor of a wrong product is wrong.

## Result on the claim fixed in advance

12 problems, 2 chains each per cell. "Without check" are the stored chains of SR4 (cell 000) and SR5 (cell 100).

| cell | told | solved without check | solved with check | difference (95% interval over problems) | my guess before |
|---|---|---|---|---|---|
| 000 | nothing | 1 of 24 (0.04) | 0 of 24 (0.00) | -0.04 (-0.12 to 0.00) | 0.10 |
| 100 | S | 3 of 24 (0.12) | 4 of 24 (0.17) | +0.04 (0.00 to +0.12) | 0.22 |

**T1 (mean difference over both cells is above 0): FAILED.** Mean difference 0.000, 95% interval -0.062 to
+0.062. I had given it a 0.3 chance.

189 sessions, 0 failed sessions, 2.10 USD nominal. Pilots before it: about 0.07 USD.

## The check reached Haiku and Haiku obeyed it

- All scored laws that were not exact got a term check: 226 checks in cell 000, 170 in cell 100.
- The product term was confirmed at least once in 23 of 48 chains (11 and 12 of 24). My guess was 0.5.
- After the first confirmation Haiku kept the product term right in 126 of 141 later checks.

One chain as an example (`bio_L3_s2`, nothing told, chain r0). True law:
`4.01*sqrt(P) - 1.02*sin(0.425*t)*P + 0.79*log(1+t)**2`. In session 2 the check said: product term RIGHT, the
term in P RIGHT, the term in t NOT right. In the two sessions after that Haiku kept the two right parts unchanged
and scored six more laws that differed only in the term in t: `t*exp(b*t)`, `exp(b*t)`, `t/(1+b*t)`, `sin(b*t)`,
`t**a*exp(b*t)`, and a polynomial plus an exponential. It never tried a squared logarithm.

## Why it did not help (post hoc, stored sessions only, not in the rules file)

**1. The check confirms, it does not suggest.** In 25 of 48 chains Haiku never found the product term, and there
the check could only say "NOT right" each time.

**2. Where the check narrowed the law to one wrong term, Haiku still did not propose the right form.** 13 chains
were told "product term right, S right, only the term in this input is wrong". 11 of them had at least one
session left. 2 of those 11 then proposed the right form, and both solved.

**3. The missing forms split into two groups.** "Proposed" = some scored law has a term in the right input with
the unusual form (text patterns; a solved chain counts as proposed). Entries are proposed / solved of chains.

| arm | free power, tanh, Gaussian, x log x (6 problems) | squared log, sine of a root, exp of a root (6 problems) |
|---|---|---|
| check, nothing told | 0 / 0 of 12 | 0 / 0 of 12 |
| check, S told | 4 / 3 of 12 | 1 / 1 of 12 |
| no check, nothing told | 3 / 1 of 12 | 0 / 0 of 12 |
| no check, S told | 3 / 2 of 12 | 1 / 1 of 12 |
| no check, S and CS told in the text | 10 / 10 of 12 | 1 / 1 of 12 |

- **Rare forms** (right column): Haiku proposes them in 3 of 60 chains, whatever it is told. On the easier level 6
  (stored chains of SR2, where the rest of the law is simple) it is 4 of 24. This is a limit of what Haiku
  proposes, not of search or of knowing which term is wrong. The stored Fable sessions proposed a rare form in 4
  of 5 sessions and solved all 4 (5 sessions only).
- **Common forms** (left column): here hiding is visible. Told S and CS, Haiku proposes and solves 10 of 12; told
  only S, 3 or 4 of 12, the same with and without the check. On level 6 it is 7 proposed, 5 solved of 8.
  In the 12 common-form chains of cell 100 with the check, the product term was confirmed in 5 and the law was
  narrowed to one wrong term in 3; so mostly the check never came into play (point 1).

When a chain does propose the right form it nearly always solves: 19 solved of 23 chains that proposed it, over
the five arms.

## What this says about hiding (SR5)

- Half of the level 3 problems have an unusual term that Haiku does not propose under any hint short of the form
  itself. They sit at the floor in every cell where U is not told. This part of the hint table is a proposal
  limit and has nothing to do with terms hiding each other.
- On the other half hiding is real (10 of 12 against 3 or 4 of 12), and a check that says which term is wrong did
  not undo it. The step that blocks is finding the product term without help.
- Not answered: whether a verdict from a tool is used as well as the same fact written in the problem text. Only 3
  common-form chains reached the fully narrowed state (1 solved).

## Known work, and what is claimed

Nothing new in the method. The check is a partially linear model (Engle et al. 1986, Robinson 1988) with additive
spline curves (Hastie and Tibshirani 1986), fitted by variable projection (Golub and Pereyra 1973); the idea of
testing one part of a law apart from the rest is in AI Feynman (2020). SR6 applies them as a tool reply for a
language model. The result is negative. No new literature search was made for this stage.

## Limits

- The chains without the check are stored ones (one and two days older), not run alongside. Same model alias,
  same problems, same budget.
- 12 problems and 2 chains per problem and cell. The interval of T1 (-0.06 to +0.06) rules out a large gain, not
  a small one.
- "With check" also changed three lines of the system prompt, so it is the check plus its instructions.
- The split into common and rare forms was made after seeing the data. Each form has 1 to 3 problems.
- "Proposed" rests on text patterns. I read two chains by hand to check them. One solved chain in the
  "S and CS told" arm solved with a form the patterns do not see; it counts as proposed because it solved.
- The check is all or nothing for the product term. A check per factor of the product is possible and was not
  built.
- One cheap model, one wording of the tool reply.

## Cost and load

Main run 16:15 to 17:32 on spruce, 12 workers, 189 Haiku sessions, 2.10 USD nominal (cap 2.5). Three pilots,
about 0.07 USD. No Fable or Sonnet session. No failed session, no memory event. Each term check is a one-thread
process of 1 to 2 s under `prlimit`.
