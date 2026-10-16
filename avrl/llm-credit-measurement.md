# Measuring human credit in LLM-assisted written work

Status: research write-up, Oct 15 2026. Findings only, no new research.

## Question

Can we measure how much human credit is deserved in LLM-assisted written work?

## Short answer

- **Disclosure policy is solved.** Every venue just requires disclosure plus human accountability.
- **Graded credit is not solved.** No accepted method says "this piece is 60% yours".
- The best available method is **counterfactual removal** (Offloading Score), because it captures load-bearing help rather than word counts.
- Nobody prices **idea vs prose** or **verification/selection**. Those are the gaps worth building into.

## Methods found

| Method | Reference | What it measures | Weakness |
|---|---|---|---|
| Word-provenance tracking | (no single paper) | Which words were typed by a human vs generated | Defeated by retyping the model's output |
| Prompt-output information overlap | arXiv 2506.03501 | BERTScore recall plus a token-level RoBERTa regressor: how much of the output is carried by the prompt | Overlap is not the same as credit; paraphrase hides it |
| Information-theoretic overlap | arXiv 2408.14792 (ACL 2026) | Information-theoretic version of the same prompt-to-output idea | Same: measures transfer of information, not value of the contribution |
| Counterfactual removal, the Offloading Score | arXiv 2605.29392 | Removes AI at each stage of the work and measures the quality drop | Needs a defined pipeline of stages and a quality scorer |
| Shapley-value credit over contributors, AME | arXiv 2606.16075 | Fair attribution across multiple contributors (human and model) | Needs a value function over coalitions, which is expensive to evaluate |
| Code-survival rate (Copilot) | Industry analog | Fraction of AI-suggested code that survives in the final artifact | Survival is not authorship; code only |

### Notes per method

- **Word provenance** is the intuitive approach and the easiest to game. Retyping launders it completely.
- **Prompt-output overlap** (2506.03501, 2408.14792) asks how much of the output was already in the prompt. High overlap means the human supplied the content and the model supplied surface form. This is the closest existing thing to pricing "a precise prompt carries the contribution".
- **Offloading Score** (2605.29392) is the only method in the set that captures *load-bearing* help. It asks what breaks if the AI is taken out at a given stage, rather than counting words.
- **AME** (2606.16075) gives a principled split across contributors via Shapley values, and composes naturally with counterfactual removal as the value function.
- **Copilot code survival** is the industry analog: a cheap, observable proxy for "did the suggestion actually matter".

## The gap

None of the methods above price two things:

1. **Idea vs prose.** A precise prompt can carry the whole contribution while the model types every word. Word counts and provenance say "AI wrote 100%", overlap metrics partly catch it, but nothing assigns it a credit value.
2. **Verification and selection.** Catching a wrong claim, or picking among candidate outputs, is real work with no footprint in the final text.

A CHI 2025 study (ACM DOI 3706598.3713522) found people weight credit by four factors:

- replaceability
- effort
- intentionality
- directness

These are human judgments of fairness, and they are a candidate target for calibrating any automatic score.

## Recommendation

Build a **counterfactual harness on your own session transcripts**:

1. Take a writing session transcript (human turns plus model turns).
2. Categorize each human turn: **idea**, **constraint**, **correction**, **selection**.
3. Ablate human turns by category, one category at a time.
4. Regenerate the output with the ablated transcript.
5. Score quality per ablation.
6. The quality drop per category is that category's measured credit.

This directly prices idea-vs-prose (idea and constraint ablations) and verification/selection (correction and selection ablations), the two gaps above. AME/Shapley can then split credit across categories when ablations interact. The CHI 2025 factors give a human-judgment baseline to check the scores against.

## Link to the thesis

This rhymes with the AVRL thesis framing of **verification as the scarce resource**. If correction and selection turns carry measurable credit, that is empirical support for the claim that verification is where human value concentrates.

## Open items (for building, not research)

- Define the quality scorer for regenerated output.
- Decide the turn-category labeling method (manual first, classifier later).
- Decide whether ablation means deleting a turn or replacing it with a neutral placeholder.
