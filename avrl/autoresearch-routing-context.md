# Model choice and context injection in autoresearch swarms

Status: research write-up, Oct 15 2026. Findings only, no new research.

## Question

How do autoresearch swarms choose which model does what, and how do they inject context? What are the cheapest evidence-backed upgrades for the avrl `autoresearch.workflow.js`?

## Systems compared

| System | Models | Role structure | Context injection | Selection / routing |
|---|---|---|---|---|
| Karpathy autoresearch | One model | Single loop | `program.md` plus git history | Keep / revert |
| AlphaEvolve | Flash for breadth, Pro for deep fixes | Generator ensemble | Prompt sampler over a program database | Evolutionary selection over the database |
| AI Scientist v2 | Sonnet codes, GPT-4o reviews | Fixed roles | Per-role prompts | None (fixed assignment) |
| AI co-scientist | One Gemini | Six role prompts | Persistent context memory | Elo tournament |
| ShinkaEvolve | Ensemble | Mutation calls | Archive of programs | UCB1 bandit per call picks the model |
| OpenEvolve | Weighted ensemble, or one model per island | Island evolution | MAP-Elites "inspirations" plus top performers | Weighted sampling, or island assignment |
| **avrl `autoresearch.workflow.js`** | **Sonnet for every role** | **Four sequential agents: plan, run, analyze, critic** | **Files only: backlog, synthesis doc, PROJECT.md, ledger, newest result JSONs** | **None** |

### Patterns visible in the table

- Only Karpathy's loop and avrl use a single model with no routing.
- Heterogeneity splits into three kinds: by task difficulty (AlphaEvolve Flash/Pro), by role (AI Scientist v2), and learned per call (ShinkaEvolve UCB1).
- AI co-scientist gets role diversity from prompts alone, with one model.
- OpenEvolve is the one system that deliberately injects non-winning material ("inspirations") alongside top performers.

## Evidence

- **"Rethinking Mixture-of-Agents" (arXiv 2502.00674):** sampling one strong model repeatedly beat mixed-model pools. Proposer quality mattered more than diversity.
- **arXiv 2607.18235:** 30 budget-matched harnesses over 3.1M rollouts. No universally best harness. Early pruning and budget reallocation beat any fixed ensemble.

Reading the two together: model mixing for its own sake is not supported. Spend the strong model where quality matters most, and prune and reallocate rather than fix a roster.

## Where avrl stands

`autoresearch.workflow.js` runs four sequential Sonnet agents (plan, run, analyze, critic). Same model in every role. Context arrives only through files: backlog, synthesis doc, PROJECT.md, ledger, newest result JSONs. No model heterogeneity.

## Cheapest evidence-backed upgrades

1. **Put the strongest model on the critic role only.** Verification is where one wrong number poisons the ledger, so this is where extra model quality buys the most. It is a one-role change and keeps cost nearly flat. Consistent with 2502.00674 (proposer quality matters, no need to mix across all roles).
2. **Give the planner two context pools:** (a) best results, and (b) a sampled set of killed or inconclusive experiments as "inspirations". This is the OpenEvolve pattern. It stops the planner from only seeing winners and re-proposing dead ends or missing near-misses.

Not recommended on current evidence: a mixed-model ensemble across roles, or a bandit router. 2502.00674 and 2607.18235 do not support the added complexity at this scale.

## Proposed experiment: ledger-context ablation

Run the avrl harness on itself, varying what the planner sees:

| Arm | Planner context |
|---|---|
| A | Full logs |
| B | Compressed ledger |
| C | Compressed ledger plus inspirations (sampled killed/inconclusive experiments) |

Measure:

- **Experiments-to-verdict** (primary)
- **Token cost** (secondary)

Success criterion: arm C or B reaches verdicts in fewer experiments than A at lower token cost, or arm C beats B on experiments-to-verdict at comparable cost. Pair this with the critic-upgrade change as a second, independent factor if budget allows.

## References

- Rethinking Mixture-of-Agents, arXiv 2502.00674
- Harness comparison, 30 harnesses / 3.1M rollouts, arXiv 2607.18235
- Karpathy autoresearch, AlphaEvolve, AI Scientist v2, AI co-scientist, ShinkaEvolve, OpenEvolve (systems as described in the table above)
