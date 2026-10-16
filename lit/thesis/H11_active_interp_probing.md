# H11: Active Interpretability and Safety Probing Under Budget

## 1. Hypothesis and SOFisher mapping

A learned policy that chooses which model component, feature, or checkpoint to probe next with an expensive interpretability method, conditioned on cheap signals and prior probe outcomes, will find circuits and unsafe features with far fewer expensive interventions than uniform or heuristic sweeps, because relevant components cluster by layer, feature neighbourhood, and training-checkpoint adjacency. This would let a lab run continuous interpretability monitoring during training instead of periodic, budget-capped audits.

SOFisher mapping:
- Slide -> the model's full component space (layers x heads x SAE features x checkpoints x prompt set).
- Field of view -> one component/checkpoint/prompt combination (e.g., layer 8 head 3 on checkpoint 40k for a harmful-instruction prompt).
- Expensive measurement -> attribution patching sweep, full activation-patching causal intervention, or an SAE feature audit against a behavioural probe suite.
- Cheap features -> activation norms, gradient magnitude, attribution-patching linear approximation, SAE reconstruction loss, prior checkpoint's probe verdict.
- Autocorrelation -> circuits concentrate in a handful of heads/layers (ACDC, EAP results below); unsafe SAE features and developmental phase transitions cluster in nearby checkpoints (Timaeus).
- Policy -> a bandit or RL agent over the component graph, analogous to SOFisher's DQN over slide tiles.
- Reward -> faithfulness/causal effect size discovered per unit of expensive-measurement compute spent.

## 2. Prior art

1. ACDC, Conmy et al. 2023, NeurIPS Spotlight. https://arxiv.org/abs/2304.14997 - selected 68 of ~32,000 possible edges in GPT-2 Small's Greater-Than circuit, matching prior manual work, via automated activation-patching pruning.
2. Attribution patching, Nanda 2023 blog/AlignmentForum. https://www.neelnanda.io/mechanistic-interpretability/attribution-patching - a gradient-based linear approximation that patches all 4.7M of GPT-3's neurons in three forward/backward passes instead of one pass per neuron.
3. Edge Attribution Patching, Syed, Rager, Conmy 2023 (BlackboxNLP 2024). https://arxiv.org/abs/2310.10348 - achieves circuit recovery competitive with ACDC using only two forward passes and one backward pass total, versus ACDC's per-edge forward-pass cost.
4. Efficient circuit discovery cost-minimality analysis, 2024/ICLR 2025. https://arxiv.org/abs/2407.00886 - reports ACDC returns smaller circuits on 9 of 10 tested behaviours while EAP-IG greedy is far cheaper in forward passes, formalising a cost-vs-circuit-size tradeoff frontier.
5. Scaling Monosemanticity, Templeton et al. 2024, Anthropic. https://transformer-circuits.pub/2024/scaling-monosemanticity/ - trained sparse autoencoders with up to 34 million features on Claude 3 Sonnet's mid-layer residual stream.
6. Gemma Scope, DeepMind 2024. https://arxiv.org/abs/2408.05147 - over 400 SAEs and 30 million+ learned features across Gemma 2 2B/9B/27B, consuming over 20% of GPT-3's training compute and 20 PiB of saved activations, released specifically to lower the cost barrier to SAE research.
7. Sparse Autoencoders Enable Scalable and Reliable Circuit Identification, 2024. https://arxiv.org/abs/2405.12522 - uses SAE features as circuit-discovery units to keep the search space tractable at scale.
8. MIB: A Mechanistic Interpretability Benchmark, Mueller et al. 2025, ICML. https://arxiv.org/abs/2504.13151 - across 4 tasks and 5 models, attribution/mask-optimization methods lead circuit localization, while SAE features are found no better than raw neurons for causal-variable localization, a negative result on the current cheap-feature quality bar this thesis needs to beat.
9. PAHQ, 2025. https://arxiv.org/abs/2510.23264 - accelerates automated circuit discovery through mixed-precision inference, another point on the cost-reduction frontier competing with an active-selection policy.
10. Certified Interventional Fidelity, 2026. https://arxiv.org/abs/2607.08349 - adds sequential, anytime-valid uncertainty quantification to adaptive interpretability workflows, i.e., statistically principled stopping rules for exactly the sequential-decision setting H11 needs.
11. MechRL, 2026. https://arxiv.org/abs/2605.26343 - frames circuit discovery as sequential edge removal guided by a faithfulness reward; a single frozen RL policy recovers faithful circuits across 12 training behaviours in GPT-2 small and transfers with zero per-behaviour search, the closest existing work to H11's policy-over-components idea, but it amortises discovery cost across behaviours, not across an imaging-style expensive/cheap-feature budget split.
12. Towards Developmental Interpretability, Timaeus, 2024. https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability - Local Learning Coefficient estimation reveals training-checkpoint phase transitions that are invisible in the loss curve, direct evidence for checkpoint-adjacency clustering.
13. Pythia suite, Biderman et al. 2023, EleutherAI. https://github.com/EleutherAI/pythia - 154 checkpoints per model across 8 sizes trained on identical data order, the standard substrate for checkpoint-adjacency interpretability experiments.
14. Automated Interpretability-Driven Model Auditing and Control research agenda, Oxford AIGI, 2026. https://aigi.ox.ac.uk/wp-content/uploads/2026/01/Automated_interp_Research_Agenda.pdf - lays out continuous, budget-aware auditing as an open research direction rather than a solved problem, framing motivation but UNVERIFIED as a source of any specific benchmark number.
15. Decoding Hidden Deception in Reasoning LLMs, 2026. https://arxiv.org/abs/2606.17478 - activation-based deception probes; reported probe-accuracy figures for midlayer deception detection are UNVERIFIED here since they came from a secondary synthesis, not a direct read of the paper's tables.

## 3. What is genuinely open as of 2026

No existing system learns a reusable, cross-axis (layer x feature x checkpoint x prompt) policy that spends a fixed expensive-measurement budget adaptively using cheap signals plus prior probe outcomes as state. ACDC, EAP, and PAHQ each optimize the cost of a single circuit-discovery run; MechRL amortises across behaviours but stays within one checkpoint and does not use cheap pre-measurement features to decide where to look next, it uses RL only for the pruning trajectory itself. MIB shows current cheap features (raw SAE activations) are not yet good enough to substitute for causal localization, so the value of a cheap-to-expensive triage policy over a naive sweep is unproven. Developmental interpretability shows checkpoint clustering exists but has not been combined with a component-selection policy. A year is needed because this requires: (a) building or reusing an SAE/attribution-patching cheap-feature pipeline across many checkpoints, which does not exist off the shelf for a full training run; (b) training and validating a policy against several strong non-learned baselines (uniform sweep, EAP-IG greedy, cheap-feature-only ranking); (c) testing transfer across model families, since MechRL's within-GPT-2 transfer result is the only existing precedent and is unverified outside its own paper.

## 4. Three-condition check

- Spatial/structural autocorrelation of targets: supported. ACDC circuits concentrate in a small edge subset (68/32,000, source 1); Timaeus shows checkpoints cluster into developmental stages (source 12); MIB's circuit-localization leaderboard shows components repeatedly implicated across tasks rather than uniformly spread (source 8).
- Expensive measurement costs far more than cheap features: supported. Attribution patching is offered explicitly as a cheap linear stand-in because full activation patching needs one forward pass per activation versus three total passes for the approximation (source 2); Gemma Scope's SAE training burned over 20% of GPT-3's total training compute just to produce the cheap-feature layer, showing even "cheap" features are not free at scale (source 6).
- Cheap features alone cannot rank targets reliably: supported but weaker than SOFisher's case. MIB shows SAE features are no better than raw neurons for causal-variable localization (source 8), meaning today's cheap signals are a weak ranker on their own, which is the gap an active policy must close, not a settled advantage the way it was in SOFisher's imaging domain. This is the shakiest of the three legs and should be the first thing tested empirically, not assumed.

## 5. Year plan

Phase 1 (months 1-3): Build the cheap-feature and expensive-measurement pipeline on Pythia-160M/410M using its public 154 checkpoints (source 13). Cheap features: activation norms, EAP linear attribution scores (source 3). Expensive measurement: ACDC-style activation patching (source 1) plus one behavioural probe suite. First experiment, runnable on 4 A100s in under a week: fix 3 known circuits/behaviours (e.g., IOI, greater-than, induction) across 20 checkpoints, compare a simple contextual-bandit policy against uniform sampling and EAP-IG-greedy ranking on faithfulness recovered per expensive probe spent. Metric: faithfulness (behavioural score recovered, as used in ACDC/EAP) per unit of expensive-measurement compute, at matched total budget. Strongest existing baseline: EAP-IG greedy ranking (source 4), since it is the cheapest strong non-learned baseline reported in the literature.

Phase 2 (months 4-8): Extend the policy to span layers, SAE features (using Gemma Scope, source 6), and checkpoints jointly on Gemma 2 2B; test transfer across behaviours and across the checkpoint axis, following MechRL's transfer protocol (source 11) but adding cheap-feature state and a genuine budget constraint MechRL does not impose.

Phase 3 (months 9-12): Apply the policy to a safety-relevant target (SAE feature audits for a known red-team behaviour set) and to live monitoring during a short continued-training run, benchmarking wall-clock and GPU-hour cost against a naive periodic full-sweep audit.

## 6. Kill criteria

- If cheap features (activation norms, EAP scores) show near-zero correlation with which components an expensive intervention confirms as causal (Phase 1, first two weeks), kill: the third SOFisher condition fails and no policy can outperform random search.
- If a trained bandit/RL policy does not beat EAP-IG-greedy ranking at matched expensive-measurement budget on the Phase 1 benchmark, kill or pivot to publishing the negative result, since EAP-IG-greedy is already cheap, non-learned, and available today.
- If transfer across checkpoints or model scale collapses to near-random performance by Phase 2 (i.e., the policy only ever works on the exact setting it trained on), kill the "continuous monitoring" framing and scope down to a single-checkpoint tool.

## 7. Impact on a lab's stack

If it works, this replaces ad hoc, human-scheduled interpretability audits with a standing, budget-capped policy that runs alongside training and flags checkpoints/components worth an expensive look, directly usable for the sensitive-pretraining-data auditor concept already logged in this project's idea backlog. If it fails past Phase 1, the negative result on cheap-feature rankability (leg three of the SOFisher check) is still a useful, publishable finding about the limits of current interpretability triage signals, and the cheap/expensive pipeline built in Phase 1 remains reusable for any future circuit-discovery work in the lab regardless of whether the active-policy layer pays off.

SCORECARD
novelty: 6
feasibility_1yr: 6
lab_impact: 7
sofisher_fit: 6
strongest_baseline: EAP-IG greedy ranking (edge attribution patching with integrated gradients)
one_line_pitch: Train a budget-aware policy that picks which layer, feature, or checkpoint to run an expensive circuit-discovery or SAE audit on next, using cheap activation and attribution signals, so a lab can monitor a training run for circuits and unsafe features continuously instead of via periodic full sweeps.
