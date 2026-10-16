# H1: does low confidence predict proxy_error over training?

Assumption: proxy_error is logged for every group in every harness analysed here (gold is computed for logging in the code and GSM8K harnesses, a logging-only judge call in the judge7b harness), so every row is usable. Checked directly: the null fraction of proxy_error was 0.0 for all runs below.

| run | task | arm | n groups | err rate (first block) | err rate (last block) | AUC (first block) | AUC (last block) | AUC (whole run, 95% CI) |
|---|---|---|---|---|---|---|---|---|
| e10b_proxy_s1 | code, OLMo (visible-test grader) | proxy | 2400 | 0.147 | 0.155 | 0.883 | 0.875 | 0.881 [0.861, 0.901] |
| e10b_full_s1 | code, OLMo (visible-test grader) | full | 2400 | 0.147 | 0.083 | 0.906 | 0.909 | 0.904 [0.882, 0.925] |
| e10b_threshold_s1 | code, OLMo (visible-test grader) | threshold | 2400 | 0.155 | 0.110 | 0.830 | 0.919 | 0.882 [0.860, 0.905] |
| e10b_random_s1 | code, OLMo (visible-test grader) | random | 2400 | 0.145 | 0.142 | 0.832 | 0.893 | 0.897 [0.876, 0.915] |
| e10b_threshold_s2 | code, OLMo (visible-test grader) | threshold | 2400 | 0.155 | 0.120 | 0.881 | 0.908 | 0.905 [0.883, 0.925] |
| e10b_random_s2 | code, OLMo (visible-test grader) | random | 2400 | 0.158 | 0.170 | 0.867 | 0.890 | 0.897 [0.881, 0.917] |
| e10c_proxy600_s1 | code, OLMo (visible-test grader) | proxy | 4800 | 0.147 | 0.160 | 0.882 | 0.841 | 0.862 [0.846, 0.875] |
| hannah_e10b_proxy | code, OLMo (visible-test grader) | proxy | 2400 | 0.158 | 0.172 | 0.898 | 0.907 | 0.904 [0.885, 0.923] |
| hannah_e10b_full | code, OLMo (visible-test grader) | full | 2400 | 0.177 | 0.072 | 0.877 | 0.898 | 0.912 [0.892, 0.931] |
| hannah_e10b_random | code, OLMo (visible-test grader) | random | 2400 | 0.133 | 0.152 | 0.887 | 0.934 | 0.916 [0.899, 0.932] |
| hannah_e10b_threshold | code, OLMo (visible-test grader) | threshold | 2400 | 0.160 | 0.117 | 0.861 | 0.928 | 0.913 [0.893, 0.932] |
| hannah_e10d_proxy | code, Qwen (visible-test grader) | proxy | 2400 | 0.220 | 0.217 | 0.893 | 0.858 | 0.864 [0.845, 0.883] |
| hannah_e10d_full | code, Qwen (visible-test grader) | full | 2400 | 0.200 | 0.203 | 0.854 | 0.838 | 0.862 [0.842, 0.881] |
| hannah_e10d_random | code, Qwen (visible-test grader) | random | 2400 | 0.207 | 0.220 | 0.858 | 0.820 | 0.861 [0.840, 0.881] |
| hannah_e10d_threshold | code, Qwen (visible-test grader) | threshold | 2400 | 0.217 | 0.215 | 0.855 | 0.851 | 0.866 [0.848, 0.886] |
| hannah_judge7b_full | open-ended, 7B judge | full | 1200 | 0.077 | 0.072 | 0.681 | 0.698 | 0.666 [0.603, 0.719] |
| hannah_judge7b_random | open-ended, 7B judge | random | 1200 | 0.085 | 0.050 | 0.707 | 0.714 | 0.694 [0.631, 0.747] |
| hannah_judge7b_thrcons | open-ended, 7B judge | thrcons | 1200 | 0.087 | 0.048 | 0.672 | 0.677 | 0.652 [0.587, 0.716] |
| hannah_judge7b_greedy | open-ended, 7B judge | greedy | 1200 | 0.080 | 0.065 | 0.718 | 0.659 | 0.672 [0.609, 0.729] |
| hannah_judge7b_s1_full | open-ended, 7B judge | full | 1200 | 0.087 | 0.075 | 0.698 | 0.637 | 0.665 [0.608, 0.723] |
| hannah_judge7b_s1_random | open-ended, 7B judge | random | 1200 | 0.092 | 0.035 | 0.707 | 0.641 | 0.693 [0.628, 0.754] |
| hannah_judge7b_s1_thrcons | open-ended, 7B judge | thrcons | 1200 | 0.105 | 0.035 | 0.679 | 0.638 | 0.677 [0.615, 0.737] |
| hannah_judge7b_s1_greedy | open-ended, 7B judge | greedy | 1200 | 0.085 | 0.045 | 0.665 | 0.703 | 0.707 [0.649, 0.764] |
| hannah_judge7b_s2_full | open-ended, 7B judge | full | 1200 | 0.122 | 0.080 | 0.691 | 0.704 | 0.686 [0.629, 0.741] |
| hannah_judge7b_s2_random | open-ended, 7B judge | random | 1200 | 0.085 | 0.070 | 0.672 | 0.635 | 0.660 [0.596, 0.720] |
| hannah_judge7b_s2_thrcons | open-ended, 7B judge | thrcons | 1200 | 0.083 | 0.060 | 0.622 | 0.709 | 0.664 [0.601, 0.725] |
| hannah_judge7b_s2_greedy | open-ended, 7B judge | greedy | 1200 | 0.083 | 0.060 | 0.700 | 0.691 | 0.702 [0.642, 0.762] |
| m10_proxy_s0 | GSM8K (vote agreement) | proxy | 800 | 0.135 | 0.145 | 0.900 | 0.869 | 0.880 [0.843, 0.911] |
| m10_proxy_s1 | GSM8K (vote agreement) | proxy | 800 | 0.175 | 0.085 | 0.950 | 0.947 | 0.922 [0.895, 0.947] |
| m14_threshold_b10_s0 | GSM8K (vote agreement) | threshold | 800 | 0.155 | 0.105 | 0.877 | 0.880 | 0.898 [0.862, 0.929] |
| m14_threshold_b10_s1 | GSM8K (vote agreement) | threshold | 800 | 0.125 | 0.070 | 0.930 | 0.975 | 0.931 [0.902, 0.956] |

In code, OLMo (visible-test grader), confidence's AUC for predicting proxy_error rises (0.873 to 0.900) over training; threshold and proxy whole-run AUCs are close (0.900 vs 0.882). In code, Qwen (visible-test grader), confidence's AUC for predicting proxy_error falls (0.865 to 0.842) over training; threshold and proxy whole-run AUCs are close (0.866 vs 0.864). In open-ended, 7B judge, confidence's AUC for predicting proxy_error stays flat (~0.684 to ~0.675) over training. In GSM8K (vote agreement), confidence's AUC for predicting proxy_error stays flat (~0.914 to ~0.918) over training; threshold and proxy whole-run AUCs are close (0.915 vs 0.901). Per-completion correctness isn't in the log (proxy_error is one bool per group), so as a group-level stand-in: across the code runs, the fraction of groups with >=1 visible-test pass that still had proxy_error true moves from 0.199 (first block) to 0.169 (last block), while the mean visible_pass_count moves from 4.17 to 5.99.
