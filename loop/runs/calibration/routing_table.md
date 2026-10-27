# Routing table

| role | model |
|---|---|
| coder constant | qwen35-4b |
| coder schedule | gemma4-12b |
| coder structural | haiku |
| repairer | haiku |
| planner_cheap (also planner) | qwen35-4b |
| analyst | sonnet |
| voter | gemma4-12b |
| critic | opus |

## Evidence (k/n passed, Wilson 95% lower bound vs bar; infra failures excluded)

| model | class | k/n | lb | bar | qualified | mean cost | mean latency s | top failure |
|---|---|---|---|---|---|---|---|---|
| gemma4-12b | constant | 15/16 | 0.717 | 0.6 | yes | 0.0016 | 6.9 | apply |
| gemma4-12b | schedule | 16/16 | 0.806 | 0.6 | yes | 0.0017 | 7.2 | - |
| gemma4-12b | structural | 10/16 | 0.386 | 0.75 | no | 0.0018 | 14.9 | apply |
| gemma4-12b | repair | 7/12 | 0.320 | 0.6 | no | 0.0016 | 6.1 | apply |
| gemma4-12b | planner | 12/12 | 0.757 | 0.6 | yes | 0.0005 | 12.2 | - |
| gemma4-12b | analyst | 8/12 | 0.391 | 0.6 | no | 0.0003 | 4.9 | numbers |
| haiku | constant | 16/16 | 0.806 | 0.6 | yes | 0.0022 | 3.3 | - |
| haiku | schedule | 16/16 | 0.806 | 0.6 | yes | 0.0018 | 5.8 | - |
| haiku | structural | 16/16 | 0.806 | 0.75 | yes | 0.0018 | 7.1 | - |
| haiku | repair | 12/12 | 0.757 | 0.6 | yes | 0.0014 | 9.9 | - |
| haiku | planner | 12/12 | 0.757 | 0.6 | yes | 0.0011 | 10.2 | - |
| haiku | analyst | 9/12 | 0.468 | 0.6 | no | 0.0008 | 6.7 | numbers |
| qwen35-4b | constant | 15/16 | 0.717 | 0.6 | yes | 0.0005 | 2.0 | apply |
| qwen35-4b | schedule | 10/16 | 0.386 | 0.6 | no | 0.0005 | 3.4 | apply |
| qwen35-4b | structural | 10/16 | 0.386 | 0.75 | no | 0.0006 | 9.2 | expect |
| qwen35-4b | repair | 9/12 | 0.468 | 0.6 | no | 0.0004 | 1.9 | expect |
| qwen35-4b | planner | 12/12 | 0.757 | 0.6 | yes | 0.0001 | 4.4 | - |
| qwen35-4b | analyst | 7/12 | 0.320 | 0.6 | no | 0.0001 | 1.7 | numbers |
| qwen35-9b | constant | 15/16 | 0.717 | 0.6 | yes | 0.0009 | 3.1 | apply |
| qwen35-9b | schedule | 12/16 | 0.505 | 0.6 | no | 0.0010 | 8.2 | touched |
| qwen35-9b | structural | 8/16 | 0.280 | 0.75 | no | 0.0012 | 31.8 | apply |
| qwen35-9b | repair | 6/12 | 0.254 | 0.6 | no | 0.0010 | 9.3 | apply |
| qwen35-9b | planner | 12/12 | 0.757 | 0.6 | yes | 0.0003 | 11.2 | - |
| qwen35-9b | analyst | 7/12 | 0.320 | 0.6 | no | 0.0002 | 4.1 | numbers |
| r1-qwen3-8b | constant | 0/16 | 0.000 | 0.6 | no | 0.0016 | 57.5 | schema |
| r1-qwen3-8b | schedule | 0/16 | 0.000 | 0.6 | no | 0.0018 | 77.1 | schema |
| r1-qwen3-8b | structural | 0/16 | 0.000 | 0.75 | no | 0.0019 | 81.8 | schema |
| r1-qwen3-8b | repair | 0/12 | 0.000 | 0.6 | no | 0.0017 | 66.2 | schema |
| r1-qwen3-8b | planner | 2/12 | 0.047 | 0.6 | no | 0.0013 | 63.1 | schema |
| r1-qwen3-8b | analyst | 0/12 | 0.000 | 0.6 | no | 0.0013 | 65.4 | schema |
| sonnet | constant | 16/16 | 0.806 | 0.6 | yes | 0.0542 | 3.0 | - |
| sonnet | schedule | 16/16 | 0.806 | 0.6 | yes | 0.0394 | 4.5 | - |
| sonnet | structural | 16/16 | 0.806 | 0.75 | yes | 0.0315 | 6.1 | - |
| sonnet | repair | 12/12 | 0.757 | 0.6 | yes | 0.0414 | 4.7 | - |
| sonnet | planner | 12/12 | 0.757 | 0.6 | yes | 0.0136 | 6.4 | - |
| sonnet | analyst | 3/12 | 0.089 | 0.6 | no | 0.0076 | 4.9 | numbers |

- calibration: 504 trials, nominal cost $3.2295, plan cost $2.8858, summed call latency 8844.6 s
