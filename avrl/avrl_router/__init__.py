"""avrl_router: standalone verification-budget diagnostic.

Given a sample of rollout groups with a cheap grader's confidence and a flag for whether the
cheap grader was wrong, it reports how informative the confidence is and how much
verification budget it takes to keep reward corruption under a target. See diagnose.py.

Library use: from avrl_router.diagnose import diagnose
CLI use: python -m avrl_router.diagnose --help
"""
