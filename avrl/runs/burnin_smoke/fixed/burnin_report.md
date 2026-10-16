# Burn-in diagnosis

n groups: 32
proxy error rate: 0.0312 (95% CI 0.0000 to 0.0938)
AUC (low confidence predicts error): 0.5000 (95% CI 0.5000 to 0.5000)

corruption by budget, threshold vs random:
  budget 0.1: threshold 0.0312 (CI 0.0000 to 0.0938) vs random 0.0281 (CI 0.0000 to 0.0844)
  budget 0.25: threshold 0.0000 (CI 0.0000 to 0.0938) vs random 0.0234 (CI 0.0000 to 0.0703)
  budget 0.5: threshold 0.0000 (CI 0.0000 to 0.0625) vs random 0.0156 (CI 0.0000 to 0.0469)

recommended budget for target corruption 0.02: 0.15
regime: uninformative - this confidence signal is close to random, verification budget buys little per call, use a different confidence signal such as response consistency or raise the budget.

budget mode: fixed, training budget stays at 0.25 as configured
