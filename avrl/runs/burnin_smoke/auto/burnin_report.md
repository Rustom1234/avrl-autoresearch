# Burn-in diagnosis

n groups: 48
proxy error rate: 0.0208 (95% CI 0.0000 to 0.0625)
AUC (low confidence predicts error): 0.5000 (95% CI 0.5000 to 0.5000)

corruption by budget, threshold vs random:
  budget 0.1: threshold 0.0000 (CI 0.0000 to 0.0625) vs random 0.0187 (CI 0.0000 to 0.0563)
  budget 0.25: threshold 0.0000 (CI 0.0000 to 0.0625) vs random 0.0156 (CI 0.0000 to 0.0469)
  budget 0.5: threshold 0.0000 (CI 0.0000 to 0.0417) vs random 0.0104 (CI 0.0000 to 0.0312)

recommended budget for target corruption 0.02: 0.10
regime: uninformative - this confidence signal is close to random, verification budget buys little per call, use a different confidence signal such as response consistency or raise the budget.

budget mode: auto, chosen budget for the rest of training: 0.10
