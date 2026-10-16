# Burn-in diagnosis

n groups: 200
proxy error rate: 0.1600 (95% CI 0.1150 to 0.2100)
AUC (low confidence predicts error): 0.8569 (95% CI 0.7755 to 0.9266)

corruption by budget, threshold vs random:
  budget 0.1: threshold 0.0800 (CI 0.0350 to 0.1250) vs random 0.1440 (CI 0.0989 to 0.1890)
  budget 0.25: threshold 0.0400 (CI 0.0150 to 0.0600) vs random 0.1200 (CI 0.0824 to 0.1575)
  budget 0.5: threshold 0.0300 (CI 0.0050 to 0.0450) vs random 0.0800 (CI 0.0549 to 0.1050)

recommended budget for target corruption 0.02: 0.85
regime: strong - use the confidence-ranked threshold at the recommended budget (0.85).

budget mode: auto, chosen budget for the rest of training: 0.85
