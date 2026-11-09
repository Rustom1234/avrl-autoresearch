# Burn-in diagnosis

n groups: 200
proxy error rate: 0.1600 (95% CI 0.1150 to 0.2100)
AUC (low confidence predicts error): 0.8569 (95% CI 0.7755 to 0.9266)

corruption by budget, threshold vs random:
  budget 0.1: threshold 0.0800 (CI 0.0350 to 0.1250) vs random 0.1440 (CI 0.0989 to 0.1890)
  budget 0.25: threshold 0.0400 (CI 0.0150 to 0.0600) vs random 0.1200 (CI 0.0824 to 0.1575)
  budget 0.5: threshold 0.0300 (CI 0.0050 to 0.0450) vs random 0.0800 (CI 0.0549 to 0.1050)

grid (0.05 steps), threshold vs random corruption:
  budget 0.05: threshold 0.1200 vs random 0.1520
  budget 0.10: threshold 0.0800 vs random 0.1440
  budget 0.15: threshold 0.0400 vs random 0.1360
  budget 0.20: threshold 0.0400 vs random 0.1280
  budget 0.25: threshold 0.0400 vs random 0.1200
  budget 0.30: threshold 0.0300 vs random 0.1120
  budget 0.35: threshold 0.0300 vs random 0.1040
  budget 0.40: threshold 0.0300 vs random 0.0960
  budget 0.45: threshold 0.0300 vs random 0.0880
  budget 0.50: threshold 0.0300 vs random 0.0800
  budget 0.55: threshold 0.0300 vs random 0.0720
  budget 0.60: threshold 0.0300 vs random 0.0640
  budget 0.65: threshold 0.0300 vs random 0.0560
  budget 0.70: threshold 0.0300 vs random 0.0480
  budget 0.75: threshold 0.0300 vs random 0.0400
  budget 0.80: threshold 0.0250 vs random 0.0320
  budget 0.85: threshold 0.0200 vs random 0.0240
  budget 0.90: threshold 0.0200 vs random 0.0160
  budget 0.95: threshold 0.0100 vs random 0.0080
  budget 1.00: threshold 0.0000 vs random 0.0000

realizable candidates (n_groups_per_step=8, pick=upper):
  budget 0.1250 (k=1): threshold point 0.0600, threshold upper 0.1050, random point 0.1400
  budget 0.2500 (k=2): threshold point 0.0400, threshold upper 0.0600, random point 0.1200
  budget 0.3750 (k=3): threshold point 0.0300, threshold upper 0.0550, random point 0.1000
  budget 0.5000 (k=4): threshold point 0.0300, threshold upper 0.0450, random point 0.0800
  budget 0.6250 (k=5): threshold point 0.0300, threshold upper 0.0350, random point 0.0600
  budget 0.7500 (k=6): threshold point 0.0300, threshold upper 0.0251, random point 0.0400
  budget 0.8750 (k=7): threshold point 0.0200, threshold upper 0.0150, random point 0.0200
  budget 1.0000 (k=8): threshold point 0.0000, threshold upper 0.0000, random point 0.0000
recommended under point rule: 0.2500
recommended under upper rule: 0.6250

target: relative 0.0400 (base error 0.1600)
recommended budget for target corruption rel:0.25: 0.15
regime: strong - use the confidence-ranked threshold at the recommended budget (0.15).

budget mode: auto, chosen budget for the rest of training: 0.62
