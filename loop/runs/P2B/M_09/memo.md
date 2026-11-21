Incumbent mean_adjusted is 1.1145 over 16 runs. Every slot has one run, so gaps near 0.002 may be noise. Nothing has been accepted yet.

Helped slightly, but all rejected: learnable query gain (1.1116, -0.0029), gradient clipping (1.1131, -0.0014), a separate Muon warmdown (1.1132, -0.0013), MATRIX_LR 0.06 (1.1124, -0.0021). All are cheap and add no compute per step. Weight decay 0.01 on the AdamW groups was -0.0005, which is likely noise.

Hurt: depth 10 (+0.0196), a 2^20 token batch (+0.0671), warmup 0.03 (+0.0095), a value residual (+0.0080), SwiGLU (+0.0041), warmdown 0.7 (+0.0017), a weight EMA (+0.0012). Anything that adds compute per step or cuts the step count looks bad under the fixed 300 s budget.

Slots 7 and 14 failed to patch, and slots 11 and 15 crashed. The 2^18 batch and the Muon momentum warmup are still untested.

Next: retry those two with patches that apply. Then stack query gain, clipping, the Muon warmdown and MATRIX_LR 0.06, and replicate the stack against noise.
