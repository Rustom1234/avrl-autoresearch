Helped: two accepted changes that make each step cheaper under the fixed 300 s budget. DEPTH 8 to 6 gave 1.0828/1.0866 (-0.0298), and total batch 2**17 gave 1.0610/1.0623 (-0.0231). Both carry a tripwire flag, so check it before building on them. Incumbent mean is 1.0617 over 2 runs.

Did not help: larger or wider models (ASPECT_RATIO 80 at 1.1010, DEPTH 10 at 1.1456), HEAD_DIM 64 (+0.0030), MLP 3x (+0.0046), WARMUP_RATIO 0.02, shorter warmdown (+0.0031). Small gains such as SCALAR_LR 0.25 (-0.0012), gradient clipping (-0.0006), and removing LR scaling (-0.0005) were rejected as likely noise. Three runs crashed or aborted: two from AdamW with bogus imports, one a repeated zero-init.

Next: smaller models (DEPTH 5, lower ASPECT_RATIO) and a smaller batch (2**16). Replicate any gain under 0.005. Avoid optimizer rewrites.
