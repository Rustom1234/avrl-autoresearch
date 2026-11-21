Helped: search-structure changes. The physics relaxation (slot 0, -1.2801) and the two-phase coarse/refine search (slot 3, -0.3475) gave the big gains. Smaller gains came from the population seed (slot 7, -0.0166), basin-hopping (slot 8, -0.0183) and the reinsertion move (slot 13, -0.0046). The incumbent is -2.6360, which beats the best known 2.635.

Did not help: new starts and seedings (mirror +0.0217, corner star +0.0228, Riesz +0.0209, torus +0.0116), the penalty objective (+0.0170), annealing (+0.0183) and the trust region (+0.0503). Most basin-hopping, polish and search variants returned +0.0000, so this basin looks saturated. Many slots were aborted as duplicates of code already in train.py.

Next: replicate the incumbent once to check noise and the validity margin. Then try a longer time budget (within about 4 minutes) or a different fixed seed. Avoid seeding changes and ideas already in train.py.
