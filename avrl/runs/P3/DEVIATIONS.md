# P3 deviations from PREREG.md

PREREG.md is frozen (PREREG.sha256). Entries here record anything done differently from it, with time and reason.

1. 2026-11-06T03:21:05Z, clerical. PREREG.md says it was frozen at "22:20 local, 03:20Z"; PREREG.sha256 was written at 22:21 local, 03:21:05Z. No data, code or Claude call existed in between. The first Claude call was at 03:21:30Z.

2. 2026-11-09, wording only. Three phrases in PREREG.md were reworded for the public release: the proposal is no longer called a scout spec, the launching shell is named instead of the parent shell, and the CLI's project instruction file is no longer named. No rule, number, item, code hash or procedure changed. PREREG.sha256 (3390824cc0ccad277f4c95ecc38ab7acdebfd2f4b2d14d1ed3726716c1002e12) was computed on the text before this rewording, so it no longer matches the file byte for byte.

(No changes to the preregistered design, code, items or decision rules as of the Day 0 gate. The code hashes in PREREG.md were re-checked after Day 0 and still match.)
