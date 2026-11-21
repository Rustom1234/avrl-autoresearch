#!/bin/bash
# Calibration jobs for one host, run one after another: 3 baseline jobs and one job each at 0.9 and 1.1 times the epochs.
cd /homes/rdubash/active-audit/loop/cifar
H=$(hostname -s); PY=/homes/rdubash/speedrun/.venv/bin/python
for j in base_1:base/train.py e090_1:calib/train_e090.py base_2:base/train.py e110_1:calib/train_e110.py base_3:base/train.py; do
  n=${j%%:*}; f=${j##*:}
  [ -f calib/jobs/${H}_$n/result.json ] || $PY harness.py $f --out calib/jobs/${H}_$n --runs 24 --sealed --seed ${n##*_} --mem-gb 8 > calib/jobs/${H}_$n.log 2>&1
done
