#!/bin/bash
# Submit the three S1 jobs on hannah in a dependency chain. Usage (on the hannah login node): bash launch_s1_hannah.sh <STEPS>
STEPS=${1:?steps per arm}
cd ~/avrl
ja=$(sbatch --parsable --export=ALL,JOB=a,STEPS=$STEPS run_s1_hannah.sbatch) || exit 1
jb=$(sbatch --parsable --dependency=afterany:$ja --export=ALL,JOB=b,STEPS=$STEPS run_s1_hannah.sbatch) || exit 1
jc=$(sbatch --parsable --dependency=afterany:$jb --export=ALL,JOB=c,STEPS=$STEPS run_s1_hannah.sbatch) || exit 1
echo "S1 jobs: a=$ja b=$jb c=$jc (steps $STEPS)"
squeue -u rustom -h -o "%.8i %.14j %.10T %.8M %R %E"
