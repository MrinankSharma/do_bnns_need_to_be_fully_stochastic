#!/bin/bash

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode: pass a file with first the job tag and second the commands per line

# Inspired by https://github.com/y0ast/slurm-for-ml/blob/master/generic.sh

#SBATCH --cpus-per-task=6 # increase cpus for more memory effectively
#SBATCH --gres=gpu:4 #ask for 4 gpus now
#SBATCH --partition=short
#SBATCH --time=11:59:59

set -e # fail fully on first line failure
echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

TIME_STR=$(date '+%m-%d_%H-%M-%S')
# Train the models
FILENAME="${TIME_STR}.txt"
echo "srun --output ${HOME}/slurm_outputs/targetted_uncertainty/${FILENAME}.out singularity exec --nv --bind ${DATA}:${DATA} ${HOME}/ub.sif python3.9 $JOB_CMD"
srun --output ${HOME}/slurm_outputs/targetted_uncertainty/${FILENAME}.out singularity exec --nv --bind ${DATA}:${DATA} ${HOME}/ub.sif python3.9 $JOB_CMD