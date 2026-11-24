#!/bin/bash
# Shows each rank only the GCD of its node-local rank, then runs the command.
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
