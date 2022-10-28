#!/bin/bash -x
# start job as: qsub job.sh
#PBS -l nodes=64:ppn=8
# using a walltime of at most 5 minutes is supposed to give higher priority
#PBS -l walltime=00:15:00
##PBS -e /work/deep38/error.txt
##PBS -o /work/deep38/output.txt
#PBS -N ipic3d
# combine standard error and standard output
#PBS -j oe
#PBS -M Alec.Johnson@wis.kuleuven.be
# when to send mail: a=abort,b=beginning,e=end; default: -m a
#PBS -m abe
### start of jobscript

cd $PBS_O_WORKDIR
echo "workdir: $PBS_O_WORKDIR"

export XLEN=16
export YLEN=16
# export OMP_NUM_THREADS=8
# NUM_PROCS = nodes * ppn / OMP_NUM_THREADS
NUM_PROCS=$(($XLEN*$YLEN))
DATA=data

mpiexec -np $NUM_PROCS ./iPic3D $DATA/parameters.inp | tee out.${XLEN}x${YLEN}.txt
