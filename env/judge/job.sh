#!/bin/bash -x
# see https://computing.llnl.gov/tutorials/moab/
# see http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUDGE/Userinfo/Quick_Introduction.html
# judge uses moab, so start job as: msub job.sh
#MSUB -l nodes=64:ppn=8
# using a walltime of at most 5 minutes is supposed to give higher priority
#MSUB -l walltime=00:05:00
##MSUB -e /homec/deep/deep38/judge/work/error.txt
##MSUB -o /homec/deep/deep38/judge/work/output.txt
#MSUB -N ipic3d
# combine standard error and standard output
#MSUB -j oe
#MSUB -M Alec.Johnson@wis.kuleuven.be
# when to send mail: a=abort,b=beginning,e=end; default: -m a
#MSUB -m abe
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
