#!/bin/bash
#
# template SLURM script for use with MPI
# rush this script with: sbatch job.sh
# monitor jobs with:
#   showq
#    squeue -u `whoami`
# kill jobs with:
#   scancel <jobId>
#
# alternatively you can submit an interactive
# job on a single node with
#
#   srun --pty -n 16 -t 01:00:00 -p development /bin/bash -l
#
# or equivalently: idev -pe 16 1 -m 60 # 1 = single node
#
# job name
#SBATCH -J ipic3d
##SBATCH -o ipic3d.%j.out
##SBATCH -e ipic3d.%j.err
# queue (development or normal)
#SBATCH -p normal
# number of nodes, not cores (16 cores per node)
#TemplateLine: #SBATCH -N $NUM_NODES
#SBATCH -N 1
# total number of MPI tasks (if omitted, n=N)
#TemplateLine: #SBATCH -n $NUM_PROCS
#SBATCH -n 16
# maximum time
#SBATCH -t 00:05:00
#SBATCH --mail-user=Alec.Johnson@wis.kuleuven.be
#SBATCH --mail-type=ALL  

#module load ipic
#module list

#TemplateLine: export XLEN=$XLEN
export XLEN=4
#TemplateLine: export YLEN=$YLEN
export YLEN=4
# export OMP_NUM_THREADS=8
# NUM_PROCS = nodes * ppn / OMP_NUM_THREADS
#TemplateLine: NUM_PROCS=$NUM_PROCS
NUM_PROCS=$(($XLEN*$YLEN))
DATA=data

# if running on Xeon
if false
then
  # use ibrun for MPI codes, not mpirun or srun
  ibrun -np $NUM_PROCS ./iPic3D "$DATA"/parameters.inp | tee out.${XLEN}x${YLEN}.txt
# if running on MIC
else
  set -x
  scontrol show hostname | sed 's/$/-mic0/' > "$DATA"/machinefile
  #ibrun.symm -np $NUM_PROCS -machinefile machinefile -env LD_LIBRARY_PATH $MIC_LD_LIBRARY_PATH -m ./iPic3D "$DATA"/parameters.inp | tee out.${XLEN}x${YLEN}.mic.txt
  mpiexec.hydra -np $NUM_PROCS -machinefile "$DATA"/machinefile -env LD_LIBRARY_PATH $MIC_LD_LIBRARY_PATH ./iPic3D "$DATA"/parameters.inp | tee out.${XLEN}x${YLEN}.mic.txt
fi
