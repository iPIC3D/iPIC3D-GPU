
basic: # basic.c
	mpicc -o basic basic.c

run.deep:
	mpiexec.hydra -np 4 ./basic

run.stampede: # basic
	mpirun -np 4 ./basic
