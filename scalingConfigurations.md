## **Configurations for running the scaling tests on different machines**

---
### **El Capitan**
**Architecture per node**  <br>
APU: AMD300A <br>
**Software**  <br>
Compiler:  AMD Clang  <br>
ROCm version: ROCm/6.3.1  <br>
MPI: MPICH~8.1.31 <br>
**Hardware**  
1 MPI rank per APU, 21 OMP threads per MPI rank

---
### **Tuolumne**
**Architecture per node**  <br>
APU: AMD300A <br>
**Software**  <br>
Compiler: AMD Clang++  <br>
ROCm version: ROCm/6.3.1  <br>
MPI: MPICH~8.1.31  <br>
**Hardware**  
1 MPI rank per APU, 21 OMP threads per MPI rank

---
### **LUMI-G**
**Architecture per node**  <br>
CPU: 1x 64-core AMD EPYC 7A53 "Trento" + 8x 64GB DDR4  <br>
GPU: 4x AMD MI250x 128GB  <br>
**Software**  
Compiler: clang/17.0  <br>
ROCm version: rocm/6.0.3  <br>
MPI: cray-mpich/8.1.29  <br>
**Hardware**  
1 MPI rank per GCD, 6 OMP threads per MPI rank

---
### **MareNostrum5 ACC**  
**Architecture per node**  
CPU: 2x 40-cores Intel Xeon Platinum 8460Y + 16x DIMM 32GB 4800MHz DDR5  <br>
GPU: 4x NVIDIA Hopper H100 64GB HBM2  <br>
**Software**  
Compiler: gcc/11.4.0  <br>
Cuda version: cuda/12.2  <br>
MPI: openmpi/4.1.5    <br>
**Hardware**  
1 MPI rank per GPU, 20 OMP threads per MPI rank

---
### **Leonardo Booster**  
**Architecture per node**  
CPU: 1x 32-cores Intel Xeon 8358 + 8x 64GB 3200MHz DDR4  <br>
GPU: 4x NVIDIA custom Ampere 64GB HBM2  <br>
**Software**  
Compiler: gcc/12.2 <br>
Cuda version: cuda/12.3 <br>
MPI: openmpi/4.1.6 <br>
**Hardware**
1 MPI rank per GPU, 6 OMP threads per MPI rank

---
### **Lassen**  
**Architecture per node**  
CPU: IBM POWER9  <br>
GPU: Nvidia V100  <br>
**Software**  
Compiler: gcc/12.2.1 <br>
Cuda version: cuda/12.2.2 <br>
MPI: spectrum-mpi/spectrum-mpi-rolling-release-gcc-12.2.1 <br>
**Hardware**
1 MPI rank per GPU, 10 OMP threads per MPI rank

---
