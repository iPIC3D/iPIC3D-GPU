## **Configurations for running the scaling tests on different machines**

---
### **El Capitan**
**Architecture per node**  
APU: AMD300A 
**Software**  
Compiler:   
ROCm version:   
MPI:  
**Hardware**  
1 MPI rank per APU, X OMP threads per MPI rank

---
### **Tuolumne**
**Architecture per node**  
APU: AMD300A 
**Software**  
Compiler:   
ROCm version:   
MPI:  
**Hardware**  
1 MPI rank per APU, X OMP threads per MPI rank

---
### **LUMI-G**
**Architecture per node**  
CPU: 1x 64-core AMD EPYC 7A53 "Trento" + 8x 64GB DDR4  
GPU: 4x AMD MI250x 128GB  
**Software**  
Compiler: clang/17.0  
ROCm version: rocm/6.0.3  
MPI: cray-mpich/8.1.29  
**Hardware**  
1 MPI rank per GCD, 6 OMP threads per MPI rank

---
### **MareNostrum5 ACC**  
**Architecture per node**  
CPU: 2x 40-cores Intel Xeon Platinum 8460Y + 16x DIMM 32GB 4800MHz DDR5  
GPU: 4x NVIDIA Hopper H100 64GB HBM2  
**Software**  
Compiler: gcc/11.4.0  
Cuda version: cuda/12.2  
MPI: openmpi/4.1.5    
**Hardware**  
1 MPI rank per GPU, 20 OMP threads per MPI rank

---
### **Leonardo Booster**  
**Architecture per node**  
CPU: 1x 32-cores Intel Xeon 8358 + 8x 64GB 3200MHz DDR4  
GPU: 4x NVIDIA custom Ampere 64GB HBM2  
**Software**  
Compiler: gcc/12.2
Cuda version: cuda/12.3
MPI: openmpi/4.1.6
**Hardware**
1 MPI rank per GPU, 6 OMP threads per MPI rank

---
### **Lassen**  
**Architecture per node**  
CPU:   
GPU:   
**Software**  
Compiler: 
Cuda version: 
MPI: 
**Hardware**
1 MPI rank per GPU, X OMP threads per MPI rank

---
## **iPIC3D Magnetosphere Simulation Parameters**
