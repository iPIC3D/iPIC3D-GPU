#ifndef _MOVERKERNEL_CUH_
#define _MOVERKERNEL_CUH_

#include "Particles3D.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "EMfields3D.h"
#include "gridCUDA.cuh"



/*
class moverOutflowParameter
{
    public:

    uint32_t nop; // number of particles (of the species)

    bool isBoundaryProcess_P;
    bool openXleft, openYleft, openZleft,
            openXright, openYright, openZright;


    __host__ moverOutflowParameter(VirtualTopology3D* vct, Particles3D* p3D)
    {

        isBoundaryProcess_P = vct->isBoundaryProcess_P();
        // The below is OpenBC outflow for all other boundaries
        using namespace BCparticles;

        // the open boundary out condition 3
        openXleft = !vct->getPERIODICX_P() && vct->noXleftNeighbor_P() && p3D->bcPfaceXleft == OPENBCOut;
        openYleft = !vct->getPERIODICY_P() && vct->noYleftNeighbor_P() && p3D->bcPfaceYleft == OPENBCOut;
        openZleft = !vct->getPERIODICZ_P() && vct->noZleftNeighbor_P() && p3D->bcPfaceZleft == OPENBCOut;

        openXright = !vct->getPERIODICX_P() && vct->noXrghtNeighbor_P() && p3D->bcPfaceXright == OPENBCOut;
        openYright = !vct->getPERIODICY_P() && vct->noYrghtNeighbor_P() && p3D->bcPfaceYright == OPENBCOut;
        openZright = !vct->getPERIODICZ_P() && vct->noZrghtNeighbor_P() && p3D->bcPfaceZright == OPENBCOut;
    }

};
*/

class moverParameter
{

public: //particle arrays

    particleArrayCUDA* pclsArray; // default main array

    arrayCUDA<uint32_t>* departureArray; // a helper array for marking exiting particles


public: // common parameter

    cudaCommonType dt;

    cudaCommonType qom;

    cudaCommonType c;

    int NiterMover;

    int DFIELD_3or4;

    cudaCommonType umax, umin, vmax, vmin, wmax, wmin;

    // moverOutflowParameter outflowParam;


public:


    __host__ moverParameter(Particles3D* p3D, VirtualTopology3D* vct)
        : dt(p3D->dt), qom(p3D->qom), c(p3D->c), NiterMover(p3D->NiterMover), DFIELD_3or4(::DFIELD_3or4),
        umax(p3D->umax), umin(p3D->umin), vmax(p3D->vmax), vmin(p3D->vmin), wmax(p3D->wmax), wmin(p3D->wmin)
    {
        // create the particle array, stream 0
        pclsArray = particleArrayCUDA(p3D).copyToDevice();
        departureArray = arrayCUDA<uint32_t, 64>(p3D->getNOP() * 1.5).copyToDevice();

    }

    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ moverParameter(Particles3D* p3D, particleArrayCUDA* pclsArrayCUDAPtr)
        : dt(p3D->dt), qom(p3D->qom), c(p3D->c), NiterMover(p3D->NiterMover), DFIELD_3or4(::DFIELD_3or4),
        umax(p3D->umax), umin(p3D->umin), vmax(p3D->vmax), vmin(p3D->vmin), wmax(p3D->wmax), wmin(p3D->wmin)
    {
        // create the particle array, stream 0
        pclsArray = pclsArrayCUDAPtr;

    }
};

__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaCommonType> fieldForPcls,
                            grid3DCUDA *grid);

#endif