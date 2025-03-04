#ifndef _GRID_CUDA_H_
#define _GRID_CUDA_H_


#include <iostream>
#include <stdexcept>

#include "Grid3DCU.h"
#include "cudaTypeDef.cuh"

class grid3DCUDA{

public:

    /** number of cells - X direction, including + 2 (guard cells) */
    int nxc;
    /** number of nodes - X direction, including + 2 extra nodes for guard cells */
    int nxn;
    /** number of cell - Y direction, including + 2 (guard cells) */
    int nyc;
    /** number of nodes - Y direction, including + 2 extra nodes for guard cells */
    int nyn;
    /** number of cell - Z direction, including + 2 (guard cells) */
    int nzc;
    /** number of nodes - Z direction, including + 2 extra nodes for guard cells */
    int nzn;
    /** dx = space step - X direction */
    cudaParticleType dx;
    /** dy = space step - Y direction */
    cudaParticleType dy;
    /** dz = space step - Z direction */
    cudaParticleType dz;
    /** invdx = 1/dx */
    cudaParticleType invdx;
    /** invdy = 1/dy */
    cudaParticleType invdy;
    /** invdz = 1/dz */
    cudaParticleType invdz;
    /** volume of mesh cell*/
    cudaParticleType VOL;
    /** invol = inverse of volume*/
    cudaParticleType invVOL;

    /** local grid boundaries coordinate of proper subdomain */
    cudaParticleType xStart, xEnd, yStart, yEnd, zStart, zEnd;
    cudaParticleType xStart_g, yStart_g, zStart_g;

    // calculate them 
    // cudaCommonType *node_xcoord;
    // cudaCommonType *node_ycoord;
    // cudaCommonType *node_zcoord;

private:

    const static bool suppress_runaway_particle_instability = true;

    cudaCommonType epsilon;
    cudaCommonType nxc_minus_epsilon;
    cudaCommonType nyc_minus_epsilon;
    cudaCommonType nzc_minus_epsilon;

    /** index of last cell including ghost cells */
    // (precomputed for speed)
    int cxlast; // nxc-1;
    int cylast; // nyc-1;
    int czlast; // nzc-1;

    __host__ __device__ static void get_weights(cudaParticleType weights[8],
                            cudaParticleType w0x, cudaParticleType w0y, cudaParticleType w0z,
                            cudaParticleType w1x, cudaParticleType w1y, cudaParticleType w1z)
    {
        weights[0] = w1x*w1y*w1z; // weight111
        weights[1] = w0x*w1y*w1z; // weight011
        weights[2] = w0x*w0y*w1z; // weight001
        weights[3] = w1x*w0y*w1z; // weight101
        weights[4] = w1x*w1y*w0z; // weight110
        weights[5] = w0x*w1y*w0z; // weight010
        weights[6] = w0x*w0y*w0z; // weight000
        weights[7] = w1x*w0y*w0z; // weight100
    }

    __host__ __device__ void make_grid_position_safe(cudaParticleType& cx_pos, cudaParticleType& cy_pos, cudaParticleType& cz_pos)const
    {
        // if the position is outside the domain, then map
        // it to the edge of the guarded subdomain
        //
        if (cx_pos < epsilon) cx_pos = epsilon;
        if (cy_pos < epsilon) cy_pos = epsilon;
        if (cz_pos < epsilon) cz_pos = epsilon;
        if (cx_pos > nxc_minus_epsilon) cx_pos = nxc_minus_epsilon;
        if (cy_pos > nyc_minus_epsilon) cy_pos = nyc_minus_epsilon;
        if (cz_pos > nzc_minus_epsilon) cz_pos = nzc_minus_epsilon;
    }

    __host__ __device__ void make_cell_coordinates_safe(int& cx, int& cy, int& cz)const
    {
        // if the cell is outside the domain, then treat it as
        // in the nearest ghost cell.
        //
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;
        if (cz < 0) cz = 0;
        if (cx > cxlast) cx = cxlast; //nxc-1;
        if (cy > cylast) cy = cylast; //nyc-1;
        if (cz > czlast) cz = czlast; //nzc-1;
    }

public:

    __host__ void initCommonElement(Grid3DCU* grid){
        nxc = grid->getNXC();
        nyc = grid->getNYC();
        nzc = grid->getNZC();

        nxn = grid->getNXN();
        nyn = grid->getNYN();
        nzn = grid->getNZN();

        dx = grid->getDX();
        dy = grid->getDY();
        dz = grid->getDZ();

        invdx = grid->get_invdx();
        invdy = grid->get_invdy();
        invdz = grid->get_invdz();

        VOL = grid->getVOL();
        invVOL = grid->getInvVOL();

        xStart = grid->getXstart();
        yStart = grid->getYstart();
        zStart = grid->getZstart();

        xEnd = grid->getXend();
        yEnd = grid->getYend();
        zEnd = grid->getZend();

        xStart_g = xStart - dx;
        yStart_g = yStart - dy;
        zStart_g = zStart - dz;

        epsilon = grid->getEpsilon();
        nxc_minus_epsilon = nxc - epsilon;
        nyc_minus_epsilon = nyc - epsilon;
        nzc_minus_epsilon = nzc - epsilon;
    }

    // __host__ void initDynamicElement(){
    //     using namespace std;
    //     auto size{nxn * sizeof(cudaCommonType)};
    //     cudaErrChk(cudaMalloc(&node_xcoord, size * 3));
    //     node_ycoord = node_xcoord + size;
    //     node_zcoord = node_ycoord + size;
    // }

    __host__ grid3DCUDA(Grid3DCU* grid){

        // init the normal elements from the grid
        initCommonElement(grid);
        // dynamic allocation and copy to device 
        //initDynamicElement();

    }

    __host__ __device__ void get_safe_cell_and_weights(
    cudaParticleType xpos, cudaParticleType ypos, cudaParticleType zpos,
    int &cx, int& cy, int& cz,
    cudaParticleType weights[8])const
    {
    //convert_xpos_to_cxpos(xpos,ypos,zpos,cx_pos,cy_pos,cz_pos);
    // gxStart marks start of guarded domain (including ghosts)
    const cudaParticleType rel_xpos = xpos - xStart_g;
    const cudaParticleType rel_ypos = ypos - yStart_g;
    const cudaParticleType rel_zpos = zpos - zStart_g;
    // cell position (in guarded array)
    cudaParticleType cx_pos = rel_xpos * invdx;
    cudaParticleType cy_pos = rel_ypos * invdy;
    cudaParticleType cz_pos = rel_zpos * invdz;
    //
    if(suppress_runaway_particle_instability)
        make_grid_position_safe(cx_pos,cy_pos,cz_pos);
    //
    cx = int(floor(cx_pos));
    cy = int(floor(cy_pos));
    cz = int(floor(cz_pos));
    // this was the old algorithm.
    if(!suppress_runaway_particle_instability)
        make_cell_coordinates_safe(cx,cy,cz);
    //assert_cell_coordinates_safe(cx,cy,cz); 

    // fraction of distance from the left
    const cudaParticleType w0x = cx_pos - cx;
    const cudaParticleType w0y = cy_pos - cy;
    const cudaParticleType w0z = cz_pos - cz;
    // fraction of the distance from the right of the cell
    const cudaParticleType w1x = 1.f-w0x;
    const cudaParticleType w1y = 1.f-w0y;
    const cudaParticleType w1z = 1.f-w0z;

    get_weights(weights, w0x, w0y, w0z, w1x, w1y, w1z);
    }


    __device__ double getXN(uint32_t index){return xStart + (index - 1) * dx;}
    __device__ double getYN(uint32_t index){return yStart + (index - 1) * dy;}
    __device__ double getZN(uint32_t index){return zStart + (index - 1) * dz;}

};





#endif