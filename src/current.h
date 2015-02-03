// current.h

#ifndef CURRENT_H_
#define CURRENT_H_

// compute J using the standard derivatives as in [4] eq. (2.9)
__global__ void JClassic(struct varsDev_t d, int dimX, int dimY, int dimZ);

// compute the electric current density using mimetic operators (Stokes method)
__global__ void JStokes(struct varsDev_t d, int dimX, int dimY, int dimZ);

// compute the electric current density via the 4th orderStokes method
__global__ void JStokes4th(struct varsDev_t d, int dimX, int dimY, int dimZ);

// compute the electric current density via the Stokes method using a quintilateral
__global__ void JStokesQuint(struct varsDev_t d, int dimX, int dimY, int dimZ);

// compute the electric current density via the Stokes method using additional triangles
__global__ void JStokesTri(struct varsDev_t d, int dimX, int dimY, int dimZ);

// determine which routine should be used for the current calculation
void current(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct varsDev_t d, struct parameters_t p);

#endif /* CURRENT_H_ */
