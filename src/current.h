// current.h

#ifndef CURRENT_H_
#define CURRENT_H_

// Compute J using the standard derivatives as in [4] eq. (2.9).
__global__ void JClassic(struct VarsDev d, int dimX, int dimY, int dimZ);

// Compute the electric current density using mimetic operators (Stokes method).
__global__ void JStokes(struct VarsDev d, int dimX, int dimY, int dimZ);

// Compute the electric current density via the 4th orderStokes method.
__global__ void JStokes4th(struct VarsDev d, int dimX, int dimY, int dimZ);

// Compute the electric current density via the Stokes method using a quintilateral.
__global__ void JStokesQuint(struct VarsDev d, int dimX, int dimY, int dimZ);

// Compute the electric current density via the Stokes method using additional triangles.
__global__ void JStokesTri(struct VarsDev d, int dimX, int dimY, int dimZ);

// Determine which routine should be used for the current calculation.
void current(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct VarsDev d, struct Parameters p);

#endif /* CURRENT_H_ */
