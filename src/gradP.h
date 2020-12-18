// gradP.h

#ifndef GRADP_H_
#define GRADP_H_

// Compute grad(p) using the standard derivatives.
__global__ void gradPClassic(struct VarsDev d, int dimX, int dimY, int dimZ);

// Determine which routine should be used for the current calculation.
void gradP(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct VarsDev d, struct Parameters params);

#endif /* GRADP_H_ */
