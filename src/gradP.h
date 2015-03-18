// gradP.h

#ifndef GRADP_H_
#define GRADP_H_

// compute grad(p) using the standard derivatives
__global__ void gradPClassic(struct varsDev_t d, int dimX, int dimY, int dimZ);

// determine which routine should be used for the current calculation
void gradP(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct varsDev_t d, struct parameters_t p);

#endif /* GRADP_H_ */
