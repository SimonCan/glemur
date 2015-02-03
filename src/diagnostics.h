// diagnostics.h

#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

// do some reductions
int reductions(struct red_t *red, struct varsDev_t d, struct parameters_t p, int blockSize[3], int gridSize[3], dim3 dimGrid, dim3 dimBlock);

// prepare the dump from the current state
int prepareDump(struct red_t red, struct varsHost_t h, struct varsDev_t d, struct parameters_t p, int blockSize[3], int gridSize[3], dim3 dimGrid, dim3 dimBlock,
		dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d);

#endif /* DIAGNOSTICS_H_ */
