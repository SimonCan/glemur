// diagnostics.h

#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

// Perform data reduction.
int reductions(struct Reduction *red, struct VarsDev d, struct Parameters params, int blockSize[3], int gridSize[3], dim3 dimGrid, dim3 dimBlock);

// Prepare the dump from the current state.
int prepareDump(struct Reduction red, struct VarsHost h, struct VarsDev d, struct Parameters params, int blockSize[3], int gridSize[3], dim3 dimGrid, dim3 dimBlock,
        dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d);

#endif /* DIAGNOSTICS_H_ */
