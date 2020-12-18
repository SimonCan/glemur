// boundaries.h

#ifndef BOUNDARIES_H_
#define BOUNDARIES_H_

#include "global.h"

// In case of a distorted initial grid we need to add the distortion to the "second" ghost cells.
__device__ REAL addDist(int i, int j, REAL z, int l);

// Update B and detJac at the boundary (previously done within B_JacB0 (inefficient)).
__global__ void updateBbound(struct VarsDev d, int face);

// Set vector field 'field' to be periodic.
__global__ void setPeriFace(REAL *field, int face);

// Set 'field' to be periodic (host code).
void setPeriHost(REAL *field, struct Parameters params);

// Set the grid to be periodic.
__global__ void setGridPeriFace(REAL *xb, int face);

// Set the grid to be periodic (host code).
void setGridPeriHost(REAL *xb, struct Parameters params);

void setBbound(dim3 dimGrid2dXY, dim3 dimGrid2dXZ, dim3 dimGrid2dYZ, dim3 dimBlock2d, struct VarsDev d);

void setPeri(dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d, REAL *dev_field, struct Parameters params);

void setGridPeri(dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d, REAL *dev_xb, struct Parameters params);

#endif /* BOUNDARIES_H_ */
