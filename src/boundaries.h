/*
 * boundaries.h
 *
 *  Created on: 13 Oct 2014
 *      Author: iomsn
 */

#ifndef BOUNDARIES_H_
#define BOUNDARIES_H_

#include "global.h"

// in case of a distorted initial grid we need to add the distortion to the "second" ghost cells
__device__ REAL addDist(int i, int j, REAL z, int l);

// update B and detJac at the boundary (previously done within B_JacB0 (inefficient))
__global__ void updateBbound(struct varsDev_t d, int face);

// set vector field 'field' to be periodic
__global__ void setPeriFace(REAL *field, int face);

// set 'field' to be periodic (host code)
void setPeriHost(REAL *field, struct parameters_t p);

// set the grid to be periodic
__global__ void setGridPeriFace(REAL *xb, int face);

// set the grid to be periodic (host code)
void setGridPeriHost(REAL *xb, struct parameters_t p);

void setBbound(dim3 dimGrid2dXY, dim3 dimGrid2dXZ, dim3 dimGrid2dYZ, dim3 dimBlock2d, struct varsDev_t d);

void setPeri(dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d, REAL *dev_field, struct parameters_t p);

void setGridPeri(dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d, REAL *dev_xb, struct parameters_t p);

#endif /* BOUNDARIES_H_ */
