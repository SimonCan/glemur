// rungeKutta.h

#ifndef RUNGEKUTTA_H_
#define RUNGEKUTTA_H_

#include "global.h"

// compute B = J.B0/Delta, first step in time stepping
__global__ void B_JacB0(struct varsDev_t d, int dimX, int dimY, int dimZ);

// compute k_n (kk)
__global__ void kk(struct varsDev_t d, int dimX, int dimY, int dimZ, int n, REAL dt);

// compute the new distorted grid
__global__ void xNewStar(struct varsDev_t d, int dimX, int dimY, int dimZ);

// compute the averaged Lorentz force
__device__ void jxbAver(struct varsDev_t d, REAL *Js, REAL *Bs, REAL *xbs, REAL jxb[6][3], int i, int j, int k, int p, int q, int r);

#endif /* RUNGEKUTTA_H_ */
