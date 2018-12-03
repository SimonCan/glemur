// rungeKutta.h

#ifndef RUNGEKUTTA_H_
#define RUNGEKUTTA_H_

#include "global.h"

// Assign values to the coefficients for the adaptive time step Runge-Kutta.
void assignCoefficients(REAL a[5], REAL b[7][6], REAL c[6], REAL cs[6]);

// Compute B = J.B0/Delta, first step in time stepping.
__global__ void B_JacB0(struct VarsDev d, int dimX, int dimY, int dimZ);

// Compute k_n (kk).
__global__ void kk(struct VarsDev d, int dimX, int dimY, int dimZ, int n, REAL dt);

// Compute the new distorted grid.
__global__ void xNewStar(struct VarsDev d, int dimX, int dimY, int dimZ);

// Compute the averaged Lorentz force.
__device__ void jxbAver(struct VarsDev d, REAL *Js, REAL *Bs, REAL *xbs, REAL jxb[6][3], int i, int j, int k, int p, int q, int r);

#endif /* RUNGEKUTTA_H_ */
