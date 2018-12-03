// maths.h

#ifndef MATHS_H_
#define MATHS_H_

#include "global.h"

// Calculate the three dimensional cross product c = axb.
__device__ void cross(REAL a[3], REAL b[3], REAL c[3]);

// Calculate the scalar product c = a.b.
__device__ REAL dot(REAL a[3], REAL b[3]);

// Calculate the norm of a 3d vector.
__device__ REAL norm(REAL a[3]);

// Normalize this 3d vector.
__device__ void normalize(REAL a[3]);

// Atomic addition for doubles.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val);
#endif

// Determine the maximum value in an array.
__global__ void maxBlock(REAL *var, REAL *maxVal, int size);

// Determine the maximum value of a device variable.
REAL findDevMax(REAL *dev_var, int size);

// Invert sign of device variable.
__global__ void invert(REAL *var, int size);

// Determine the maximum value in an array.
__device__ REAL maxValue(REAL *a, int len);

// Compute the norm of a vector.
__global__ void mag(REAL *vec, REAL *mag);

// Compute the norm of JxB/B**2.
__global__ void JxB_B2(REAL *B, REAL *J, REAL *JxB_B2, int dimX, int dimY, int dimZ);

// Compute the norm of J.B/B**2.
__global__ void JB_B2(REAL *B, REAL *J, REAL *JB_B2);

// Compute the force-free parameter epsilon*.
__global__ void epsilonStar(REAL *xb, REAL *JB_B2, REAL *epsStar, int dimZ);

// Compute the error of B-ez.
__global__ void B_1ez(REAL *B, REAL *B_1ez2);

// Compute the error of xb-xbAnalytical.
__global__ void xb_XbAn(REAL *xb, REAL *xb_xbAn);

// Global sum on the device.
__global__ void sumBlock(REAL *var, REAL *sum, int size, int stride);

// Compute the sum of a device variable.
REAL sumGlobal(REAL *dev_var, int size);

// Compute the directional cell volume.
__global__ void dirCellVol(REAL *xb, REAL *cellVol, int dimX, int dimY, int dimZ);

// Compute the minimal cell volume.
__global__ void gridWedgeMin(REAL *xb, REAL *wedge, int dimX, int dimY, int dimZ);

// Compute the "convexity" of the cells.
__global__ void gridConvexity(REAL *xb, REAL *convexity, int dimX, int dimY, int dimZ);

// Compute the magnetic energy.
__global__ void B2_det(REAL *B, REAL *detJac, REAL *B2_det);

// Compute the kinetic energy.
__global__ void U2_det(REAL *uu, REAL *detJac, REAL *U2_det);

#endif /* MATHS_H_ */
