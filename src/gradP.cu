// gradP.cu
//
// Routines for computing grad(p).
//

#include "gradP.h"
#include "maths.h"

// Compute grad(p) using the standard derivatives.
__global__ void gradPClassic(struct VarsDev d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, a, b;
    int u, v;                   // auxiliary variables for reading in the block edges
    REAL jac[3][3], jac1[3][3]; // Jacobian matrix and its inverse
    REAL d2x_dX2[3][3][3];      // second order derivative of xb
    REAL det1;                  // inverse determinant

    // Shared memory for faster communication, the size is assigned dynamically.
    extern __shared__ REAL s[];
    REAL *xbs = s;  // position vector at beginning of time step

    // Copy from global memory to shared memory for faster computation.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        det1 = 1/d.detJac[(i+1) + (j+1)*(dev_params.nx+2) + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)];
        for (l = 0; l < 3; l++) {
            // Assign the inner values without the boundaries.
            xbs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            // Assign the block boundaries.
            if (p == 0) {
                xbs[ l + (p+0)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+0)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3]; }
            if (((p == (blockDim.x-1)) && (i < (dev_params.nx-1))) || (i == (dev_params.nx-1))) {
                xbs[ l + (p+2)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+2)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3]; }
            if (q == 0) {
                xbs[ l + (p+1)*3 + (q+0)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+0)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3]; }
            if (((q == (blockDim.y-1)) && (j < (dev_params.ny-1))) || (j == (dev_params.ny-1))) {
                xbs[ l + (p+1)*3 + (q+2)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+2)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3]; }
            if (r == 0) {
                xbs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+0)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+0)*(dev_params.nx+2)*(dev_params.ny+2)*3]; }
            if (((r == (blockDim.z-1)) && (k < (dev_params.nz-1))) || (k == (dev_params.nz-1))) {
                xbs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+2)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+2)*(dev_params.nx+2)*(dev_params.ny+2)*3]; }
        }
    }
    __syncthreads();
    // Assign the block edges.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for (l = 0; l < 3; l++) {
            if ((((p == 0) && (q == 0)) || ((p == 0) && (q == (blockDim.y-1))) || ((p == (blockDim.x-1)) && (q == 0)) || ((p == (blockDim.x-1)) && (q == (blockDim.y-1)))
                || ((i == 0) && (j == (dev_params.ny-1))) || ((i == (dev_params.nx-1)) && (j == 0)) || ((i == (dev_params.nx-1)) && (j == (dev_params.ny-1))))
                || (((p == 0) || (p == (blockDim.x-1))) && (j == (dev_params.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (i == (dev_params.nx-1)))) {
                u = (p+1)/blockDim.x; v = (q+1)/blockDim.y;
                if (i == (dev_params.nx-1)) u = 1;
                if (j == (dev_params.ny-1)) v = 1;
                xbs[ l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+2*u)*3    + (j+2*v)*(dev_params.nx+2)*3  + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
            if ((((p == 0) && (r == 0)) || ((p == 0) && (r == (blockDim.z-1))) || ((p == (blockDim.x-1)) && (r == 0)) || ((p == (blockDim.x-1)) && (r == (blockDim.z-1)))
                || ((i == 0) && (k == (dev_params.nz-1))) || ((i == (dev_params.nx-1)) && (k == 0)) || ((i == (dev_params.nx-1)) && (k == (dev_params.nz-1))))
                || (((p == 0) || (p == (blockDim.x-1))) && (k == (dev_params.nz-1))) || (((r == 0) || (r == (blockDim.z-1))) && (i == (dev_params.nx-1)))) {
                u = (p+1)/blockDim.x; v = (r+1)/blockDim.z;
                if (i == (dev_params.nx-1)) u = 1;
                if (k == (dev_params.nz-1)) v = 1;
                xbs[ l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
                d.xb[l + (i+2*u)*3    + (j+1)*(dev_params.nx+2)*3    + (k+2*v)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
            if ((((r == 0) && (q == 0)) || ((r == 0) && (q == (blockDim.y-1))) || ((r == (blockDim.z-1)) && (q == 0)) || ((r == (blockDim.z-1)) && (q == (blockDim.y-1)))
                || ((j == 0) && (k == (dev_params.nz-1))) || ((j == (dev_params.ny-1)) && (k == 0)) || ((j == (dev_params.ny-1)) && (k == (dev_params.nz-1))))
                || (((r == 0) || (r == (blockDim.z-1))) && (j == (dev_params.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (k == (dev_params.nz-1)))) {
                u = (q+1)/blockDim.y; v = (r+1)/blockDim.z;
                if (j == (dev_params.ny-1)) u = 1;
                if (k == (dev_params.nz-1)) v = 1;
                xbs[ l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
                d.xb[l + (i+1)*3      + (j+2*u)*(dev_params.nx+2)*3  + (k+2*v)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
        }
    }
    __syncthreads();

    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        // compute the Jacobian matrix, assuming the initial grid X was rectilinear and equidistant.
        for (l = 0; l < 3; l++) {
            jac[l][0] = (xbs[l + (p+2)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] -
                         xbs[l + (p+0)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3])*dev_params.dx1/2;
            jac[l][1] = (xbs[l + (p+1)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
                         xbs[l + (p+1)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_params.dy1/2;
            jac[l][2] = (xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
                         xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_params.dz1/2;
        }

        // Inverse Jacobian matrix.
        jac1[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])*det1;
        jac1[0][1] = (jac[0][2]*jac[2][1] - jac[0][1]*jac[2][2])*det1;
        jac1[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1])*det1;
        jac1[1][0] = (jac[1][2]*jac[2][0] - jac[1][0]*jac[2][2])*det1;
        jac1[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0])*det1;
        jac1[1][2] = (jac[0][2]*jac[1][0] - jac[0][0]*jac[1][2])*det1;
        jac1[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0])*det1;
        jac1[2][1] = (jac[0][1]*jac[2][0] - jac[0][0]*jac[2][1])*det1;
        jac1[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0])*det1;

        // Second order derivatives of x.
        for (l = 0; l < 3; l++) {
            d2x_dX2[l][0][0] = (xbs[  l + (p+2)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] -
                                2*xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] +
                                xbs[  l + (p+0)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3])*dev_params.dx1*dev_params.dx1;
            d2x_dX2[l][1][1] = (xbs[  l + (p+1)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
                                2*xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] +
                                xbs[  l + (p+1)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_params.dy1*dev_params.dy1;
            d2x_dX2[l][2][2] = (xbs[  l + (p+1)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
                                2*xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] +
                                xbs[  l + (p+1)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_params.dz1*dev_params.dz1;
            d2x_dX2[l][0][1] = (xbs[  l + (p+2)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
                                xbs[  l + (p+2)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3] -
                                xbs[  l + (p+0)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] +
                                xbs[  l + (p+0)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_params.dx1*dev_params.dy1/4;
            d2x_dX2[l][0][2] = (xbs[  l + (p+2)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
                                xbs[  l + (p+2)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3] -
                                xbs[  l + (p+0)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] +
                                xbs[  l + (p+0)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_params.dx1*dev_params.dz1/4;
            d2x_dX2[l][1][2] = (xbs[  l + (p+1)*3 + (q+2)*dimX*3  + (r+2)*dimX*dimY*3] -
                                xbs[  l + (p+1)*3 + (q+0)*dimX*3  + (r+2)*dimX*dimY*3] -
                                xbs[  l + (p+1)*3 + (q+2)*dimX*3  + (r+0)*dimX*dimY*3] +
                                xbs[  l + (p+1)*3 + (q+0)*dimX*3  + (r+0)*dimX*dimY*3])*dev_params.dy1*dev_params.dz1/4;
            d2x_dX2[l][1][0] = d2x_dX2[l][0][1];
            d2x_dX2[l][2][0] = d2x_dX2[l][0][2];
            d2x_dX2[l][2][1] = d2x_dX2[l][1][2];
        }

        // Add all up.
        d.gradP[0 + i*3 + j*dev_params.nx*3  + k*dev_params.nx*dev_params.ny*3] = 0;
        d.gradP[1 + i*3 + j*dev_params.nx*3  + k*dev_params.nx*dev_params.ny*3] = 0;
        d.gradP[2 + i*3 + j*dev_params.nx*3  + k*dev_params.nx*dev_params.ny*3] = 0;
        for (a = 0; a < 3; a++)
            for (b = 0; b < 3; b++)
                for (l = 0; l < 3; l++) {
                    d.gradP[0 + i*3 + j*dev_params.nx*3 + k*dev_params.nx*dev_params.ny*3] -= d2x_dX2[a][b][l]*jac1[b][a]*jac1[l][0];
                    d.gradP[1 + i*3 + j*dev_params.nx*3 + k*dev_params.nx*dev_params.ny*3] -= d2x_dX2[a][b][l]*jac1[b][a]*jac1[l][1];
                    d.gradP[2 + i*3 + j*dev_params.nx*3 + k*dev_params.nx*dev_params.ny*3] -= d2x_dX2[a][b][l]*jac1[b][a]*jac1[l][2];
                }
        d.gradP[0 + i*3 + j*dev_params.nx*3  + k*dev_params.nx*dev_params.ny*3] *= dev_params.beta*det1;
        d.gradP[1 + i*3 + j*dev_params.nx*3  + k*dev_params.nx*dev_params.ny*3] *= dev_params.beta*det1;
        d.gradP[2 + i*3 + j*dev_params.nx*3  + k*dev_params.nx*dev_params.ny*3] *= dev_params.beta*det1;
    }
}


// Determine which routine should be used for the current calculation.
void gradP(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct VarsDev d, struct Parameters p)
{
    if (strncmp(p.pMethod, "Classic ", 8) == 0)
        gradPClassic
            <<<dimGrid, dimBlock, (blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*3*sizeof(*(d.xb))>>>
            (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
}
