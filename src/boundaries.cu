// boundaries.cu
//
// Boundary routines for host and device arrays.
//


#include "boundaries.h"
#include "maths.h"

// In case of a distorted initial grid we need to add the distortion to the "second" ghost cells.
__device__ REAL addDist(int i, int j, REAL z, int l)
{
    REAL x, y;
    REAL distX, distY;

    x = dev_params.dx*i + dev_params.Ox;
    y = dev_params.dy*j + dev_params.Oy;

    distX = 0;
    distY = 0;

    if (dev_params.initDistCode == 0) {
        distY = -dev_params.initShearA * sin(dev_params.initShearK*2*PI*(x+dev_params.Ox-dev_params.dx/2)/(dev_params.Lx+dev_params.dx)) * z;
        distX = -dev_params.initShearB * sin(dev_params.initShearK*2*PI*(distY+y+dev_params.Oy-dev_params.dy/2)/(dev_params.Ly+dev_params.dy)) * z;
    }
    if (dev_params.initDistCode == 1) {
        distX = 0;
        distY = dev_params.initShearA * exp(-dev_params.initShearK*x*x)*(1-y*y)*(exp(-1*z+1) - 0.15366);
    }

    if (l == 0)
        return distX;
    else
        return distY;
}


// Update B and detJac at the boundary (previously done within B_JacB0 (inefficient)).
__global__ void updateBbound(struct VarsDev d, int face)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int b, l, side;
    REAL Bx, By, Bz, detJac1;      // local variables (registers)

    __shared__ REAL xbs[3][3][18][18], B0s[3][16][16], jacs[3][3][16][16];

    // xy faces
    if (face == 2) {
        for (side = 0; side < 2; side++) {
            if (dev_params.zPeri == false) {
                if ((i < dev_params.nx) && (j < dev_params.ny)) {
                    // Copy from global memory to shared memory for faster computation.
                    for (l = 0; l < 3; l++) {
                        for (b = 0; b < 2; b++)
                            xbs[l][b+1-side][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (b+side*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        B0s[l][p][q] = d.B0[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + side*(dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                }
                __syncthreads();
                if ((i < dev_params.nx) && (j < dev_params.ny)) {
                    for (l = 0; l < 2; l++)
                        xbs[l][0+2*side][p+1][q+1] = xbs[l][1][p+1][q+1] - addDist(i, j, xbs[2][1][p+1][q+1], l)
                            + addDist(i, j, 2*dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, l);
                    xbs[2][0+2*side][p+1][q+1] = xbs[2][1][p+1][q+1] + (2*side-1)*dev_params.dz;
                }
                __syncthreads();
                if ((i < dev_params.nx) && (j < dev_params.ny)) {
                    // Get xbs at the edges.
                    if (p == 0) {
                        xbs[0][1][0][q+1]  = xbs[0][1][1][q+1] - dev_params.dx - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0) +
                            addDist(i-1, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0);
                        xbs[1][1][0][q+1]  = xbs[1][1][1][q+1] - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1) +
                            addDist(i-1, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1);
                        xbs[2][1][0][q+1]  = xbs[2][1][1][q+1];
                        xbs[0][1][17][q+1] = xbs[0][1][16][q+1] + dev_params.dx - addDist(i+15, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0) +
                            addDist(i+16, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0);
                        xbs[1][1][17][q+1] = xbs[1][1][16][q+1] - addDist(i+15, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1) +
                            addDist(i+16, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1);
                        xbs[2][1][17][q+1] = xbs[2][1][16][q+1];
                    }
                    if (i == dev_params.nx-1) {
                        xbs[0][1][p+2][q+1] = xbs[0][1][p+1][q+1] + dev_params.dx - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0) +
                            addDist(i+1, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0);
                        xbs[1][1][p+2][q+1] = xbs[1][1][p+1][q+1] - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1) +
                            addDist(i+1, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1);
                        xbs[2][1][p+2][q+1] = xbs[2][1][p+1][q+1];
                    }
                    if (q == 0) {
                        xbs[0][1][p+1][0]  = xbs[0][1][p+1][1] - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0) +
                            addDist(i, j-1, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0);
                        xbs[1][1][p+1][0]  = xbs[1][1][p+1][1] - dev_params.dy - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1) +
                            addDist(i, j-1, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1);
                        xbs[2][1][p+1][0]  = xbs[2][1][p+1][1];
                        xbs[0][1][p+1][17] = xbs[0][1][p+1][16] - addDist(i, j+15, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0) +
                            addDist(i, j+16, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0);
                        xbs[1][1][p+1][17] = xbs[1][1][p+1][16] + dev_params.dy - addDist(i, j+15, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1) +
                            addDist(i, j+16, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1);
                        xbs[2][1][p+1][17] = xbs[2][1][p+1][16];
                    }
                    if (j == dev_params.ny-1) {
                        xbs[0][1][p+1][q+2] = xbs[0][1][p+1][q+1] - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0) +
                            addDist(i, j+1, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 0);
                        xbs[1][1][p+1][q+2] = xbs[1][1][p+1][q+1] + dev_params.dy - addDist(i, j, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1) +
                            addDist(i, j+1, dev_params.dz*(2*side-1) + dev_params.Oz + dev_params.Lz*side, 1);
                        xbs[2][1][p+1][q+2] = xbs[2][1][p+1][q+1];
                    }
                }
            }
            else {
                if ((i < dev_params.nx) && (j < dev_params.ny)) {
                    // Copy from global memory to shared memory for faster computation.
                    if (side == 0) {
                        for (l = 0; l < 2; l++) {
                            xbs[l][0][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz-1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][1][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][2][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        }
                        xbs[2][0][p+1][q+1] = d.xb[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz-1)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Lz - dev_params.dz;
                        xbs[2][1][p+1][q+1] = d.xb[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Lz - dev_params.dz;
                        xbs[2][2][p+1][q+1] = d.xb[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        for (l = 0; l < 3; l++)
                            B0s[l][p][q] = d.B0[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                    else {
                        for (l = 0; l < 2; l++) {
                            xbs[l][0][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][1][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][2][p+1][q+1] = d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 2*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        }
                        xbs[2][0][p+1][q+1] = d.xb[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        xbs[2][1][p+1][q+1] = d.xb[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Lz + dev_params.dz;
                        xbs[2][2][p+1][q+1] = d.xb[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 2*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Lz + dev_params.dz;
                        for (l = 0; l < 3; l++)
                            B0s[l][p][q] = d.B0[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                }
                __syncthreads();
                if ((i < dev_params.nx) && (j < dev_params.ny)) {
                    // Get xbs at the edges.
                    for (l = 0; l < 2; l++) {
                        if (p == 0)
                            xbs[l][1][0][q+1]  = d.xb[l + (i+0)*3 + (j+1)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if ((p == blockDim.x-1) || (i == dev_params.nx-1))
                            xbs[l][1][p+2][q+1] = d.xb[l + (i+2)*3 + (j+1)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if (q == 0)
                            xbs[l][1][p+1][0]  = d.xb[l + (i+1)*3 + (j+0)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if ((q == blockDim.y-1) || (j == dev_params.ny-1))
                            xbs[l][1][p+1][q+2] = d.xb[l + (i+1)*3 + (j+2)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                    if (p == 0)
                        xbs[2][1][0][q+1]  = d.xb[2 + (i+0)*3 + (j+1)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lz + dev_params.dz);
                    if ((p == blockDim.x-1) || (i == dev_params.nx-1))
                        xbs[2][1][p+2][q+1] = d.xb[2 + (i+2)*3 + (j+1)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lz + dev_params.dz);
                    if (q == 0)
                        xbs[2][1][p+1][0]  = d.xb[2 + (i+1)*3 + (j+0)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lz + dev_params.dz);
                    if ((q == blockDim.y-1) || (j == dev_params.ny-1))
                        xbs[2][1][p+1][q+2] = d.xb[2 + (i+1)*3 + (j+2)*(dev_params.nx+2)*3 + (side+(1-side)*dev_params.nz)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lz + dev_params.dz);
                }
            }
            __syncthreads();
            if ((i < dev_params.nx) && (j < dev_params.ny)) {
                for (l = 0; l < 3; l++) {
                    jacs[l][0][p][q] = (xbs[l][1][p+2][q+1] - xbs[l][1][p+0][q+1]) * dev_params.dx1 / 2;
                    jacs[l][1][p][q] = (xbs[l][1][p+1][q+2] - xbs[l][1][p+1][q+0]) * dev_params.dy1 / 2;
                    jacs[l][2][p][q] = (xbs[l][2][p+1][q+1] - xbs[l][0][p+1][q+1]) * dev_params.dz1 / 2;
                }
                detJac1 = 1/(jacs[0][0][p][q]*jacs[1][1][p][q]*jacs[2][2][p][q] + jacs[0][1][p][q]*jacs[1][2][p][q]*jacs[2][0][p][q] + jacs[0][2][p][q]*jacs[1][0][p][q]*jacs[2][1][p][q] -
                             jacs[0][0][p][q]*jacs[1][2][p][q]*jacs[2][1][p][q] - jacs[0][1][p][q]*jacs[1][0][p][q]*jacs[2][2][p][q] - jacs[0][2][p][q]*jacs[1][1][p][q]*jacs[2][0][p][q]);

                // Compute the boundary magnetic field from B0.
                Bx = (jacs[0][0][p][q]*B0s[0][p][q] + jacs[0][1][p][q]*B0s[1][p][q] + jacs[0][2][p][q]*B0s[2][p][q])*detJac1;
                By = (jacs[1][0][p][q]*B0s[0][p][q] + jacs[1][1][p][q]*B0s[1][p][q] + jacs[1][2][p][q]*B0s[2][p][q])*detJac1;
                if (dev_params.zPeri == true)
                    Bz = (jacs[2][0][p][q]*B0s[0][p][q] + jacs[2][1][p][q]*B0s[1][p][q] + jacs[2][2][p][q]*B0s[2][p][q])*detJac1;
                else
                    Bz = B0s[2][p][q];  // Set B.n to B0.n on the boundary.

                // Copy back to global memory.
                d.B[0 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + side*(dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = Bx;
                d.B[1 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + side*(dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = By;
                d.B[2 + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + side*(dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = Bz;
                d.detJac[(i+1) + (j+1)*(dev_params.nx+2) + side*(dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)] = 1/detJac1;
            }
            __syncthreads();
        }
    }

    // xz faces
    if (face == 1) {
        for (side = 0; side < 2; side++) {
            if (dev_params.yPeri == false) {
                if ((i < dev_params.nx) && (j < dev_params.nz)) {
                    // Copy from global memory to shared memory for faster computation.
                    for (l = 0; l < 3; l++) {
                        for (b = 0; b < 2; b++)
                            xbs[l][b+1-side][p+1][q+1] = d.xb[l + (i+1)*3 + (b+side*dev_params.ny)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        B0s[l][p][q] = d.B0[l + (i+1)*3 + side*(dev_params.ny+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                }
                __syncthreads();
                if ((i < dev_params.nx) && (j < dev_params.nz)) {
                    for (l = 0; l < 3; l += 2)
                        xbs[l][0+2*side][p+1][q+1] = xbs[l][1][p+1][q+1];
                    xbs[1][0+2*side][p+1][q+1] = xbs[1][1][p+1][q+1] + (2*side-1)*dev_params.dy;
                }
                __syncthreads();
                if ((i < dev_params.nx) && (j < dev_params.nz)) {
                    // Get xbs at the edges.
                    if (p == 0) {
                        xbs[0][1][0][q+1]  = xbs[0][1][1][q+1] - dev_params.dx;
                        xbs[1][1][0][q+1]  = xbs[1][1][1][q+1];
                        xbs[2][1][0][q+1]  = xbs[2][1][1][q+1];
                        xbs[0][1][17][q+1] = xbs[0][1][16][q+1] + dev_params.dx;
                        xbs[1][1][17][q+1] = xbs[1][1][16][q+1];
                        xbs[2][1][17][q+1] = xbs[2][1][16][q+1];
                    }
                    if (i == dev_params.nx-1) {
                        xbs[0][1][p+2][q+1] = xbs[0][1][p+1][q+1] + dev_params.dx;
                        xbs[1][1][p+2][q+1] = xbs[1][1][p+1][q+1];
                        xbs[2][1][p+2][q+1] = xbs[2][1][p+1][q+1];
                    }
                    if (q == 0) {
                        xbs[0][1][p+1][0]  = xbs[0][1][p+1][1];
                        xbs[1][1][p+1][0]  = xbs[1][1][p+1][1];
                        xbs[2][1][p+1][0]  = xbs[2][1][p+1][1] - dev_params.dz;
                        xbs[0][1][p+1][17] = xbs[0][1][p+1][16];
                        xbs[1][1][p+1][17] = xbs[1][1][p+1][16];
                        xbs[2][1][p+1][17] = xbs[2][1][p+1][16] + dev_params.dz;
                    }
                    if (j == dev_params.nz-1) {
                        xbs[0][1][p+1][q+2] = xbs[0][1][p+1][q+1];
                        xbs[1][1][p+1][q+2] = xbs[1][1][p+1][q+1];
                        xbs[2][1][p+1][q+2] = xbs[2][1][p+1][q+1] + dev_params.dz;
                    }
                }
            }
            else {
                if ((i < dev_params.nx) && (j < dev_params.nz)) {
                    // Copy from global memory to shared memory for faster computation.
                    if (side == 0) {
                        for (l = 0; l < 3; l += 2) {
                            xbs[l][0][p+1][q+1] = d.xb[l + (i+1)*3 + (dev_params.ny-1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][1][p+1][q+1] = d.xb[l + (i+1)*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][2][p+1][q+1] = d.xb[l + (i+1)*3 + 1*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        }
                        xbs[1][0][p+1][q+1] = d.xb[1 + (i+1)*3 + (dev_params.ny-1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Ly - dev_params.dy;
                        xbs[1][1][p+1][q+1] = d.xb[1 + (i+1)*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Ly - dev_params.dy;
                        xbs[1][2][p+1][q+1] = d.xb[1 + (i+1)*3 + 1*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        for (l = 0; l < 3; l++)
                            B0s[l][p][q] = d.B0[l + (i+1)*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                    else {
                        for (l = 0; l < 3; l += 2) {
                            xbs[l][0][p+1][q+1] = d.xb[l + (i+1)*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][1][p+1][q+1] = d.xb[l + (i+1)*3 + 1*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][2][p+1][q+1] = d.xb[l + (i+1)*3 + 2*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        }
                        xbs[1][0][p+1][q+1] = d.xb[1 + (i+1)*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        xbs[1][1][p+1][q+1] = d.xb[1 + (i+1)*3 + 1*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Ly + dev_params.dy;
                        xbs[1][2][p+1][q+1] = d.xb[1 + (i+1)*3 + 2*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Ly + dev_params.dy;
                        for (l = 0; l < 3; l++)
                            B0s[l][p][q] = d.B0[l + (i+1)*3 + 1*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                }
                __syncthreads();
                if ((i < dev_params.nx) && (j < dev_params.nz)) {
                    // Get xbs at the edges.
                    for (l = 0; l < 3; l += 2) {
                        if (p == 0)
                            xbs[l][1][0][q+1]  = d.xb[l + (i+0)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if ((p == blockDim.x-1) || (i == dev_params.nx-1))
                            xbs[l][1][p+2][q+1] = d.xb[l + (i+2)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if (q == 0)
                            xbs[l][1][p+1][0]  = d.xb[l + (i+1)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if ((q == blockDim.z-1) || (j == dev_params.nz-1))
                            xbs[l][1][p+1][q+2] = d.xb[l + (i+1)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+2)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                    if (p == 0)
                        xbs[1][1][0][q+1]  = d.xb[1 + (i+0)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Ly + dev_params.dy);
                    if ((p == blockDim.x-1) || (i == dev_params.nx-1))
                        xbs[1][1][p+2][q+1] = d.xb[1 + (i+2)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Ly + dev_params.dy);
                    if (q == 0)
                        xbs[1][1][p+1][0]  = d.xb[1 + (i+1)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+0)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Ly + dev_params.dy);
                    if ((q == blockDim.z-1) || (j == dev_params.nz-1))
                        xbs[1][1][p+1][q+2] = d.xb[1 + (i+1)*3 + (side+(1-side)*dev_params.ny)*(dev_params.nx+2)*3 + (j+2)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Ly + dev_params.dy);
                }
            }
            __syncthreads();
            if ((i < dev_params.nx) && (j < dev_params.nz)) {
                for (l = 0; l < 3; l++) {
                    jacs[l][0][p][q] = (xbs[l][1][p+2][q+1] - xbs[l][1][p+0][q+1]) * dev_params.dx1 / 2;
                    jacs[l][1][p][q] = (xbs[l][2][p+1][q+1] - xbs[l][0][p+1][q+1]) * dev_params.dy1 / 2;
                    jacs[l][2][p][q] = (xbs[l][1][p+1][q+2] - xbs[l][1][p+1][q+0]) * dev_params.dz1 / 2;
                }
                detJac1 = 1/(jacs[0][0][p][q]*jacs[1][1][p][q]*jacs[2][2][p][q] + jacs[0][1][p][q]*jacs[1][2][p][q]*jacs[2][0][p][q] + jacs[0][2][p][q]*jacs[1][0][p][q]*jacs[2][1][p][q] -
                             jacs[0][0][p][q]*jacs[1][2][p][q]*jacs[2][1][p][q] - jacs[0][1][p][q]*jacs[1][0][p][q]*jacs[2][2][p][q] - jacs[0][2][p][q]*jacs[1][1][p][q]*jacs[2][0][p][q]);

                // Compute the boundary magnetic field from B0.
                Bx = (jacs[0][0][p][q]*B0s[0][p][q] + jacs[0][1][p][q]*B0s[1][p][q] + jacs[0][2][p][q]*B0s[2][p][q])*detJac1;
                if (dev_params.yPeri == true)
                    By = (jacs[1][0][p][q]*B0s[0][p][q] + jacs[1][1][p][q]*B0s[1][p][q] + jacs[1][2][p][q]*B0s[2][p][q])*detJac1;
                else
                    By = B0s[1][p][q];    // set B.n to B0.n on the boundary
                Bz = (jacs[2][0][p][q]*B0s[0][p][q] + jacs[2][1][p][q]*B0s[1][p][q] + jacs[2][2][p][q]*B0s[2][p][q])*detJac1;

                // Copy back to global memory.
                d.B[0 + (i+1)*3 + side*(dev_params.ny+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = Bx;
                d.B[1 + (i+1)*3 + side*(dev_params.ny+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = By;
                d.B[2 + (i+1)*3 + side*(dev_params.ny+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = Bz;
                d.detJac[(i+1) + side*(dev_params.ny+1)*(dev_params.nx+2) + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)] = 1/detJac1;
            }
            __syncthreads();
        }
    }

    // yz faces
    if (face == 0) {
        for (side = 0; side < 2; side++) {
            if (dev_params.xPeri == false) {
                if ((i < dev_params.ny) && (j < dev_params.nz)) {
                    // Copy from global memory to shared memory for faster computation.
                    for (l = 0; l < 3; l++) {
                        for (b = 0; b < 2; b++)
                            xbs[l][b+1-side][p+1][q+1] = d.xb[l + (b+side*dev_params.nx)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        B0s[l][p][q] = d.B0[l + side*(dev_params.nx+1)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                }
                __syncthreads();
                if ((i < dev_params.ny) && (j < dev_params.nz)) {
                    for (l = 1; l < 3; l++)
                        xbs[l][0+2*side][p+1][q+1] = xbs[l][1][p+1][q+1];
                    xbs[0][0+2*side][p+1][q+1] = xbs[0][1][p+1][q+1] + (2*side-1)*dev_params.dx;
                }
                __syncthreads();
                if ((i < dev_params.ny) && (j < dev_params.nz)) {
                    // Get xbs at the edges.
                    if (p == 0) {
                        xbs[0][1][0][q+1]  = xbs[0][1][1][q+1];
                        xbs[1][1][0][q+1]  = xbs[1][1][1][q+1] - dev_params.dy;
                        xbs[2][1][0][q+1]  = xbs[2][1][1][q+1];
                        xbs[0][1][17][q+1] = xbs[0][1][16][q+1];
                        xbs[1][1][17][q+1] = xbs[1][1][16][q+1] + dev_params.dy;
                        xbs[2][1][17][q+1] = xbs[2][1][16][q+1];
                    }
                    if (i == dev_params.ny-1) {
                        xbs[0][1][p+2][q+1] = xbs[0][1][p+1][q+1];
                        xbs[1][1][p+2][q+1] = xbs[1][1][p+1][q+1] + dev_params.dy;
                        xbs[2][1][p+2][q+1] = xbs[2][1][p+1][q+1];
                    }
                    if (q == 0) {
                        xbs[0][1][p+1][0]  = xbs[0][1][p+1][1];
                        xbs[1][1][p+1][0]  = xbs[1][1][p+1][1];
                        xbs[2][1][p+1][0]  = xbs[2][1][p+1][1] - dev_params.dz;
                        xbs[0][1][p+1][17] = xbs[0][1][p+1][16];
                        xbs[1][1][p+1][17] = xbs[1][1][p+1][16];
                        xbs[2][1][p+1][17] = xbs[2][1][p+1][16] + dev_params.dz;
                    }
                    if (j == dev_params.nz-1) {
                        xbs[0][1][p+1][q+2] = xbs[0][1][p+1][q+1];
                        xbs[1][1][p+1][q+2] = xbs[1][1][p+1][q+1];
                        xbs[2][1][p+1][q+2] = xbs[2][1][p+1][q+1] + dev_params.dz;
                    }
                }
            }
            else {
                if ((i < dev_params.ny) && (j < dev_params.nz)) {
                    // Copy from global memory to shared memory for faster computation.
                    if (side == 0) {
                        for (l = 1; l < 3; l++) {
                            xbs[l][0][p+1][q+1] = d.xb[l + (dev_params.nx-1)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][1][p+1][q+1] = d.xb[l + (dev_params.nx+0)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][2][p+1][q+1] = d.xb[l + 1*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        }
                        xbs[0][0][p+1][q+1] = d.xb[0 + (dev_params.nx-1)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Lx - dev_params.dx;
                        xbs[0][1][p+1][q+1] = d.xb[0 + (dev_params.nx+0)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Lx - dev_params.dx;
                        xbs[0][2][p+1][q+1] = d.xb[0 + 1*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        for (l = 0; l < 3; l++)
                            B0s[l][p][q] = d.B0[l + (dev_params.nx+0)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                    else {
                        for (l = 1; l < 3; l++) {
                            xbs[l][0][p+1][q+1] = d.xb[l + (dev_params.nx+0)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][1][p+1][q+1] = d.xb[l + 1*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                            xbs[l][2][p+1][q+1] = d.xb[l + 2*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        }
                        xbs[0][0][p+1][q+1] = d.xb[0 + (dev_params.nx+0)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        xbs[0][1][p+1][q+1] = d.xb[0 + 1*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Lx + dev_params.dx;
                        xbs[0][2][p+1][q+1] = d.xb[0 + 2*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Lx + dev_params.dx;
                        for (l = 0; l < 3; l++)
                            B0s[l][p][q] = d.B0[l + 1*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                }
                __syncthreads();
                if ((i < dev_params.ny) && (j < dev_params.nz)) {
                    // Get xbs at the edges.
                    for (l = 1; l < 3; l++) {
                        if (p == 0)
                            xbs[l][1][0][q+1]  = d.xb[l + (side+(1-side)*dev_params.nx)*3 + (i+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if ((p == blockDim.y-1) || (i == dev_params.ny-1))
                            xbs[l][1][p+2][q+1] = d.xb[l + (side+(1-side)*dev_params.nx)*3 + (i+2)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if (q == 0)
                            xbs[l][1][p+1][0]  = d.xb[l + (side+(1-side)*dev_params.nx)*3 + (i+1)*(dev_params.nx+2)*3 + (j+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                        if ((q == blockDim.z-1) || (j == dev_params.nz-1))
                            xbs[l][1][p+1][q+2] = d.xb[l + (side+(1-side)*dev_params.nx)*3 + (i+1)*(dev_params.nx+2)*3 + (j+2)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                    }
                    if (p == 0)
                        xbs[0][1][0][q+1]  = d.xb[0 + (side+(1-side)*dev_params.nx)*3 + (i+0)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lx + dev_params.dx);
                    if ((p == blockDim.y-1) || (i == dev_params.ny-1))
                        xbs[0][1][p+2][q+1] = d.xb[0 + (side+(1-side)*dev_params.nx)*3 + (i+2)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lx + dev_params.dx);
                    if (q == 0)
                        xbs[0][1][p+1][0]  = d.xb[0 + (side+(1-side)*dev_params.nx)*3 + (i+1)*(dev_params.nx+2)*3 + (j+0)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lx + dev_params.dx);
                    if ((q == blockDim.z-1) || (j == dev_params.nz-1))
                        xbs[0][1][p+1][q+2] = d.xb[0 + (side+(1-side)*dev_params.nx)*3 + (i+1)*(dev_params.nx+2)*3 + (j+2)*(dev_params.nx+2)*(dev_params.ny+2)*3] + (2*side-1)*(dev_params.Lx + dev_params.dx);
                }
            }
            __syncthreads();
            if ((i < dev_params.ny) && (j < dev_params.nz)) {
                for (l = 0; l < 3; l++) {
                    jacs[l][0][p][q] = (xbs[l][2][p+1][q+1] - xbs[l][0][p+1][q+1]) * dev_params.dx1 / 2;
                    jacs[l][1][p][q] = (xbs[l][1][p+2][q+1] - xbs[l][1][p+0][q+1]) * dev_params.dy1 / 2;
                    jacs[l][2][p][q] = (xbs[l][1][p+1][q+2] - xbs[l][1][p+1][q+0]) * dev_params.dz1 / 2;
                }
                detJac1 = 1/(jacs[0][0][p][q]*jacs[1][1][p][q]*jacs[2][2][p][q] + jacs[0][1][p][q]*jacs[1][2][p][q]*jacs[2][0][p][q] + jacs[0][2][p][q]*jacs[1][0][p][q]*jacs[2][1][p][q] -
                             jacs[0][0][p][q]*jacs[1][2][p][q]*jacs[2][1][p][q] - jacs[0][1][p][q]*jacs[1][0][p][q]*jacs[2][2][p][q] - jacs[0][2][p][q]*jacs[1][1][p][q]*jacs[2][0][p][q]);

                // Compute the boundary magnetic field from B0.
                if (dev_params.xPeri == true)
                    Bx = (jacs[0][0][p][q]*B0s[0][p][q] + jacs[0][1][p][q]*B0s[1][p][q] + jacs[0][2][p][q]*B0s[2][p][q])*detJac1;
                else
                    Bx = B0s[0][p][q];    // set B.n to B0.n on the boundary
                By = (jacs[1][0][p][q]*B0s[0][p][q] + jacs[1][1][p][q]*B0s[1][p][q] + jacs[1][2][p][q]*B0s[2][p][q])*detJac1;
                Bz = (jacs[2][0][p][q]*B0s[0][p][q] + jacs[2][1][p][q]*B0s[1][p][q] + jacs[2][2][p][q]*B0s[2][p][q])*detJac1;

                // Copy back to global memory.
                d.B[0 + side*(dev_params.nx+1)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = Bx;
                d.B[1 + side*(dev_params.nx+1)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = By;
                d.B[2 + side*(dev_params.nx+1)*3 + (i+1)*(dev_params.nx+2)*3 + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] = Bz;
                d.detJac[side*(dev_params.nx+1) + (i+1)*(dev_params.nx+2) + (j+1)*(dev_params.nx+2)*(dev_params.ny+2)] = 1/detJac1;
            }
            __syncthreads();
        }
    }
}


// Set vector field 'field' to be periodic.
__global__ void setPeriFace(REAL *field, int face)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int l;

    if (face == 2) {
        if ((i < dev_params.nx+2) && (j < dev_params.ny+2)) {
            for (l = 0; l < 3; l++) {
                field[l + i*3 + j*(dev_params.nx+2)*3 + 0*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                field[l + i*3 + j*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                field[l + i*3 + j*(dev_params.nx+2)*3 + (dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                field[l + i*3 + j*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
        }
    }

    if (face == 1) {
        if ((i < dev_params.nx+2) && (j < dev_params.nz+2)) {
            for (l = 0; l < 3; l++) {
                field[l + i*3 + 0*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                field[l + i*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
                field[l + i*3 + (dev_params.ny+1)*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                field[l + i*3 + 1*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
        }
    }

    if (face == 0) {
        if ((i < dev_params.ny+2) && (j < dev_params.nz+2)) {
            for (l = 0; l < 3; l++) {
                field[l + 0*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                field[l + (dev_params.nx+0)*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
                field[l + (dev_params.nx+1)*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                field[l + 1*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
        }
    }
}


// Set 'field' to be periodic (host code).
void setPeriHost(REAL *field, struct Parameters p)
{
    int i, j, l;

    if (p.zPeri) {
        for (i = 0; i < p.nx+2; i++)
            for (j = 0; j < p.ny+2; j++)
                for (l = 0; l < 3; l++) {
                    field[l + i*3 + j*(p.nx+2)*3 + 0*(p.nx+2)*(p.ny+2)*3] =
                    field[l + i*3 + j*(p.nx+2)*3 + (p.nz+0)*(p.nx+2)*(p.ny+2)*3];
                    field[l + i*3 + j*(p.nx+2)*3 + (p.nz+1)*(p.nx+2)*(p.ny+2)*3] =
                    field[l + i*3 + j*(p.nx+2)*3 + 1*(p.nx+2)*(p.ny+2)*3];
                }
    }

    if (p.yPeri) {
        for (i = 0; i < p.nx+2; i++)
            for (j = 0; j < p.nz+2; j++)
                for (l = 0; l < 3; l++) {
                    field[l + i*3 + 0*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    field[l + i*3 + (p.ny+0)*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                    field[l + i*3 + (p.ny+1)*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    field[l + i*3 + 1*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                }
    }

    if (p.xPeri) {
        for (i = 0; i < p.ny+2; i++)
            for (j = 0; j < p.nz+2; j++)
                for (l = 0; l < 3; l++) {
                    field[l + 0*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    field[l + (p.nx+0)*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                    field[l + (p.nx+1)*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    field[l + 1*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                }
    }
}


// Set the grid to be periodic.
__global__ void setGridPeriFace(REAL *xb, int face)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int l;

    if (face == 2) {
        if ((i < dev_params.nx+2) && (j < dev_params.ny+2)) {
            for (l = 0; l < 2; l++) {
                xb[l + i*3 + j*(dev_params.nx+2)*3 + 0*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                xb[l + i*3 + j*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
                xb[l + i*3 + j*(dev_params.nx+2)*3 + (dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                xb[l + i*3 + j*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
            xb[2 + i*3 + j*(dev_params.nx+2)*3 + 0*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb[2 + i*3 + j*(dev_params.nx+2)*3 + (dev_params.nz+0)*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Lz - dev_params.dz;
            xb[2 + i*3 + j*(dev_params.nx+2)*3 + (dev_params.nz+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb[2 + i*3 + j*(dev_params.nx+2)*3 + 1*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Lz + dev_params.dz;
        }
    }

    if (face == 1) {
        if ((i < dev_params.nx+2) && (j < dev_params.nz+2)) {
            for (l = 0; l < 3; l += 2) {
                xb[l + i*3 + 0*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                xb[l + i*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
                xb[l + i*3 + (dev_params.ny+1)*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                xb[l + i*3 + 1*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
            xb[1 + i*3 + 0*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb[1 + i*3 + (dev_params.ny+0)*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Ly - dev_params.dy;
            xb[1 + i*3 + (dev_params.ny+1)*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb[1 + i*3 + 1*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Ly + dev_params.dy;
        }
    }

    if (face == 0) {
        if ((i < dev_params.ny+2) && (j < dev_params.nz+2)) {
            for (l = 1; l < 3; l++) {
                xb[l + 0*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                xb[l + (dev_params.nx+0)*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
                xb[l + (dev_params.nx+1)*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
                xb[l + 1*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3];
            }
            xb[0 + 0*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb[0 + (dev_params.nx+0)*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] - dev_params.Lx - dev_params.dx;
            xb[0 + (dev_params.nx+1)*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb[0 + 1*3 + i*(dev_params.nx+2)*3 + j*(dev_params.nx+2)*(dev_params.ny+2)*3] + dev_params.Lx + dev_params.dx;
        }
    }
}


// Set the grid to be periodic (host code).
void setGridPeriHost(REAL *xb, struct Parameters p)
{
    int i, j, l;

    if (p.zPeri) {
        for (i = 0; i < p.nx+2; i++)
            for (j = 0; j < p.ny+2; j++)
                for (l = 0; l < 2; l++) {
                    xb[l + i*3 + j*(p.nx+2)*3 + 0*(p.nx+2)*(p.ny+2)*3] =
                    xb[l + i*3 + j*(p.nx+2)*3 + (p.nz+0)*(p.nx+2)*(p.ny+2)*3];
                    xb[l + i*3 + j*(p.nx+2)*3 + (p.nz+1)*(p.nx+2)*(p.ny+2)*3] =
                    xb[l + i*3 + j*(p.nx+2)*3 + 1*(p.nx+2)*(p.ny+2)*3];
                }
                xb[2 + i*3 + j*(p.nx+2)*3 + 0*(p.nx+2)*(p.ny+2)*3] =
                xb[2 + i*3 + j*(p.nx+2)*3 + (p.nz+0)*(p.nx+2)*(p.ny+2)*3] - p.Lz - p.dz;
                xb[2 + i*3 + j*(p.nx+2)*3 + (p.nz+1)*(p.nx+2)*(p.ny+2)*3] =
                xb[2 + i*3 + j*(p.nx+2)*3 + 1*(p.nx+2)*(p.ny+2)*3] + p.Lz + p.dz;
    }

    if (p.yPeri) {
        for (i = 0; i < p.nx+2; i++)
            for (j = 0; j < p.nz+2; j++)
                for (l = 0; l < 3; l += 2) {
                    xb[l + i*3 + 0*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    xb[l + i*3 + (p.ny+0)*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                    xb[l + i*3 + (p.ny+1)*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    xb[l + i*3 + 1*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                }
                xb[1 + i*3 + 0*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                xb[1 + i*3 + (p.ny+0)*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] - p.Ly - p.dy;
                xb[1 + i*3 + (p.ny+1)*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                xb[1 + i*3 + 1*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] + p.Ly + p.dy;
    }

    if (p.xPeri) {
        for (i = 0; i < p.ny+2; i++)
            for (j = 0; j < p.nz+2; j++)
                for (l = 1; l < 3; l++) {
                    xb[l + 0*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    xb[l + (p.nx+0)*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                    xb[l + (p.nx+1)*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                    xb[l + 1*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3];
                }
                xb[0 + 0*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                xb[0 + (p.nx+0)*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] - p.Lx - p.dx;
                xb[0 + (p.nx+1)*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] =
                xb[0 + 1*3 + i*(p.nx+2)*3 + j*(p.nx+2)*(p.ny+2)*3] + p.Lx + p.dx;
    }
}


void setBbound(dim3 dimGrid2dXY, dim3 dimGrid2dXZ, dim3 dimGrid2dYZ, dim3 dimBlock2d, struct VarsDev d)
{
    updateBbound<<<dimGrid2dXY, dimBlock2d>>>(d, 2); cudaDeviceSynchronize();
    updateBbound<<<dimGrid2dXZ, dimBlock2d>>>(d, 1); cudaDeviceSynchronize();
    updateBbound<<<dimGrid2dYZ, dimBlock2d>>>(d, 0); cudaDeviceSynchronize();
}


void setPeri(dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d, REAL *dev_field, struct Parameters p)
{
    if (p.zPeri) setPeriFace<<<dimGrid2dPlusXY, dimBlock2d>>>(dev_field, 2);
    if (p.yPeri) setPeriFace<<<dimGrid2dPlusXZ, dimBlock2d>>>(dev_field, 1);
    if (p.xPeri) setPeriFace<<<dimGrid2dPlusYZ, dimBlock2d>>>(dev_field, 0);
}


void setGridPeri(dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d, REAL *dev_xb, struct Parameters p)
{
    if (p.zPeri) setGridPeriFace<<<dimGrid2dPlusXY, dimBlock2d>>>(dev_xb, 2);
    if (p.yPeri) setGridPeriFace<<<dimGrid2dPlusXZ, dimBlock2d>>>(dev_xb, 1);
    if (p.xPeri) setGridPeriFace<<<dimGrid2dPlusYZ, dimBlock2d>>>(dev_xb, 0);
}



