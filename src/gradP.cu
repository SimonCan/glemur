// gradP.cu
//
// Routines for computing grad(p)

#include "gradP.h"
#include "maths.h"

// compute grad(P) using the standard derivatives
__global__ void gradPClassic(struct varsDev_t d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, a, b;
    REAL jac[3][3], jac1[3][3]; // Jacobian matrix and its inverse
    REAL d2x_dX2[3][3][3];      // second order derivative of xb
    REAL det1;					// inverse determinant

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;	// position vector at beginning of time step

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		det1 = 1/d.detJac[(i+1) + (j+1)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)];
 	    printf("det1 = %f\n", det1);
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3           + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0) {
            	xbs[ l + (p+0)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
			    d.xb[l + (i+0)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1))) {
                xbs[ l + (p+2)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+2)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (q == 0) {
                xbs[ l + (p+1)*3 + (q+0)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+0)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1))) {
                xbs[ l + (p+1)*3 + (q+2)*dimX*3         + (r+1)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+2)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (r == 0) {
                xbs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+0)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1))) {
                xbs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+2)*dimX*dimY*3] =
                d.xb[l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
        }
    }
    __syncthreads();

	if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		// compute the Jacobian matrix, assuming the initial grid X was rectilinear and equidistant
		for (l = 0; l < 3; l++) {
			jac[l][0] = (xbs[l + (p+2)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] -
					     xbs[l + (p+0)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dx1/2;
			jac[l][1] = (xbs[l + (p+1)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
					     xbs[l + (p+1)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dy1/2;
			jac[l][2] = (xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
					     xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_p.dz1/2;
		}

		// inverse Jacobian matrix
		jac1[0][0] = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])*det1;
		jac1[0][1] = (jac[0][2]*jac[2][1] - jac[0][1]*jac[2][2])*det1;
		jac1[0][2] = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1])*det1;
		jac1[1][0] = (jac[1][2]*jac[2][0] - jac[1][0]*jac[2][2])*det1;
		jac1[1][1] = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0])*det1;
		jac1[1][2] = (jac[0][2]*jac[1][0] - jac[0][0]*jac[1][2])*det1;
		jac1[2][0] = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0])*det1;
		jac1[2][1] = (jac[0][1]*jac[2][0] - jac[0][0]*jac[2][1])*det1;
		jac1[2][2] = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0])*det1;

		// second order derivatives of x
		for (l = 0; l < 3; l++) {
			d2x_dX2[l][0][0] = (xbs[  l + (p+2)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] -
								2*xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] +
								xbs[  l + (p+0)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dx1*dev_p.dx1;
			d2x_dX2[l][1][1] = (xbs[  l + (p+1)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
								2*xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] +
								xbs[  l + (p+1)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dy1*dev_p.dy1;
			d2x_dX2[l][2][2] = (xbs[  l + (p+1)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
								2*xbs[l + (p+1)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] +
								xbs[  l + (p+1)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_p.dz1*dev_p.dz1;
			d2x_dX2[l][0][1] = (xbs[  l + (p+2)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
								xbs[  l + (p+2)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3] -
								xbs[  l + (p+0)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] +
								xbs[  l + (p+0)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dx1*dev_p.dy1/4;
			d2x_dX2[l][0][2] = (xbs[  l + (p+2)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
								xbs[  l + (p+2)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3] -
								xbs[  l + (p+0)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] +
								xbs[  l + (p+0)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_p.dx*dev_p.dz/4;
			d2x_dX2[l][1][2] = (xbs[  l + (p+1)*3 + (q+2)*dimX*3  + (r+2)*dimX*dimY*3] -
								xbs[  l + (p+1)*3 + (q+0)*dimX*3  + (r+2)*dimX*dimY*3] -
								xbs[  l + (p+1)*3 + (q+2)*dimX*3  + (r+0)*dimX*dimY*3] +
								xbs[  l + (p+1)*3 + (q+0)*dimX*3  + (r+0)*dimX*dimY*3])*dev_p.dy*dev_p.dz/4;
			d2x_dX2[l][1][0] = d2x_dX2[l][0][1];
			d2x_dX2[l][2][0] = d2x_dX2[l][0][2];
			d2x_dX2[l][2][1] = d2x_dX2[l][1][2];
		}

		// add all up
		d.gradP[0 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] = 0;
		d.gradP[1 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] = 0;
		d.gradP[2 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] = 0;
		for (a = 0; a < 3; a++)
			for (b = 0; b < 3; b++)
				for (l = 0; l < 3; l++) {
					d.gradP[0 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] += d2x_dX2[a][b][l]*jac1[b][a]*jac1[l][0];
					d.gradP[1 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] += d2x_dX2[a][b][l]*jac1[b][a]*jac1[l][1];
					d.gradP[2 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] += d2x_dX2[a][b][l]*jac1[b][a]*jac1[l][2];
				}
		d.gradP[0 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] *= dev_p.beta*det1;
		d.gradP[1 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] *= dev_p.beta*det1;
		d.gradP[2 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] *= dev_p.beta*det1;
	}
}


// determine which routine should be used for the current calculation
void gradP(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct varsDev_t d, struct parameters_t p)
{
	if (strncmp(p.pMethod, "Classic ", 8) == 0)
		gradPClassic
			<<<dimGrid, dimBlock, (blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*(3*2+1)*sizeof(*(d.xb))>>>
			(d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
}
