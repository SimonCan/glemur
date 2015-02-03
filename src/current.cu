// current.cu
//
// Routines for computing J = curl(B)

#include "current.h"
#include "maths.h"

// compute J using the standard derivatives as in [4] eq. (2.9)
__global__ void JClassic(struct varsDev_t d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, a, b;
    int u, v;					// auxiliary variables for reading in the block edges
    REAL jac[3][3], jac1[3][3]; // Jacobian matrix and its inverse
    REAL d2x_dX2[3][3][3];      // second order derivative of xb
    REAL dB0_dX[3][3];          // derivative of B0
    REAL dDet1_dX[3];           // derivative of detJac**-1
    REAL det1;					// inverse determinant
    REAL B0l[3];				// local B0 for shorter notation

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;                       // position vector at beginning of time step
    REAL *B0s      = &xbs[3*dimX*dimY*dimZ];  // B at t=0
    REAL *detJacS  = &B0s[3*dimX*dimY*dimZ];  // det(Jac)

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            B0s[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.B0[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            B0l[l] = B0s[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3];
            if (l == 0)
				detJacS[(p+1) + (q+1)*dimX         + (r+1)*dimX*dimY] =
				d.detJac[ (i+1) + (j+1)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)];
            // assign the block boundaries
            if (p == 0) {
            	xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    B0s[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B0[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
				if (l == 0)
					detJacS[(p+0) + (q+1)*dimX         + (r+1)*dimX*dimY] =
					d.detJac[ (i+0) + (j+1)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)]; }
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1))) {
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
 			    B0s[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
 			    d.B0[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
				if (l == 0)
					detJacS[(p+2) + (q+1)*dimX         + (r+1)*dimX*dimY] =
					d.detJac[ (i+2) + (j+1)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)]; }
            if (q == 0) {
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    B0s[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B0[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
				if (l == 0)
					detJacS[(p+1) + (q+0)*dimX         + (r+1)*dimX*dimY] =
					d.detJac[ (i+1) + (j+0)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)]; }
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1))) {
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    B0s[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B0[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
				if (l == 0)
					detJacS[(p+1) + (q+2)*dimX         + (r+1)*dimX*dimY] =
					d.detJac[ (i+1) + (j+2)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)]; }
            if (r == 0) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    B0s[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
			    d.B0[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
				if (l == 0)
					detJacS[(p+1) + (q+1)*dimX         + (r+0)*dimX*dimY] =
					d.detJac[ (i+1) + (j+1)*(dev_p.nx+2) + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)]; }
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1))) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    B0s[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
			    d.B0[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
				if (l == 0)
					detJacS[(p+1) + (q+1)*dimX         + (r+2)*dimX*dimY] =
					d.detJac[ (i+1) + (j+1)*(dev_p.nx+2) + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)]; }
        }
    }
    __syncthreads();
	// assign the block edges
	if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		for (l = 0; l < 3; l++) {
			if ((((p == 0) && (q == 0)) || ((p == 0) && (q == (blockDim.y-1))) || ((p == (blockDim.x-1)) && (q == 0)) || ((p == (blockDim.x-1)) && (q == (blockDim.y-1)))
				|| ((i == 0) && (j == (dev_p.ny-1))) || ((i == (dev_p.nx-1)) && (j == 0)) || ((i == (dev_p.nx-1)) && (j == (dev_p.ny-1))))
			    || (((p == 0) || (p == (blockDim.x-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (q+1)/blockDim.y;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (j == (dev_p.ny-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((p == 0) && (r == 0)) || ((p == 0) && (r == (blockDim.z-1))) || ((p == (blockDim.x-1)) && (r == 0)) || ((p == (blockDim.x-1)) && (r == (blockDim.z-1)))
				|| ((i == 0) && (k == (dev_p.nz-1))) || ((i == (dev_p.nx-1)) && (k == 0)) || ((i == (dev_p.nx-1)) && (k == (dev_p.nz-1))))
				|| (((p == 0) || (p == (blockDim.x-1))) && (k == (dev_p.nz-1))) || (((r == 0) || (r == (blockDim.z-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (r+1)/blockDim.z;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((r == 0) && (q == 0)) || ((r == 0) && (q == (blockDim.y-1))) || ((r == (blockDim.z-1)) && (q == 0)) || ((r == (blockDim.z-1)) && (q == (blockDim.y-1)))
				|| ((j == 0) && (k == (dev_p.nz-1))) || ((j == (dev_p.ny-1)) && (k == 0)) || ((j == (dev_p.ny-1)) && (k == (dev_p.nz-1))))
				|| (((r == 0) || (r == (blockDim.z-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (k == (dev_p.nz-1)))) {
            	u = (q+1)/blockDim.y; v = (r+1)/blockDim.z;
            	if (j == (dev_p.ny-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
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

		det1 = 1/detJacS[(p+1) + (q+1)*dimX + (r+1)*dimX*dimY];

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

		// derivative of B0
		for (l = 0; l < 3; l++) {
			dB0_dX[l][0] = (B0s[l + (p+2)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3] -
						    B0s[l + (p+0)*3 + (q+1)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dx1/2;
			dB0_dX[l][1] = (B0s[l + (p+1)*3 + (q+2)*dimX*3  + (r+1)*dimX*dimY*3] -
						    B0s[l + (p+1)*3 + (q+0)*dimX*3  + (r+1)*dimX*dimY*3])*dev_p.dy1/2;
			dB0_dX[l][2] = (B0s[l + (p+1)*3 + (q+1)*dimX*3  + (r+2)*dimX*dimY*3] -
						    B0s[l + (p+1)*3 + (q+1)*dimX*3  + (r+0)*dimX*dimY*3])*dev_p.dz1/2;
		}

		// derivative of the inverse determinant of the Jacobian
		dDet1_dX[0] = (1/detJacS[(p+2) + (q+1)*dimX  + (r+1)*dimX*dimY] -
					   1/detJacS[(p+0) + (q+1)*dimX  + (r+1)*dimX*dimY])*dev_p.dx1/2;
		dDet1_dX[1] = (1/detJacS[(p+1) + (q+2)*dimX  + (r+1)*dimX*dimY] -
					   1/detJacS[(p+1) + (q+0)*dimX  + (r+1)*dimX*dimY])*dev_p.dy1/2;
		dDet1_dX[2] = (1/detJacS[(p+1) + (q+1)*dimX  + (r+2)*dimX*dimY] -
					   1/detJacS[(p+1) + (q+1)*dimX  + (r+0)*dimX*dimY])*dev_p.dz1/2;

		// add all up
		d.J[0 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] = 0;
		d.J[1 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] = 0;
		d.J[2 + i*3 + j*dev_p.nx*3  + k*dev_p.nx*dev_p.ny*3] = 0;
		for (a = 0; a < 3; a++)
			for (b = 0; b < 3; b++) {
				d.J[0 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] +=
						jac1[a][1]*(d2x_dX2[2][a][b]*B0l[b]*det1 + jac[2][b]*dB0_dX[b][a]*det1 + jac[2][b]*B0l[b]*dDet1_dX[a]) -
						jac1[a][2]*(d2x_dX2[1][a][b]*B0l[b]*det1 + jac[1][b]*dB0_dX[b][a]*det1 + jac[1][b]*B0l[b]*dDet1_dX[a]);
				d.J[1 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] +=
						jac1[a][2]*(d2x_dX2[0][a][b]*B0l[b]*det1 + jac[0][b]*dB0_dX[b][a]*det1 + jac[0][b]*B0l[b]*dDet1_dX[a]) -
						jac1[a][0]*(d2x_dX2[2][a][b]*B0l[b]*det1 + jac[2][b]*dB0_dX[b][a]*det1 + jac[2][b]*B0l[b]*dDet1_dX[a]);
				d.J[2 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] +=
						jac1[a][0]*(d2x_dX2[1][a][b]*B0l[b]*det1 + jac[1][b]*dB0_dX[b][a]*det1 + jac[1][b]*B0l[b]*dDet1_dX[a]) -
						jac1[a][1]*(d2x_dX2[0][a][b]*B0l[b]*det1 + jac[0][b]*dB0_dX[b][a]*det1 + jac[0][b]*B0l[b]*dDet1_dX[a]);
			}
	}
}


// compute the electric current density using mimetic operators (Stokes method)
__global__ void JStokes(struct varsDev_t d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, m;

    REAL dx1[3], dx2[3], dx3[3], dx4[3];   // difference vectors for the Stokes method
    REAL dx1Xdx2[3], dx2Xdx3[3], dx3Xdx4[3], dx4Xdx1[3]; // cross products for the Stokes method
    REAL I;            	// electric current through a quadrilateral
    REAL detN;			// determinant of N
    REAL detN1;			// inverse determinant of N
    REAL NA[3][3];		// n.A with area weighted normals
    REAL NA1[3][3];		// inverse to NA
    REAL II[3];        	// the three intermediate currents from the Stokes method (area weighted)

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;                      	// position vector at beginning of time step
    REAL *Bs       = &xbs[3*dimX*dimY*dimZ]; 	// current magnetic field
    REAL *Js	   = &Bs[3*dimX*dimY*dimZ];  	// electric current density

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            Bs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.B[  l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0) {
            	xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1))) {
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
 			    Bs[ l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
 			    d.B[  l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (q == 0) {
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1))) {
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (r == 0) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1))) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		// quadrilateral 0
		I = 0;
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3] -
					 xbs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			I +=     (Bs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3] +
					  Bs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l] +
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3])/2 * dx2[l] +
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l] +
					 (Bs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
		}
		// to save computation time, compute the cross products and norms used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		II[0] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			NA[0][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
		}

		// quadrilateral 1
		I = 0;
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			I +=     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3])/2 * dx1[l] +
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3])/2     * dx2[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l] +
					 (Bs[l + (p+1)*3 + q*dimX*3     + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx4[l];
		}
		// to save computation time, compute the cross products and norm used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		II[1] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			NA[1][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
		}

		// quadrilateral 2
		I = 0;
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			I     += (Bs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3])/2     * dx1[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + p*3     + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l] +
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx3[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + r*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
		}
		// to save computation time, compute the cross products and norm used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		II[2] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			NA[2][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
		}

		// compute the inverse to NA
		detN = NA[0][0]*NA[1][1]*NA[2][2] + NA[0][1]*NA[1][2]*NA[2][0] + NA[0][2]*NA[1][0]*NA[2][1] -
			   NA[0][0]*NA[1][2]*NA[2][1] - NA[0][1]*NA[1][0]*NA[2][2] - NA[0][2]*NA[1][1]*NA[2][0];
		detN1 = 1/detN;
		NA1[0][0] = (NA[1][1]*NA[2][2] - NA[1][2]*NA[2][1])*detN1;
		NA1[0][1] = (NA[0][2]*NA[2][1] - NA[0][1]*NA[2][2])*detN1;
		NA1[0][2] = (NA[0][1]*NA[1][2] - NA[0][2]*NA[1][1])*detN1;
		NA1[1][0] = (NA[1][2]*NA[2][0] - NA[1][0]*NA[2][2])*detN1;
		NA1[1][1] = (NA[0][0]*NA[2][2] - NA[0][2]*NA[2][0])*detN1;
		NA1[1][2] = (NA[0][2]*NA[1][0] - NA[0][0]*NA[1][2])*detN1;
		NA1[2][0] = (NA[1][0]*NA[2][1] - NA[1][1]*NA[2][0])*detN1;
		NA1[2][1] = (NA[0][1]*NA[2][0] - NA[0][0]*NA[2][1])*detN1;
		NA1[2][2] = (NA[0][0]*NA[1][1] - NA[0][1]*NA[1][0])*detN1;

		// compute J from Jn as in eq. (18), ref [1]
		for (l = 0; l < 3; l++) {
			Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] = 0;
			d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] = 0;
			for (m = 0; m < 3; m++) {
				Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] += NA1[l][m]*II[m]; // area weighted
				d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] += NA1[l][m]*II[m];
			}
		}
    }
    __syncthreads();
}


// compute the electric current density via the 4th orderStokes method
__global__ void JStokes4th(struct varsDev_t d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, m;

    REAL dx1[3], dx2[3], dx3[3], dx4[3], dx5[3], dx6[3], dx7[3], dx8[3];   // difference vectors for the Stokes method
    REAL dxA[3], dxB[3], dxC[3], dxD[3]; // additional difference vectors for the Stokes method
    REAL dxAXdxB[3], dxBXdxC[3], dxCXdxD[3], dxDXdxA[3]; // cross products for the Stokes method
    REAL I;         // electric current through a quadrilateral
    REAL NA[3][3];	// n.A with area weighted normals
    REAL NA1[3][3];	// inverse to NA
    REAL detNA;		// determinant of NA
    REAL II[3];     // the three intermediate currents from the Stokes method (area weighted)
    int u, v;		// auxiliary variables for reading in the block edges

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;                      // position vector at beginning of time step
    REAL *Bs       = &xbs[3*dimX*dimY*dimZ]; // current magnetic field
    REAL *Js	   = &Bs[3*dimX*dimY*dimZ];  // electric current density

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            Bs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.B[  l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0) {
            	xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1))) {
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
 			    Bs[ l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
 			    d.B[  l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (q == 0) {
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1))) {
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (r == 0) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1))) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
        }
    }
    __syncthreads();
	// assign the block edges
	if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		for (l = 0; l < 3; l++) {
			if ((((p == 0) && (q == 0)) || ((p == 0) && (q == (blockDim.y-1))) || ((p == (blockDim.x-1)) && (q == 0)) || ((p == (blockDim.x-1)) && (q == (blockDim.y-1)))
				|| ((i == 0) && (j == (dev_p.ny-1))) || ((i == (dev_p.nx-1)) && (j == 0)) || ((i == (dev_p.nx-1)) && (j == (dev_p.ny-1))))
			    || (((p == 0) || (p == (blockDim.x-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (q+1)/blockDim.y;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (j == (dev_p.ny-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.B[  l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((p == 0) && (r == 0)) || ((p == 0) && (r == (blockDim.z-1))) || ((p == (blockDim.x-1)) && (r == 0)) || ((p == (blockDim.x-1)) && (r == (blockDim.z-1)))
				|| ((i == 0) && (k == (dev_p.nz-1))) || ((i == (dev_p.nx-1)) && (k == 0)) || ((i == (dev_p.nx-1)) && (k == (dev_p.nz-1))))
				|| (((p == 0) || (p == (blockDim.x-1))) && (k == (dev_p.nz-1))) || (((r == 0) || (r == (blockDim.z-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (r+1)/blockDim.z;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.B[  l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((r == 0) && (q == 0)) || ((r == 0) && (q == (blockDim.y-1))) || ((r == (blockDim.z-1)) && (q == 0)) || ((r == (blockDim.z-1)) && (q == (blockDim.y-1)))
				|| ((j == 0) && (k == (dev_p.nz-1))) || ((j == (dev_p.ny-1)) && (k == 0)) || ((j == (dev_p.ny-1)) && (k == (dev_p.nz-1))))
				|| (((r == 0) || (r == (blockDim.z-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (k == (dev_p.nz-1)))) {
            	u = (q+1)/blockDim.y; v = (r+1)/blockDim.z;
            	if (j == (dev_p.ny-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.B[  l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		// octilateral 0
		I = 0;
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx7[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
				     xbs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx8[l] = xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			I     += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l] +
					 (Bs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l] +
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l] +
					 (Bs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l] +
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
					 (Bs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l] +
					 (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx7[l] +
					 (Bs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx8[l];
			dxA[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dxB[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dxC[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dxD[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		// compute the interpolated area enclosed by this octilateral in analogy to eq. (14), ref. [1]
		II[0] = I;
		// to save computation time, compute the cross products and norms in analogy to eq. (14), ref. [1]
		cross(dx2, dx3, dxAXdxB);
		cross(dx4, dx5, dxBXdxC);
		cross(dx6, dx7, dxCXdxD);
		cross(dx8, dx1, dxDXdxA);
		for (l = 0; l < 3; l++)
			NA[0][l] = (dxAXdxB[l] + dxBXdxC[l] + dxCXdxD[l] + dxDXdxA[l])/2;
		cross(dxA, dxB, dxAXdxB);
		cross(dxB, dxC, dxBXdxC);
		cross(dxC, dxD, dxCXdxD);
		cross(dxD, dxA, dxDXdxA);
		// compute the surface normal to this octilateral in analogy to eq. (15), ref [1]
		for (l = 0; l < 3; l++)
			NA[0][l] += (dxAXdxB[l] + dxBXdxC[l] + dxCXdxD[l] + dxDXdxA[l])/4;

		// octilateral 1
		I = 0;
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx7[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3];
			dx8[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			I +=     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx1[l] +
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx2[l] +
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx3[l] +
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
					 (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l] +
					 (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx7[l] +
					 (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx8[l];
			dxA[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dxB[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dxC[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dxD[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
		}
		// compute the interpolated area enclosed by this octilateral in analogy to eq. (14), ref. [1]
		II[1] = I;
		// to save computation time, compute the cross products and norms in analogy to eq. (14), ref. [1]
		cross(dx2, dx3, dxAXdxB);
		cross(dx4, dx5, dxBXdxC);
		cross(dx6, dx7, dxCXdxD);
		cross(dx8, dx1, dxDXdxA);
		for (l = 0; l < 3; l++)
			NA[1][l] = (dxAXdxB[l] + dxBXdxC[l] + dxCXdxD[l] + dxDXdxA[l])/2;
		cross(dxA, dxB, dxAXdxB);
		cross(dxB, dxC, dxBXdxC);
		cross(dxC, dxD, dxCXdxD);
		cross(dxD, dxA, dxDXdxA);
		// compute the surface normal to this octilateral in analogy to eq. (15), ref [1]
		for (l = 0; l < 3; l++)
			NA[1][l] += (dxAXdxB[l] + dxBXdxC[l] + dxCXdxD[l] + dxDXdxA[l])/4;

		// octilateral 2
		I = 0;
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx7[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx8[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			I +=     (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx1[l] +
					 (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx3[l] +
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx4[l] +
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx7[l] +
					 (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx8[l];
			dxA[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dxB[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dxC[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dxD[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		// compute the interpolated area enclosed by this octilateral in analogy to eq. (14), ref. [1]
		II[2] = I;
		// to save computation time, compute the cross products and norms in analogy to eq. (14), ref. [1]
		cross(dx2, dx3, dxAXdxB);
		cross(dx4, dx5, dxBXdxC);
		cross(dx6, dx7, dxCXdxD);
		cross(dx8, dx1, dxDXdxA);
		for (l = 0; l < 3; l++)
			NA[2][l] = (dxAXdxB[l] + dxBXdxC[l] + dxCXdxD[l] + dxDXdxA[l])/2;
		cross(dxA, dxB, dxAXdxB);
		cross(dxB, dxC, dxBXdxC);
		cross(dxC, dxD, dxCXdxD);
		cross(dxD, dxA, dxDXdxA);
		// compute the surface normal to this octilateral in analogy to eq. (15), ref [1]
		for (l = 0; l < 3; l++)
			NA[2][l] += (dxAXdxB[l] + dxBXdxC[l] + dxCXdxD[l] + dxDXdxA[l])/4;

		// compute the inverse to NA
		detNA = NA[0][0]*NA[1][1]*NA[2][2] + NA[0][1]*NA[1][2]*NA[2][0] + NA[0][2]*NA[1][0]*NA[2][1] -
			    NA[0][0]*NA[1][2]*NA[2][1] - NA[0][1]*NA[1][0]*NA[2][2] - NA[0][2]*NA[1][1]*NA[2][0];

		NA1[0][0] = (NA[1][1]*NA[2][2] - NA[1][2]*NA[2][1])/detNA;
		NA1[0][1] = (NA[0][2]*NA[2][1] - NA[0][1]*NA[2][2])/detNA;
		NA1[0][2] = (NA[0][1]*NA[1][2] - NA[0][2]*NA[1][1])/detNA;
		NA1[1][0] = (NA[1][2]*NA[2][0] - NA[1][0]*NA[2][2])/detNA;
		NA1[1][1] = (NA[0][0]*NA[2][2] - NA[0][2]*NA[2][0])/detNA;
		NA1[1][2] = (NA[0][2]*NA[1][0] - NA[0][0]*NA[1][2])/detNA;
		NA1[2][0] = (NA[1][0]*NA[2][1] - NA[1][1]*NA[2][0])/detNA;
		NA1[2][1] = (NA[0][1]*NA[2][0] - NA[0][0]*NA[2][1])/detNA;
		NA1[2][2] = (NA[0][0]*NA[1][1] - NA[0][1]*NA[1][0])/detNA;

//   	compute J from Jn as in eq. (18), ref [1]
		for (l = 0; l < 3; l++) {
			Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] = 0;
			for (m = 0; m < 3; m++) {
				Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] += NA1[l][m]*II[m]; // area weighted
			}
		}
    }
    __syncthreads();

	// J into global memory
	if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz))
	   for (l = 0; l < 3; l++)
		   d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] = Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3];
}


// compute the electric current density via the Stokes method using a quintilateral
__global__ void JStokesQuint(struct varsDev_t d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, m, u, v;

    REAL dx1[3], dx2[3], dx3[3], dx4[3], dx5[3], dx6[3];   // difference vectors for the Stokes method
    REAL dx1Xdx2[3], dx2Xdx3[3], dx3Xdx4[3], dx4Xdx1[3], dx5Xdx6[3]; // cross products for the Stokes method
    REAL I;        		// electric current through a quadrilateral
    REAL detN;			// determinant of N
    REAL detN1;			// inverse determinant of N
    REAL NA[3][3];		// n.A with area weighted normals
    REAL NA1[3][3];		// inverse to NA
    REAL II[3];         // the three intermediate currents from the Stokes method (area weighted)
    int  triangle;      // triangle number where the inner point lies in

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;                      	// position vector at beginning of time step
    REAL *Bs       = &xbs[3*dimX*dimY*dimZ]; 	// current magnetic field
    REAL *Js	   = &Bs[3*dimX*dimY*dimZ];  	// electric current density

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            Bs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.B[  l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0) {
            	xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1))) {
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
 			    Bs[ l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
 			    d.B[  l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (q == 0) {
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1))) {
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (r == 0) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1))) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
        }
    }
    __syncthreads();
	// assign the block edges
	if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		for (l = 0; l < 3; l++) {
			if ((((p == 0) && (q == 0)) || ((p == 0) && (q == (blockDim.y-1))) || ((p == (blockDim.x-1)) && (q == 0)) || ((p == (blockDim.x-1)) && (q == (blockDim.y-1)))
				|| ((i == 0) && (j == (dev_p.ny-1))) || ((i == (dev_p.nx-1)) && (j == 0)) || ((i == (dev_p.nx-1)) && (j == (dev_p.ny-1))))
			    || (((p == 0) || (p == (blockDim.x-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (q+1)/blockDim.y;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (j == (dev_p.ny-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.B[  l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((p == 0) && (r == 0)) || ((p == 0) && (r == (blockDim.z-1))) || ((p == (blockDim.x-1)) && (r == 0)) || ((p == (blockDim.x-1)) && (r == (blockDim.z-1)))
				|| ((i == 0) && (k == (dev_p.nz-1))) || ((i == (dev_p.nx-1)) && (k == 0)) || ((i == (dev_p.nx-1)) && (k == (dev_p.nz-1))))
				|| (((p == 0) || (p == (blockDim.x-1))) && (k == (dev_p.nz-1))) || (((r == 0) || (r == (blockDim.z-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (r+1)/blockDim.z;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.B[  l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((r == 0) && (q == 0)) || ((r == 0) && (q == (blockDim.y-1))) || ((r == (blockDim.z-1)) && (q == 0)) || ((r == (blockDim.z-1)) && (q == (blockDim.y-1)))
				|| ((j == 0) && (k == (dev_p.nz-1))) || ((j == (dev_p.ny-1)) && (k == 0)) || ((j == (dev_p.ny-1)) && (k == (dev_p.nz-1))))
				|| (((r == 0) || (r == (blockDim.z-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (k == (dev_p.nz-1)))) {
            	u = (q+1)/blockDim.y; v = (r+1)/blockDim.z;
            	if (j == (dev_p.ny-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.B[  l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		// quadrilateral 0
		I = 0; triangle = 0;
		// determine if the point lies outside the cell
		for (l = 0; l < 3; l++) {
			dx1[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		cross(dx1, dx2, dx1Xdx2);
		if ((dot(dx1Xdx2, dx5) < 0) or (dot(dx1Xdx2, dx6) > 0)) {
			for (l = 0; l < 3; l++) {
				dx5[l] = xbs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
						 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
				dx6[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
						 xbs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
				triangle = 2; } }
		if (triangle != 0) {
			cross(dx2, dx3, dx2Xdx3);
			if ((dot(dx2Xdx3, dx5) < 0) or (dot(dx2Xdx3, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 3; } } }
		if (triangle != 0) {
		cross(dx3, dx4, dx3Xdx4);
			if ((dot(dx3Xdx4, dx5) < 0) or (dot(dx3Xdx4, dx6) > 4)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 4; } } }
		cross(dx4, dx1, dx4Xdx1);
		if (triangle != 0) {
			if ((dot(dx4Xdx1, dx5) < 0) or (dot(dx4Xdx1, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 1; } } }
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			if (triangle != 1)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l];
			if (triangle != 2)
				I += (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l];
			if (triangle != 3)
				I += (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l];
			if (triangle != 4)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
			if (triangle == 1)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 2)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 3)
				I += (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 4)
				I += (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l];
		}
		// to save computation time, compute the cross products and norms used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		if (triangle > 0)
			cross(dx5, dx6, dx5Xdx6);
		II[0] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			NA[0][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
			if (triangle > 0)
				NA[0][l] += dx5Xdx6[l]/2;
		}

		// quadrilateral 1
		I = 0; triangle = 0;
		// determine if the point lies outside the cell
		for (l = 0; l < 3; l++) {
			dx1[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		cross(dx1, dx2, dx1Xdx2);
		if ((dot(dx1Xdx2, dx5) < 0) or (dot(dx1Xdx2, dx6) > 0)) {
			for (l = 0; l < 3; l++) {
				dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3] -
						 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
				dx6[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
						 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3];
				triangle = 2; } }
		if (triangle != 0) {
			cross(dx2, dx3, dx2Xdx3);
			if ((dot(dx2Xdx3, dx5) < 0) or (dot(dx2Xdx3, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3];
					triangle = 3; } } }
		if (triangle != 0) {
		cross(dx3, dx4, dx3Xdx4);
			if ((dot(dx3Xdx4, dx5) < 0) or (dot(dx3Xdx4, dx6) > 4)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					triangle = 4; } } }
		cross(dx4, dx1, dx4Xdx1);
		if (triangle != 0) {
			if ((dot(dx4Xdx1, dx5) < 0) or (dot(dx4Xdx1, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 1; } } }
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			if (triangle != 1)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l];
			if (triangle != 2)
				I += (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx2[l];
			if (triangle != 3)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l];
			if (triangle != 4)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx4[l];
			if (triangle == 1)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 2)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 3)
				I += (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 4)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l];
		}
		// to save computation time, compute the cross products and norm used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		if (triangle > 0)
			cross(dx5, dx6, dx5Xdx6);
		II[1] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			NA[1][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
			if (triangle > 0)
				NA[1][l] += dx5Xdx6[l]/2;
		}

		// quadrilateral 2
		I = 0; triangle = 0;
		// determine if the point lies outside the cell
		for (l = 0; l < 3; l++) {
			dx1[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		cross(dx1, dx2, dx1Xdx2);
		if ((dot(dx1Xdx2, dx5) < 0) or (dot(dx1Xdx2, dx6) > 0)) {
			for (l = 0; l < 3; l++) {
				dx5[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
						 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
				dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
						 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
				triangle = 2; } }
		if (triangle != 0) {
			cross(dx2, dx3, dx2Xdx3);
			if ((dot(dx2Xdx3, dx5) < 0) or (dot(dx2Xdx3, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					triangle = 3; } } }
		if (triangle != 0) {
		cross(dx3, dx4, dx3Xdx4);
			if ((dot(dx3Xdx4, dx5) < 0) or (dot(dx3Xdx4, dx6) > 4)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 4; } } }
		cross(dx4, dx1, dx4Xdx1);
		if (triangle != 0) {
			if ((dot(dx4Xdx1, dx5) < 0) or (dot(dx4Xdx1, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
					triangle = 1; } } }
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			if (triangle != 1)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx1[l];
			if (triangle != 2)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l];
			if (triangle != 3)
				I += (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx3[l];
			if (triangle != 4)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
			if (triangle == 1)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 2)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 3)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l];
			if (triangle == 4)
				I += (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l];
		}
		// to save computation time, compute the cross products and norm used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		if (triangle > 0)
			cross(dx5, dx6, dx5Xdx6);
		II[2] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			NA[2][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
			if (triangle > 0)
				NA[2][l] += dx5Xdx6[l]/2;
		}

		// compute the inverse to NA
		detN = NA[0][0]*NA[1][1]*NA[2][2] + NA[0][1]*NA[1][2]*NA[2][0] + NA[0][2]*NA[1][0]*NA[2][1] -
			   NA[0][0]*NA[1][2]*NA[2][1] - NA[0][1]*NA[1][0]*NA[2][2] - NA[0][2]*NA[1][1]*NA[2][0];
		detN1 = 1/detN;
		NA1[0][0] = (NA[1][1]*NA[2][2] - NA[1][2]*NA[2][1])*detN1;
		NA1[0][1] = (NA[0][2]*NA[2][1] - NA[0][1]*NA[2][2])*detN1;
		NA1[0][2] = (NA[0][1]*NA[1][2] - NA[0][2]*NA[1][1])*detN1;
		NA1[1][0] = (NA[1][2]*NA[2][0] - NA[1][0]*NA[2][2])*detN1;
		NA1[1][1] = (NA[0][0]*NA[2][2] - NA[0][2]*NA[2][0])*detN1;
		NA1[1][2] = (NA[0][2]*NA[1][0] - NA[0][0]*NA[1][2])*detN1;
		NA1[2][0] = (NA[1][0]*NA[2][1] - NA[1][1]*NA[2][0])*detN1;
		NA1[2][1] = (NA[0][1]*NA[2][0] - NA[0][0]*NA[2][1])*detN1;
		NA1[2][2] = (NA[0][0]*NA[1][1] - NA[0][1]*NA[1][0])*detN1;

		// compute J from Jn as in eq. (18), ref [1]
		for (l = 0; l < 3; l++) {
			Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] = 0;
			d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] = 0;
			for (m = 0; m < 3; m++) {
				Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] += NA1[l][m]*II[m]; // area weighted
				d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] += NA1[l][m]*II[m];
			}
		}
    }
    __syncthreads();
}


// compute the electric current density via the Stokes method using additional triangles
__global__ void JStokesTri(struct varsDev_t d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, m, u, v;

    REAL dx1[3], dx2[3], dx3[3], dx4[3], dx5[3], dx6[3];   // difference vectors for the Stokes method
    REAL dx1Xdx2[3], dx2Xdx3[3], dx3Xdx4[3], dx4Xdx1[3], dx5Xdx6[3]; // cross products for the Stokes method
    REAL I;            	// electric current through a quadrilateral
    REAL detN;			// determinant of N
    REAL detN1;			// inverse determinant of N
    REAL NA[3][3];		// n.A with area weighted normals
    REAL NA1[3][3];		// inverse to NA
    REAL II[3];        	// the three intermediate currents from the Stokes method (area weighted)
    int  triangle;      // triangle number where the inner point lies in

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;                      	// position vector at beginning of time step
    REAL *Bs       = &xbs[3*dimX*dimY*dimZ]; 	// current magnetic field
    REAL *Js	   = &Bs[3*dimX*dimY*dimZ];  	// electric current density

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            Bs[ l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.B[  l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0) {
            	xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1))) {
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
 			    Bs[ l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
 			    d.B[  l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (q == 0) {
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1))) {
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (r == 0) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1))) {
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
			    Bs[ l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
			    d.B[  l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3]; }
        }
    }
    __syncthreads();
	// assign the block edges
	if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		for (l = 0; l < 3; l++) {
			if ((((p == 0) && (q == 0)) || ((p == 0) && (q == (blockDim.y-1))) || ((p == (blockDim.x-1)) && (q == 0)) || ((p == (blockDim.x-1)) && (q == (blockDim.y-1)))
				|| ((i == 0) && (j == (dev_p.ny-1))) || ((i == (dev_p.nx-1)) && (j == 0)) || ((i == (dev_p.nx-1)) && (j == (dev_p.ny-1))))
			    || (((p == 0) || (p == (blockDim.x-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (q+1)/blockDim.y;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (j == (dev_p.ny-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+2*u)*3    + (q+2*v)*dimX*3          + (r+1)*dimX*dimY*3] =
            	d.B[  l + (i+2*u)*3    + (j+2*v)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((p == 0) && (r == 0)) || ((p == 0) && (r == (blockDim.z-1))) || ((p == (blockDim.x-1)) && (r == 0)) || ((p == (blockDim.x-1)) && (r == (blockDim.z-1)))
				|| ((i == 0) && (k == (dev_p.nz-1))) || ((i == (dev_p.nx-1)) && (k == 0)) || ((i == (dev_p.nx-1)) && (k == (dev_p.nz-1))))
				|| (((p == 0) || (p == (blockDim.x-1))) && (k == (dev_p.nz-1))) || (((r == 0) || (r == (blockDim.z-1))) && (i == (dev_p.nx-1)))) {
            	u = (p+1)/blockDim.x; v = (r+1)/blockDim.z;
            	if (i == (dev_p.nx-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+2*u)*3    + (q+1)*dimX*3            + (r+2*v)*dimX*dimY*3] =
            	d.B[  l + (i+2*u)*3    + (j+1)*(dev_p.nx+2)*3    + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
			if ((((r == 0) && (q == 0)) || ((r == 0) && (q == (blockDim.y-1))) || ((r == (blockDim.z-1)) && (q == 0)) || ((r == (blockDim.z-1)) && (q == (blockDim.y-1)))
				|| ((j == 0) && (k == (dev_p.nz-1))) || ((j == (dev_p.ny-1)) && (k == 0)) || ((j == (dev_p.ny-1)) && (k == (dev_p.nz-1))))
				|| (((r == 0) || (r == (blockDim.z-1))) && (j == (dev_p.ny-1))) || (((q == 0) || (q == (blockDim.y-1))) && (k == (dev_p.nz-1)))) {
            	u = (q+1)/blockDim.y; v = (r+1)/blockDim.z;
            	if (j == (dev_p.ny-1)) u = 1;
            	if (k == (dev_p.nz-1)) v = 1;
            	xbs[l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.xb[ l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            	Bs[ l + (p+1)*3      + (q+2*u)*dimX*3          + (r+2*v)*dimX*dimY*3] =
            	d.B[  l + (i+1)*3      + (j+2*u)*(dev_p.nx+2)*3  + (k+2*v)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            }
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
		// quadrilateral 0
		I = 0; triangle = 0;
		// determine if the point lies outside the cell
		for (l = 0; l < 3; l++) {
			dx1[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		cross(dx1, dx2, dx1Xdx2);
		if ((dot(dx1Xdx2, dx5) < 0) or (dot(dx1Xdx2, dx6) > 0)) {
			for (l = 0; l < 3; l++) {
				dx5[l] = xbs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
						 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
				dx6[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
						 xbs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
				triangle = 2; } }
		if (triangle != 0) {
			cross(dx2, dx3, dx2Xdx3);
			if ((dot(dx2Xdx3, dx5) < 0) or (dot(dx2Xdx3, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 3; } } }
		if (triangle != 0) {
		cross(dx3, dx4, dx3Xdx4);
			if ((dot(dx3Xdx4, dx5) < 0) or (dot(dx3Xdx4, dx6) > 4)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 4; } } }
		cross(dx4, dx1, dx4Xdx1);
		if (triangle != 0) {
			if ((dot(dx4Xdx1, dx5) < 0) or (dot(dx4Xdx1, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 1; } } }
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			if (triangle == 0)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l] +
				     (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l] +
				     (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l] +
				     (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
			if (triangle == 1)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
					 (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l];
			if (triangle == 2)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l];
			if (triangle == 3)
				I += (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l];
			if (triangle == 4)
				I += (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
		}
		// to save computation time, compute the cross products and norms used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		if (triangle > 0)
			cross(dx5, dx6, dx5Xdx6);
		II[0] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			if (triangle == 0)
				NA[0][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
			else
				NA[0][l] += dx5Xdx6[l]/2;
		}

		// quadrilateral 1
		I = 0; triangle = 0;
		// determine if the point lies outside the cell
		for (l = 0; l < 3; l++) {
			dx1[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		cross(dx1, dx2, dx1Xdx2);
		if ((dot(dx1Xdx2, dx5) < 0) or (dot(dx1Xdx2, dx6) > 0)) {
			for (l = 0; l < 3; l++) {
				dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3] -
						 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
				dx6[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
						 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3];
				triangle = 2; } }
		if (triangle != 0) {
			cross(dx2, dx3, dx2Xdx3);
			if ((dot(dx2Xdx3, dx5) < 0) or (dot(dx2Xdx3, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3];
					triangle = 3; } } }
		if (triangle != 0) {
		cross(dx3, dx4, dx3Xdx4);
			if ((dot(dx3Xdx4, dx5) < 0) or (dot(dx3Xdx4, dx6) > 4)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					triangle = 4; } } }
		cross(dx4, dx1, dx4Xdx1);
		if (triangle != 0) {
			if ((dot(dx4Xdx1, dx5) < 0) or (dot(dx4Xdx1, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 1; } } }
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			if (triangle == 0)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l] +
				     (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx2[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l] +
				     (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx4[l];
			if (triangle == 1)
				I += (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l];
			if (triangle == 2)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx2[l];
			if (triangle == 3)
				I += (Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l];
			if (triangle == 4)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx4[l];
		}
		// to save computation time, compute the cross products and norm used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		if (triangle > 0)
			cross(dx5, dx6, dx5Xdx6);
		II[1] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			if (triangle == 0)
				NA[1][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
			else
				NA[1][l] += dx5Xdx6[l]/2;
		}

		// quadrilateral 2
		I = 0; triangle = 0;
		// determine if the point lies outside the cell
		for (l = 0; l < 3; l++) {
			dx1[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx5[l] = xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx6[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
		}
		cross(dx1, dx2, dx1Xdx2);
		if ((dot(dx1Xdx2, dx5) < 0) or (dot(dx1Xdx2, dx6) > 0)) {
			for (l = 0; l < 3; l++) {
				dx5[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
						 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
				dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
						 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
				triangle = 2; } }
		if (triangle != 0) {
			cross(dx2, dx3, dx2Xdx3);
			if ((dot(dx2Xdx3, dx5) < 0) or (dot(dx2Xdx3, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
					triangle = 3; } } }
		if (triangle != 0) {
		cross(dx3, dx4, dx3Xdx4);
			if ((dot(dx3Xdx4, dx5) < 0) or (dot(dx3Xdx4, dx6) > 4)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
					triangle = 4; } } }
		cross(dx4, dx1, dx4Xdx1);
		if (triangle != 0) {
			if ((dot(dx4Xdx1, dx5) < 0) or (dot(dx4Xdx1, dx6) > 0)) {
				for (l = 0; l < 3; l++) {
					dx5[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
							 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
					dx6[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
							 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
					triangle = 1; } } }
		for (l = 0; l < 3; l++) { // eq. (13), ref [1]
			dx1[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3];
			dx2[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
					 xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			dx3[l] = xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
					 xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3];
			dx4[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] -
					 xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
			if (triangle == 0)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx1[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx2[l] +
				     (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx3[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx4[l];
			if (triangle == 1)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
					 (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx1[l];
			if (triangle == 2)
				I += (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx2[l];
			if (triangle == 3)
				I += (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])/2 * dx3[l];
			if (triangle == 4)
				I += (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx5[l] +
				     (Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] +
					  Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx6[l] -
					 (Bs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
					  Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3])/2 * dx4[l];
		}
		// to save computation time, compute the cross products and norm used in eq. (14), ref. [1]
		cross(dx1, dx2, dx1Xdx2);
		cross(dx2, dx3, dx2Xdx3);
		cross(dx3, dx4, dx3Xdx4);
		cross(dx4, dx1, dx4Xdx1);
		if (triangle > 0)
			cross(dx5, dx6, dx5Xdx6);
		II[2] = I;
		// compute the surface normal to this quadrilateral, as in eq. (15), ref [1]
		for (l = 0; l < 3; l++) {
			if (triangle == 0)
				NA[2][l] = (dx1Xdx2[l] + dx2Xdx3[l] + dx3Xdx4[l] + dx4Xdx1[l])/4;
			else
				NA[2][l] += dx5Xdx6[l]/2;
		}

		// compute the inverse to NA
		detN = NA[0][0]*NA[1][1]*NA[2][2] + NA[0][1]*NA[1][2]*NA[2][0] + NA[0][2]*NA[1][0]*NA[2][1] -
			   NA[0][0]*NA[1][2]*NA[2][1] - NA[0][1]*NA[1][0]*NA[2][2] - NA[0][2]*NA[1][1]*NA[2][0];
		detN1 = 1/detN;
		NA1[0][0] = (NA[1][1]*NA[2][2] - NA[1][2]*NA[2][1])*detN1;
		NA1[0][1] = (NA[0][2]*NA[2][1] - NA[0][1]*NA[2][2])*detN1;
		NA1[0][2] = (NA[0][1]*NA[1][2] - NA[0][2]*NA[1][1])*detN1;
		NA1[1][0] = (NA[1][2]*NA[2][0] - NA[1][0]*NA[2][2])*detN1;
		NA1[1][1] = (NA[0][0]*NA[2][2] - NA[0][2]*NA[2][0])*detN1;
		NA1[1][2] = (NA[0][2]*NA[1][0] - NA[0][0]*NA[1][2])*detN1;
		NA1[2][0] = (NA[1][0]*NA[2][1] - NA[1][1]*NA[2][0])*detN1;
		NA1[2][1] = (NA[0][1]*NA[2][0] - NA[0][0]*NA[2][1])*detN1;
		NA1[2][2] = (NA[0][0]*NA[1][1] - NA[0][1]*NA[1][0])*detN1;

		// compute J from Jn as in eq. (18), ref [1]
		for (l = 0; l < 3; l++) {
			Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] = 0;
			d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] = 0;
			for (m = 0; m < 3; m++) {
				Js[l + p*3 + q*(dimX-2)*3 + r*(dimX-2)*(dimY-2)*3] += NA1[l][m]*II[m]; // area weighted
				d.J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3] += NA1[l][m]*II[m];
			}
		}
    }
    __syncthreads();
}


// determine which routine should be used for the current calculation
void current(dim3 dimGrid, dim3 dimBlock, int blockSize[3], struct varsDev_t d, struct parameters_t p)
{
	if (strncmp(p.jMethod, "Stokes ", 7) == 0)
		JStokes
			<<<dimGrid, dimBlock, (3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*2 +
								   3*blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*(d.xb))>>>
			(d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
	if (strncmp(p.jMethod, "Stokes4th ", 10) == 0)
		JStokes4th
			<<<dimGrid, dimBlock, (3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*2 +
								   3*blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*(d.xb))>>>
			(d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
	if (strncmp(p.jMethod, "StokesQuint ", 12) == 0)
		JStokesQuint
			<<<dimGrid, dimBlock, (3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*2 +
								   3*blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*(d.xb))>>>
			(d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
	if (strncmp(p.jMethod, "StokesTri ", 10) == 0)
		JStokesTri
			<<<dimGrid, dimBlock, (3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*2 +
								   3*blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*(d.xb))>>>
			(d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
	if (strncmp(p.jMethod, "Classic ", 8) == 0)
		JClassic
			<<<dimGrid, dimBlock, (blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*(3*2+1)*sizeof(*(d.xb))>>>
			(d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
}
