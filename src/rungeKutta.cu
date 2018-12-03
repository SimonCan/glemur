// rungeKutta.cu
//
// Routines for the Runge-Kutta time stepping.
//

#include "rungeKutta.h"
#include "maths.h"

// Assign values to the coefficients for the adaptive time step Runge-Kutta.
void assignCoefficients(REAL a[5], REAL b[7][6], REAL c[6], REAL cs[6])
{
    // Not used for the moment, since we do not have any explicit time dependence.
    a[0] = 0; a[1] = 0.2; a[2] = 0.3; a[3] = 0.6; a[4] = 1; a[5] = 0.875;

    b[0][0] = 0;           b[0][1] = 0;        b[0][2] = 0;          b[0][3] = 0;             b[0][4] = 0;            b[0][5] = 0;
    b[1][0] = 0.2;         b[1][1] = 0;        b[1][2] = 0;          b[1][3] = 0;             b[1][4] = 0;            b[1][5] = 0;
    b[2][0] = 3/40.;       b[2][1] = 9/40.;    b[2][2] = 0;          b[2][3] = 0;             b[2][4] = 0;            b[2][5] = 0;
    b[3][0] = 0.3;         b[3][1] = -0.9;     b[3][2] = 1.2;        b[3][3] = 0;             b[3][4] = 0;            b[3][5] = 0;
    b[4][0] = -11/54.;     b[4][1] = 2.5;      b[4][2] = -70/27.;    b[4][3] = 35/27.;        b[4][4] = 0;            b[4][5] = 0;
    b[5][0] = 1631/55296.; b[5][1] = 175/512.; b[5][2] = 575/13824.; b[5][3] = 44275/110592.; b[5][4] = 253/4096.;    b[5][5] = 0;
    b[6][0] = 0;           b[6][1] = 0;        b[6][2] = 0;          b[6][3] = 0;             b[6][4] = 0;            b[6][5] = 0;

    c[0]  = 37/378.;     c[1]  = 0; c[2]  = 250/621.;     c[3]  = 125/594.;     c[4]  = 0;          c[5]  = 512/1771.;
    cs[0] = 2825/27648.; cs[1] = 0; cs[2] = 18575/48384.; cs[3] = 13525/55296.; cs[4] = 277/14336.; cs[5] = 0.25;
}

// Compute B = J.B0/Delta, first step in time stepping.
__global__ void B_JacB0(struct VarsDev d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l;
    REAL jac[3][3]; // Jacobian matrix
    REAL detJac1;   // inverse of the determinant of the Jacobian

    // Shared memory for faster communication. The size is assigned dynamically.
    extern __shared__ REAL s[];
    REAL *B0s = s;                      // magnetic field at t = 0
    REAL *xbs = &B0s[3*dimX*dimY*dimZ]; // position vector at beginning of time step
    REAL *Bs  = &xbs[3*dimX*dimY*dimZ]; // current magnetic field

    // Copy from global memory to shared memory for faster computation.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for (l = 0; l < 3; l++) {
            // Assign the inner values without the boundaries.
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            d.xb[ l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            // Assign the block boundaries.
            if (p == 0)
                xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+0)*3      + (j+1)*(dev_params.nx+2)*3  + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            if (((p == (blockDim.x-1)) && (i < (dev_params.nx-1))) || (i == (dev_params.nx-1)))
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+2)*3      + (j+1)*(dev_params.nx+2)*3  + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            if (q == 0)
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+0)*(dev_params.nx+2)*3  + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            if (((q == (blockDim.y-1)) && (j < (dev_params.ny-1))) || (j == (dev_params.ny-1)))
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+2)*(dev_params.nx+2)*3  + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            if (r == 0)
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_params.nx+2)*3  + (k+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            if (((r == (blockDim.z-1)) && (k < (dev_params.nz-1))) || (k == (dev_params.nz-1)))
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                d.xb[ l + (i+1)*3      + (j+1)*(dev_params.nx+2)*3  + (k+2)*(dev_params.nx+2)*(dev_params.ny+2)*3];

            // Assign the magnetic field for t = 0.
            B0s[ l + (p+1)*3 + (q+1)*dimX*3              + (r+1)*dimX*dimY*3] =
            d.B0[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
        }
    }
    __syncthreads();

    // Compute the Jacobian matrix, assuming the initial grid X was rectilinear and equidistant.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for (l = 0; l < 3; l++) {
            jac[l][0] = (xbs[l + (p+2)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] -
                         xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3]) * dev_params.dx1 / 2;
            jac[l][1] = (xbs[l + (p+1)*3 + (q+2)*dimX*3 + (r+1)*dimX*dimY*3] -
                         xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3]) * dev_params.dy1 / 2;
            jac[l][2] = (xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+2)*dimX*dimY*3] -
                         xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3]) * dev_params.dz1 / 2;
        }
        detJac1 = 1./(jac[0][0]*jac[1][1]*jac[2][2] + jac[0][1]*jac[1][2]*jac[2][0] + jac[0][2]*jac[1][0]*jac[2][1] -
                  jac[0][0]*jac[1][2]*jac[2][1] - jac[0][1]*jac[1][0]*jac[2][2] - jac[0][2]*jac[1][1]*jac[2][0]);
    }

    // Compute the magnetic field from B0.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz))
       for (l = 0; l < 3; l++)
           Bs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] =
              (jac[l][0]*B0s[0 + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
               jac[l][1]*B0s[1 + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] +
               jac[l][2]*B0s[2 + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3])*detJac1;

    // Write B and detJac into global memory.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
       for (l = 0; l < 3; l++)
           d.B[ l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] =
           Bs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3];
       d.detJac[(i+1) + (j+1)*(dev_params.nx+2) + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)] = 1./detJac1;
    }
}


// Compute k_n (kk).
__global__ void kk(struct VarsDev d, int dimX, int dimY, int dimZ, int n, REAL dt)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, m, nVar;      // nVar = number of variables
    REAL epsilon;        // parameter for the Lorentz force
    REAL jxbTmp[6][3];   // contains the adjacent values for JxB in case of averaging

    // Shared memory for faster communication, the size is assigned dynamically.
    extern __shared__ REAL s[];
    REAL *xbs      = s;                                                                                     // position vector at beginning of time step
    REAL *Bs       = &xbs[3*(dimX-2)*(dimY-2)*(dimZ-2)];                                                    // current magnetic field
    REAL *Js       = &Bs[3*(dimX-2)*(dimY-2)*(dimZ-2)];                                                     // electric current density
    REAL *ks       = &Js[3*(dimX-2)*(dimY-2)*(dimZ-2)];                                                     // intermediate steps for RK
    REAL *Us       = &ks[6*6*(dimX-2)*(dimY-2)*(dimZ-2)*dev_params.inertia];                                // velocity field
    REAL *detJacS  = &Us[3*(dimX-2)*(dimY-2)*(dimZ-2)*dev_params.inertia];                                  // determinant of the Jacobian (1/density)
    REAL *gradPS   = &ks[3*6*(dimX-2)*(dimY-2)*(dimZ-2)*(1-dev_params.inertia)*dev_params.pressure
                         + (6*6+3+1)*(dimX-2)*(dimY-2)*(dimZ-2)*dev_params.inertia*dev_params.pressure];    // pressure gradient

    nVar = 3;
    if (dev_params.inertia == true)
        nVar = 6;

    // Copy from global memory to shared memory for faster computation.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for (l = 0; l < 3; l++) {
            xbs[ l + (p+0)*3 + (q+0)*blockDim.x*3   + (r+0)*blockDim.x*blockDim.y*3] =
            d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            Bs[ l + (p+0)*3 + (q+0)*blockDim.x*3   + (r+0)*blockDim.x*blockDim.y*3] =
            d.B[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            Js[ l + (p+0)*3 + (q+0)*blockDim.x*3   + (r+0)*blockDim.x*blockDim.y*3] =
            d.J[l + (i+0)*3 + (j+0)*dev_params.nx*3     + (k+0)*dev_params.nx*dev_params.ny*3];
            if (dev_params.inertia == true)
                Us[  l + (p+0)*3 + (q+0)*blockDim.x*3 + (r+0)*blockDim.x*blockDim.y*3] =
                d.uu[l + (i+0)*3 + (j+0)*dev_params.nx*3   + (k+0)*dev_params.nx*dev_params.ny*3];
            if (dev_params.pressure == true)
                gradPS[ l + (p+0)*3 + (q+0)*blockDim.x*3 + (r+0)*blockDim.x*blockDim.y*3] =
                d.gradP[l + (i+0)*3 + (j+0)*dev_params.nx*3   + (k+0)*dev_params.nx*dev_params.ny*3];
        }
        if (dev_params.inertia == true)
            detJacS[ p + q*blockDim.x + r*blockDim.x*blockDim.y] =
            d.detJac[(i+1) + (j+1)*(dev_params.nx+2) + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)];
        for (l = 0; l < nVar; l++)
            // Read previous kk (if any).
            for(m = 0; m < n; m++)
                ks[  l + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + m*blockDim.x*blockDim.y*blockDim.z*nVar] =
                d.kk[l + i*nVar + j*dev_params.nx*nVar   + k*dev_params.nx*dev_params.ny*nVar     + m*dev_params.nx*dev_params.ny*dev_params.nz*nVar];
    }
    __syncthreads();

    // Compute the vector k for this intermediate step.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        if (dev_params.inertia == false) {
            cross(&Js[0 + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3],
                 &Bs[0 + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3],
                 &ks[0 + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3 + n*blockDim.x*blockDim.y*blockDim.z*3]);
            if (dev_params.jxbAver == true) {
                jxbAver(d, Js, Bs,    xbs, jxbTmp, i, j, k, p, q, r);
                for (l = 0; l < 3; l++) {
                    ks[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3 + n*blockDim.x*blockDim.y*blockDim.z*3] *= (1-dev_params.jxbAverWeight);
                    for (m = 0; m < 6; m++)
                        ks[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3 + n*blockDim.x*blockDim.y*blockDim.z*3] += jxbTmp[m][l];
                }
            }
            if (dev_params.epsilonProf == true)
               epsilon = (-(i-(dev_params.nx)/2.)/(dev_params.nx)*(i+1-(dev_params.nx)/2.)/(dev_params.nx) * 4 + 1) *
                         (-(j-(dev_params.ny)/2.)/(dev_params.ny)*(j+1-(dev_params.ny)/2.)/(dev_params.ny) * 4 + 1) *
                         (-(k-(dev_params.nz)/2.)/(dev_params.nz)*(k+1-(dev_params.nz)/2.)/(dev_params.nz) * 4 + 1);
           else
               epsilon = 1;
           for (l = 0; l < 3; l++) {
               ks[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3 +
                  n*blockDim.x*blockDim.y*blockDim.z*3] *= dt*epsilon;
                  // Add grad(p).
               if (dev_params.pressure == true)
                   ks[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3 +
                       n*blockDim.x*blockDim.y*blockDim.z*3] += -dt*gradPS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
           }
        }
        else {
            cross(&Js[0 + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3],
                 &Bs[0 + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3],
                 &ks[3 + p*6 + q*blockDim.x*6 + r*blockDim.x*blockDim.y*6 + n*blockDim.x*blockDim.y*blockDim.z*6]);
            for (l = 0; l < 3; l++) {
                ks[l + p*6 + q*blockDim.x*6 + r*blockDim.x*blockDim.y*6 + n*blockDim.x*blockDim.y*blockDim.z*6] =
                    dt*Us[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
                ks[3+l + p*6 + q*blockDim.x*6 + r*blockDim.x*blockDim.y*6 + n*blockDim.x*blockDim.y*blockDim.z*6] *=
                    dt*detJacS[p + q*blockDim.x + r*blockDim.x*blockDim.y];
                ks[3+l + p*6 + q*blockDim.x*6 + r*blockDim.x*blockDim.y*6 + n*blockDim.x*blockDim.y*blockDim.z*6] -=
                    dt*dev_params.nu*Us[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3]
                    *detJacS[p + q*blockDim.x + r*blockDim.x*blockDim.y];
               // Add grad(p).
               if (dev_params.pressure == true)
                   ks[3+l + p*6 + q*blockDim.x*6 + r*blockDim.x*blockDim.y*6 + n*blockDim.x*blockDim.y*blockDim.z*6] +=
                       -dt*gradPS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
            }
        }
    }

    // Update xbs = xbs + bb*k + ... and Us = Us + bb*k + ... .
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for(l = 0; l < 3; l++)
            for(m = 0; m < (n+1); m++) {
                xbs[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] += (b_ij[m + (n+1)*6] - b_ij[m + n*6]) *
                    ks[ l + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar +
                    m*blockDim.x*blockDim.y*blockDim.z*nVar];
                if (dev_params.inertia == true)
                    Us[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] += (b_ij[m + (n+1)*6] - b_ij[m + n*6]) *
                        ks[3+l + p*6 + q*blockDim.x*6 + r*blockDim.x*blockDim.y*6 +
                        m*blockDim.x*blockDim.y*blockDim.z*6];
            }
    }
    __syncthreads();

    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        // Write xbs, us and ks into global memory.
       for (l = 0; l < 3; l++) {
           // Since other blocks might still read from xb use xb_tmp.
           d.xb_tmp[ l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] =
           xbs[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
           d.kk[l + i*nVar + j*dev_params.nx*nVar   + k*dev_params.nx*dev_params.ny*nVar     + n*dev_params.nx*dev_params.ny*dev_params.nz*nVar] =
           ks[  l + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar];
           if (dev_params.inertia == true) {
               d.uu_tmp[ l + i*3 + j*dev_params.nx*3 + k*dev_params.nx*dev_params.ny*3] =
               Us[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
               d.kk[l+3 + i*nVar + j*dev_params.nx*nVar   + k*dev_params.nx*dev_params.ny*nVar     + n*dev_params.nx*dev_params.ny*dev_params.nz*nVar] =
               ks[  l+3 + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar];
           }
        }
    }
    __syncthreads();
}


// Compute the new distorted grid.
__global__ void xNewStar(struct VarsDev d, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, n, nVar;              // nVar = number of variables
    __shared__ REAL maxDelta_s;  // maximum error in this block

    // Shared memory for faster communication, the size is assigned dynamically.
    extern __shared__ REAL s[];
    REAL *xb_newS  = s;
    REAL *xb_star  = &xb_newS[3*(dimX-2)*(dimY-2)*(dimZ-2)]; // 'dim' includes the boundaries
    REAL *delta    = &xb_star[3*(dimX-2)*(dimY-2)*(dimZ-2)]; // error for each grid point
    REAL *xbs      = &delta[3*(dimX-2)*(dimY-2)*(dimZ-2)];   // position vector at beginning of time step
    REAL *ks       = &xbs[3*(dimX-2)*(dimY-2)*(dimZ-2)];     // intermediate steps for RK
    REAL *Us       = &ks[6*6*(dimX-2)*(dimY-2)*(dimZ-2)*dev_params.inertia];    // velocity field
    REAL *uu_newS  = &Us[3*(dimX-2)*(dimY-2)*(dimZ-2)*dev_params.inertia];
    REAL *uu_star  = &uu_newS[3*(dimX-2)*(dimY-2)*(dimZ-2)*dev_params.inertia]; // velocity field at beginning of time step

    nVar = 3;
    if (dev_params.inertia == true)
        nVar = 6;

    // Copy from global memory to shared memory for faster computation.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for (l = 0; l < 3; l++) {
            // Assign only the inner values without the boundaries.
            xbs[ l + p*3     + q*blockDim.x*3       + r*blockDim.x*blockDim.y*3] =
            d.xb[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
            if (dev_params.inertia == true)
                Us[  l + (p+0)*3 + (q+0)*blockDim.x*3 + (r+0)*blockDim.x*blockDim.y*3] =
                d.uu[l + (i+0)*3 + (j+0)*dev_params.nx*3   + (k+0)*dev_params.nx*dev_params.ny*3];
        }
        // Read kk.
        for (l = 0; l < nVar; l++)
            for(n = 0; n < 6; n++)
                ks[  l + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar] =
                d.kk[l + i*nVar + j*dev_params.nx*nVar   + k*dev_params.nx*dev_params.ny*nVar     + n*dev_params.nx*dev_params.ny*dev_params.nz*nVar];
    }
    __syncthreads();

    // Compute xb for the next full time step.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
       for (l = 0; l < 3; l++) {
           xb_newS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] = xbs[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
           xb_star[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] = xbs[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
           if ((l < 2) || (dev_params.zUpdate == true)) { // switch off z-update for diagnostics if requested
               for (n = 0; n < 6; n++) {
                   xb_newS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] +=
                   c_i[n]*ks[l + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar];
                   xb_star[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] +=
                   cs_i[n]*ks[l + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar];
               }
           }
       }
    }
    // Compute uu for the next full time step.
    if ((dev_params.inertia == true) && (i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
       for (l = 0; l < 3; l++) {
           uu_newS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] = Us[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
           uu_star[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] = Us[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
           for (n = 0; n < 6; n++) {
               uu_newS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] +=
               c_i[n]*ks[l+3 + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar];
               uu_star[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] +=
               cs_i[n]*ks[l+3 + p*nVar + q*blockDim.x*nVar + r*blockDim.x*blockDim.y*nVar + n*blockDim.x*blockDim.y*blockDim.z*nVar];
           }
       }
    }
    __syncthreads();

    // Determine the error (only for xb).
    for(l = 0; l < 3; l++) {
        if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz))
            delta[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] =
                abs(xb_newS[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] -
                    xb_star[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3]);
        else
            delta[l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3] = 0;
    }
    __syncthreads();
    if ((p == 0) && (q == 0) && (r == 0)) {
        maxDelta_s = maxValue(delta, blockDim.x*blockDim.y*blockDim.z*3);
        // Communicate the error into global memory.
        d.maxDelta[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y] = maxDelta_s;
    }
    __syncthreads();

    // Copy the latest values into global memory.
    if ((i < dev_params.nx) && (j < dev_params.ny) && (k < dev_params.nz)) {
        for(l = 0; l < 3; l++) {
            d.xb_new[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3] =
            xb_newS[ l + p*3     + q*blockDim.x*3       + r*blockDim.x*blockDim.y*3];
            if (dev_params.inertia == true) {
                d.uu_new[l + i*3 + j*dev_params.nx*3   + k*dev_params.nx*dev_params.ny*3] =
                uu_newS[ l + p*3 + q*blockDim.x*3 + r*blockDim.x*blockDim.y*3];
            }
        }
    }
}


// Compute the averaged Lorentz force.
__device__ void jxbAver(struct VarsDev d, REAL *Js, REAL *Bs, REAL *xs, REAL jxb[6][3], int i, int j, int k, int p, int q, int r) {
    REAL jj[6][3] = {0}, bb[6][3] = {0}, xx[6][3] = {0}; // adjacent values
    REAL diff[3];   // difference vector between central and adjacent points
    REAL dist[6];   // distance between central and adjacent points used for the weighing
    REAL weightTot; // sum of all weights
    int l, m;

    // x0
    if (i > 0)
        for (l = 0; l < 3; l++) {
            jj[0][l] = d.J[l + (i-1)*3 + (j+0)*dev_params.nx*3 + (k+0)*dev_params.nx*dev_params.ny*3];
            bb[0][l] = d.B[l + (i+0)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
        }
    else
        for (l = 0; l < 3; l++) {
            jj[0][l] = 0;
            bb[0][l] = 0;
        }

    // x1
    if (i < dev_params.nx-1)
        for (l = 0; l < 3; l++) {
            jj[1][l] = d.J[l + (i+1)*3 + (j+0)*dev_params.nx*3 + (k+0)*dev_params.nx*dev_params.ny*3];
            bb[1][l] = d.B[l + (i+2)*3 + (j+1)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
        }
    else
        for (l = 0; l < 3; l++) {
            jj[1][l] = 0;
            bb[1][l] = 0;
        }

    // y0
    if (j > 0)
        for (l = 0; l < 3; l++) {
            jj[2][l] = d.J[l + (i+0)*3 + (j-1)*dev_params.nx*3 + (k+0)*dev_params.nx*dev_params.ny*3];
            bb[2][l] = d.B[l + (i+1)*3 + (j+0)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
        }
    else
        for (l = 0; l < 3; l++) {
            jj[2][l] = 0;
            bb[2][l] = 0;
        }

    // y1
    if (j < dev_params.ny-1)
        for (l = 0; l < 3; l++) {
            jj[3][l] = d.J[l + (i+0)*3 + (j+1)*dev_params.nx*3 + (k+0)*dev_params.nx*dev_params.ny*3];
            bb[3][l] = d.B[l + (i+1)*3 + (j+2)*(dev_params.nx+2)*3 + (k+1)*(dev_params.nx+2)*(dev_params.ny+2)*3];
        }
    else
        for (l = 0; l < 3; l++) {
            jj[3][l] = 0;
            bb[3][l] = 0;
        }

    // z0
    if (k > 0)
        for (l = 0; l < 3; l++) {
            jj[4][l] = d.J[l + (i+0)*3 + (j+0)*dev_params.nx*3 + (k-1)*dev_params.nx*dev_params.ny*3];
            bb[4][l] = d.B[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+0)*(dev_params.nx+2)*(dev_params.ny+2)*3];
        }
    else
        for (l = 0; l < 3; l++) {
            jj[4][l] = 0;
            bb[4][l] = 0;
        }

    // z1
    if (k < dev_params.nz-1)
        for (l = 0; l < 3; l++) {
            jj[5][l] = d.J[l + (i+0)*3 + (j+0)*dev_params.nx*3 + (k+1)*dev_params.nx*dev_params.ny*3];
            bb[5][l] = d.B[l + (i+1)*3 + (j+1)*(dev_params.nx+2)*3 + (k+2)*(dev_params.nx+1)*(dev_params.ny+1)*3];
        }
    else
        for (l = 0; l < 3; l++) {
            jj[5][l] = 0;
            bb[5][l] = 0;
        }

    for (m = 0; m < 6; m++)
        cross(&jj[m][0], &bb[m][0], &jxb[m][0]);

    // Do the weighing.
    for (m = 0; m < 6; m++)
        for (l = 0; l < 3; l++)
            jxb[m][l] *= dev_params.jxbAverWeight/6;
}

