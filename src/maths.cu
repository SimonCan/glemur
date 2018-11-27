// maths.cu
//
// Functions like the norm, cross product, etc.
//

#include "maths.h"

// calculate the three dimensional cross product c = axb
__device__ void cross(REAL a[3], REAL b[3], REAL c[3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}


// calculate the scalar product c = a.b
__device__ REAL dot(REAL a[3], REAL b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


// calculate the norm of a 3d vector
__device__ REAL norm(REAL a[3]) {
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}


// normalize this 3d vector
__device__ void normalize(REAL a[3]) {
    REAL s;
    int l;

    s = norm(a);
    if (s > 0)
        for (l = 0; l < 3; l++)
            a[l] = a[l]/s;
}


// atomic addition for doubles
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,  __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


// determine the maximum value in an array
__global__ void maxBlock(REAL *var, REAL *maxVal, int size) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int p = threadIdx.x;
    int l;
    REAL maxl;
    __shared__ REAL vars[64];
    __shared__ REAL maxVals[32];

    if (2*i < size)
        vars[2*p] = var[2*i];
    else
        vars[2*p] = var[0];
    if ((2*i+1) < size)
        vars[2*p+1] = var[2*i+1];
    else
        vars[2*p+1] = var[0];

    if (vars[2*p] < vars[2*p+1])
        maxVals[p] = vars[2*p+1];
    else
        maxVals[p] = vars[2*p];
    __syncthreads();

    maxl = maxVals[0];
    if (p == 0) {
        for (l = 0; l < 32; l++)
            if (maxl < maxVals[l])
                maxl = maxVals[l];
        maxVal[blockIdx.x] = maxl;
    }

    // TODO: implement faster way
//    for (l == 2; l < 32; l *= 2) {
//        if (p == p/l)
//        maxVals[2*p/l];
//        maxVals[2*p/l+1];
//        maxVals[2*p/l] =;
//        __synchthreads();
//    }
}


// determine the maximum value of a device variable
REAL findDevMax(REAL *dev_var, int size) {
    REAL *dev_max;
    REAL *dev_tmp;
    REAL maxVal;
    cudaError_t errCuda;

    // allocate global memory on GPU device
    errCuda = cudaMalloc((void**)&dev_max, size*sizeof(REAL));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for dev_max\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(dev_max, dev_var, size*sizeof(REAL), cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'var' within the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&dev_tmp, int(ceil(size/64.))*sizeof(REAL));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for dev_tmp\n"); exit(EXIT_FAILURE); }

    while (size > 1) {
        maxBlock<<<int(ceil(size/64.)), 32>>>(dev_max, dev_tmp, size);
        cudaDeviceSynchronize();
        errCuda = cudaMemcpy(dev_max, dev_tmp, int(ceil(size/64.))*sizeof(REAL), cudaMemcpyDeviceToDevice);
        if (cudaSuccess != errCuda) { printf("error: could not copy 'max' within the device\n"); exit(EXIT_FAILURE); }
        cudaDeviceSynchronize();
        size = int(ceil(size/64.));
    }

    errCuda = cudaMemcpy(&maxVal, dev_max, 1*sizeof(REAL), cudaMemcpyDeviceToHost);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'max' to the host\n"); exit(EXIT_FAILURE); }

    cudaFree(dev_max); cudaFree(dev_tmp);

    return maxVal;
}


// invert sign of device variable
__global__ void invert(REAL *var, int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < size)
        var[i] = -var[i];
}


// determine the maximum value in an array
__device__ REAL maxValue(REAL *a, int len) {
    int i;
    REAL max = 0;

    for (i = 0; i < len; i++)
        if (a[i] > max)
            max = a[i];

    return max;
}


// compute the norm of a vector
__global__ void mag(REAL *vec, REAL *mag) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz))
        mag[i + j*dev_p.nx + k*dev_p.nx*dev_p.ny] = norm(&vec[0 + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3]);
}


// compute the norm of JxB/B**2
__global__ void JxB_B2(REAL *B, REAL *J, REAL *JxB_B2, int dimX, int dimY, int dimZ) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l;
    REAL B2;

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *Bs = s;                                 // magnetic field
    REAL *Js = &s[3 * dimX * dimY * dimZ];         // electric current density
    REAL *JxBs = &Js[3 * dimX * dimY * dimZ];     // JxB

    // copy from global memory into shared memory
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            Bs[l + p*3 + q*dimX*3 + r*dimX*dimY*3] = B[l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            Js[l + p*3 + q*dimX*3 + r*dimX*dimY*3] = J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3];
        }

        cross(&Js[0 + p*3 + q*dimX*3 + r*dimX*dimY*3],
                &Bs[0 + p*3 + q*dimX*3 + r*dimX*dimY*3],
                &JxBs[0 + p*3 + q*dimX*3 + r*dimX*dimY*3]);

        B2 = dot(&Bs[0 + p*3 + q*dimX*3 + r*dimX*dimY*3], &Bs[0 + p*3 + q*dimX*3 + r*dimX*dimY*3]);

        // return result into global memory
        JxB_B2[i + j*dev_p.nx + k*dev_p.nx*dev_p.ny] = norm(&JxBs[0 + p*3 + q*dimX*3 + r*dimX*dimY*3])/B2;
    }
}


// compute the norm of J.B/B**2
__global__ void JB_B2(REAL *B, REAL *J, REAL *JB_B2)
{
    int i, j;
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int l;
    REAL Bl[3], Jl[3], JB, B2;

    // the index for the most central location in the xy-plane
    i = dev_p.nx/2; j = dev_p.ny/2;

    // copy from global memory into local memory
    if (k < dev_p.nz) {
        for (l = 0; l < 3; l++) {
            Bl[l] = B[l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            Jl[l] = J[l + i*3 + j*dev_p.nx*3 + k*dev_p.nx*dev_p.ny*3];
        }

        JB = dot(Jl, Bl);
        B2 = dot(Bl, Bl);

        // return result into global memory
        JB_B2[k] = JB/B2;
    }
}


// compute the force-free parameter epsilon*
__global__ void epsilonStar(REAL *xb, REAL *JB_B2, REAL *epsStar, int dimZ)
{
    int i, j;
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int p = threadIdx.x;
    int iter;    // number of iterations through the global memory
    int m, n;
    REAL epsStarL[256];    // stores the differences in alph*

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *z      = s;
    REAL *JB_B2s = &z[dimZ*2];

    // the index for the most central location in the xy-plane
    i = dev_p.nx/2; j = dev_p.ny/2;

    iter = int(dev_p.nz-1)/dimZ + 1;

    // copy from global to shared memory
    if (k < dev_p.nz) {
        z[p]      = xb[2 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
        JB_B2s[p] = JB_B2[k];
    }

    for (m = 0; m < iter; m++) {
        if (k < dev_p.nz) {
            if (dimZ*m >= dev_p.nz) {
                z[p + dimZ]      = xb[2 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1+dimZ*m-dev_p.nz)*(dev_p.nx+2)*(dev_p.ny+2)*3];
                JB_B2s[p + dimZ] = JB_B2[k + dimZ*m - dev_p.nz];
            }
            else if (k + dimZ*m >= dev_p.nz) {
                z[p + dimZ]      = z[p];
                JB_B2s[p + dimZ] = JB_B2s[p];
            }
            else {
                z[p + dimZ]      = xb[2 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1+dimZ*m)*(dev_p.nx+2)*(dev_p.ny+2)*3];
                JB_B2s[p + dimZ] = JB_B2[k + dimZ*m];
            }
        }
        __syncthreads();

        // compute the differences
        if (k < dev_p.nz)
            for (n = 0; n < dimZ; n++) {
                if (z[n + dimZ] != z[p]) {
                    epsStarL[n] = SQRT2 * abs((JB_B2s[n + dimZ] - JB_B2s[p])/(z[n + dimZ] - z[p]));
                }
                else
                    epsStarL[n] = 0;
            }
        __syncthreads();

        if (k < dev_p.nz)
            for (n = 0; n < dimZ; n++)
                if ((n + blockDim.x*m) < dev_p.nz)
                    epsStar[n + blockDim.x*m + dev_p.nz*k] = epsStarL[n];
    }
}


// compute the error of B-ez
__global__ void B_1ez(REAL *B, REAL *B_1ez2)
{
    int m = threadIdx.x + blockDim.x * blockIdx.x;
    int p = threadIdx.x;
    int i = m%dev_p.nx;
    int k = m/(dev_p.nx*dev_p.ny);
    int j = m/(dev_p.nx) - k*dev_p.ny;
    int l;
    REAL sum;

    __shared__ REAL B_1ezS[3*256];
    __shared__ REAL B_1ez2S[256];

    if (m < (dev_p.nx*dev_p.ny*dev_p.nz)) {
        B_1ezS[0 + 3*p] = B[0 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
        B_1ezS[1 + 3*p] = B[1 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
        B_1ezS[2 + 3*p] = B[2 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]-1.;
    }
    else {
        B_1ezS[0 + 3*p] = 0;
        B_1ezS[1 + 3*p] = 0;
        B_1ezS[2 + 3*p] = 0;
    }

    B_1ez2S[p] = dot(&B_1ezS[0 + 3*p], &B_1ezS[0 + 3*p]);
    __syncthreads();

    // precompute partial sum
    if (p == 0) {
        sum = 0;
        for (l = 0; l < blockDim.x; l++)
            sum += B_1ez2S[l];
        B_1ez2[blockIdx.x] = sum;
    }
}


// compute the error of xb-xbAnalytical
__global__ void xb_XbAn(REAL *xb, REAL *xb_xbAn)
{
    int m = threadIdx.x + blockDim.x * blockIdx.x;
    int p = threadIdx.x;
    int i = m%dev_p.nx;
    int k = m/(dev_p.nx*dev_p.ny);
    int j = m/(dev_p.nx) - k*dev_p.ny;
    int l;
    REAL sum;
    REAL x, y, z;  // physical coordinates of the initial analytical field

    __shared__ REAL xb_xbAnS[3*256];
    __shared__ REAL xb_xbAn2S[256];

    // need to recompute the coordinates
    x = i*dev_p.dx + dev_p.Ox;
    y = j*dev_p.dy + dev_p.Oy;
    z = k*dev_p.dz + dev_p.Oz;

    if (m < (dev_p.nx*dev_p.ny*dev_p.nz)) {
        xb_xbAnS[0 + 3*p] = xb[0 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]
                            - cos(exp(-(x*x+y*y)/(dev_p.ar*dev_p.ar)-z*z/(dev_p.az*dev_p.az))*dev_p.phi1)*x
                               - sin(exp(-(x*x+y*y)/(dev_p.ar*dev_p.ar)-z*z/(dev_p.az*dev_p.az))*dev_p.phi1)*y;
        xb_xbAnS[1 + 3*p] = xb[1 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]
                            + sin(exp(-(x*x+y*y)/(dev_p.ar*dev_p.ar)-z*z/(dev_p.az*dev_p.az))*dev_p.phi1)*x
                            - cos(exp(-(x*x+y*y)/(dev_p.ar*dev_p.ar)-z*z/(dev_p.az*dev_p.az))*dev_p.phi1)*y;
        xb_xbAnS[2 + 3*p] = xb[2 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3] - z;
    }
    else {
        xb_xbAnS[0 + 3*p] = 0;
        xb_xbAnS[1 + 3*p] = 0;
        xb_xbAnS[2 + 3*p] = 0;
    }


    xb_xbAn2S[p] = dot(&xb_xbAnS[0 + 3*p], &xb_xbAnS[0 + 3*p]);
    __syncthreads();

    // precompute partial sum
    if (p == 0) {
        sum = 0;
        for (l = 0; l < blockDim.x; l++)
            sum += xb_xbAn2S[l];
        xb_xbAn[blockIdx.x] = sum;
    }
}


// global sum on the device
__global__ void sumBlock(REAL *var, REAL *sum, int size, int stride)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int p = threadIdx.x;
    int l;
    __shared__ REAL sumS[32];

    if (i*2 < size)
        if ((i*2+1) < size)
            sumS[p] = var[i*2] + var[(i*2+1)];
        else
            sumS[p] = var[i*2];
    else
        sumS[p] = 0;
    __syncthreads();

    for (l = 16; l > 0; l /= 2) {
        if (p < l)
            sumS[p] = sumS[p] + sumS[p+l];
        __syncthreads();
    }

    if (p == 0)
        sum[blockIdx.x*stride] = sumS[0];
}


// compute the sum of a device variable
REAL sumGlobal(REAL *dev_var, int size)
{
    cudaError_t     errCuda;    // error returned by device functions
    REAL            *dev_sum;
    REAL             sum;
    int                i;

    errCuda = cudaMalloc((void**)&dev_sum, ceil(size/64.)*sizeof(*dev_sum));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for dev_sum\n"); exit(EXIT_FAILURE); }

    sumBlock<<<ceil(size/64.), 32>>>(dev_var, dev_sum, size, 1);
    cudaDeviceSynchronize();
    size = ceil(size/64.);    // size of dev_sum
    while (size > 1) {
        sumBlock<<<ceil(size/64.), 32>>>(dev_sum, dev_sum, size, 64);
        cudaDeviceSynchronize();
        for (i = 0; i < ceil(size/64.); i++) {
            errCuda = cudaMemcpy(&dev_sum[i], &dev_sum[i*64], 1*sizeof(*dev_sum), cudaMemcpyDeviceToDevice);
            if (cudaSuccess != errCuda) { printf("error: could not copy 'sum' within the device\n"); exit(EXIT_FAILURE); }
        }
        cudaDeviceSynchronize();
        size = int(ceil(size/64.));    // new and reduced size
    }

    errCuda = cudaMemcpy(&sum, dev_sum, 1*sizeof(*dev_sum), cudaMemcpyDeviceToHost);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'sum' to the host\n"); exit(EXIT_FAILURE); }

    cudaFree(dev_sum);

    return sum;
}


// compute the directional cell volume
__global__ void dirCellVol(REAL *xb, REAL *cellVol, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l;

    REAL dx1[3], dx2[3], dx3[3];   // difference vectors to the neighbours
    REAL dx1Xdx2[3];               // cross product

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs      = s;                       // grid points
    REAL *cellVolS = &xbs[3*dimX*dimY*dimZ];  // directional cell volume

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+0)*3 + (q+0)*dimX*3         + (r+0)*dimX*dimY*3] =
            xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries; only one direction is needed
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1)))
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+0)*dimX*dimY*3] =
                xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1)))
                xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1)))
                xbs[l + (p+0)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            dx1[l] = xbs[l + (p+1)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3] - xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3];
            dx2[l] = xbs[l + (p+0)*3 + (q+1)*dimX*3 + (r+0)*dimX*dimY*3] - xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3];
            dx3[l] = xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+1)*dimX*dimY*3] - xbs[l + (p+0)*3 + (q+0)*dimX*3 + (r+0)*dimX*dimY*3];
        }
        cross(dx1, dx2, dx1Xdx2);
        cellVolS[p + q*(dimX-1) + r*(dimX-1)*(dimY-1)] = dot(dx1Xdx2, dx3);

        // write into global memory
       cellVol[i + j*dev_p.nx + k*dev_p.nx*dev_p.ny] = cellVolS[p + q*(dimX-1) + r*(dimX-1)*(dimY-1)];
    }
}


// compute the minimal cell volume
__global__ void gridWedgeMin(REAL *xb, REAL *wedge, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, u, v, w;

    REAL dx1[3], dx2[3], dx3[3];   // difference vectors to the neighbours
    REAL dx1Xdx2[3];               // cross product
    REAL wedgeCurrent, wedgeMin;   // wedge products

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs        = s;    // grid points

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0)
                xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1)))
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (q == 0)
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1)))
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (r == 0)
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1)))
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (u = 0; u <= 2; u += 2)
            for (v = 0; v <= 2; v += 2)
                for (w = 0; w <= 2; w += 2) {
                    for (l = 0; l < 3; l++) {
                        dx1[l] = xbs[l + (p+u)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] - xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
                        dx2[l] = xbs[l + (p+1)*3 + (q+v)*dimX*3 + (r+1)*dimX*dimY*3] - xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
                        dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+w)*dimX*dimY*3] - xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
                    }
                    cross(dx1, dx2, dx1Xdx2);
                    wedgeCurrent = dot(dx1Xdx2, dx3);

                    if (u+v+w == 0)
                        wedgeMin = -wedgeCurrent;

                    if ((u+v+w == 0) || (u+v+w == 4))
                        wedgeCurrent = -wedgeCurrent;

                    if (wedgeCurrent < wedgeMin)
                        wedgeMin = wedgeCurrent;
                }

        // copy back to global memory
        wedge[i + j*dev_p.nx + k*dev_p.nx*dev_p.ny] = wedgeMin;
    }
}


// compute the "convexity" of the cells
__global__ void gridConvexity(REAL *xb, REAL *convexity, int dimX, int dimY, int dimZ)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    int l, u, v, w;

    REAL dx1[3], dx2[3], dx3[3];   // difference vectors to the neighbours
    REAL dx1Xdx2[3];               // cross product
    REAL wedge;                       // wedge product

    // shared memory for faster communication, the size is assigned dynamically
    extern __shared__ REAL s[];
    REAL *xbs        = s;                       // grid cells
    REAL *convexityS = &xbs[3*dimX*dimY*dimZ];  // convexity of the grid point

    // copy from global memory to shared memory for faster computation
    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        for (l = 0; l < 3; l++) {
            // assign the inner values without the boundaries
            xbs[l + (p+1)*3 + (q+1)*dimX*3         + (r+1)*dimX*dimY*3] =
            xb[ l + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            // assign the block boundaries
            if (p == 0)
                xbs[l + (p+0)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+0)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((p == (blockDim.x-1)) && (i < (dev_p.nx-1))) || (i == (dev_p.nx-1)))
                xbs[l + (p+2)*3      + (q+1)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+2)*3      + (j+1)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (q == 0)
                xbs[l + (p+1)*3      + (q+0)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+0)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((q == (blockDim.y-1)) && (j < (dev_p.ny-1))) || (j == (dev_p.ny-1)))
                xbs[l + (p+1)*3      + (q+2)*dimX*3          + (r+1)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+2)*(dev_p.nx+2)*3  + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (r == 0)
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+0)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+0)*(dev_p.nx+2)*(dev_p.ny+2)*3];
            if (((r == (blockDim.z-1)) && (k < (dev_p.nz-1))) || (k == (dev_p.nz-1)))
                xbs[l + (p+1)*3      + (q+1)*dimX*3          + (r+2)*dimX*dimY*3] =
                xb[ l + (i+1)*3      + (j+1)*(dev_p.nx+2)*3  + (k+2)*(dev_p.nx+2)*(dev_p.ny+2)*3];
        }
    }
    __syncthreads();

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        convexityS[p + q*(dimX-2) + r*(dimX-2)*(dimY-2)] = 1;

        for (u = 0; u <= 2; u += 2)
            for (v = 0; v <= 2; v += 2)
                for (w = 0; w <= 2; w += 2) {
                    for (l = 0; l < 3; l++) {
                        dx1[l] = xbs[l + (p+u)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3] - xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
                        dx2[l] = xbs[l + (p+1)*3 + (q+v)*dimX*3 + (r+1)*dimX*dimY*3] - xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
                        dx3[l] = xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+w)*dimX*dimY*3] - xbs[l + (p+1)*3 + (q+1)*dimX*3 + (r+1)*dimX*dimY*3];
                    }
                    cross(dx1, dx2, dx1Xdx2);
                    wedge = dot(dx1Xdx2, dx3);
                    if ((wedge < 0) && ((u+v+w == 2) || (u+v+w == 6)))
                        convexityS[p + q*(dimX-2) + r*(dimX-2)*(dimY-2)] = -1;
                    if ((wedge > 0) && ((u+v+w == 0) || (u+v+w == 4)))
                        convexityS[p + q*(dimX-2) + r*(dimX-2)*(dimY-2)] = -1;
                }

        // copy back to global memory
        convexity[i + j*dev_p.nx + k*dev_p.nx*dev_p.ny] = convexityS[p + q*(dimX-2) + r*(dimX-2)*(dimY-2)];
    }
}


// compute the magnetic energy
__global__ void B2_det(REAL *B, REAL *detJac, REAL *B2_det)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    REAL B2_detL = 0;
    int bound;    // contains number of boundaries of this point

    __shared__ REAL sumS;

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        B2_detL = dot(&B[ 0 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3],
                      &B[ 0 + (i+1)*3 + (j+1)*(dev_p.nx+2)*3 + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)*3]) *
                  detJac[     (i+1)   + (j+1)*(dev_p.nx+2)   + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)];
        // account faces, edges and corners to only 1/2, 1/4 and 1/8
        bound = (i == 0) + (j == 0) + (k == 0) + (i == (dev_p.nx-1)) + (j == (dev_p.ny-1)) + (k == (dev_p.nz-1));
        bound = 1 << bound;
        B2_detL /= bound;
    }
    else
        B2_detL = 0;

    // add up values
    if (p+q+r == 0)
        sumS = 0;
    __syncthreads();
    atomicAdd(&sumS, B2_detL);
    __syncthreads();
    if (p+q+r == 0)
        B2_det[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y] = sumS;
}


// compute the kinetic energy
__global__ void U2_det(REAL *uu, REAL *detJac, REAL *U2_det)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int p = threadIdx.x;
    int q = threadIdx.y;
    int r = threadIdx.z;
    REAL U2_detL = 0;
    int bound;    // contains number of boundaries of this point

    __shared__ REAL sumS;

    if ((i < dev_p.nx) && (j < dev_p.ny) && (k < dev_p.nz)) {
        U2_detL = dot(&uu[ 0 + i*3   + j*dev_p.nx*3       + k*dev_p.nx*dev_p.ny*3],
                      &uu[ 0 + i*3   + j*dev_p.nx*3       + k*dev_p.nx*dev_p.ny*3]) *
                  detJac[      (i+1) + (j+1)*(dev_p.nx+2) + (k+1)*(dev_p.nx+2)*(dev_p.ny+2)];
        // account faces, edges and corners to only 1/2, 1/4 and 1/8
        bound = (i == 0) + (j == 0) + (k == 0) + (i == (dev_p.nx-1)) + (j == (dev_p.ny-1)) + (k == (dev_p.nz-1));
        bound = 1 << bound;
        U2_detL /= bound;
    }
    else
        U2_detL = 0;

    // add up values
    if (p+q+r == 0)
        sumS = 0;
    __syncthreads();
    atomicAdd(&sumS, U2_detL);
    __syncthreads();
    if (p+q+r == 0)
        U2_det[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y] = sumS;
}
