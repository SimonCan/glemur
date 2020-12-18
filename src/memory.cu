// memory.cu
//
// Memory allocation and freeing routines for host and device memory.
//

#include "memory.h"
#include "global.h"

// Allocate host and device memory.
int allocateMemory(struct VarsHost *h, struct VarsDev *d, struct Parameters params, int gridSize[3]) {
    cudaError_t	errCuda;	// error returned by device functions

    // Allocate host memory.
    h->B0 = (REAL *)malloc(3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(h->B0)));
    if (h->B0 == NULL) { printf("error: could not allocate memory for B0\n"); return -1; }
    h->B = (REAL *)malloc(3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(h->B)));
    if (h->B == NULL) { printf("error: could not allocate memory for B\n"); return -1; }
    h->xb = (REAL *)malloc(3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(h->xb)));
    if (h->xb == NULL) { printf("error: could not allocate memory for xb\n"); return -1; }
    h->JJ = (REAL *)malloc(3*params.nx*params.ny*params.nz*sizeof(*(h->JJ)));
    if (h->JJ == NULL) { printf("error: could not allocate memory for JJ\n"); return -1; }
    h->detJac = (REAL *)malloc((params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(h->detJac)));
    if (h->detJac == NULL) { printf("error: could not allocate memory for detJac\n"); return -1; }
    if (params.dumpCellVol == 1) {
        h->cellVol = (REAL *)malloc(params.nx*params.ny*params.nz*sizeof(*(h->cellVol)));
        if (h->cellVol == NULL) { printf("error: could not allocate memory for cellVol\n"); return -1; } }
    if ((params.dumpConvexity == 1) || (params.redConvex == 1)) {
        h->convexity = (REAL *)malloc(params.nx*params.ny*params.nz*sizeof(*(h->convexity)));
        if (h->cellVol == NULL) { printf("error: could not allocate memory for convexity\n"); return -1; } }
    if ((params.dumpWedgeMin == 1) || (params.redWedgeMin == 1)) {
        h->wedgeMin = (REAL *)malloc(params.nx*params.ny*params.nz*sizeof(*(h->wedgeMin)));
        if (h->wedgeMin == NULL) { printf("error: could not allocate memory for wedgeMin\n"); return -1; } }
    if (params.inertia == true) {
        h->uu = (REAL *)malloc(3*params.nx*params.ny*params.nz*sizeof(*(h->uu)));
        if (h->uu == NULL) { printf("error: could not allocate memory for uu\n"); return -1; } }

    // Allocate global memory on GPU device.
    errCuda = cudaMalloc((void**)&(d->B0), 3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(d->B0)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for B0\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&(d->B), 3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(d->B)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for B\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&(d->xb), 3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(d->xb)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for xb\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&(d->J), 3*params.nx*params.ny*params.nz*sizeof(*(d->J)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for J\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&(d->xb_new), 3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(d->xb_new)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for xb_new\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&(d->xb_tmp), 3*(params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(d->xb_tmp)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for xb_tmp\n"); exit(EXIT_FAILURE); }
    if (params.inertia == false) {
        errCuda = cudaMalloc((void**)&(d->kk), 3*params.nx*params.ny*params.nz*6*sizeof(*(d->kk)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for kk\n"); exit(EXIT_FAILURE); } }
    else {
        errCuda = cudaMalloc((void**)&(d->kk), 6*params.nx*params.ny*params.nz*6*sizeof(*(d->kk)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for kk\n"); exit(EXIT_FAILURE); } }
    if (params.pressure == true) {
        errCuda = cudaMalloc((void**)&(d->gradP), 3*params.nx*params.ny*params.nz*sizeof(*(d->gradP)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for gradP\n"); exit(EXIT_FAILURE); } }
    errCuda = cudaMalloc((void**)&(d->maxDelta), gridSize[0]*gridSize[1]*gridSize[2]*sizeof(*(d->maxDelta)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for maxDelta\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMalloc((void**)&(d->detJac), (params.nx+2)*(params.ny+2)*(params.nz+2)*sizeof(*(d->detJac)));
    if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for detJac\n"); exit(EXIT_FAILURE); }
    if (params.inertia == true) {
        errCuda = cudaMalloc((void**)&(d->uu), 3*params.nx*params.ny*params.nz*sizeof(*(d->uu)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for uu\n"); exit(EXIT_FAILURE); }
        errCuda = cudaMalloc((void**)&(d->uu_new), 3*params.nx*params.ny*params.nz*sizeof(*(d->uu_new)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for uu_new\n"); exit(EXIT_FAILURE); }
        errCuda = cudaMalloc((void**)&(d->uu_tmp), 3*params.nx*params.ny*params.nz*sizeof(*(d->uu_tmp)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for uu_tmp\n"); exit(EXIT_FAILURE); }
    }

    return 0;
}


// Free host and device memory.
int freeMemory(struct VarsHost *h, struct VarsDev *d, struct Parameters params) {
    free(h->B0); free(h->B); free(h->JJ); free(h->xb); free(h->detJac);
    cudaFree(d->kk); cudaFree(d->xb_new); cudaFree(d->maxDelta); cudaFree(d->xb_tmp);
    cudaFree(d->detJac);
    cudaFree(d->xb); cudaFree(d->B0); cudaFree(d->B); cudaFree(d->J);
    if (params.dumpCellVol == 1)
        free(h->cellVol);
    if ((params.dumpConvexity == 1) || (params.redConvex == 1))
        free(h->convexity);
    if ((params.dumpWedgeMin == 1) || (params.redWedgeMin == 1))
        free(h->wedgeMin);
    if (params.inertia == true) {
        free(h->uu);
        cudaFree(d->uu); cudaFree(d->uu_new); cudaFree(d->uu_tmp);
    }

    return 0;
}

