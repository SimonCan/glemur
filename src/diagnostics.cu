// diagnostics.cu
//
// Further calculations with the data and data reduction.
//

#include "global.h"
#include "maths.h"
#include "boundaries.h"
#include "io.h"

// Perform data reduction.
int reductions(struct Reduction *red, struct VarsDev d, struct Parameters p, int blockSize[3], int gridSize[3], dim3 dimGrid, dim3 dimBlock) {
    cudaError_t	errCuda;	// error returned by device functions

    if (p.redJMax == true) {
        errCuda = cudaMalloc((void**)&(d.Jmag), p.nx*p.ny*p.nz*sizeof(*(d.Jmag)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.Jmag\n"); exit(EXIT_FAILURE); }
        mag<<<dimGrid, dimBlock>>>(d.J, d.Jmag);
        cudaDeviceSynchronize();
        red->JMax = findDevMax(d.Jmag, p.nx*p.ny*p.nz);
        cudaFree(d.Jmag);
    }
    if (p.redJxB_B2Max == true) {
        errCuda = cudaMalloc((void**)&(d.JxB_B2), p.nx*p.ny*p.nz*sizeof(*(d.JxB_B2)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.JxB_B2\n"); exit(EXIT_FAILURE); }
        JxB_B2
            <<<dimGrid, dimBlock, 3*blockSize[0]*blockSize[1]*blockSize[2]*3 * sizeof(*(d.B))>>>
            (d.B, d.J, d.JxB_B2, blockSize[0], blockSize[1], blockSize[2]);
        cudaDeviceSynchronize();
        red->JxB_B2Max = findDevMax(d.JxB_B2, p.nx*p.ny*p.nz);
        cudaFree(d.JxB_B2);
    }
    if (p.redEpsilonStar == true) {
        // TODO: change 256 to maxThreads per block for any CUDA device
        errCuda = cudaMalloc((void**)&(d.JB_B2), p.nz*sizeof(*(d.JB_B2)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.JB_B2\n"); exit(EXIT_FAILURE); }
        errCuda = cudaMalloc((void**)&(d.epsStar), p.nz*p.nz*sizeof(*(d.epsStar)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.epsStar\n"); exit(EXIT_FAILURE); }
        JB_B2
            <<<int(p.nz-1)/256+1, 256>>>
            (d.B, d.J, d.JB_B2);
        cudaDeviceSynchronize();
        epsilonStar
            <<<int(p.nz-1)/256+1, 256, 512*2 * sizeof(*(d.xb))>>>
            (d.xb, d.JB_B2, d.epsStar, 256);
        cudaDeviceSynchronize();
        red->epsilonStar = findDevMax(d.epsStar, p.nz*p.nz);
        cudaFree(d.JB_B2);
        cudaFree(d.epsStar);
    }
    if (p.redErrB_1ez == true) {
        // TODO: change 256 to maxThreads per block for any CUDA device
        errCuda = cudaMalloc((void**)&(d.B_1ez2), (int(p.nx*p.ny*p.nz-1)/256+1)*sizeof(*(d.B_1ez2)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.B_1ez2\n"); exit(EXIT_FAILURE); }
        B_1ez
            <<<int(p.nx*p.ny*p.nz-1)/256+1, 256>>>
            (d.B, d.B_1ez2);
        cudaDeviceSynchronize();
        red->errB_1ez = sqrt(sumGlobal(d.B_1ez2, int(p.nx*p.ny*p.nz-1)/256+1))/sqrt(p.nx*p.ny*p.nz);
        cudaFree(d.B_1ez2);
    }
    if (p.redErrXb_XbAn == true) {
        // TODO: change 256 to maxThreads per block for any CUDA device
        errCuda = cudaMalloc((void**)&(d.xb_xbAn), (int(p.nx*p.ny*p.nz-1)/256+1)*sizeof(*(d.xb_xbAn)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.xb_xbAn\n"); exit(EXIT_FAILURE); }
        xb_XbAn
            <<<int(p.nx*p.ny*p.nz-1)/256+1, 256>>>
            (d.xb, d.xb_xbAn);
        cudaDeviceSynchronize();
        red->errXb_XbAn = sqrt(sumGlobal(d.xb_xbAn, int(p.nx*p.ny*p.nz-1)/256+1))/sqrt(p.nx*p.ny*p.nz);
        cudaFree(d.xb_xbAn);
    }
    if ((p.redB2 == true) || (p.redB2f == true)) {
        errCuda = cudaMalloc((void**)&(d.B2_det), gridSize[0]*gridSize[1]*gridSize[2]*sizeof(*(d.B2_det)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.B2_det\n"); exit(EXIT_FAILURE); }
        B2_det
            <<<dimGrid, dimBlock>>>
            (d.B, d.detJac, d.B2_det);
        cudaDeviceSynchronize();
        red->B2 = sumGlobal(d.B2_det, gridSize[0]*gridSize[1]*gridSize[2]) * p.dx*p.dy*p.dz;
        cudaFree(d.B2_det);
    }
    if (p.redConvex == true) {
        errCuda = cudaMalloc((void**)&(d.convexity), p.nx*p.ny*p.nz*sizeof(*(d.convexity)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.convexity\n"); exit(EXIT_FAILURE); }
        gridConvexity
            <<<dimGrid, dimBlock, (3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2) +
                                   blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*(d.xb))>>>
            (d.xb, d.convexity, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
        cudaDeviceSynchronize();
        // Find the minimum.
        invert<<<int(p.nx*p.ny*p.nz-1)/256+1, 256>>>(d.convexity, int(p.nx*p.ny*p.nz));
        cudaDeviceSynchronize();
        red->convex = -findDevMax(d.convexity, p.nx*p.ny*p.nz);
        invert<<<int(p.nx*p.ny*p.nz-1)/256+1, 256>>>(d.convexity, int(p.nx*p.ny*p.nz));
        cudaDeviceSynchronize();
        cudaFree(d.convexity);
    }
    if (p.redWedgeMin == true) {
        errCuda = cudaMalloc((void**)&(d.wedgeMin), p.nx*p.ny*p.nz*sizeof(*(d.wedgeMin)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.wedgeMin\n"); exit(EXIT_FAILURE); }
        gridWedgeMin
            <<<dimGrid, dimBlock, 3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*sizeof(*(d.xb))>>>
                (d.xb, d.wedgeMin, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
        cudaDeviceSynchronize();
        // Find the minimum.
        invert<<<int(p.nx*p.ny*p.nz-1)/256+1, 256>>>(d.wedgeMin, int(p.nx*p.ny*p.nz));
        cudaDeviceSynchronize();
        red->wedgeMin = -findDevMax(d.wedgeMin, p.nx*p.ny*p.nz);
        invert<<<int(p.nx*p.ny*p.nz-1)/256+1, 256>>>(d.wedgeMin, int(p.nx*p.ny*p.nz));
        cudaDeviceSynchronize();
        cudaFree(d.wedgeMin);
    }
    if ((p.redU2 == true) && (p.inertia == true)) {
        errCuda = cudaMalloc((void**)&(d.U2_det), gridSize[0]*gridSize[1]*gridSize[2]*sizeof(*(d.U2_det)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.U2_det\n"); exit(EXIT_FAILURE); }
        U2_det
            <<<dimGrid, dimBlock>>>
            (d.uu, d.detJac, d.U2_det);
        cudaDeviceSynchronize();
        red->U2 = sumGlobal(d.U2_det, gridSize[0]*gridSize[1]*gridSize[2]) * p.dx*p.dy*p.dz;
        cudaFree(d.U2_det);
    }
    if ((p.redUMax == true) && (p.inertia == true)) {
        errCuda = cudaMalloc((void**)&(d.Umag), p.nx*p.ny*p.nz*sizeof(*(d.Umag)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.Umag\n"); exit(EXIT_FAILURE); }
        mag<<<dimGrid, dimBlock>>>(d.uu, d.Umag);
        cudaDeviceSynchronize();
        red->UMax = findDevMax(d.Umag, p.nx*p.ny*p.nz);
        cudaFree(d.Umag);
    }

    return 0;
}


// Prepare the dump from the current state.
int prepareDump(struct Reduction red, struct VarsHost h, struct VarsDev d, struct Parameters p, int blockSize[3], int gridSize[3], dim3 dimGrid, dim3 dimBlock,
        dim3 dimGrid2dPlusXY, dim3 dimGrid2dPlusXZ, dim3 dimGrid2dPlusYZ, dim3 dimBlock2d) {
    cudaError_t         errCuda;          // error returned by device functions

    setPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d, d.B, p);
    cudaDeviceSynchronize();

    errCuda = cudaMemcpy(h.B, d.B, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(h.B)), cudaMemcpyDeviceToHost);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'B' to the host\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(h.xb, d.xb, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(h.xb)), cudaMemcpyDeviceToHost);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'xb' to the host\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(h.JJ, d.J, 3*p.nx*p.ny*p.nz*sizeof(*(h.JJ)), cudaMemcpyDeviceToHost);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'JJ' to the host\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(h.detJac, d.detJac, (p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(h.detJac)), cudaMemcpyDeviceToHost);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'detJac' to the host\n"); exit(EXIT_FAILURE); }
    // Compute the sigend cell volume.
    if (p.dumpCellVol == true) {
        errCuda = cudaMalloc((void**)&(d.cellVol), p.nx*p.ny*p.nz*sizeof(*(d.cellVol)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.cellVol\n"); exit(EXIT_FAILURE); }
        dirCellVol
            <<<dimGrid, dimBlock, (3*(blockSize[0]+1)*(blockSize[1]+1)*(blockSize[2]+1) +
                                   blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*(d.xb))>>>
            (d.xb, d.cellVol, blockSize[0]+1, blockSize[1]+1, blockSize[2]+1);
        cudaDeviceSynchronize();
        errCuda = cudaMemcpy(h.cellVol, d.cellVol, p.nx*p.ny*p.nz*sizeof(*(h.cellVol)), cudaMemcpyDeviceToHost);
        if (cudaSuccess != errCuda) { printf("error: could not copy 'cellVol' to the host\n"); exit(EXIT_FAILURE); }
        cudaFree(d.cellVol);
    }
    if (p.dumpConvexity == true) {
        errCuda = cudaMalloc((void**)&(d.convexity), p.nx*p.ny*p.nz*sizeof(*(d.convexity)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.convexity\n"); exit(EXIT_FAILURE); }
        gridConvexity
            <<<dimGrid, dimBlock, (3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2) +
                                   blockSize[0]*blockSize[1]*blockSize[2])*sizeof(*d.xb)>>>
            (d.xb, d.convexity, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
        cudaDeviceSynchronize();
        errCuda = cudaMemcpy(h.convexity, d.convexity, p.nx*p.ny*p.nz*sizeof(*(h.convexity)), cudaMemcpyDeviceToHost);
        if (cudaSuccess != errCuda) { printf("error: could not copy 'convexity' to the host\n"); exit(EXIT_FAILURE); }
        cudaFree(d.convexity);
    }
    if (p.dumpWedgeMin == true) {
        errCuda = cudaMalloc((void**)&(d.wedgeMin), p.nx*p.ny*p.nz*sizeof(*(d.wedgeMin)));
        if (cudaSuccess != errCuda) { printf("error: could not allocate device memory for d.wedgeMin\n"); exit(EXIT_FAILURE); }
        gridWedgeMin
            <<<dimGrid, dimBlock, 3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*sizeof(*(d.xb))>>>
                (d.xb, d.wedgeMin, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
        cudaDeviceSynchronize();
        errCuda = cudaMemcpy(h.wedgeMin, d.wedgeMin, p.nx*p.ny*p.nz*sizeof(*(d.wedgeMin)), cudaMemcpyDeviceToHost);
        if (cudaSuccess != errCuda) { printf("error: could not copy 'wedgeMin' to the host\n"); exit(EXIT_FAILURE); }
        cudaFree(d.wedgeMin);
    }
    if (p.inertia == true) {
        errCuda = cudaMemcpy(h.uu, d.uu, 3*p.nx*p.ny*p.nz*sizeof(*(h.uu)), cudaMemcpyDeviceToHost);
        if (cudaSuccess != errCuda) { printf("error: could not copy 'U' to the host\n"); exit(EXIT_FAILURE); }
    }

    return 0;
}
