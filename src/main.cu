/*
 GLEMuR Gpu Lagrangian mimEtic Magnetic Relaxation

 Solves the magneto frictional equations for a relaxation problem in 3D
 as described by [1] or equations including inertia and velocity damping.
 Makes use of the mimetic approach for computing the current
 as described in [1,2,3].
 Uses standard ODE solvers for the time stepping.
 Published in SIAM J. Sci. Comput., 36:952 (2014).

 References:
 [1] Pontin et al., ApJ 700:1449 (2009)
 [2] Douglas and Gunn, Num. Math. 6:428 (1964)
 [3] Comput. Math. Appl., 33:81 (1997)
 [4] Craig I. J. and Sneyd A. D., ApJ, 311:451 (1986)
 */

#include "main.h"
#include "io.h"
#include "init.h"
#include "memory.h"
#include "diagnostics.h"
#include "maths.h"
#include "current.h"
#include "gradP.h"
#include "rungeKutta.h"
#include "boundaries.h"

// Make fortran read namelist routine available.
extern "C" {
extern void readnamelist(struct Parameters *params);
}

__constant__ struct Parameters dev_params; // simulation parameters on the device
__constant__ REAL   a_i[6];                // coefficients a_i for the adaptive step size Runge-Kutta
__constant__ REAL   b_ij[7*6];             // coefficients b_ij for the adaptive step size Runge-Kutta
__constant__ REAL   c_i[6];                // coefficients c_i for the adaptive step size Runge-Kutta
__constant__ REAL   cs_i[6];               // coefficients c*_i for the adaptive step size Runge-Kutta

int      endian;             // tells the code whether machine uses big or little endian
int      one = 1;            // used for determining the endianness
char     *pEndian = (char *)&one;

int main(int argc, char* argv[]) {
    printf("GLEMuR\n\n");
    printf("PRECISION = %i\n", PRECISION);

    // How many CUDA devices are on this machine?
    errCuda = cudaGetDeviceCount(&deviceCount);
    if (cudaSuccess != errCuda) {
        printf("error: could not locate CUDA devices\n");
        return -1;
    }
    if (deviceCount == 0) {
        printf("error: there are no CUDA compatible devices on this machine\n");
        return -1;
    }
    printf("CUDA device count = %i\n", deviceCount);
    activeDevice = 0;

    // Parse the command line arguments.
    while ((parsedCom = getopt(argc, argv, "d:")) != -1) {
        switch (parsedCom) {
        case 'd':
            activeDevice = strtol(optarg, NULL, 10);
            if ((activeDevice == 0) and (optarg[0] != '0')) {
                printf("error: device number is not integer\n");
                return -1;
            }
            if ((activeDevice > deviceCount - 1) or (activeDevice < 0)) {
                printf("error: device number is not valid\n");
                return -1;
            }
            break;
        }
    }
    cudaSetDevice (activeDevice);
    printf("running on device %li\n", activeDevice);

    // Write CUDA device information into file.
    err = writeCudaInfo(deviceCount, activeDevice);
    if (err != 0) {
        printf("error: could not write device properties into cuda.info\n");
        return -1;
    }

    // Determine the endianness on this machine.
    if (pEndian[0] == 1) {
        printf("endian = little\n");
        endian = LITTLE_ENDIAN;
    } else {
        printf("endian = big\n");
        endian = BIG_ENDIAN;
    }

    // Increase the bank size for shared memory in case of double precision.
    if (PRECISION == 64) {
        printf("bank size = 8 bytes\n");
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    } else {
        printf("bank size = 4 bytes\n");
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    }

    // Read parameters from the input file.
    readnamelist (&params);

    // Check if the finite difference method is valid.
    if ((strncmp(params.jMethod, "Classic ", 8) != 0)
            && (strncmp(params.jMethod, "Stokes ", 7) != 0)
            && (strncmp(params.jMethod, "Stokes4th ", 10) != 0)
            && (strncmp(params.jMethod, "StokesQuint ", 12) != 0)
            && (strncmp(params.jMethod, "StokesTri ", 10) != 0)) {
        printf("error: invalid method for computing J: %s\n", params.jMethod);
        return -1;
    } else
        printf("method for J: %s\n", params.jMethod);

    printf("initial B: %s\n", params.bInit);
    if (params.inertia == true)
        printf("initial U: %s\n", params.uInit);
    if (params.pressure == true)
        printf("include pressure gradient with beta = %f\n", params.beta);

    if (params.fRestart == true)
        readGrid(params);

    // Figure out how to paralallelize.
    // Optimized block size for shared memory usage distinguishing single and double precision and inertia equation.
    blockSize[0] = 5 + BLCK_EXT;
    blockSize[1] = 5 + BLCK_EXT + not (params.inertia);
    blockSize[2] = 5 + BLCK_EXT + not (params.inertia);
    if (params.inertia)
        blockSize[2] = 4 + BLCK_EXT; // needed to limit the amount of shared memory usage to 48 kB for double precision
    gridSize[0] = (params.nx - 1) / blockSize[0] + 1;
    gridSize[1] = (params.ny - 1) / blockSize[1] + 1;
    gridSize[2] = (params.nz - 1) / blockSize[2] + 1;
    dim3 dimBlock(blockSize[0], blockSize[1], blockSize[2]);
    dim3 dimGrid(gridSize[0], gridSize[1], gridSize[2]);
    printf("blockSize[0], blockSize[1], blockSize[2] = %i, %i, %i\n",
            blockSize[0], blockSize[1], blockSize[2]);
    printf("gridSize[0],  gridSize[1],  gridSize[2]  = %i, %i, %i\n",
            gridSize[0], gridSize[1], gridSize[2]);
    // Grid size for 2d operations, e.g. boundary settings.
    gridSize2[0] = (params.nx + 1) / 16 + 1;
    gridSize2[1] = (params.ny + 1) / 16 + 1;
    gridSize2[2] = (params.nz + 1) / 16 + 1;
    dim3 dimGrid2dPlusXY(gridSize2[0], gridSize2[1]);
    dim3 dimGrid2dPlusXZ(gridSize2[0], gridSize2[2]);
    dim3 dimGrid2dPlusYZ(gridSize2[1], gridSize2[2]);
    dim3 dimBlock2d(16, 16); // should be the square root of the maximum threads per block
    gridSize2[0] = (params.nx - 1) / 16 + 1;
    gridSize2[1] = (params.ny - 1) / 16 + 1;
    gridSize2[2] = (params.nz - 1) / 16 + 1;
    dim3 dimGrid2dXY(gridSize2[0], gridSize2[1]);
    dim3 dimGrid2dXZ(gridSize2[0], gridSize2[2]);
    dim3 dimGrid2dYZ(gridSize2[1], gridSize2[2]);

    // Allocate host and device memory.
    allocateMemory(&h, &d, params, gridSize);

    // Initialize or load the simulation state.
    initResiduals(params, &red);
    if (params.fRestart == false) {
        remove("data/save.vtk");
        initState(h, params, &red);
        initDistortion(h.xb, params);
        setPeriHost(h.B0, params);
        setGridPeriHost(h.xb, params);
        writeB0(h.B0, params);
        dt = params.dt0;
        t = 0;
        writeTs(params, red, t, dt, 0, 0.);
    } else {
        readState(h, params, &t, &dt);
        setPeriHost(h.B0, params);
        setGridPeriHost(h.xb, params);
    }

    // Copy current state into device memory.
    errCuda = cudaMemcpy(d.B0, h.B0,
            3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.B0)),
            cudaMemcpyHostToDevice);
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'B0' to the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpy(d.xb, h.xb,
            3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.xb)),
            cudaMemcpyHostToDevice);
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'xb' to the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpy(d.B, d.B0,
            3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.B)),
            cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'B0' within the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpy(d.xb_tmp, d.xb,
            3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.xb_tmp)),
            cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'xb' within the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpy(d.xb_new, d.xb,
            3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.xb_new)),
            cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'xb' within the device\n");
        exit (EXIT_FAILURE);
    }
    if (params.inertia == true) {
        errCuda = cudaMemcpy(d.uu, h.uu,
                3 * params.nx * params.ny * params.nz * sizeof(*(d.uu)),
                cudaMemcpyHostToDevice);
        if (cudaSuccess != errCuda) {
            printf("error: could not copy 'uu' to the device\n");
            exit (EXIT_FAILURE);
        }
    }

    // Copy the simulation parameters into constant memory.
    errCuda = cudaMemcpyToSymbol(dev_params, &params, sizeof(struct Parameters));
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'params' to the device\n");
        exit (EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    // Assign coefficients.
    assignCoefficients(host_a_i, host_b_ij, host_c_i, host_cs_i);
    errCuda = cudaMemcpyToSymbol(a_i, host_a_i, 6 * sizeof(*a_i));
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'a_i' to the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpyToSymbol(b_ij, host_b_ij, 7 * 6 * sizeof(*b_ij));
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'b_ij' to the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpyToSymbol(c_i, host_c_i, 6 * sizeof(*c_i));
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'c_i' to the device\n");
        exit (EXIT_FAILURE);
    }
    errCuda = cudaMemcpyToSymbol(cs_i, host_cs_i, 6 * sizeof(*cs_i));
    if (cudaSuccess != errCuda) {
        printf("error: could not copy 'cs_i' to the device\n");
        exit (EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    // Perform the time stepping.
    for (timeIndex = 0; (timeIndex < params.nCycle) && (dt > params.dtMin); timeIndex++) {
        for (n = 0; n < 6; n++) {
            // B = jac.B0/Delta
            B_JacB0
            <<<dimGrid, dimBlock, 3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*3*sizeof(*(d.xb))>>>
            (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
            cudaDeviceSynchronize();
            setBbound(dimGrid2dXY, dimGrid2dXZ, dimGrid2dYZ, dimBlock2d, d);
            setPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ,
                    dimBlock2d, d.B, params);
            cudaDeviceSynchronize();

            current(dimGrid, dimBlock, blockSize, d, params);
            cudaDeviceSynchronize();
            if (params.pressure == true)
                gradP(dimGrid, dimBlock, blockSize, d, params);
            cudaDeviceSynchronize();
            
            // Intermediate steps for the Runge-Kutta method.
            kk
            <<<dimGrid, dimBlock, blockSize[0]*blockSize[1]*blockSize[2]*(3*(9+7*params.inertia+params.pressure)+params.inertia) * sizeof(*(d.xb))>>>
            (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2, n, dt);
            cudaDeviceSynchronize();
            setGridPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ,
                    dimBlock2d, d.xb, params);
            setGridPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ,
                    dimBlock2d, d.xb_tmp, params);
            cudaDeviceSynchronize();

            errCuda = cudaMemcpy(d.xb, d.xb_tmp,
                    3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.xb)),
                    cudaMemcpyDeviceToDevice);
            if (cudaSuccess != errCuda) {
                printf("error: could not copy 'xb' within the device during RK step, "
                       "error = %s\n", cudaGetErrorString(errCuda));
                exit (EXIT_FAILURE);
            }
            if (params.inertia == true) {
                errCuda = cudaMemcpy(d.uu, d.uu_tmp,
                        3 * params.nx * params.ny * params.nz * sizeof(*(d.uu)),
                        cudaMemcpyDeviceToDevice);
                if (cudaSuccess != errCuda) {
                    printf("error: could not copy 'uu' within the device during RK step, "
                           "error = %s\n", cudaGetErrorString(errCuda));
                    exit (EXIT_FAILURE);
                }
            }
            cudaDeviceSynchronize();
        }
        // Compute the next full step.
        xNewStar
        <<<dimGrid, dimBlock, (3*(10+9*params.inertia)*blockSize[0]*blockSize[1]*blockSize[2])* sizeof(*(d.xb))>>>
        (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
        cudaDeviceSynchronize();
        setGridPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ,
                dimBlock2d, d.xb_new, params);
        cudaDeviceSynchronize();

        // Maximum error in the domain.
        maxDelta = findDevMax(d.maxDelta,
                gridSize[0] * gridSize[1] * gridSize[2]);

        if (maxDelta < params.maxError) {
            // Progress time.
            t += dt;
            errCuda = cudaMemcpy(d.xb, d.xb_new,
                    3 * (params.nx + 2) * (params.ny + 2) * (params.nz + 2) * sizeof(*(d.xb)),
                    cudaMemcpyDeviceToDevice);
            if (cudaSuccess != errCuda) {
                printf("error: could not copy 'xb' within the device\n");
                exit (EXIT_FAILURE);
            }
            if (params.inertia == true) {
                errCuda = cudaMemcpy(d.uu, d.uu_new,
                        3 * params.nx * params.ny * params.nz * sizeof(*(d.uu)),
                        cudaMemcpyDeviceToDevice);
                if (cudaSuccess != errCuda) {
                    printf("error: could not copy 'uu' within the device\n");
                    exit (EXIT_FAILURE);
                }
            }
            cudaDeviceSynchronize();

            // Compute the current magnetic field and electric current density from xb and B0.
            B_JacB0
            <<<dimGrid, dimBlock, 3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*3 * sizeof(*(d.xb))>>>
            (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
            cudaDeviceSynchronize();
            setBbound(dimGrid2dXY, dimGrid2dXZ, dimGrid2dYZ, dimBlock2d, d);
            setPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ,
                    dimBlock2d, d.B, params);
            cudaDeviceSynchronize();

            current(dimGrid, dimBlock, blockSize, d, params);
            cudaDeviceSynchronize();

            // Write diagnosis on the screen and into time series file.
            if (timeIndex % params.nTs == 0) {
                reductions(&red, d, params, blockSize, gridSize, dimGrid, dimBlock);
                writeTs(params, red, t, dt, timeIndex, maxDelta);
            }

            // Adapt the time step.
            if (maxDelta > 0)
                dt = dt * pow(0.9 * params.maxError / maxDelta, REAL(0.2));

            // Write out the state file.
            if ((ceil((t - dt) / params.dtDump) != ceil(t / params.dtDump))
                    || (ceil((t - dt) / params.dtSave) != ceil(t / params.dtSave))
                    || (timeIndex == (params.nCycle - 1))) {
                prepareDump(red, h, d, params, blockSize, gridSize, dimGrid,
                        dimBlock, dimGrid2dPlusXY, dimGrid2dPlusXZ,
                        dimGrid2dPlusYZ, dimBlock2d);
                if (ceil((t - dt) / params.dtDump) != ceil(t / params.dtDump))
                    dumpState(h, params, t, dt, ceil((t - dt) / params.dtDump));
                if ((ceil((t - dt) / params.dtSave) != ceil(t / params.dtSave))
                        || (timeIndex == (params.nCycle - 1)))
                    dumpState(h, params, t, dt, -1);
            }
        } else {
            // Adapt the time step.
            timeIndex -= 1;
            if (maxDelta > 0)
                dt = dt * pow(0.9 * params.maxError / maxDelta, REAL(0.2));
        }
    }

    if (dt < params.dtMin)
        printf("dt < dtMin = %e, stopping simulation\n", params.dtMin);

    freeMemory(&h, &d, params);

    printf("glemur terminated successfully\n");
    
    return 0;
}
