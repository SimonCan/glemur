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

// make fortran read namelist routine available
extern "C" { extern void readnamelist(struct parameters_t *p); }

// some globally defined variables
int  endian;    // tells the code whether machine uses big or little endian
int  one = 1;   // used for determining the endianness
char *pEndian = (char *)&one;

__constant__ struct parameters_t dev_p; // simulation parameters on the device
__constant__ REAL   a_i[6];             // coefficients a_i for the adaptive step size RG
__constant__ REAL   b_ij[7*6];          // coefficients b_ij for the adaptive step size RG
__constant__ REAL   c_i[6];             // coefficients c_i for the adaptive step size RG
__constant__ REAL   c_star_i[6];        // coefficients c*_i for the adaptive step size RG

// assign values to the coefficients for the adaptive time step RG
void assignCoefficients(REAL a[5], REAL b[7][6], REAL c[6], REAL cs[6])
{
    // not used for the moment, since no explicit time dependence
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


int main(int argc, char* argv[])
{
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
    // parse the command line parameters
    while ((c = getopt(argc, argv, "d:")) != -1) {
    	switch(c) {
    		case 'd':
    			activeDevice = strtol(optarg, NULL, 10);
        		printf("case 'd'\n");
    			if ((activeDevice == 0) and (optarg[0] != '0')) {
    				printf("error: device number is not integer\n");
    				return -1;
    			}
    			if ((activeDevice > deviceCount-1) or (activeDevice < 0)) {
    				printf("error: device number is not valid\n");
    				return -1;
    			}
    			break;
    	}
    }
    cudaSetDevice(activeDevice);
    printf("Running on device %li\n", activeDevice);

    err = writeCudaInfo(deviceCount, activeDevice); // write CUDA device info into file
    if (err != 0) {
        printf("error: could not write device properties into 'gpu.info'\n");
        return -1;
    }

    // determine the endianness on this machine
    if (pEndian[0] == 1) {
        printf("endian = little\n");
        endian = LITTLE_ENDIAN; }
    else {
        printf("endian = big\n");
        endian = BIG_ENDIAN; }

    // increase the bank size for shared memory in case of double precision
    if (PRECISION == 64) {
    	printf("bank size = 8 bytes\n");
    	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }
    else {
    	printf("bank size = 4 bytes\n");
    	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    }

    // read parameters from input file
    readnamelist(&p);

    // check if the method is valid
    if ((strncmp(p.jMethod, "Classic ", 8) != 0) &&
        (strncmp(p.jMethod, "Stokes ", 7) != 0) &&
    	(strncmp(p.jMethod, "Stokes4th ", 10) != 0) &&
    	(strncmp(p.jMethod, "StokesQuint ", 12) != 0) &&
    	(strncmp(p.jMethod, "StokesTri ", 10) != 0)) {
    	printf("error: invalid method for computing J: %s\n", p.jMethod);
    	return -1;
    }
    else
    	printf("method for J: %s\n", p.jMethod);

    printf("initial B: %s\n", p.bInit);
    if (p.inertia == true)
    	printf("initial U: %s\n", p.uInit);
    if (p.pressure == true)
    	printf("include pressure gradient with beta = %f\n", p.beta);

    if (p.fRestart == true)
        readGrid(p);

    // figure out how to paralallelize
    // optimized block size for shared memory usage distinguishing single and double precision and inertia equation
    blockSize[0] = 5 + BLCK_EXT;
    blockSize[1] = 5 + BLCK_EXT + not(p.inertia);
    blockSize[2] = 5 + BLCK_EXT + not(p.inertia);
    if (p.inertia)
		blockSize[2] = 4 + BLCK_EXT;	// needed to limit the amount of shared memory usage to 48 kB for double precision
    gridSize[0] = (p.nx - 1)/blockSize[0] + 1;
    gridSize[1] = (p.ny - 1)/blockSize[1] + 1;
    gridSize[2] = (p.nz - 1)/blockSize[2] + 1;
    dim3 dimBlock(blockSize[0], blockSize[1], blockSize[2]);
    dim3 dimGrid(gridSize[0], gridSize[1], gridSize[2]);
    printf("blockSize[0], blockSize[1], blockSize[2] = %i, %i, %i\n", blockSize[0], blockSize[1], blockSize[2]);
    printf("gridSize[0],  gridSize[1],  gridSize[2]  = %i, %i, %i\n", gridSize[0], gridSize[1], gridSize[2]);
    // grid size for 2d operations, e.g. boundary settings
    gridSize2[0] = (p.nx + 1)/16 + 1;
    gridSize2[1] = (p.ny + 1)/16 + 1;
    gridSize2[2] = (p.nz + 1)/16 + 1;
    dim3 dimGrid2dPlusXY(gridSize2[0], gridSize2[1]);
    dim3 dimGrid2dPlusXZ(gridSize2[0], gridSize2[2]);
    dim3 dimGrid2dPlusYZ(gridSize2[1], gridSize2[2]);
    dim3 dimBlock2d(16, 16);  // should be the square root of the maximum threads per block
    gridSize2[0] = (p.nx - 1)/16 + 1;
    gridSize2[1] = (p.ny - 1)/16 + 1;
    gridSize2[2] = (p.nz - 1)/16 + 1;
    dim3 dimGrid2dXY(gridSize2[0], gridSize2[1]);
    dim3 dimGrid2dXZ(gridSize2[0], gridSize2[2]);
    dim3 dimGrid2dYZ(gridSize2[1], gridSize2[2]);

    // allocate memory
    allocateMemory(&h, &d, p, gridSize);

    if (p.fRestart == false) {
        remove("data/save.vtk");
    	initResiduals(p, &red);
        initState(h, p, &red);
        initDistortion(h.xb, p);
		setPeriHost(h.B0, p);
		setGridPeriHost(h.xb, p);
        writeB0(h.B0, p);
        dt = p.dt0;
        t = 0;
        writeTs(p, red, t, dt, 0, 0.);
    }
    else {
    	initResiduals(p, &red);
        readState(h, p, &t, &dt);
		setPeriHost(h.B0, p);
		setGridPeriHost(h.xb, p);
    }

    // copy current state into device memory
    errCuda = cudaMemcpy(d.B0, h.B0, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.B0)), cudaMemcpyHostToDevice);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'B0' to the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(d.xb, h.xb, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.xb)), cudaMemcpyHostToDevice);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'xb' to the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(d.B, d.B0, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.B)), cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'B0' within the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(d.xb_tmp, d.xb, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.xb_tmp)), cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'xb' within the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpy(d.xb_new, d.xb, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.xb_new)), cudaMemcpyDeviceToDevice);
    if (cudaSuccess != errCuda) { printf("error: could not copy 'xb' within the device\n"); exit(EXIT_FAILURE); }
    if (p.inertia == true) {
		errCuda = cudaMemcpy(d.uu, h.uu, 3*p.nx*p.ny*p.nz*sizeof(*(d.uu)), cudaMemcpyHostToDevice);
		if (cudaSuccess != errCuda) { printf("error: could not copy 'uu' to the device\n"); exit(EXIT_FAILURE); } }

    // copy parameters into constant memory
    errCuda = cudaMemcpyToSymbol(dev_p, &p, sizeof(struct parameters_t));
    if (cudaSuccess != errCuda) { printf("error: could not copy 'p' to the device\n"); exit(EXIT_FAILURE); }
    cudaDeviceSynchronize();

    // assign coefficients
    assignCoefficients(host_a_i, host_b_ij, host_c_i, host_c_star_i);
    errCuda = cudaMemcpyToSymbol(a_i, host_a_i, 6*sizeof(*a_i));
    if (cudaSuccess != errCuda) { printf("error: could not copy 'a_i' to the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpyToSymbol(b_ij, host_b_ij, 7*6*sizeof(*b_ij));
    if (cudaSuccess != errCuda) { printf("error: could not copy 'b_ij' to the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpyToSymbol(c_i, host_c_i, 6*sizeof(*c_i));
    if (cudaSuccess != errCuda) { printf("error: could not copy 'c_i' to the device\n"); exit(EXIT_FAILURE); }
    errCuda = cudaMemcpyToSymbol(c_star_i, host_c_star_i, 6*sizeof(*c_star_i));
    if (cudaSuccess != errCuda) { printf("error: could not copy 'c_star_i' to the device\n"); exit(EXIT_FAILURE); }
    cudaDeviceSynchronize();

    // perform the time stepping
    for (it = 0; (it < p.nCycle) && (dt > p.dtMin); it++)
    {
        for (n = 0; n < 6; n++) {
        	// B = J.B0/Delta
            B_JacB0
                <<<dimGrid, dimBlock, 3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*3*sizeof(*(d.xb))>>>
                (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
            cudaDeviceSynchronize();
            setBbound(dimGrid2dXY, dimGrid2dXZ, dimGrid2dYZ, dimBlock2d, d);
			setPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d, d.B, p);
			cudaDeviceSynchronize();

            current(dimGrid, dimBlock, blockSize, d, p);
            cudaDeviceSynchronize();
            if (p.pressure == true)
            	gradP(dimGrid, dimBlock, blockSize, d, p);
            cudaDeviceSynchronize();
            // intermediate steps for the Runge-Kutta method
            kk
                <<<dimGrid, dimBlock, blockSize[0]*blockSize[1]*blockSize[2]*(3*(9+7*p.inertia+p.pressure)+p.inertia) * sizeof(*(d.xb))>>>
                (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2, n, dt);
            cudaDeviceSynchronize();
            setGridPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d, d.xb, p);
            setGridPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d, d.xb_tmp, p);
			cudaDeviceSynchronize();

            errCuda = cudaMemcpy(d.xb, d.xb_tmp, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.xb)), cudaMemcpyDeviceToDevice);
            if (cudaSuccess != errCuda) { printf("error: could not copy 'xb' within the device during RK step, "
            		"error = %s\n", cudaGetErrorString(errCuda)); exit(EXIT_FAILURE); }
            if (p.inertia == true) {
				errCuda = cudaMemcpy(d.uu, d.uu_tmp, 3*p.nx*p.ny*p.nz*sizeof(*(d.uu)), cudaMemcpyDeviceToDevice);
				if (cudaSuccess != errCuda) { printf("error: could not copy 'uu' within the device during RK step, "
						"error = %s\n", cudaGetErrorString(errCuda)); exit(EXIT_FAILURE); }
            }
            cudaDeviceSynchronize();
        }
        // compute the next full step
        xNewStar
            <<<dimGrid, dimBlock, (3*(10+9*p.inertia)*blockSize[0]*blockSize[1]*blockSize[2])* sizeof(*(d.xb))>>>
           (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
        cudaDeviceSynchronize();
        setGridPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d, d.xb_new, p);
		cudaDeviceSynchronize();

		// miximum error in the domain
        maxDelta = findDevMax(d.maxDelta, gridSize[0]*gridSize[1]*gridSize[2]);

        if (maxDelta < p.maxError) {
            t += dt;
            errCuda = cudaMemcpy(d.xb, d.xb_new, 3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(d.xb)), cudaMemcpyDeviceToDevice);
            if (cudaSuccess != errCuda) { printf("error: could not copy 'xb' within the device\n"); exit(EXIT_FAILURE); }
            if (p.inertia == true) {
				errCuda = cudaMemcpy(d.uu, d.uu_new, 3*p.nx*p.ny*p.nz*sizeof(*(d.uu)), cudaMemcpyDeviceToDevice);
				if (cudaSuccess != errCuda) { printf("error: could not copy 'uu' within the device\n"); exit(EXIT_FAILURE); }
            }
            cudaDeviceSynchronize();

            // compute the current magnetic field and electric current density from xb and B0
            B_JacB0
                <<<dimGrid, dimBlock, 3*(blockSize[0]+2)*(blockSize[1]+2)*(blockSize[2]+2)*3 * sizeof(*(d.xb))>>>
                (d, blockSize[0]+2, blockSize[1]+2, blockSize[2]+2);
            cudaDeviceSynchronize();
            setBbound(dimGrid2dXY, dimGrid2dXZ, dimGrid2dYZ, dimBlock2d, d);
            setPeri(dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d, d.B, p);
			cudaDeviceSynchronize();

            current(dimGrid, dimBlock, blockSize, d, p);
            cudaDeviceSynchronize();

            // write diagnosis on the screen and into time series file
            if (it % p.nTs == 0) {
            	reductions(&red, d, p, blockSize, gridSize, dimGrid, dimBlock);
                writeTs(p, red, t, dt, it, maxDelta);
            }

            // adapt the time step
            if (maxDelta > 0)
            	dt = dt*pow(0.9*p.maxError/maxDelta, REAL(0.2));

            // write out the state file
            if ((ceil((t-dt)/p.dtDump) != ceil(t/p.dtDump)) || (ceil((t-dt)/p.dtSave) != ceil(t/p.dtSave))
            		|| (it == (p.nCycle-1))) {
            	prepareDump(red, h, d, p, blockSize, gridSize, dimGrid, dimBlock,
            			dimGrid2dPlusXY, dimGrid2dPlusXZ, dimGrid2dPlusYZ, dimBlock2d);
            	if (ceil((t-dt)/p.dtDump) != ceil(t/p.dtDump))
            		dumpState(h, p, t, dt, ceil((t-dt)/p.dtDump));
            	if ((ceil((t-dt)/p.dtSave) != ceil(t/p.dtSave)) || (it == (p.nCycle-1)))
            		dumpState(h, p, t, dt, -1);
            }
        }
        else {
			it -= 1;
			// adapt the time step
			if (maxDelta > 0)
				dt = dt*pow(0.9*p.maxError/maxDelta, REAL(0.2));
        }
    }

    if (dt < p.dtMin)
    	printf("dt < dtMin = %e, stopping simulation\n", p.dtMin);

    freeMemory(&h, &d, p);

    printf("glemur terminated successfully\n");
    return 0;
}
