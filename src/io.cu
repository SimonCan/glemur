// io.cu
//
// Reading and writing routines for the snap shots and the time series.
//

#include "global.h"
#include "io.h"

// store the information about the CUDA devices in 'cuda.info'
int writeCudaInfo(int deviceCount, long int activeDevice)
{
    FILE                    *fd;
    struct cudaDeviceProp   devProp;  // cuda device properties
    cudaError_t             errCuda;
    int                     dev;      // device index

    fd = fopen("cuda.info", "w");
    if (fd == NULL) {
        printf("error: could not open file 'cuda.info'\n");
        exit(EXIT_FAILURE);
    }

    fprintf(fd, "number of CUDA devices: %i\n\n", deviceCount);
    fprintf(fd, "active device: %li\n\n", activeDevice);
    fprintf(fd, "PRECISION = %i\n", PRECISION);

    for(dev = 0; dev < deviceCount; dev++) {
        errCuda = cudaGetDeviceProperties(&devProp, dev);
        if (cudaSuccess != errCuda) {
            printf("error: could not obtain device properties for device number %i\n", dev);
            exit(EXIT_FAILURE); }

        fprintf(fd, "device number                   %i\n",  dev);
        fprintf(fd, "name                            %s\n",  devProp.name);
        fprintf(fd, "global memory [bytes]           %lu\n", devProp.totalGlobalMem);
        fprintf(fd, "shared memory per block [bytes] %lu\n", devProp.sharedMemPerBlock);
        fprintf(fd, "registers per block             %d\n",  devProp.regsPerBlock);
        fprintf(fd, "threads per warp                %d\n",  devProp.warpSize);
        fprintf(fd, "max memory pitch [bytes]        %lu\n", devProp.memPitch);
        fprintf(fd, "max threads per block           %d\n",  devProp.maxThreadsPerBlock);
        fprintf(fd, "max threads per block in x      %d\n",  devProp.maxThreadsDim[0]);
        fprintf(fd, "max threads per block in y      %d\n",  devProp.maxThreadsDim[1]);
        fprintf(fd, "max threads per block in z      %d\n",  devProp.maxThreadsDim[2]);
        fprintf(fd, "max blocks per grid in x        %d\n",  devProp.maxGridSize[0]);
        fprintf(fd, "max blocks per grid in y        %d\n",  devProp.maxGridSize[1]);
        fprintf(fd, "max blocks per grid in z        %d\n",  devProp.maxGridSize[2]);
        fprintf(fd, "clock rate [kHz]                %d\n", devProp.clockRate); //
        fprintf(fd, "constant memory [bytes]         %lu\n", devProp.totalConstMem);
        fprintf(fd, "compute capability major        %d\n",  devProp.major);
        fprintf(fd, "compute capability minor        %d\n",  devProp.minor);
        fprintf(fd, "texture alignment               %lu\n", devProp.textureAlignment);
        fprintf(fd, "device overlap                  %d\n",  devProp.deviceOverlap);
        fprintf(fd, "multiprocessors                 %d\n",  devProp.multiProcessorCount);
        fprintf(fd, "kernel exec timeout             %d\n",  devProp.kernelExecTimeoutEnabled);
        fprintf(fd, "integrated GPU                  %d\n",  devProp.integrated);
        fprintf(fd, "can map host memory             %d\n",  devProp.canMapHostMemory);
        fprintf(fd, "computing mode                  %d\n",  devProp.computeMode);
        fprintf(fd, "max size 1D textures            %d\n",  devProp.maxTexture1D);
        fprintf(fd, "max size 2D textures in x       %d\n",  devProp.maxTexture2D[0]);
        fprintf(fd, "max size 2D textures in y       %d\n",  devProp.maxTexture2D[1]);
        fprintf(fd, "max size 3D textures in x       %d\n",  devProp.maxTexture3D[0]);
        fprintf(fd, "max size 3D textures in y       %d\n",  devProp.maxTexture3D[1]);
        fprintf(fd, "max size 3D textures in z       %d\n",  devProp.maxTexture3D[2]);
        fprintf(fd, "concurrent kernels              %d\n",  devProp.concurrentKernels);
        fprintf(fd, "ECC enabled                     %d\n",  devProp.ECCEnabled);
        fprintf(fd, "PCI bus ID                      %d\n",  devProp.pciBusID);
        fprintf(fd, "PCI device ID                   %d\n",  devProp.pciDeviceID);
        fprintf(fd, "PCI domain ID                   %d\n",  devProp.pciDomainID);
        fprintf(fd, "TCC drive                       %d\n",  devProp.tccDriver);
        fprintf(fd, "asynchronous engine count       %d\n",  devProp.asyncEngineCount);
        fprintf(fd, "unified addressing with host    %d\n",  devProp.unifiedAddressing);
        fprintf(fd, "memory clock rate [kHz]         %d\n",  devProp.memoryClockRate);
        fprintf(fd, "memory bus width [bits]         %d\n",  devProp.memoryBusWidth);
        fprintf(fd, "l2 cache size [bytes]           %d\n",  devProp.l2CacheSize);
        fprintf(fd, "maximum threads per multiproc.  %d\n",  devProp.maxThreadsPerMultiProcessor);
        fprintf(fd, "\n");
    }

    fclose(fd);

    return 0;
}


// swap byte order for the correct endianess
REAL floatSwap(REAL value) {
    union v {
        REAL  f;
        UINT  i;
    };

    union  v val;
    UINT   temp;
	#if (PRECISION == 64)
    	unsigned int left, right;
	#endif

	if (endian == LITTLE_ENDIAN) {
		val.f = value;
		#if (PRECISION == 32)
			val.i  = htonl(val.i);
		#endif
		#if (PRECISION == 64)
			right = (unsigned int) val.i;
			val.i = val.i >> 32;
			left  = (unsigned int) val.i;
			left  = htonl(left);
			right = htonl(right);
			val.i = right;
			val.i = val.i << 32;
			val.i = val.i | (UINT) left;
		#endif
		return *(REAL*)&val.i;
	}
	else
		return value;
}


// swap byte order for the correct endianess
int intSwap(int value){
    union v {
        int             f;
        unsigned int    i;
    };

    union        v val;
    unsigned int temp;

    val.f = value;
    temp  = htonl(val.i);

    if (endian == LITTLE_ENDIAN)
        return *(int*)&temp;
    else
        return value;
}


// read the number of grid points from the last snapshot
int readGrid(struct parameters_t p)
{
    FILE *  fd;
    char    line[256];   // text line in the save file
    char    tmp[256];
    int     i;

    fd = fopen("data/save.vtk", "r");
    if (fd == NULL) {
        printf("error: could not open file 'data/save.vtk'\n");
        exit(EXIT_FAILURE);
    }

    // jump over the header
    for (i = 0; i < 5; i++)
        fgets(line, sizeof(line), fd);

    strncpy(tmp, line+11, 9); p.nx = atoi(tmp)-2;
    strncpy(tmp, line+21, 9); p.ny = atoi(tmp)-2;
    strncpy(tmp, line+31, 9); p.nz = atoi(tmp)-2;

    fclose(fd);
    return 0;
}


// read the state and the parameters from the latest dump file
int readState(struct varsHost_t h, struct parameters_t p, REAL *t, REAL *dt)
{
    FILE *  fd;
    char    line[256];   // text line in the save file
    char    tmp[256];
    char    *pos;        // position in a string
    int     numParams;   // number of fields with parameters
    int     i, j, k, l;
    REAL    legacy;	 	 // for backwards compatibility with older files

    fd = fopen("data/save.vtk", "r");
    if (fd == NULL) {
        printf("error: could not open file 'data/save.vtk'\n");
        exit(EXIT_FAILURE);
    }

    // jump over the header
    for (i = 0; i < 5; i++)
        fgets(line, sizeof(line), fd);

    // read how many fields there are
    fgets(line, sizeof(line), fd); strncpy(tmp, line+17, 2);
    numParams = atoi(tmp);

    for (i = 0; i < numParams; i++) {
        fgets(line, sizeof(line), fd);
        pos = strstr(line, " ");
        strncpy(tmp, line, pos-line);

        if (strncmp(tmp, "t", pos-line) == 0) {
            fread(t, sizeof(REAL), 1, fd); *t = floatSwap(*t);
        }
        if (strncmp(tmp, "dt", pos-line) == 0) {
            fread(dt, sizeof(REAL), 1, fd); *dt = floatSwap(*dt);
        }
        if (strncmp(tmp, "nx_ny_nz", pos-line) == 0) {
            fread(&p.nx, sizeof(int), 1, fd); p.nx = intSwap(p.nx);
            fread(&p.ny, sizeof(int), 1, fd); p.ny = intSwap(p.ny);
            fread(&p.nz, sizeof(int), 1, fd); p.nz = intSwap(p.nz);
        }
        if (strncmp(tmp, "Lx_Ly_Lz", pos-line) == 0) {
            fread(&p.Lx, sizeof(REAL), 1, fd); p.Lx = floatSwap(p.Lx);
            fread(&p.Ly, sizeof(REAL), 1, fd); p.Ly = floatSwap(p.Ly);
            fread(&p.Lz, sizeof(REAL), 1, fd); p.Lz = floatSwap(p.Lz);
        }
        if (strncmp(tmp, "Ox_Oy_Oz", pos-line) == 0) {
            fread(&p.Ox, sizeof(REAL), 1, fd); p.Ox = floatSwap(p.Ox);
            fread(&p.Oy, sizeof(REAL), 1, fd); p.Oy = floatSwap(p.Oy);
            fread(&p.Oz, sizeof(REAL), 1, fd); p.Oz = floatSwap(p.Oz);
        }
        if (strncmp(tmp, "dx_dy_dz", pos-line) == 0) {
            fread(&p.dx, sizeof(REAL), 1, fd); p.dx = floatSwap(p.dx);
            fread(&p.dy, sizeof(REAL), 1, fd); p.dy = floatSwap(p.dy);
            fread(&p.dz, sizeof(REAL), 1, fd); p.dz = floatSwap(p.dz);
        }
        if (strncmp(tmp, "ampl", pos-line) == 0) {
            fread(&p.ampl, sizeof(REAL), 1, fd); p.ampl = floatSwap(p.ampl);
        }
        if (strncmp(tmp, "phi1_phi2", pos-line) == 0) {
            fread(&p.phi1, sizeof(REAL), 1, fd); p.phi1 = floatSwap(p.phi1);
            fread(&p.phi2, sizeof(REAL), 1, fd); p.phi2 = floatSwap(p.phi2);
        }
        if (strncmp(tmp, "rxhalf_ryhalf", pos-line) == 0) {
            fread(&legacy, sizeof(REAL), 1, fd);
            fread(&legacy, sizeof(REAL), 1, fd);
        }
    }

    // read the grid data
    fgets(line, sizeof(line), fd);
    fread(h.xb, sizeof(REAL), 3*(p.nx+2)*(p.ny+2)*(p.nz+2), fd);
    for (i = 0; i < 3*(p.nx+2)*(p.ny+2)*(p.nz+2); i++)
        h.xb[i] = floatSwap(h.xb[i]);

    // read the velocity field. need to jump over B-field first
    if (p.inertia == true) {
    	REAL* uu_tmp;
    	uu_tmp = (REAL *)malloc(3*(p.nx+2)*(p.ny+2)*(p.nz+2)*sizeof(*(uu_tmp)));
        if (uu_tmp == NULL) { printf("error: could not allocate memory for uu_tmp\n"); return -1; }
        fgets(line, sizeof(line), fd);
		fread(uu_tmp, sizeof(REAL), 3*(p.nx+2)*(p.ny+2)*(p.nz+2), fd);	// jump over B
		fgets(line, sizeof(line), fd);
		fread(uu_tmp, sizeof(REAL), 3*(p.nx+2)*(p.ny+2)*(p.nz+2), fd);
	    for (k = 0; k < p.nz; k++)
	        for (j = 0; j < p.ny; j++)
	            for (i = 0; i < p.nx; i++)
	            	for (l = 0; l < 3; l++)
	            		h.uu[l + 3*i + 3*p.nx*j + 3*p.nx*p.ny*k] = floatSwap(uu_tmp[l + 3*(i+1) + 3*(p.nx+2)*(j+1) + 3*(p.nx+2)*(p.ny+2)*(k+1)]);
    }

    fclose(fd);

    // read the initial magnetic field B0
    fd = fopen("data/B0.vtk", "r");
    if (fd == NULL) {
        printf("error: could not open file 'data/save.vtk'\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < 9; i++)
        fgets(line, sizeof(line), fd);

    fread(h.B0, sizeof(REAL), 3*(p.nx+2)*(p.ny+2)*(p.nz+2), fd);
    for (i = 0; i < 3*(p.nx+2)*(p.ny+2)*(p.nz+2); i++)
        h.B0[i] = floatSwap(h.B0[i]);

    fclose(fd);

    return 0;
}


// dump the initial magnetic field
int writeB0(REAL *B0, struct parameters_t p)
{
    FILE    *fd;
    int     i, j, k;
    REAL    swapped[1]; // needed to swap byte order to big endian

    fd = fopen("data/B0.vtk", "w");
    if (fd == NULL) {
        printf("error: could not open file 'data/B0.vtk'\n");
        exit(EXIT_FAILURE);
    }

    // write common header
    fprintf(fd, "# vtk DataFile Version 2.0\n");
    fprintf(fd, "GLEMuR B0 dump\n");
    fprintf(fd, "BINARY\n");
    fprintf(fd, "DATASET STRUCTURED_POINTS\n");
    fprintf(fd, "DIMENSIONS %9i %9i %9i\n", p.nx+2, p.ny+2, p.nz+2);
    fprintf(fd, "ORIGIN %8.12f %8.12f %8.12f\n", p.Ox-p.dx, p.Oy-p.dy, p.Oz-p.dz);
    fprintf(fd, "SPACING %8.12f %8.12f %8.12f\n", p.Lx/(p.nx-1), p.Ly/(p.ny-1), p.Lz/(p.nz-1));
    fprintf(fd, "POINT_DATA %9i\n", (p.nx+2)*(p.ny+2)*(p.nz+2));
    fprintf(fd, "VECTORS bfield %s\n", REAL_STR);

    for (k = 0; k < p.nz+2; k++) {
        for (j = 0; j < p.ny+2; j++) {
            for (i = 0; i < p.nx+2; i++) {
                swapped[0] = floatSwap(B0[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
                swapped[0] = floatSwap(B0[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
                swapped[0] = floatSwap(B0[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
            }
        }
    }

    fclose(fd);
    return 0;
}


// dump the current state
int dumpState(struct varsHost_t h, struct parameters_t p, REAL t, REAL dt, int n)
{
    FILE    *fd;
    int     i, j, k;
    REAL    swapped[1]; // needed to swap byte order to big endian
    char	fileName[20];

    if (n == -1)
        sprintf(fileName, "data/save.vtk");
    else
        sprintf(fileName, "data/dump%d.vtk", n);
    fd = fopen(fileName, "w");

    if (fd == NULL) {
        printf("error: could not open file '%s'\n", fileName);
        exit(EXIT_FAILURE);
    }

    // write common header
    fprintf(fd, "# vtk DataFile Version 2.0\n");
    fprintf(fd, "GLEMuR data dump\n");
    fprintf(fd, "BINARY\n");
    fprintf(fd, "DATASET STRUCTURED_GRID\n");
    fprintf(fd, "DIMENSIONS %9i %9i %9i\n", p.nx+2, p.ny+2, p.nz+2);

    // parameters as meta data
    writeParams(p, t, dt, fd);

    // write structured grid xb
    fprintf(fd, "POINTS %9i %s\n", (p.nx+2)*(p.ny+2)*(p.nz+2), REAL_STR);
    for (k = 0; k < p.nz+2; k++) {
        for (j = 0; j < p.ny+2; j++) {
            for (i = 0; i < p.nx+2; i++) {
                swapped[0] = floatSwap(h.xb[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
                swapped[0] = floatSwap(h.xb[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
                swapped[0] = floatSwap(h.xb[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
            }
        }
    }

    fprintf(fd, "POINT_DATA %9i\n", (p.nx+2)*(p.ny+2)*(p.nz+2));

    // write magnetic field B
    fprintf(fd, "VECTORS bfield %s\n", REAL_STR);
    for (k = 0; k < p.nz+2; k++) {
        for (j = 0; j < p.ny+2; j++) {
            for (i = 0; i < p.nx+2; i++) {
                swapped[0] = floatSwap(h.B[0 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
                swapped[0] = floatSwap(h.B[1 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
                swapped[0] = floatSwap(h.B[2 + i*3 + j*(p.nx+2)*3 + k*(p.nx+2)*(p.ny+2)*3]);
                fwrite(swapped, sizeof(REAL), 1, fd);
            }
        }
    }

    // write velocity field uu
    if (p.inertia == 1) {
		fprintf(fd, "VECTORS ufield %s\n", REAL_STR);
		for (k = 0; k < p.nz+2; k++) {
			for (j = 0; j < p.ny+2; j++) {
				for (i = 0; i < p.nx+2; i++) {
                    if ((i == 0) || (i == p.nx+1) || (j == 0) || (j == p.ny+1) || (k == 0) || (k == p.nz+1)) {
                        swapped[0] = 0.;
                        fwrite(swapped, sizeof(REAL), 1, fd);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                    }
                    else {
						swapped[0] = floatSwap(h.uu[0 + (i-1)*3 + (j-1)*p.nx*3 + (k-1)*p.nx*p.ny*3]);
						fwrite(swapped, sizeof(REAL), 1, fd);
						swapped[0] = floatSwap(h.uu[1 + (i-1)*3 + (j-1)*p.nx*3 + (k-1)*p.nx*p.ny*3]);
						fwrite(swapped, sizeof(REAL), 1, fd);
						swapped[0] = floatSwap(h.uu[2 + (i-1)*3 + (j-1)*p.nx*3 + (k-1)*p.nx*p.ny*3]);
						fwrite(swapped, sizeof(REAL), 1, fd);
                    }
				}
			}
		}
    }

	// write electric current density J
    if (p.dumpJ == 1) {
        fprintf(fd, "VECTORS jfield %s\n", REAL_STR);
        for (k = 0; k < p.nz+2; k++) {
            for (j = 0; j < p.ny+2; j++) {
                for (i = 0; i < p.nx+2; i++) {
                    if ((i == 0) || (i == p.nx+1) || (j == 0) || (j == p.ny+1) || (k == 0) || (k == p.nz+1)) {
                        swapped[0] = 0.;
                        fwrite(swapped, sizeof(REAL), 1, fd);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                    }
                    else {
                        swapped[0] = floatSwap(h.JJ[0 + (i-1)*3 + (j-1)*p.nx*3 + (k-1)*p.nx*p.ny*3]);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                        swapped[0] = floatSwap(h.JJ[1 + (i-1)*3 + (j-1)*p.nx*3 + (k-1)*p.nx*p.ny*3]);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                        swapped[0] = floatSwap(h.JJ[2 + (i-1)*3 + (j-1)*p.nx*3 + (k-1)*p.nx*p.ny*3]);
                        fwrite(swapped, sizeof(REAL), 1, fd);
                    }
                }
            }
        }
    }

	// write the determinant of the Jacobian matrix
    if (p.dumpDetJac == 1) {
        fprintf(fd, "SCALARS detJac %s\n", REAL_STR);
        fprintf(fd, "LOOKUP_TABLE default\n");
        for (k = 0; k < p.nz+2; k++) {
            for (j = 0; j < p.ny+2; j++) {
                for (i = 0; i < p.nx+2; i++) {
                    swapped[0] = floatSwap(h.detJac[i + j*(p.nx+2) + k*(p.nx+2)*(p.ny+2)]);
                    fwrite(swapped, sizeof(REAL), 1, fd);
                }
            }
        }
    }

	// write the directional cell volume
    if (p.dumpCellVol == 1) {
        fprintf(fd, "SCALARS cellVol %s\n", REAL_STR);
        fprintf(fd, "LOOKUP_TABLE default\n");
        for (k = 0; k < p.nz+2; k++) {
            for (j = 0; j < p.ny+2; j++) {
                for (i = 0; i < p.nx+2; i++) {
                	if ((i*j*k == 0) || (i == p.nx+1) || (j == p.ny+1) || (k == p.nz+1))
                		swapped[0] = floatSwap(0);
                	else
                		swapped[0] = floatSwap(h.cellVol[(i-1) + (j-1)*p.nx + (k-1)*p.nx*p.ny]);
                    fwrite(swapped, sizeof(REAL), 1, fd);
                }
            }
        }
    }

	// write the convexity of the cells around the grid point
    if (p.dumpConvexity == 1) {
        fprintf(fd, "SCALARS convexity %s\n", REAL_STR);
        fprintf(fd, "LOOKUP_TABLE default\n");
        for (k = 0; k < p.nz+2; k++) {
            for (j = 0; j < p.ny+2; j++) {
                for (i = 0; i < p.nx+2; i++) {
                	if ((i*j*k == 0) || (i == p.nx+1) || (j == p.ny+1) || (k == p.nz+1))
                		swapped[0] = floatSwap(0);
                	else
                		swapped[0] = floatSwap(h.convexity[(i-1) + (j-1)*p.nx + (k-1)*p.nx*p.ny]);
                    fwrite(swapped, sizeof(REAL), 1, fd);
                }
            }
        }
    }

	// write the minimum of the wedge products
    if (p.dumpWedgeMin == 1) {
        fprintf(fd, "SCALARS wedgeMin %s\n", REAL_STR);
        fprintf(fd, "LOOKUP_TABLE default\n");
        for (k = 0; k < p.nz+2; k++) {
            for (j = 0; j < p.ny+2; j++) {
                for (i = 0; i < p.nx+2; i++) {
                	if ((i*j*k == 0) || (i == p.nx+1) || (j == p.ny+1) || (k == p.nz+1))
                		swapped[0] = floatSwap(0);
                	else
                		swapped[0] = floatSwap(h.wedgeMin[(i-1) + (j-1)*p.nx + (k-1)*p.nx*p.ny]);
                    fwrite(swapped, sizeof(REAL), 1, fd);
                }
            }
        }
    }

    fclose(fd);
    return 0;
}


// writes parameters in the dumping files
int writeParams(struct parameters_t p, REAL t, REAL dt, FILE *fd)
{
    REAL    swappedF[1]; // needed to swap byte order to big endian
    int     swappedI[1]; // needed to swap byte order to big endian

    fprintf(fd, "FIELD parameters %i\n", 9);
    fprintf(fd, "t 1 1 %s\n", REAL_STR);
    swappedF[0] = floatSwap(t); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "dt 1 1 %s\n", REAL_STR);
    swappedF[0] = floatSwap(dt); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "nx_ny_nz 1 3 int\n");
    swappedI[0] = intSwap(p.nx); fwrite(swappedI, sizeof(int), 1, fd);
    swappedI[0] = intSwap(p.ny); fwrite(swappedI, sizeof(int), 1, fd);
    swappedI[0] = intSwap(p.nz); fwrite(swappedI, sizeof(int), 1, fd);
    fprintf(fd, "Lx_Ly_Lz 1 3 %s\n", REAL_STR);
    swappedF[0] = floatSwap(p.Lx); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.Ly); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.Lz); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "Ox_Oy_Oz 1 3 %s\n", REAL_STR);
    swappedF[0] = floatSwap(p.Ox); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.Oy); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.Oz); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "dx_dy_dz 1 3 %s\n", REAL_STR);
    swappedF[0] = floatSwap(p.dx); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.dy); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.dz); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "ampl 1 1 %s\n", REAL_STR);
    swappedF[0] = floatSwap(p.ampl); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "phi1_phi2 1 2 %s\n", REAL_STR);
    swappedF[0] = floatSwap(p.phi1); fwrite(swappedF, sizeof(REAL), 1, fd);
    swappedF[0] = floatSwap(p.phi2); fwrite(swappedF, sizeof(REAL), 1, fd);
    fprintf(fd, "maxError 1 1 %s\n", REAL_STR);
    swappedF[0] = floatSwap(p.maxError); fwrite(swappedF, sizeof(REAL), 1, fd);

    return 0;
}


// write out the time series
int writeTs(struct parameters_t p, struct red_t red, REAL t, REAL dt, int it, REAL maxDelta)
{
    FILE  * fd;

    if (t == 0) {
        fd = fopen("data/time_series.dat", "w+");
        if (fd == NULL) {
            printf("error: could not open file 'data/time_series.dat'\n");
            exit(EXIT_FAILURE);
        }

        printf("#%7s", "it");
        printf("%13s", "t");
        printf("%13s", "dt");
        printf("%13s", "maxError");
        if (p.redJMax == true)
            printf("%13s", "JMax");
        if (p.redJxB_B2Max == true)
            printf("%13s", "JxB_B2Max");
        if (p.redEpsilonStar == true)
            printf("%13s", "epsilonStar");
        if (p.redErrB_1ez == true)
            printf("%13s", "errB_1ez");
        if (p.redErrXb_XbAn == true)
            printf("%13s", "errXb_XbAn");
        if (p.redB2 == true)
            printf("%13s", "B2");
        if (p.redB2f == true)
            printf("%13s", "B2f");
        if (p.redConvex == true)
            printf("%13s", "convex");
        if (p.redWedgeMin == true)
            printf("%13s", "wedgeMin");
        if ((p.redU2 == true) && (p.inertia == true))
            printf("%13s", "U2");
        if ((p.redUMax == true) && (p.inertia == true))
            printf("%13s", "UMax");
        printf("\n");

        fprintf(fd, "#%7s", "it");
        fprintf(fd, "%13s", "t");
        fprintf(fd, "%13s", "dt");
        fprintf(fd, "%13s", "maxDelta");
        if (p.redJMax == true)
            fprintf(fd, "%13s", "JMax");
        if (p.redJxB_B2Max == true)
            fprintf(fd, "%13s", "JxB_B2Max");
        if (p.redEpsilonStar == true)
            fprintf(fd, "%13s", "epsilonStar");
        if (p.redErrB_1ez == true)
            fprintf(fd, "%13s", "errB_1ez");
        if (p.redErrXb_XbAn == true)
            fprintf(fd, "%13s", "errXb_XbAn");
        if (p.redB2 == true)
            fprintf(fd, "%13s", "B2");
        if (p.redB2f == true)
            fprintf(fd, "%13s", "B2f");
        if (p.redConvex == true)
            fprintf(fd, "%13s", "convex");
        if (p.redWedgeMin == true)
            fprintf(fd, "%13s", "wedgeMin");
        if ((p.redU2 == true) && (p.inertia == true))
            fprintf(fd, "%13s", "U2");
        if ((p.redUMax == true) && (p.inertia == true))
        	fprintf(fd, "%13s", "UMax");
        fprintf(fd, "\n");

        fclose(fd);
    }
    else {
        fd = fopen("data/time_series.dat", "a");
        if (fd == NULL) {
            printf("error: could not open file 'data/time_series.dat'\n");
            exit(EXIT_FAILURE);
        }

        printf("%8i ", it);
        printf("%12.5e ", t);
        printf("%12.5e ", dt);
        printf("%12.5e ", maxDelta);
        if (p.redJMax == true)
            printf("%12.5e ", red.JMax);
        if (p.redJxB_B2Max == true)
            printf("%12.5e ", red.JxB_B2Max);
        if (p.redEpsilonStar == true)
            printf("%12.5e ", red.epsilonStar);
        if (p.redErrB_1ez == true)
            printf("%12.5e ", red.errB_1ez);
        if (p.redErrXb_XbAn == true)
            printf("%12.5e ", red.errXb_XbAn);
        if (p.redB2 == true)
            printf("%12.5e ", red.B2);
        if (p.redB2f == true)
            printf("%12.5e ", red.B2-red.B2res);
        if (p.redConvex == true)
            printf("%12.5e ", red.convex);
        if (p.redWedgeMin == true)
            printf("%12.5e ", red.wedgeMin);
        if ((p.redU2 == true) && (p.inertia == true))
            printf("%12.5e ", red.U2);
        if ((p.redUMax == true) && (p.inertia == true))
            printf("%12.5e ", red.UMax);
        printf("\n");

        fprintf(fd, "%8i ", it);
        fprintf(fd, "%12.5e ", t);
        fprintf(fd, "%12.5e ", dt);
        fprintf(fd, "%12.5e ", maxDelta);
        if (p.redJMax == true)
            fprintf(fd, "%12.5e ", red.JMax);
        if (p.redJxB_B2Max == true)
            fprintf(fd, "%12.5e ", red.JxB_B2Max);
        if (p.redEpsilonStar == true)
            fprintf(fd, "%12.5e ", red.epsilonStar);
        if (p.redErrB_1ez == true)
        	fprintf(fd, "%12.5e ", red.errB_1ez);
        if (p.redErrXb_XbAn == true)
            fprintf(fd, "%12.5e ", red.errXb_XbAn);
        if (p.redB2 == true)
            fprintf(fd, "%12.5e ", red.B2);
        if (p.redB2f == true)
            fprintf(fd, "%12.5e ", red.B2-red.B2res);
        if (p.redConvex == true)
            fprintf(fd, "%12.5e ", red.convex);
        if (p.redWedgeMin == true)
            fprintf(fd, "%12.5e ", red.wedgeMin);
        if ((p.redU2 == true) && (p.inertia == true))
            fprintf(fd, "%12.5e ", red.U2);
        if ((p.redUMax == true) && (p.inertia == true))
        	fprintf(fd, "%12.5e ", red.UMax);
        fprintf(fd, "\n");

        fclose(fd);
    }

    return 0;
}
