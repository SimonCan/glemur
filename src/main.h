// main.h

#ifndef MAIN_H_
#define MAIN_H_

#include "global.h"

int					c;			  	// used for parsing command line parameters
long int			activeDevice; 	// number of the CUDA device to be used

int                 it;           	// time step index
struct parameters_t p;            	// contains the user specified parameters
struct red_t        red;          	// contains all the reduced values
REAL                t, dt;        	// time and delta time

// host memory
struct varsHost_t	h;			  	// struct with all the host variables
int                 err;          	// error returned by host functions

// device memory
struct varsDev_t	d;			  	// struct with all the device variables
cudaError_t         errCuda;      	// error returned by device functions
int                 deviceCount;  	// number of CUDA devices
int                 blockSize[3]; 	// number of threads per dimension per block
int                 gridSize[3];  	// number of blocks per dimension
int					gridSize2[3]; 	// 2d grid size

// time stepping variables
REAL                maxDelta;     	// maximum value for the error during one time step
int                 n;            	// increment for the vectors kk

// coefficients for the RG adaptive time step method
REAL   host_a_i[6];               	// coefficients a_i for the adaptive step size RG
REAL   host_b_ij[7][6];           	// coefficients b_ij for the adaptive step size RG
REAL   host_c_i[6];               	// coefficients c_i for the adaptive step size RG
REAL   host_c_star_i[6];          	// coefficients c*_i for the adaptive step size RG

#endif /* ODEMI_H_ */
