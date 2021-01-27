// main.h

#ifndef MAIN_H_
#define MAIN_H_

#include "global.h"

int                 parsedCom;        // used for parsing command line parameters
long int            activeDevice;     // number of the CUDA device to be used

int                 timeIndex;        // time step index
Parameters          params;           // contains the user specified parameters
struct Reduction    red;              // contains all the reduced values
REAL                t, dt;            // time and delta time

// Host memory.
struct VarsHost     h;                // struct with all the host variables
int                 err;              // error returned by host functions

// Device memory.
struct VarsDev      d;                // struct with all the device variables
cudaError_t         errCuda;          // error returned by device functions
int                 deviceCount;      // number of CUDA devices on the system
int                 blockSize[3];     // number of threads per dimension per block
int                 gridSize[3];      // number of blocks per dimension
int                 gridSize2[3];     // 2d grid size

// Time stepping variables.
REAL                maxDelta;         // maximum value for the error during one time step
int                 n;                // increment for the vectors kk

// Coefficients for the Runge-Kutta adaptive time step method.
REAL   host_a_i[6];                   // coefficients a_i for the adaptive step size Runge-Kutta
REAL   host_b_ij[7][6];               // coefficients b_ij for the adaptive step size Runge-Kutta
REAL   host_c_i[6];                   // coefficients c_i for the adaptive step size Runge-Kutta
REAL   host_cs_i[6];                  // coefficients c*_i for the adaptive step size Runge-Kutta


#endif
