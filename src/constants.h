// constant.h

#ifndef CONSTANT_H_
#define CONSTANT_H_

#include "global.h"

// constants on the device for fast access
__constant__ struct parameters_t dev_p; // simulation parameters on the device
__constant__ REAL   a_i[6];             // coefficients a_i for the adaptive step size RG
__constant__ REAL   b_ij[7*6];          // coefficients b_ij for the adaptive step size RG
__constant__ REAL   c_i[6];             // coefficients c_i for the adaptive step size RG
__constant__ REAL   c_star_i[6];        // coefficients c*_i for the adaptive step size RG

#endif /* CONSTANT_H_ */
