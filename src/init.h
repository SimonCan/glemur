// init.h

#ifndef INIT_H_
#define INIT_H_

// Set the residual magnetic energy for the corresponding configurations.
void initResiduals(struct Parameters params, struct Reduction *red);

// Create the initial magnetic field B0 and the initial grid xb.
int initState(struct VarsHost h, struct Parameters params, struct Reduction *red);

// Add a distortion to the initial grid xb. note that B0 refers to the undistorted grid.
int initDistortion(REAL *xb, struct Parameters params);

#endif /* INIT_H_ */
