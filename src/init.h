// init.h

#ifndef INIT_H_
#define INIT_H_

// set the residual magnetic energy for the corresponding configurations
int initResiduals(struct parameters_t p, struct red_t *red);

// create the initial magnetic field B0 and the initial grid xb
int initState(struct varsHost_t h, struct parameters_t p, struct red_t *red);

// add a distortion to the initial grid xb. note that B0 refers to the undistorted grid
int initDistortion(REAL *xb, struct parameters_t p);

#endif /* INIT_H_ */
