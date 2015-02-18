// io.h

#ifndef IO_H_
#define IO_H_

// store the information about the CUDA devices in 'cuda.info'
int writeCudaInfo(int deviceCount, long int activeDevice);

// swap byte order for the correct endianness
REAL floatSwap(REAL value);

// read the number of grid points from the last snapshot
int readGrid(struct parameters_t p);

// read the state from the latest dump file
int readState(struct varsHost_t h, struct parameters_t p, REAL *t, REAL *dt);

// dump the initial magnetic field
int writeB0(REAL *B0, struct parameters_t p);

// dump the current state
int dumpState(struct varsHost_t h, struct parameters_t p, REAL t, REAL dt, int n);

// writes parameters in the dumping files
int writeParams(struct parameters_t p, REAL t, REAL dt, FILE *fd);

// write out the time series
int writeTs(struct parameters_t p, struct red_t red, REAL t, REAL dt, int it, REAL maxDelta);

#endif /* IO_H_ */
