// io.h

#ifndef IO_H_
#define IO_H_

// Store the information about the CUDA devices in 'cuda.info'.
int writeCudaInfo(int deviceCount, long int activeDevice);

// Swap byte order for the correct endianness.
REAL floatSwap(REAL value);

// Read the number of grid points from the last snapshot.
int readGrid(struct Parameters params);

// Read the state from the latest dump file.
int readState(struct VarsHost h, struct Parameters p, REAL *t, REAL *dt);

// Dump the initial magnetic field.
int writeB0(REAL *B0, struct Parameters params);

// Dump the current state.
int dumpState(struct VarsHost h, struct Parameters params, REAL t, REAL dt, int n);

// Write parameters in the dumping files.
int writeParams(struct Parameters params, REAL t, REAL dt, FILE *fd);

// Write out the time series.
int writeTs(struct Parameters params, struct Reduction red, REAL t, REAL dt, int it, REAL maxDelta);

#endif /* IO_H_ */
