// blobsDomes.h

#ifndef BLOBSDOMES_H_
#define BLOBSDOMES_H_

// create the initial magnetic field B0 for one dome and blobs with variable twist k
int initBlobsDome(struct varsHost_t h, struct parameters_t p);

// create the initial magnetic field B0 for one dome and blobs with variable twist k for short box (-8 .. 8)
int initBlobsDomeShort(struct varsHost_t h, struct parameters_t p);

// create the initial magnetic field B0 for two domes and blobs
int initBlobsDomes2(struct varsHost_t h, struct parameters_t p);

#endif /* BLOBSDOMES_H_ */
