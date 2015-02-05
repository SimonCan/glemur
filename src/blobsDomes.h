// blobsDomes.h

#ifndef BLOBSDOMES_H_
#define BLOBSDOMES_H_

// initialize the blobsDomes configuration
int initBlobsDomes(struct varsHost_t h, struct parameters_t p);

// create the initial magnetic field B0 for two domes and blobs with twist k = 0.35
int initBlobsDomesK035(struct varsHost_t h, struct parameters_t p);

// create the initial magnetic field B0 for two domes and blobs with twist k = 0.5
int initBlobsDomesK05(struct varsHost_t h, struct parameters_t p);

// create the initial magnetic field B0 for two domes and blobs with variable twist k
int initBlobsDomesK(struct varsHost_t h, struct parameters_t p);

// initialize the blobsDomeSingle configuration
int initBlobsDomeSingle(struct varsHost_t h, struct parameters_t p);

#endif /* BLOBSDOMES_H_ */
