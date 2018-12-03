// blobsDomes.h

#ifndef BLOBSDOMES_H_
#define BLOBSDOMES_H_

// create the initial magnetic field B0 for one dome and blobs with variable twist k
int initBlobsDome(struct VarsHost h, struct Parameters p);

// create the initial magnetic field B0 for one dome and blobs with variable twist k for short box (-8 .. 8)
int initBlobsDomeShort(struct VarsHost h, struct Parameters p);

// create the initial magnetic field B0 for two domes and blobs
int initBlobsDomes2(struct VarsHost h, struct Parameters p);

#endif /* BLOBSDOMES_H_ */
