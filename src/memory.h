// memory.h

#ifndef MEMORY_H_
#define MEMORY_H_

// allocate host and device memory
int allocateMemory(struct varsHost_t *h, struct varsDev_t *d, struct parameters_t p, int gridSize[3]);

// free host and device memory
int freeMemory(struct varsHost_t *h, struct varsDev_t *d, struct parameters_t p);

#endif /* MEMORY_H_ */
