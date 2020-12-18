// memory.h

#ifndef MEMORY_H_
#define MEMORY_H_

// Allocate host and device memory.
int allocateMemory(struct VarsHost *h, struct VarsDev *d, struct Parameters params, int gridSize[3]);

// Free host and device memory.
int freeMemory(struct VarsHost *h, struct VarsDev *d, struct Parameters params);

#endif /* MEMORY_H_ */
