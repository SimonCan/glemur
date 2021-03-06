# Makefile for glemur

EXEC = glemur
TMP = tmp.o
OBJ_CU_MAIN = main.o io.o init.o memory.o diagnostics.o maths.o current.o gradP.o rungeKutta.o boundaries.o
# Add initial conditions.
OBJ_CU = $(OBJ_CU_MAIN) blobsDomes.o readExternal.o
VPATH = initialConditions
OBJ = $(OBJ_CU) readNamelist.o $(TMP)
SRC = main.cu
PRECISION = 64
#DEBUG = -G -g
DEBUG = 
LIB_PATH = $(DYLD_LIBRARY)
#OPT = -rdc=true -m64 -arch sm_30 -L$(LIB_PATH) -lcuda -lcudart -dc -lm -IinitialConditions
OPT = -rdc=true -m64 --gpu-architecture=sm_35 -L$(LIB_PATH) -lcuda -lcudart -dc -lm -IinitialConditions

all: $(EXEC)

$(EXEC): $(OBJ)
	$(FC) -v -o $(EXEC) $(OBJ) -L$(LIB_PATH) -lcuda -lcudart -lstdc++
	rm $(TMP)

tmp.o: $(OBJ_CU)
	$(NVCC) -o $(TMP) $(OBJ_CU) -rdc=true --gpu-architecture=sm_35 -dlink -L$(LIB_PATH) -lcuda -lcudart -lstdc++

readNamelist.o: readNamelist.f90
	$(FC) -cpp -DPRECISION=$(PRECISION) -c readNamelist.f90

main.o: main.cu main.h global.h io.o init.o memory.o diagnostics.o maths.o current.o gradP.o rungeKutta.o boundaries.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) main.cu
		
io.o: io.cu io.h global.h blobsDomes.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) io.cu

init.o: init.cu init.h global.h
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) init.cu
	
memory.o: memory.cu memory.h global.h
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) memory.cu

diagnostics.o: diagnostics.cu diagnostics.h maths.o maths.h global.h
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) diagnostics.cu

maths.o: maths.cu maths.h global.h
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) maths.cu

current.o: current.cu current.h maths.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) current.cu

gradP.o: gradP.cu gradP.h maths.h
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) gradP.cu

rungeKutta.o: rungeKutta.cu current.cu current.h global.h maths.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) rungeKutta.cu

boundaries.o: boundaries.cu boundaries.h global.h maths.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) boundaries.cu

# Initial conditions.
blobsDomes.o: blobsDomes.cu
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) $@ -c $<

readExternal.o: readExternal.cu
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) $@ -c $<

clean:
	rm -f $(EXEC) *.o
