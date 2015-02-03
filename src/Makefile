# Makefile for main.cu

EXEC = glemur
TMP = tmp.o
OBJ_CU = main.o io.o init.o memory.o diagnostics.o maths.o current.o rungeKutta.o boundaries.o blobsDomes.o
OBJ  = $(OBJ_CU) readNamelist.o $(TMP)
SRC  = main.cu
PRECISION = 64
#DEBUG = -G -g
DEBUG = 
LIB_PATH = /usr/local/cuda-5.5/lib64	# modify here for the path to your cuda library
#LIB_PATH = /usr/local/cuda-6.0/lib64
OPT = -rdc=true -m64 -arch sm_30 -L$(LIB_PATH) -lcuda -lcudart -dc -lm

all: $(EXEC)

$(EXEC): $(OBJ)
	$(FC) -o $(EXEC) $(OBJ) -L$(LIB_PATH) -lcuda -lcudart -lstdc++
	rm $(TMP)

tmp.o: $(OBJ_CU)
	$(NVCC) -o $(TMP) $(OBJ_CU) -rdc=true -arch sm_30 -dlink -L$(LIB_PATH) -lcuda -lcudart -lstdc++

readNamelist.o: readNamelist.f90
	$(FC) -cpp -DPRECISION=$(PRECISION)  -c readNamelist.f90

main.o: main.cu main.h global.h io.o init.o memory.o diagnostics.o maths.o current.o rungeKutta.o boundaries.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) main.cu
		
io.o: io.cu io.h global.h blobsDomes.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) io.cu

blobsDomes.o: blobsDomes.h
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) blobsDomes.cu

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

rungeKutta.o: current.cu current.h global.h maths.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) rungeKutta.cu

boundaries.o: boundaries.cu boundaries.h global.h maths.o
	$(NVCC) $(OPT) $(DEBUG) -DPRECISION=$(PRECISION) boundaries.cu

clean:
	rm -f $(EXEC) *.o
