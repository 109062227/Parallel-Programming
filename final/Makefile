CC = gcc
CXX = g++
CFLAGS = -lm -O3
conv_vec: CFLAGS += -fopenmp -mavx2 
conv_vec2: CFLAGS += -fopenmp -mavx2 
conv_mpi: CC = mpicc
conv_mpi: CXX = mpicxx
conv_mpi: CFLAGS += -fopenmp
conv_hybrid: CC = mpicc
conv_hybrid: CXX = mpicxx
conv_hybrid: CFLAGS += -fopenmp
conv_omp: CFLAGS += -fopenmp -mavx2 
conv_pthread: CFLAGS += -pthread -mavx2 
CXXFLAGS = $(CFLAGS)
TARGETS = conv_matrix conv_seq conv_vec conv_pthread conv_mpi conv_omp conv_hybrid conv_vec2

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
