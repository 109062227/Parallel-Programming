CC = icc
CXX = icpc
LDLIBS = -lpng -lm
CFLAGS = -O3


hw2a: CFLAGS += -std=c++11 -pthread -mavx512f 
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -mavx512f -ffp-contract=off
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
