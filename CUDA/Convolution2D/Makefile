CC=g++
NCC=nvcc
FLAGS=-g -lcudart 
NFLAGS=-deviceemu
SRC=$(wildcard src/*.c src/*.cpp src/*.hpp src/*.h)
NSRC=$(wildcard src/*.cu)
NOBJ=$(addsuffix .o, $(basename $(NSRC)))
LD=/usr/local/cuda/lib

2dconvol.o:cuda.o
	$(CC) $(FLAGS) -o ./bin/2dconvol.o $(SRC) $(NOBJ) -L $(LD)

cuda.o:
	$(NCC) $(NFLAGS) -c $(NSRC) 
	mv *.o src/

run: 2dconvol.o
	./bin/2dconvol.o ./bin/input.txt

clean:
	find . -name "*.o" -type f -delete
