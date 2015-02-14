CC=g++
FLAGS=-g
SRC=./src/*.*

2dconvol.o:$(SRC)
	$(CC) $(FLAGS) -o ./bin/2dconvol.o $(SRC)

run: 2dconvol.o
	./bin/2dconvol.o ./bin/input.txt

clean:
	rm -rf ./bin/2dconvol.o
