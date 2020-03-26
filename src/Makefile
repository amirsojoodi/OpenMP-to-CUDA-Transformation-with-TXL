#*******************************************************************
# Makefile
# 
# Used to build the project.
#*******************************************************************

nvcc_options= --compiler-options -Wall 
gcc_options= -fopenmp -Wall 

all: sampleCuda sampleOpenMP

sampleCuda:  Makefile sample.cu
	nvcc -o sampleCuda sample.cu $(nvcc_options)

sampleOpenMP: Makefile sample.c
	gcc -o sampleOpenMP sample.c $(gcc_options)

clean:
	rm sampleCuda sampleOpenMP
