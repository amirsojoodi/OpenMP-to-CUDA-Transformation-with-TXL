//*****************************************************************************
// sample2.c
//
// A simple program in OpenMP
//
// Project Description: A TXL transformation from OpenMP C sources to CUDA
// equivalent.
//
// For more information on TXL, visit: txl.ca
// Authors: AmirHossein Sojoodi, Nicolas Merz
// Course: ELEC-875 2020, Tom Dean
// Queen's University
//*****************************************************************************

#include<stdlib.h>
#include<omp.h>
#include<stdio.h>

#define SIZE 10000
#define DUMMY 1369

int main(){

	int *array1;
	int *array2;
	float *array3;
	float *array4;
	int size = SIZE;
	array1 = (int *)malloc(size * sizeof(int));
	array2 = (int *)malloc(size * sizeof(int));
	array3 = (float *)malloc(size * sizeof(int));
	array4 = (float *)malloc(size * sizeof(float));
	
	#pragma omp parallel
	for(int i = 0; i < size; i++){
		array1[i] = i;
        array2[i] = array1[i] + 1;
        array2[i] = array2[i] + DUMMY;
		array3[i] = array2[i] / (array1[i] + 0.5) ;
	}

	// A simple validity test
	printf("Array3[%d] = %f", size - 1, array3[size - 1]);

	free(array1);
    free(array2);
    free(array3);
	free(array4);

	return 0;
}

