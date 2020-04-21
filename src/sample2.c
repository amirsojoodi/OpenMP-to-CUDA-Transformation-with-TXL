#*******************************************************************
# sample.c
# 
# A simple array initialization in OpenMP
#*******************************************************************

#include<stdlib.h>
#include<omp.h>
#include<stdio.h>

#define SIZE 10000

int main(){

	int *array;
	int *array2;
    int *array3;
	int size = SIZE;
	array = (int *)malloc(size * sizeof(int));
	array2 = (int *)malloc(size * sizeof(int));
	array3 = (int *)malloc(size * sizeof(int));

	#pragma omp parallel
	for(int i = 0; i < size; i++){
		array[i] = i;
        array2[i] = i+1;
        array2[i] = i+1;
	}

	// A simple validity test
	printf("Array[%d] = %d", size - 1, array[size - 1]);

	free(array);
    free(array2);
    free(array3);

	return 0;
}

