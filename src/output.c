#*******************************************************************
# sample.c
# 
# A simple array initialization in OpenMP
#*******************************************************************
#include<stdlib.h>
#include<omp.h>
#include<stdio.h>
#define SIZE 10000
#define foo 10

int main () {
    int *array;
    int size = SIZE;
    array = (int *) cudaMallocManaged (size * sizeof (int));
#pragma omp parallel    
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
    printf ("Array[%d] = %d", size - 1, array [size - 1]);
    free (array);
    return 0;
}

