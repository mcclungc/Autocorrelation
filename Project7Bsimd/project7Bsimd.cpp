//Connie McClung
//CS475 Spring 2019 
//Project 7B - SIMD Implementation
//June 10, 2019

#include <omp.h>
#include "simd.p4.h"
#include <stdio.h>
#include <math.h>


int main(int argc, char* argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "OpenMp is not supported here -- sorry.\n");
	return 1;
#endif

	//read signal file into array
	FILE *fp = fopen("signal.txt", "r");
	if (fp == NULL)
	{
		fprintf(stderr, "Cannot open file 'signal.txt'\n");
		return 1;
	}
	int Size;
	fscanf(fp, "%d", &Size);
	float *Array = new float[2 * Size];
	float *Sums = new float[1 * Size];
	for (int i = 0; i < Size; i++)
	{
		fscanf(fp, "%f", &Array[i]);
		Array[i + Size] = Array[i];		// duplicate the array
	}
	fclose(fp);

	// track  the maximum performance
	double maxPerformance = 0.;

	//loop to get a reduction for every index in Sums

	double time0 = omp_get_wtime(); //start timing
	for (int shift = 0; shift < Size; shift++)
	{
		Sums[shift] = SimdMulSum(&Array[0], &Array[0 + shift], Size);

	}
	double time1 = omp_get_wtime();

	//calculate performance
	double megaMultReductsPerSecond = (double)Size *(double)Size/ (time1 - time0) / 1000000.;
	if (megaMultReductsPerSecond > maxPerformance)
	{
		maxPerformance = megaMultReductsPerSecond;
	}

	//print for entries in Sums: i, Sums[i]
	printf("Shift\tSum\n");
	//for (int i = 0; i < Size; i++)
	for (int i = 1; i < 513; i++)
	{
		printf("%d\t%10.6f\n", i, Sums[i]);
	}

	printf("Size: %d\tMaxPerformance: %10.6lf\n", Size,  maxPerformance);



	return 0;
}
