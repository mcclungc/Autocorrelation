//Connie McClung
//CS475 Spring 2019 
//Project 7B - Open MP Implementation
//June 10, 2019


#include <omp.h>
#include <stdio.h>
#include <math.h>


// setthe number of threads:
#ifndef NUMT
//#define NUMT  1 			
#define NUMT  4 			
#endif


int main()
{
	//check for openmp support
#ifndef _OPENMP
	fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
	return 1;
#endif

	//read signal file into array
	FILE *fp = fopen( "signal.txt", "r" );
	if (fp == NULL)
	{
		fprintf( stderr, "Cannot open file 'signal.txt'\n" );
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


	omp_set_num_threads(NUMT);

	// track  the maximum performance
	double maxPerformance = 0.;

	//loop to get a reduction for every index in Sums

	double time0 = omp_get_wtime(); //start timing
	
#pragma omp parallel for default(none), shared(Array, Size, Sums)
	for (int shift = 0; shift < Size; shift++)
	{
		float sum = 0.;

		for(int i = 0; i < Size; i++)
		{
			sum += Array[i] * Array[i + shift];
		}

		Sums[shift] = sum;
	}

	double time1 = omp_get_wtime();

	//calculate performance
	double megaMultReductsPerSecond = (double)Size *(double)Size / (time1 - time0) / 1000000.;
	if (megaMultReductsPerSecond > maxPerformance)
	{
		maxPerformance = megaMultReductsPerSecond;
	}

	//print entries in Sums: i, Sums[i]
	printf("Shift\tSum\n");
	//for (int i = 0; i < Size; i++)
	for (int i = 1; i < 513; i++)
	{
		printf("%d\t%10.6f\n", i, Sums[i]);
	}

	printf("Threads: %d\tMaxPerformance: %10.6lf MegaMultReducts/sec\n",NUMT, maxPerformance);

	return 0;

}
