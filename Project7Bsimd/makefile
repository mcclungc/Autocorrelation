simd.p4.o:	simd.p4.h  simd.p4.cpp
		g++  -c  simd.p4.cpp -o simd.p4.o

project7Bsimd:	project7Bsimd.cpp simd.p4.o
		g++  -o project7Bsimd project7Bsimd.cpp simd.p4.o -lm -fopenmp
