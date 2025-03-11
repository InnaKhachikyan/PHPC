

all:
	gcc -mavx2 -O0 -o vectorizedMinSearch vectorizedMinSearch.c
	gcc -mavx2 -O0 -o convolution convolution.c
	gcc -mavx2 -O0 -o prefixSum prefixSum.c
	gcc -mavx2 -O0 -o matrixVectorMul matrixVectorMul.c
	ulimit -s unlimited

test_min:
	./vectorizedMinSearch

test_conv:
	./convolution

test_prefix:
	./prefixSum

test_mult:
	./matrixVectorMul