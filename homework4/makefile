

all:
	gcc -mavx2 -O0 -o vectorizedMinSearch vectorizedMinSearch.c
	gcc -mavx2 -O0 -o convolution convolution.c
	gcc -mavx2 -O0 -o prefixSum prefixSum.c
	gcc -mavx2 -O0 -o matrixVectorMul matrixVectorMul.c
	gcc -mavx2 -O0 -o stringProcessing stringProcessing.c
	gcc -mavx2 -O0 -o imageProcessing imageProcessing.c 
	ulimit -s unlimited

test_string:
	./stringProcessing string.txt 

test_image:
	./imageProcessing inputImage.bmp
