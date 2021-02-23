#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// both M and N must be evenly divisible by SIZE, M must be evenly divisible by CHKSIZE
#define SIZE 2
#define N 2
#define M 3
#define CHKSIZE 1

//naive kernel
__global__ void EuclideanDistancesNaive(float *A, float *B, float *C, int n,
		int m) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	float result = 0.0f;

	if ((idx < n) && (idy < m)) {
		for (int i = 0; i < SIZE; i++) {
			float temp = A[(idx * SIZE) + i] - B[(idy * SIZE) + i];
			result += temp * temp;
		}
		C[(idx * m) + idy] = result;
	}
}
//optimized kernel
__global__ void EuclideanDistancesFast(const float *A, const float *B, float *C,
		const int n, const int m) {
	// n, A,  4000 this kernel assumes A is column-major A(SIZE, n)
	// m, B, 20000 this kernel assumes B is row-major    B(m, SIZE)
	// this kernel assumes C is column-major             C(m,n)
	// this kernel assumes number of threads per threadblock == SIZE
	// CHKSIZE is the number of B vectors that will be compute per block
	__shared__ float my_sB[CHKSIZE * SIZE]; // enough shared storage for CHKSIZE vectors of B
	int bx = blockIdx.x; // one block per CHKSIZE rows of B (the larger input matrix)
	while ((bx * CHKSIZE) < m) { // not used, this while loop could be used to extend a block to multiple chunks
		int tx = threadIdx.x;
		for (int i = 0; i < CHKSIZE; i++) // load vectors of B into shared memory
			my_sB[(i * SIZE) + tx] = B[(((bx * CHKSIZE) + i) * SIZE) + tx];
		__syncthreads();
		while (tx < n) {  //loop across all vectors in A
			float result[CHKSIZE];
			for (int i = 0; i < CHKSIZE; i++)
				result[i] = 0.0f;
			for (int i = 0; i < SIZE; i++) {
				float Atemp = A[(n * i) + tx];
				for (int j = 0; j < CHKSIZE; j++) { // compute all CHKSIZE B vectors with read of A
					float temp = Atemp - my_sB[i + (j * SIZE)];
					result[j] += temp * temp;
				}
			}
			for (int i = 0; i < CHKSIZE; i++) // store CHKSIZE results
				C[((i + (bx * CHKSIZE)) * n) + tx] = result[i];
			tx += blockDim.x;
		} // continue looping across vectors in A
		__syncthreads(); // necessary to prevent warps from racing ahead, if block looping is used
		bx += gridDim.x;
	}
}

float comp_euclid_sq(const float *rA, const float *rB, const int size) {

	float result = 0.0f;
	float temp;
	for (int i = 0; i < size; i++) {
		temp = (rA[i] - rB[i]);
		result += temp * temp;
	}
	return result;
}

int main_ed_gpu() {
	float cpu_time = 0.0f, et1 = 0.0f, et2 = 0.0f, et_mem = 0.0f;
	cudaEvent_t start1, start2, stop1, stop2, start_mem_copy, stop_mem_copy;
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&start_mem_copy);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	cudaEventCreate(&stop_mem_copy);

	int n = N;  //MatrixA size : n * SIZE
	int m = M; //MatrixB size : m * SIZE

	srand((unsigned) time(0));

	// Host Allocations
	float *matrixA = (float *) malloc(n * SIZE * sizeof(float));
	for (int i = 0; i < n * SIZE; i++)
		matrixA[i] = (float) i + 1;

	float *matrixB = (float *) malloc(m * SIZE * sizeof(float));
	for (int i = 0; i < m * SIZE; i++)
		matrixB[i] = (float) i + i;

	const clock_t begin_time = clock();
	float *results_kernel = (float *) malloc(n * m * sizeof(float));
	float *cpu_results_kernel = (float *) malloc(n * m * sizeof(float));
	for (int i = 0; i < n * m; i++)
		cpu_results_kernel[i] = comp_euclid_sq(matrixA + ((i / m) * SIZE),
				matrixB + (i % m) * SIZE, SIZE);
	cpu_time = float( clock () - begin_time ) /  1000;

	//Device Allocation
	cudaEventRecord(start_mem_copy);
	float *d_matrixA;
	float *d_matrixB;
	cudaMalloc((void **) &d_matrixA, n * SIZE * sizeof(float));
	cudaMalloc((void **) &d_matrixB, m * SIZE * sizeof(float));
	cudaMemcpy(d_matrixA, matrixA, n * SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, matrixB, m * SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaEventElapsedTime(&et_mem, start_mem_copy, stop_mem_copy);

	float *d_results_kernel;
	cudaMalloc((void **) &d_results_kernel, n * m * sizeof(float));

	cudaFuncSetCacheConfig(EuclideanDistancesNaive, cudaFuncCachePreferL1);
	dim3 threads3(8, 32);   // 1024 threads per block (maximum)
	dim3 blocks3(n / threads3.x, m / threads3.y); // assumes evenly divisible
	cudaEventRecord(start1);
	EuclideanDistancesNaive<<<blocks3, threads3>>>(d_matrixA, d_matrixB,
			d_results_kernel, n, m);
	cudaEventRecord(stop1);
	cudaMemcpy(results_kernel, d_results_kernel, n * m * sizeof(float),
			cudaMemcpyDeviceToHost);
//	for (int i = 0; i < n * m; i++) {
//		if (results_kernel[i] != cpu_results_kernel[i]) {
//			printf("cpu/kernel3 mismatch at %d, cpu: %f, kernel3: %f\n", i,
//					cpu_results_kernel[i], results_kernel[i]);
//			return 1;
//		}
//	}
	cudaMemset(d_results_kernel, 0, n * m * sizeof(float));
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&et1, start1, stop1);

	// transpose matrix A
	float *matrixA_T = (float *) malloc(n * SIZE * sizeof(float));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < SIZE; j++)
			matrixA_T[(j * n) + i] = matrixA[(i * SIZE) + j];
	cudaMemcpy(d_matrixA, matrixA_T, n * SIZE * sizeof(float),
			cudaMemcpyHostToDevice);

	cudaFuncSetCacheConfig(EuclideanDistancesFast, cudaFuncCachePreferL1);
	dim3 threads4(SIZE); // one thread per vector element
	dim3 blocks4(m / CHKSIZE);
	cudaEventRecord(start2);
	EuclideanDistancesFast<<<blocks4, threads4>>>(d_matrixA, d_matrixB,
			d_results_kernel, n, m);
	cudaEventRecord(stop2);
	cudaMemcpy(results_kernel, d_results_kernel, n * m * sizeof(float),
			cudaMemcpyDeviceToHost);
	// test for correct transposed result C(m,n)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%f ", results_kernel[(j * n) + i]);
			if (results_kernel[(j * n) + i]
					!= cpu_results_kernel[(i * m) + j]) {
				printf("cpu/kernel4 mismatch at %d,%d, cpu: %f, kernel4: %f\n",
						i, j, cpu_results_kernel[(i * m) + j],
						results_kernel[(j * n) + i]);
				return 1;
			}
		}
		printf("\n");
	}
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&et2, start2, stop2);
	cudaFree(d_results_kernel);

	printf("Success!\n");
	printf("CPU: %.fms, kernel1 : %.fms, kernel2 : %.fms, Mem copy: %.fms\n",
			cpu_time, et1, et2, et_mem);

	free(matrixA);
	free(matrixB);
	free(results_kernel);

	return 0;
}
