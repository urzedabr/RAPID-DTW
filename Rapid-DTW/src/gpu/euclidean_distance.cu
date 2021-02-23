#include "../../include/euclidean_distance_gpu.h"
#define SIZE 2
#define min2(x,y) (x < y ? x : y)
#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )
#define argmin3(x,y,z) ( x<y ? ( x<z ? 0:2) : (y<z ? 1:2) )

#define INFTY (1E10)

#define NUM_ROWS 45
#define NUM_COLS 23

#define BLOCK_SIZE 32

//naive kernel
__global__ void WeightedLocalCost(float *local_cost_matrix, int *time_local_cost_matrix, float *result, int size, double theta, double alpha, double beta) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < size) {
		float local_cost = local_cost_matrix[idx];
		float time_local_cost = time_local_cost_matrix[idx];
		float logistic_function = 1 / (1 + exp(alpha * (time_local_cost - beta )));
		float weighted_cost = (1 - theta) * local_cost + theta * logistic_function;

		result[idx] = weighted_cost;
	}
}

__device__ float ComputeDtwPath(
		float tiled_accumulated_cost[NUM_ROWS][NUM_COLS], int num_rows,
		int num_cols) {

	int n = num_rows - 1;
	int m = num_cols - 1;
	float cost = 0;

	while (n > 0 && m > 0) {
		int argmin = argmin3(tiled_accumulated_cost[n - 1][m - 1],
				tiled_accumulated_cost[n - 1][m], tiled_accumulated_cost[n][m - 1]);

		if (argmin == 0) {
			n--;
			m--;
		} else if (argmin == 1) {
			n--;
		} else {
			m--;
		}
		cost += tiled_accumulated_cost[n][m];
	}

	return cost;
}

__device__ void DTW_task(float tiled_accumulated_cost[NUM_ROWS][NUM_COLS], float *result,
		int num_rows, int num_cols, int i, int j) {
	float accumulated_cost = 0;
	if (i == 0 && j == 0) {
		accumulated_cost = 0;
	} else if (i == 0 || j == 0) {
		accumulated_cost = INFTY;
	} else {
		float cost = tiled_accumulated_cost[i][j];
		float value_d = tiled_accumulated_cost[i - 1][j - 1];  	//diagonal
		float value_u = tiled_accumulated_cost[i - 1][j];  		//upper
		float value_l = tiled_accumulated_cost[i][j - 1];  		// left
		accumulated_cost = min3( value_d, value_u, value_l) + cost;
	}

	int wi = i + blockIdx.x * num_rows;
	int wj = j + blockIdx.y * num_cols;
	int weighted_local_cost_matrix_num_cols = num_cols * gridDim.y;
	int index = (wi * weighted_local_cost_matrix_num_cols) + wj;
	result[index] = accumulated_cost;
	tiled_accumulated_cost[i][j] = accumulated_cost;
}

__global__ void DtwCostMatrix (float *weighted_local_cost_matrix, float *result, int num_rows, int num_cols, float *cost) {

	__shared__ float tiled_accumulated_cost[NUM_ROWS][NUM_COLS];

	for (int i = 0; i < num_rows; i++) {
		int row = i + blockIdx.x * num_rows;
		int index = threadIdx.x + row * num_cols * gridDim.y + num_cols * blockIdx.y;
		tiled_accumulated_cost[i][threadIdx.x] = weighted_local_cost_matrix[index];
	}

	 __syncthreads();

	 int tid = threadIdx.x;
	 if (tid < num_rows) {

		 // upper diagonal
		 for (int si = 0; si < num_rows; si++) {
			 if (tid <= min2(si, num_cols - 1)) {    // careful with case nx > ny
				 int i = si - tid; // start i position (si) walking up tid times
				 int j = tid;

				 DTW_task(tiled_accumulated_cost, result, num_rows, num_cols, i, j);
			 }
			 __syncthreads();
		 }

		 for (int sj = num_cols - 2; sj >= 0; sj--) {
			 // only the thread within the anti-diagonal region is called
			 if (tid <= min2(sj, num_rows - 1)) {                 // careful with case ny > nx
				 int i = num_rows - tid - 1; // last i - step from cell position (tid) to sj
				 int j = num_cols - (sj - tid) - 1;

				 DTW_task(tiled_accumulated_cost, result, num_rows, num_cols, i, j);
			 }
			 __syncthreads();
		 }
	 }

	 __syncthreads();

	 if (tid == 0)
		 cost[blockIdx.x * gridDim.y + blockIdx.y] = ComputeDtwPath(tiled_accumulated_cost, num_rows, num_cols);

	 __syncthreads();
}

__device__ int getGlobalIndex(int i, int j, int num_rows, int num_cols) {
	int wi = i + blockIdx.x * num_rows;
	int wj = j + blockIdx.y * num_cols;
	int weighted_local_cost_matrix_num_cols = num_cols * gridDim.y;
	int index = (wi * weighted_local_cost_matrix_num_cols) + wj;
	return index;
}

__device__ float ComputeGlobalDtwPath(
		float *result, int num_rows,
		int num_cols) {

	int n = num_rows - 1;
	int m = num_cols - 1;
	float cost = 0;

	while (n > 0 && m > 0) {
		int index_n_1_m_1 = getGlobalIndex(n - 1, m - 1, num_rows, num_cols);
		int index_n_1_m = getGlobalIndex(n - 1, m, num_rows, num_cols);
		int index_n_m_1 = getGlobalIndex(n, m - 1, num_rows, num_cols);

		int argmin = argmin3(result[index_n_1_m_1],
				result[index_n_1_m], result[index_n_m_1]);

		if (argmin == 0) {
			n--;
			m--;
		} else if (argmin == 1) {
			n--;
		} else {
			m--;
		}

		int index_n_m = getGlobalIndex(n, m, num_rows, num_cols);
		cost += result[index_n_m];
	}

	return cost;
}

__device__ void DTWTaskGlobal(float *weighted_local_cost_matrix, float *result,
		int num_rows, int num_cols, int i, int j) {
	int index_ij = getGlobalIndex(i, j, num_rows, num_cols);
	float accumulated_cost = 0;
	if (i == 0 && j == 0) {
		accumulated_cost = 0;
	} else if (i == 0 || j == 0) {
		accumulated_cost = INFTY;
	} else {
		int index_i_1_j_1 = getGlobalIndex(i - 1, j - 1, num_rows, num_cols);
		int index_i_1_j = getGlobalIndex(i - 1, j, num_rows, num_cols);
		int index_i_j_1 = getGlobalIndex(i, j - 1, num_rows, num_cols);

		float cost = weighted_local_cost_matrix[index_ij];
		float value_d = result[index_i_1_j_1];  	//diagonal
		float value_u = result[index_i_1_j];  		//upper
		float value_l = result[index_i_j_1];  		// left
		accumulated_cost = min3( value_d, value_u, value_l) + cost;
	}

	result[index_ij] = accumulated_cost;
}

__global__ void DtwCostMatrixGlobalMemory (float *weighted_local_cost_matrix, float *result, const int num_rows, const int num_cols, float *cost) {

		 //int window = (num_cols / num_threads); //window
		 const int window = (num_cols / blockDim.x);
		 int tid = threadIdx.x;
		 //int tid = blockIdx.x *blockDim.x + threadIdx.x;
		 int base;
		 int aux;
		 int i;
		 int j;
		 const int tidWindow = (tid * window);
		 const int tidWindowaux = tid * (window - 1);
		 //if (tid < num_rows) { //analisar se if pode ser necessário ou não...


				for (int si = 0; si < num_rows; si++) {
					base = tidWindow + (si - tid) * num_cols;
				    aux =  tidWindowaux;
					if (tid <= min2(si, num_cols-1)) {
						//printf("quantidade de janelas no passo %d\n", si);
						for (int index = base; index < base + window; index++) {
							  //printf("A tid %d calcula o elemento %d\n" , tid, index);
						      i = si - tid;
						      j = tid + aux;
						      DTWTaskGlobal(weighted_local_cost_matrix, result, num_rows, num_cols, i, j);
						      aux = aux + 1;

						   // printf("A tid %d calcula o elemento i %d e j %d\n" , tid, i ,j);


						}
					}
					__syncthreads();
				}


				 const int si = (num_rows -1 - tid) * num_cols;
				 aux = 0;
				 int auxj;
				 for (int sj = (num_cols/window) - 2; sj >= 0; sj--) {
					 base = tidWindow + si + window + aux;
					 auxj = 0;
					 aux = aux + window;
					 if (tid <= min2(sj, num_rows - 1)) {
						// printf("quantidade de janelas no passo %d\n", sj);
						  for (int index = base; index < base + window; index++) {
							  //printf("A tid %d calcula o elemento %d\n" , tid, index);
							  i = num_rows - tid - 1;
							  j = num_cols - (window * sj) - window + auxj + tidWindow;
							  DTWTaskGlobal(weighted_local_cost_matrix, result, num_rows, num_cols, i, j);
							  auxj = auxj + 1;
							  //printf("A tid %d calcula o elemento i %d e j %d\n" , tid, i ,j);

					  }

				  }
				  __syncthreads();
			   }
		 //}

		 //__syncthreads();

		 if (tid == 0)
			 cost[blockIdx.x * gridDim.y + blockIdx.y] = ComputeGlobalDtwPath(result, num_rows, num_cols);

		 //__syncthreads();
		}

//naive kernel
__global__ void EuclideanDoyDistances(int *A, int *B, int *C, int sizeA, int sizeB) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	if ((idx < sizeA) && (idy < sizeB)) {
		C[(idx * sizeB) + idy] = __vabsdiffu2(A[idx], B[idy]);
	}
}

//naive kernel
__global__ void EuclideanDistances(float *A, float *B, float *C, int sizeA, int sizeB, int num_cols) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	float result = 0.0f;

	if ((idx < sizeA) && (idy < sizeB)) {
		for (int i = 0; i < num_cols; i++) {
			float temp = A[(idx * num_cols) + i] - B[(idy * num_cols) + i];
			result += temp * temp;
		}
		C[(idx * sizeB) + idy] = sqrtf(result);
	}
}

//optimized kernel
__global__ void EuclideanDistanceFast(const float *A, const float *B, float *C,
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
				C[((i + (bx * CHKSIZE)) * n) + tx] = sqrtf(result[i]);
			tx += blockDim.x;
		} // continue looping across vectors in A
		__syncthreads(); // necessary to prevent warps from racing ahead, if block looping is used
		bx += gridDim.x;
	}
}

void ComputeEuclideanDistances(float *pattern, float *timeseries, float *result, int pattern_num_rows, int timeseries_num_rows, int num_cols) {
	float et2 = 0.0f;
	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	srand((unsigned) time(0));

	//Device Allocation
	float *d_matrixPattern;
	float *d_matrixTimeseries;
	cudaMalloc((void **) &d_matrixPattern, pattern_num_rows * num_cols * sizeof(float));
	cudaMalloc((void **) &d_matrixTimeseries, timeseries_num_rows * num_cols * sizeof(float));

	cudaMemcpy(d_matrixPattern, pattern, pattern_num_rows * num_cols * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixTimeseries, timeseries, timeseries_num_rows * num_cols * sizeof(float),
			cudaMemcpyHostToDevice);

	float *d_results_kernel;
	cudaMalloc((void **) &d_results_kernel, pattern_num_rows * timeseries_num_rows * sizeof(float));

	cudaFuncSetCacheConfig(EuclideanDistances, cudaFuncCachePreferL1);
	dim3 threads(8, 32);   // 1024 threads per block (maximum)
	// Handle cases when the number of threads is greater than the pattern and timeseries size
	int block_dim_x_size = (int) std::ceil((float)pattern_num_rows / threads.x);
	int block_dim_y_size = (int) std::ceil((float)timeseries_num_rows / threads.y);
	dim3 blocks(block_dim_x_size, block_dim_y_size); // assumes evenly divisible
	cudaEventRecord(start2);
	EuclideanDistances<<<blocks, threads>>>(d_matrixPattern, d_matrixTimeseries,
			d_results_kernel, pattern_num_rows, timeseries_num_rows, num_cols);
	cudaEventRecord(stop2);
	cudaMemcpy(result, d_results_kernel, pattern_num_rows * timeseries_num_rows * sizeof(float),
			cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&et2, start2, stop2);

	cudaFree(d_matrixPattern);
	cudaFree(d_matrixTimeseries);
	cudaFree(d_results_kernel);
}

void ComputeEuclideanDistanceDoy(int* pattern_doy, int pattern_doy_size,
		int* timeseries_doy, int timeseries_doy_size, int* result) {
	//Device Allocation
	int *d_matrix_pattern_doy;
	int *d_matrix_timeseries_doy;
	cudaMalloc((void **) &d_matrix_pattern_doy, pattern_doy_size * sizeof(int));
	cudaMalloc((void **) &d_matrix_timeseries_doy, timeseries_doy_size * sizeof(int));
	cudaMemcpy(d_matrix_pattern_doy, pattern_doy, pattern_doy_size * sizeof(int),
				cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_timeseries_doy, timeseries_doy, timeseries_doy_size * sizeof(int),
			cudaMemcpyHostToDevice);

	int *d_results_kernel;
	cudaMalloc((void **) &d_results_kernel, pattern_doy_size * timeseries_doy_size * sizeof(int));

	cudaFuncSetCacheConfig(EuclideanDoyDistances, cudaFuncCachePreferL1);
	dim3 threads(8, 32);   // 1024 threads per block (maximum)
	// Handle cases when the number of threads is greater than the pattern and timeseries size
	int block_dim_x_size = (int) std::ceil((float)pattern_doy_size / threads.x);
	int block_dim_y_size = (int) std::ceil((float)timeseries_doy_size / threads.y);
	dim3 blocks(block_dim_x_size, block_dim_y_size); // assumes evenly divisible
	EuclideanDoyDistances<<<blocks, threads>>>(d_matrix_pattern_doy, d_matrix_timeseries_doy,
			d_results_kernel, pattern_doy_size, timeseries_doy_size);
	cudaMemcpy(result, d_results_kernel, pattern_doy_size * timeseries_doy_size * sizeof(int),
			cudaMemcpyDeviceToHost);

	cudaFree(d_matrix_pattern_doy);
	cudaFree(d_matrix_timeseries_doy);
	cudaFree(d_results_kernel);
}

void ComputeWeightedLocalCostMatrix(float *local_cost_matrix, int *time_local_cost_matrix,
		int input_array_size, float *weighted_local_cost_matrix,
		double theta, double alpha, double beta) {
	//Device Allocation
	float *d_matrix_local_cost;
	int *d_matrix_time_local_cost;
	cudaMalloc((void **) &d_matrix_local_cost, input_array_size * sizeof(float));
	cudaMalloc((void **) &d_matrix_time_local_cost, input_array_size * sizeof(int));
	cudaMemcpy(d_matrix_local_cost, local_cost_matrix, input_array_size * sizeof(float),
				cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_time_local_cost, time_local_cost_matrix, input_array_size * sizeof(int),
			cudaMemcpyHostToDevice);

	float *d_results_kernel;
	cudaMalloc((void **) &d_results_kernel, input_array_size * sizeof(float));

	cudaFuncSetCacheConfig(WeightedLocalCost, cudaFuncCachePreferL1);
	int num_threads_per_block = 512;
	int num_blocks = (input_array_size / num_threads_per_block) + 1;
	WeightedLocalCost<<<num_blocks, num_threads_per_block>>>(d_matrix_local_cost, d_matrix_time_local_cost,
			d_results_kernel, input_array_size, theta, alpha, beta);
	cudaMemcpy(weighted_local_cost_matrix, d_results_kernel, input_array_size * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(d_matrix_local_cost);
	cudaFree(d_matrix_time_local_cost);
	cudaFree(d_results_kernel);
}

void ComputeDTWMatrix(float *weighted_local_cost_matrix,


		int weighted_local_cost_matrix_size, float *result, float *result_cost,
		int num_patterns, int patterns_size, int num_timeseries, int timeseries_size) {
	//Device Allocation

	cudaEvent_t start, stop;
	cudaEvent_t startapp, stopapp;

	float elapsedTime;
	float elapsedTimeapp;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startapp);
	cudaEventCreate(&stopapp);



	float *d_weighted_local_cost_matrix;

	cudaEventRecord(startapp);

	cudaMalloc((void **) &d_weighted_local_cost_matrix,
			weighted_local_cost_matrix_size * sizeof(float));
	cudaMemcpy(d_weighted_local_cost_matrix, weighted_local_cost_matrix,
			weighted_local_cost_matrix_size * sizeof(float),
			cudaMemcpyHostToDevice);

	float *d_results_kernel;
	cudaMalloc((void **) &d_results_kernel,
			weighted_local_cost_matrix_size * sizeof(float));

	float *d_cost;
	cudaMalloc((void **) &d_cost, num_timeseries * num_patterns * sizeof(float));

	cudaFuncSetCacheConfig(DtwCostMatrixGlobalMemory, cudaFuncCachePreferL1);
	int num_threads = min2(patterns_size, timeseries_size)/2;
	printf("tamanho do padrão= %d\n", patterns_size);
	printf("tamanho da serie temporal= %d\n", timeseries_size);
	//printf("numero de threads= %d\n", num_threads);
	printf("quantidade de padrões= %d\n", num_patterns);
	printf("quantidade de séries temporais= %d\n", num_timeseries);
	dim3 blocks(num_patterns, num_timeseries);
	//int blocks =  ((patterns_size /BLOCK_SIZE) +1);
	//printf("numero de blocos= %d\n", blocks);
	//int blocks = ((patterns_size / num_threads) + 1 );
	//nx*ny + BLOCK_SIZE-1)/BLOCK_SIZE
	cudaEventRecord(start);

	DtwCostMatrixGlobalMemory<<<blocks, num_threads>>>(d_weighted_local_cost_matrix,
			d_results_kernel, patterns_size, timeseries_size, d_cost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaMemcpy(result, d_results_kernel,
			weighted_local_cost_matrix_size * sizeof(float),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(result_cost, d_cost, num_timeseries * num_patterns * sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(d_cost);
	cudaFree(d_results_kernel);
	cudaFree(d_weighted_local_cost_matrix);

	cudaEventRecord(stopapp);
	cudaEventSynchronize(stopapp);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventElapsedTime(&elapsedTimeapp, startapp, stopapp);

	//printf("%f" , (float) (elapsedTime) / 1000.0);
	//printf("GPU Application Time: %f seconds \n", (float) (elapsedTimeapp) / 1000.0);

}



