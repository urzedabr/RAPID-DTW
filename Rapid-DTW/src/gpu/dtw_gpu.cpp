// reading a text file
#include "../../include/dtw_gpu.hpp"
#include "../../include/euclidean_distance_gpu.h"


void printSeries(int* pattern_doy, int pattern_doy_size, int* timeseries_doy,
		int timeseries_doy_size, float* pattern, int pattern_rows_size,
		float* timeseries, int timeseries_rows_size, int num_cols) {
	cout << "Pattern day of year: " << endl;
	for (int i = 0; i < pattern_doy_size; i++) {
		cout << pattern_doy[i] << " ";
	}

	cout << "\n\nTimeseries day of year: " << endl;
	for (int i = 0; i < timeseries_doy_size; i++) {
		cout << timeseries_doy[i] << " ";
	}

	cout << "\n\nPattern: " << endl;
	for (int i = 0; i < pattern_rows_size; i++) {
		for (int j = 0; j < num_cols; j++) {
			cout << pattern[i * num_cols + j] << " ";
		}
		cout << endl;
	}

	cout << "\n\nTimeseries: " << endl;
	for (int i = 0; i < timeseries_rows_size; i++) {
		for (int j = 0; j < num_cols; j++) {
			cout << timeseries[i * num_cols + j] << " ";
		}
		cout << endl;
	}
}

float* ComputeGpuDtw(int* pattern_doy, int pattern_doy_size,
		int* timeseries_doy, int timeseries_doy_size, float* pattern,
		int pattern_rows_size, float* timeseries, int timeseries_rows_size,
		int num_cols, int num_patterns, int patterns_size, int num_timeseries, int timeseries_size) {

	const clock_t begin_total_time = clock();

	const clock_t begin_matrix_construction_time = clock();

	const clock_t begin_time1 = clock();
	float *result = (float *) malloc(pattern_rows_size * timeseries_rows_size * sizeof(float));
	ComputeEuclideanDistances(pattern, timeseries, result, pattern_rows_size, timeseries_rows_size, num_cols);
	int end_time1 = float( clock () - begin_time1 ) * 1000 /  CLOCKS_PER_SEC;

	const clock_t begin_time2 = clock();
	int *result_doy = (int *) malloc(pattern_doy_size * timeseries_doy_size * sizeof(int));
	ComputeEuclideanDistanceDoy(pattern_doy, pattern_doy_size, timeseries_doy, timeseries_doy_size, result_doy);
	int end_time2 = float( clock () - begin_time2 )  * 1000 /  CLOCKS_PER_SEC;

	const clock_t begin_time3 = clock();
	float *result_cost = (float *) malloc(pattern_doy_size * timeseries_doy_size * sizeof(float));
	double theta = 0.5;
	double alpha = -0.1;
	double beta = 50;
	ComputeWeightedLocalCostMatrix(result,result_doy,pattern_doy_size * timeseries_doy_size,result_cost,theta, alpha, beta);
	int end_time3 = float( clock () - begin_time3 )  * 1000 /  CLOCKS_PER_SEC;
	int end_matrix_construction_time = float( clock () - begin_matrix_construction_time )  * 1000 /  CLOCKS_PER_SEC;

	const clock_t begin_time4 = clock();
	float *result_dtw = (float *) malloc(pattern_doy_size * timeseries_doy_size * sizeof(float));
	float *cost_list = (float *) malloc(num_patterns * num_timeseries * sizeof(float));
	ComputeDTWMatrix(result_cost, pattern_doy_size * timeseries_doy_size, result_dtw, cost_list, num_patterns, patterns_size, num_timeseries, timeseries_size);
	int end_time4 = float( clock () - begin_time4 )  * 1000 /  CLOCKS_PER_SEC;

	int end_time = float( clock () - begin_total_time )  * 1000 /  CLOCKS_PER_SEC;

	free(result);
	free(result_doy);
	free(result_cost);
	free(result_dtw);

	return cost_list;
}
