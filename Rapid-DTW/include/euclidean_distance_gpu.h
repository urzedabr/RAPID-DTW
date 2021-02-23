// both M and N must be evenly divisible by SIZE, M must be evenly divisible by CHKSIZE
#define CHKSIZE 1
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void ComputeEuclideanDistances(float *pattern, float *timeseries, float *result,
		int pattern_num_rows, int timeseries_num_rows, int num_cols);
void ComputeEuclideanDistanceDoy(int* pattern_doy, int pattern_doy_size,
		int* timeseries_doy, int timeseries_doy_size, int* result);
void ComputeWeightedLocalCostMatrix(float *local_cost_matrix,
		int *time_local_cost_matrix, int input_array_size,
		float *weighted_local_cost_matrix, double theta, double alpha,
		double beta);
void ComputeDTWMatrix(float *weighted_local_cost_matrix,
		int weighted_local_cost_matrix_size, float *result, float *result_cost,
		int num_patterns, int patterns_size, int num_timeseries, int timeseries_size);
