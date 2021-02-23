// reading a text file
#include "../../include/dtw_cpu.hpp"

void VectorDistance(vector<vector<double>>& pattern,
		vector<vector<double>>& timeseries, vector<vector<double>>& local_cost_matrix) {
	for (vector<double> pattern_series : pattern) {
		vector<double> local_cost;
		for (vector<double> series : timeseries) {
			vector<double>::iterator it1 = series.begin();
			vector<double>::iterator it2 = pattern_series.begin();
			double distance = 0;
			for (; it1 != series.end() && it2 != pattern_series.end();
					++it1, ++it2) {
				distance += pow(*it2 - *it1, 2);
			}
			distance = sqrt(distance);
			local_cost.push_back(distance);
		}
		local_cost_matrix.push_back(local_cost);
	}
}

void DoyVectorDistance(vector<int>& pattern_day_of_year,
		vector<int>& timeseries_day_of_year, vector<vector<int>>& time_cost_matrix) {
	for (int doy_pattern : pattern_day_of_year) {
		vector<int> time_local_cost;
		for (int doy_series : timeseries_day_of_year) {
			time_local_cost.push_back(sqrt(pow(doy_pattern - doy_series, 2)));
		}
		time_cost_matrix.push_back(time_local_cost);
	}
}

template<typename T>
void PrintSeries(vector<vector<T>>& timeseries) {
	for (vector<T> series : timeseries) {
		for (T value : series) {
			cout << value << ";";
		}
		cout << endl;
	}

	cout << endl << endl;
}

void ComputeWeightedLocalCostMatrix(vector<vector<double>>& local_cost_matrix,
		vector<vector<int>>& time_local_cost_matrix, vector<vector<double>>& weighted_local_cost_matrix,
		double theta, double alpha, double beta) {

	for (unsigned i = 0; i < local_cost_matrix.size();i++) {

		vector<double> local_cost_array = local_cost_matrix[i];
		vector<int> time_local_cost_array = time_local_cost_matrix[i];
		vector<double> weighted_cost_array;

		for (unsigned j = 0; j < local_cost_array.size();j++) {
			double local_cost = local_cost_array[j];
			double time_local_cost = time_local_cost_array[j];
			double logistic_function = 1 / (1 + exp(alpha * (time_local_cost - beta )));
			double weighted_cost = (1 - theta) * local_cost + theta * logistic_function;
			weighted_cost_array.push_back(weighted_cost);
		}

		weighted_local_cost_matrix.push_back(weighted_cost_array);
	}
}

int Argmin(vector<double> v) {
	return std::distance(v.begin(), std::min_element(v.begin(), v.end()));
}

double ComputeDtwPath(vector<vector<double>>& weighted_local_cost_matrix,
		vector<vector<double> >& accumulated_cost, int patterns_size,
		int offset_pattern, int timeseries_size, int offset_timeseries) {

	int n_index = patterns_size - 1;
	int n = offset_pattern + n_index;

	int m_index = timeseries_size - 1;
	int m = offset_timeseries + m_index;

	double cost = 0;

	while (n_index > 0 && m_index > 0) {
		int argmin = Argmin(
				{ accumulated_cost[n - 1][m - 1], accumulated_cost[n - 1][m],
						accumulated_cost[n][m - 1] });

		if (argmin == 0) {
			n--;
			m--;
			n_index--;
			m_index--;
		} else if (argmin == 1) {
			n--;
			n_index--;
		} else {
			m--;
			m_index--;
		}

		cost += accumulated_cost[n][m];
	}

	return cost;
}

void ComputeCostMatrix(vector<vector<double>>& weighted_local_cost_matrix,
		vector<vector<double>>& accumulated_cost, vector<vector<int>>& direction_matrix,
		vector<vector<int>>& starting_matrix, int patterns_size, int offset_pattern, int timeseries_size, int offset_timeseries) {

	for (unsigned j = offset_timeseries;
				j < offset_timeseries + timeseries_size; j++) {
		accumulated_cost[offset_pattern][j] = weighted_local_cost_matrix[offset_pattern][j];
		starting_matrix[offset_pattern][j] = offset_timeseries - j + 1;
	}

	for (unsigned i = offset_pattern + 1; i < offset_pattern + patterns_size;
			i++) {
		accumulated_cost[i][offset_timeseries] = weighted_local_cost_matrix[i][offset_timeseries] + accumulated_cost[i - 1][offset_timeseries];
		direction_matrix[i][offset_timeseries] = 3;
		starting_matrix[i][offset_timeseries] = 1;
	}

	direction_matrix[offset_pattern][offset_timeseries] = 3; //first element is 3
	starting_matrix[offset_pattern][offset_timeseries] = 1; //first element is 1

	for (unsigned i = offset_pattern + 1; i < offset_pattern + patterns_size;
			i++) {
		for (unsigned j = offset_timeseries + 1;
				j < offset_timeseries + timeseries_size; j++) {
			double cost = weighted_local_cost_matrix[i][j];
			double min_cost = min(
					min(accumulated_cost[i - 1][j - 1],
							accumulated_cost[i - 1][j]),
					accumulated_cost[i][j - 1]);

			//Direction matrix
			int argmin = Argmin(
							{ accumulated_cost[i - 1][j - 1], accumulated_cost[i - 1][j],
									accumulated_cost[i][j - 1] });
			if (argmin == 0) {
				direction_matrix[i][j] = 1;
			} else if (argmin == 1) {
				direction_matrix[i][j] = 3;
			} else {
				direction_matrix[i][j] = 2;
			}

			accumulated_cost[i][j] = min_cost + cost;
		}
	}
}

vector<double> Dtw(vector<vector<double>>& weighted_local_cost_matrix, vector<vector<double>>& accumulated_cost, vector<vector<int>>& direction_matrix,
		vector<vector<int>>& starting_matrix, int num_patterns, int patterns_size, int num_timeseries, int timeseries_size) {

	vector<double> cost_list;

	for (int index_pattern = 0; index_pattern < num_patterns; index_pattern++) {
		for (int index_timeseries = 0; index_timeseries < num_timeseries; index_timeseries++) {
			int offset_pattern = index_pattern * patterns_size;
			int offset_timeseries = index_timeseries * timeseries_size;
			ComputeCostMatrix(weighted_local_cost_matrix, accumulated_cost, direction_matrix, starting_matrix, patterns_size, offset_pattern, timeseries_size, offset_timeseries);
			double cost = ComputeDtwPath(weighted_local_cost_matrix, accumulated_cost, patterns_size, offset_pattern, timeseries_size, offset_timeseries);
			cost_list.push_back(cost);
		}
	}

	return cost_list;
}

vector<double> ComputeCpuDtw(vector<int>& pattern_day_of_year, vector<vector<double>>& pattern,vector<int>& timeseries_day_of_year, vector<vector<double>>& timeseries,
		int num_patterns, int patterns_size, int num_timeseries, int timeseries_size) {
	const clock_t begin_total_time = clock();

	const clock_t begin_matrix_construction_time = clock();

	// compute local cost matrix
	const clock_t begin_time1 = clock();
	vector<vector<double>> local_cost_matrix; //phi
	VectorDistance(pattern, timeseries, local_cost_matrix);
	int end_time1 = float( clock () - begin_time1 )  * 1000 /  CLOCKS_PER_SEC;

	// compute time local cost matrix
	const clock_t begin_time2 = clock();
	vector<vector<int>> time_local_cost_matrix; //psi
	DoyVectorDistance(pattern_day_of_year, timeseries_day_of_year, time_local_cost_matrix);
	int end_time2 = float( clock () - begin_time2 )  * 1000 /  CLOCKS_PER_SEC;

	// compute cost matrix
	const clock_t begin_time3 = clock();
	vector<vector<double>> weighted_local_cost_matrix; //cm
	double theta = 0.5;
	double alpha = -0.1;
	double beta = 50;
	ComputeWeightedLocalCostMatrix(local_cost_matrix, time_local_cost_matrix, weighted_local_cost_matrix, theta, alpha, beta);
	int end_time3 = float( clock () - begin_time3 )  * 1000 /  CLOCKS_PER_SEC;
	//end cost matrix
	int end_matrix_construction_time = float( clock () - begin_matrix_construction_time )  * 1000 /  CLOCKS_PER_SEC;

	// compute DTW
	const clock_t begin_time4 = clock();
	vector<vector<double>> accumulated_cost(weighted_local_cost_matrix.size(), vector<double>(weighted_local_cost_matrix[0].size(), 0));
	vector<vector<int>> direction_matrix(weighted_local_cost_matrix.size(), vector<int>(weighted_local_cost_matrix[0].size(), 1)); //dm
	vector<vector<int>> starting_matrix(weighted_local_cost_matrix.size(), vector<int>(weighted_local_cost_matrix[0].size(), 1)); //vm
	vector<double> cost = Dtw(weighted_local_cost_matrix, accumulated_cost, direction_matrix, starting_matrix, num_patterns, patterns_size, num_timeseries, timeseries_size);
	int end_time4 = float( clock () - begin_time4 )  * 1000 /  CLOCKS_PER_SEC;

	int end_time = float( clock () - begin_total_time )  * 1000 /  CLOCKS_PER_SEC;

	return cost;
}
