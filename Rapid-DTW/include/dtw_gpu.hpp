/*
 * dtw_cpu.hpp
 *
 *  Created on: 24/07/2018
 *      Author: savio
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <cmath>
#include <stdlib.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <dirent.h>
using namespace std;
using namespace boost::gregorian;

typedef std::vector<std::string> stringvec;


float* ComputeGpuDtw(int* pattern_doy, int pattern_doy_size,
		int* timeseries_doy, int timeseries_doy_size, float* pattern,
		int pattern_rows_size, float* timeseries, int timeseries_rows_size,
		int num_cols, int num_patterns, int patterns_size, int num_timeseries, int timeseries_size);

