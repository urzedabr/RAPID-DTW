/*
 * dtw_cpu.hpp
 *
 *  Created on: 24/07/2018
 *      Author: savio
 */

#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
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

vector<double> ComputeCpuDtw(vector<int>& pattern_day_of_year, vector<vector<double>>& pattern,vector<int>& timeseries_day_of_year, vector<vector<double>>& timeseries,
		int num_patterns, int patterns_size, int num_timeseries, int timeseries_size);
