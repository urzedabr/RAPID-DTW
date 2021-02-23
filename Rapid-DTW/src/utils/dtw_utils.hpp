#include <boost/algorithm/string/split.hpp> // Include for boost::split
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <chrono>
#include <boost/date_time/gregorian/gregorian.hpp>

void GetTestSeriesData(vector<int>& pattern_day_of_year,
		vector<int>& timeseries_day_of_year, vector<vector<double>>& pattern,
		vector<vector<double>>& timeseries, vector<float>& pattern_gpu,
		vector<float>& timeseries_gpu, int num_patterns, int num_timeseries,
		int pattern_rows_size, int timeseries_rows_size, int num_cols) {

	for (int n = 0; n < num_patterns; n++)
		for (int i = 0; i < pattern_rows_size; i++) {
			pattern_day_of_year.push_back(i + 1);
		}

	for (int n = 0; n < num_timeseries; n++)
		for (int i = 0; i < timeseries_rows_size; i++) {
			timeseries_day_of_year.push_back(i + i);
		}

	int count = 0;
	for (int n = 0; n < num_patterns; n++) {
		count = 0;
		for (int i = 0; i < pattern_rows_size; i++) {
			vector<double> line_vector;
			for (int j = 0; j < num_cols; j++) {
				double value = ++count;
				line_vector.push_back(value);
				pattern_gpu.push_back((float) value);
			}
			pattern.push_back(line_vector);
		}
	}

	for (int n = 0; n < num_timeseries; n++) {
		count = 0;
		for (int i = 0; i < timeseries_rows_size; i++) {
			vector<double> line_vector;
			for (int j = 0; j < num_cols; j++) {
				double value = count + count;
				line_vector.push_back(value);
				timeseries_gpu.push_back((float) value);
				count++;
			}
			timeseries.push_back(line_vector);
		}
	}
}

int GetSize(string file_name) {
	int size = 0;
	ifstream myfile(file_name.c_str());
	string line;

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			size++;
		}
		myfile.close();
	} else
		cout << "Unable to open file";

	myfile.close();

	return size;
}


int GetTimeSeriesToTest(string file_name, vector<int>& day_of_year,
		vector<vector<double>>& timeseries, vector<float>& timeseries_gpu,
		int timeseries_rows_size) {

	// generate day of year increment the ith term.
	for (int i = 0; i < timeseries_rows_size; i++) {
		day_of_year.push_back(i + 1);
	}


	int total_size = GetSize(file_name);
	ifstream myfile(file_name.c_str());
	string line;
	vector<string> lines;

	// read the time series file base
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			lines.push_back(line);
		}
	} else
		cout << "Unable to open file";
	myfile.close();

	int size = 0;
	int lines_index = 0;

	while(size < timeseries_rows_size) {
		string ts_line = lines[lines_index];
		// Split input line
		vector<string> strs;
		boost::split(strs, ts_line, boost::is_any_of(";"));

		// Put time series data into line_vector
		vector<double> line_vector;
		for (size_t i = 1; i < strs.size(); i++) {
			double value = atof(strs[i].c_str());
			line_vector.push_back(value);
			timeseries_gpu.push_back((float) value);
		}
		timeseries.push_back(line_vector);

		lines_index++;
		if (lines_index >= total_size)
			lines_index = 0;

		size++;
	}

	return size;
}

int GetSeriesData(string file_name, vector<int>& day_of_year,
		vector<vector<double>>& pattern, vector<float>& pattern_gpu, bool controlSerieSize) {

	int total_size = GetSize(file_name);
	int size = 0;
	ifstream myfile(file_name.c_str());
	string line;

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			size++;
			// Split input line
			vector<string> strs;
			boost::split(strs, line, boost::is_any_of(";"));

			// Get the day of year
			boost::gregorian::date d(
					from_simple_string(
							boost::replace_all_copy(strs[0], "\"", "")));
			day_of_year.push_back(d.day_of_year());

			// Put timeseries data into line_vector
			vector<double> line_vector;
			for (size_t i = 1; i < strs.size(); i++) {
				double value = atof(strs[i].c_str());
				line_vector.push_back(value);
				pattern_gpu.push_back((float) value);
			}
			pattern.push_back(line_vector);

			if (size == 23 && controlSerieSize)
				break;

			if (size == 22 && size == total_size && controlSerieSize) {
				for (size_t i = 1; i < strs.size(); i++) {
					double value = atof(strs[i].c_str());
					pattern_gpu.push_back((float) value);
				}
				pattern.push_back(line_vector);
				day_of_year.push_back(d.day_of_year() + 16);
				size++;
			}

			if (size == 21 && size == total_size && controlSerieSize) {
				for (int j = 0; j < 2; j++) {
					for (size_t i = 1; i < strs.size(); i++) {
						double value = atof(strs[i].c_str());
						pattern_gpu.push_back((float) value);
					}
					pattern.push_back(line_vector);
					day_of_year.push_back(d.day_of_year() + 16);
					size++;
				}
			}
		}
		myfile.close();
	} else
		cout << "Unable to open file";

	myfile.close();

	return size;
}

std::string basename(const std::string& pathname)
{
    return {std::find_if(pathname.rbegin(), pathname.rend(),
                         [](char c) { return c == '/'; }).base(),
            pathname.end()};
}
