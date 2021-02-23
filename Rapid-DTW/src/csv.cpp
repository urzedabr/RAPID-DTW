#include "../include/dtw_cpu.hpp"
#include "../include/dtw_gpu.hpp"
#include "utils/dtw_utils.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <queue>
#include <mutex>
#include <chrono>
#include <thread>

//using namespace boost::program_options;


#define PATTERN_CHUNK_SIZE 5
#define TIMESERIES_CHUNK_SIZE 541

#define PATTERN_NUM_CHUNKS 1
#define TIMESERIES_NUM_CHUNKS 1

#define PATTERN_QUEUE_MAX_SIZE (PATTERN_CHUNK_SIZE * 4)
#define TIMESERIES_QUEUE_MAX_SIZE (TIMESERIES_CHUNK_SIZE * 4)

#define NUM_EXECUTIONS 1

#define SLEEP_QUEUE_VERIFY_MS	100

queue<string> timeseries_queue;
queue<string> pattern_queue;
std::mutex pattern_queue_mutex;
std::mutex timeseries_queue_mutex;

vector<double> runDTW(vector<int>& pattern_day_of_year,vector<vector<double>>& pattern,vector<float>& pattern_gpu_vector,
		vector<int>& timeseries_day_of_year,vector<vector<double>>& timeseries,vector<float>& timeseries_gpu_vector,
		int num_patterns, int patterns_size, int num_timeseries, int timeseries_size, bool use_cpu) {
	if (use_cpu) {
		vector<double> dtw_cost_result = ComputeCpuDtw(pattern_day_of_year, pattern, timeseries_day_of_year,
				timeseries, num_patterns, patterns_size, num_timeseries,
				timeseries_size);
		return dtw_cost_result;
	} else {
		float* pattern_gpu = &pattern_gpu_vector[0];
		float* timeseries_gpu = &timeseries_gpu_vector[0];
		int* pattern_doy_gpu = &pattern_day_of_year[0];
		int* timeseries_doy_gpu = &timeseries_day_of_year[0];

		int num_cols = pattern[0].size(); // or timeseries[0].size(); They have the same number of columns

		std::vector<int> intVec;
		std::vector<double> doubleVec(intVec.begin(), intVec.end());

		float* gpu_dtw_cost_result = ComputeGpuDtw(pattern_doy_gpu, pattern_day_of_year.size(),
				timeseries_doy_gpu, timeseries_day_of_year.size(), pattern_gpu,
				pattern.size(), timeseries_gpu, timeseries.size(), num_cols, num_patterns, patterns_size, num_timeseries,
				timeseries_size);

		vector<double> dtw_cost_result;
		for (int i = 0; i < num_patterns * num_timeseries; i++)
			dtw_cost_result.push_back(gpu_dtw_cost_result[i]);

		return dtw_cost_result;
	}
}

void loadPattern(string files_directory, double pattern_num_chunks, int pattern_chunk_size)
{
	int loaded_pattern_size = 0;

	int pattern_total_size = ceil(pattern_num_chunks * (double) pattern_chunk_size);

	while (loaded_pattern_size < pattern_total_size) {
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir(files_directory.c_str())) != NULL) {
			/* print all the files and directories within directory */
			while ((ent = readdir(dir)) != NULL && (loaded_pattern_size < pattern_total_size)) {
				std::stringstream ss;
				ss << files_directory << ent->d_name;
				string filePath = ss.str();
				if (boost::starts_with(ent->d_name, "pattern")) {
					if (GetSize(filePath) != 45) {
						continue;
					}

					while (pattern_queue.size() > PATTERN_QUEUE_MAX_SIZE) {
						std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_QUEUE_VERIFY_MS));
					}

					if (loaded_pattern_size < pattern_total_size) {
						pattern_queue_mutex.lock();
						pattern_queue.push(filePath);
						pattern_queue_mutex.unlock();
						loaded_pattern_size++;
					}


				}
			}
			closedir(dir);
		} else {
			/* could not open directory */
			perror("");
			return;
		}
	}
}

void loadTimeseries(string files_directory, double timeseries_num_chunks, int timeseries_chunk_size)
{
	int loaded_timeseries_size = 0;

	int timeseries_total_size = ceil(timeseries_num_chunks * (double) timeseries_chunk_size);

	while (loaded_timeseries_size < timeseries_total_size) {
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir(files_directory.c_str())) != NULL) {
			/* print all the files and directories within directory */
			while ((ent = readdir(dir)) != NULL && (loaded_timeseries_size < timeseries_total_size)) {
				std::stringstream ss;
				ss << files_directory << ent->d_name;
				string filePath = ss.str();
				if (boost::starts_with(ent->d_name, "validation")) {
					if (GetSize(filePath) != 23 && GetSize(filePath) != 24  && GetSize(filePath) != 22 && GetSize(filePath) != 21) {
						continue;
					}

					while (timeseries_queue.size() > TIMESERIES_QUEUE_MAX_SIZE) {
						std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_QUEUE_VERIFY_MS));
					}

					if (loaded_timeseries_size < timeseries_total_size) {
						timeseries_queue_mutex.lock();
						timeseries_queue.push(filePath);
						timeseries_queue_mutex.unlock();
						loaded_timeseries_size++;
					}
				}
			}
			closedir(dir);
		} else {
			/* could not open directory */
			perror("");
			return;
		}
	}
}

void generate_results(vector<string>& pattern_files, vector<string>& timeseries_files, vector<double>& dtw_cost_result) {
	for (int j = 0; j < timeseries_files.size(); j++) {
		string timeseries_file = timeseries_files[j];

		// get time series name
		vector<string> strs;
		boost::split(strs, timeseries_file, boost::is_any_of("_"));
		string timeseries = strs[1];
		string id = strs[2];
		boost::replace_all(id, ".csv", "");

		printf("serie temporal= %s\n", timeseries.c_str());
		printf("ID= %s\n", id.c_str());
		double cost = DBL_MAX;
		string pattern = "";
		bool match = false;
		for (int i = 0; i < pattern_files.size(); i++) {
			string pattern_file = pattern_files[i];

			//get pattern name
			vector<string> strs_p;
			boost::split(strs_p, pattern_file, boost::is_any_of("_"));
			string pattern_type = strs_p[1];
			boost::replace_all(pattern_type, ".csv", "");

			double local_cost = dtw_cost_result[j + i * timeseries_files.size()];

			if (local_cost < cost) {
				cost = local_cost;
				pattern = pattern_type;

				printf("custo= %f\n", cost);
				//std::string str = "world";
				//std::printf("hello %s.\n", str.c_str());

				printf("padrão= %s\n", pattern.c_str());
			}
		}

		if (timeseries.compare(pattern) == 0)
			match = true;
	}
}

int dtwMain(string files_directory, string benchmark_file, bool use_cpu, string line) {

	const clock_t begin_time = clock();
	// Split input line
	vector<string> strs;
	boost::split(strs, line, boost::is_any_of(","));

	double pattern_num_chunks = atof(strs[0].c_str());
	int pattern_chunk_size = atoi(strs[1].c_str());
	double timeseries_num_chunks = atof(strs[2].c_str());
	int timeseries_chunk_size = atoi(strs[3].c_str());

	int pattern_total_size = ceil(pattern_num_chunks * (double) pattern_chunk_size);
	int timeseries_total_size = ceil(timeseries_num_chunks * (double) timeseries_chunk_size);

	//start pattern reader thread
	thread t_pattern (loadPattern, files_directory, pattern_num_chunks, pattern_chunk_size);
	int loaded_total_pattern = 0;
	int patterns_size = 0;

	//pattern vector
	vector<int> pattern_day_of_year;
	vector<string> pattern_files;
	vector<vector<double>> pattern;
	vector<float> pattern_gpu_vector;

	// Load all patterns
	while (loaded_total_pattern < pattern_total_size) {
		for (int i = 0; i < min(PATTERN_CHUNK_SIZE, pattern_total_size); i++) {
			while (pattern_queue.size() <= 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_QUEUE_VERIFY_MS));
			}

			pattern_queue_mutex.lock();
			string file_path = pattern_queue.front();
			pattern_queue.pop();
			pattern_queue_mutex.unlock();

			pattern_files.push_back(basename(file_path));

			// read pattern
			patterns_size = GetSeriesData(file_path, pattern_day_of_year, pattern, pattern_gpu_vector, false);

			loaded_total_pattern++;
		}
	}
	t_pattern.join();

	//start time series reader thread
	thread t_time_series (loadTimeseries, files_directory, timeseries_num_chunks, timeseries_chunk_size);
	int loaded_total_timeseries = 0;

	while (loaded_total_pattern < pattern_total_size || loaded_total_timeseries < timeseries_total_size) {

		//timeseries vector
		vector<int> timeseries_day_of_year;
		vector<vector<double>> timeseries;
		vector<string> timeseries_files;
		vector<float> timeseries_gpu_vector;

		int num_timeseries = 0, timeseries_size = 0;

		for (int i = 0; i < min(TIMESERIES_CHUNK_SIZE, timeseries_total_size); i++) {
			while (timeseries_queue.size() <= 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_QUEUE_VERIFY_MS));
			}

			timeseries_queue_mutex.lock();
			string file_path = timeseries_queue.front();
			timeseries_queue.pop();
			timeseries_queue_mutex.unlock();

			timeseries_files.push_back(basename(file_path));

			// read time series data
			timeseries_size = GetSeriesData(file_path, timeseries_day_of_year, timeseries, timeseries_gpu_vector, true);

			num_timeseries++;
			loaded_total_timeseries++;
		}

		// run NUM_EXECUTIONS times
		for (int i = 0; i < NUM_EXECUTIONS; i++) {
			vector<double> dtw_cost_result = runDTW(pattern_day_of_year, pattern, pattern_gpu_vector,
					timeseries_day_of_year, timeseries, timeseries_gpu_vector,
					loaded_total_pattern, patterns_size, num_timeseries, timeseries_size, use_cpu);
			generate_results(pattern_files, timeseries_files, dtw_cost_result);
		}
	}


	t_time_series.join();

	int end_time = float( clock () - begin_time )  * 1000 /  CLOCKS_PER_SEC;
	return end_time;
}

int dtwTest(string files_directory, string benchmark_file, bool use_cpu, string line) {

	const clock_t begin_time = clock();
	// Split input line
	vector<string> strs;
	boost::split(strs, line, boost::is_any_of(","));

	double pattern_num_chunks = atof(strs[0].c_str());
	int pattern_chunk_size = atoi(strs[1].c_str());
	double timeseries_num_chunks = atof(strs[2].c_str());
	int timeseries_chunk_size = atoi(strs[3].c_str());

	//start reader thread
	loadPattern(files_directory, 1, 1);
	loadTimeseries(files_directory, 1, 1);

	//pattern vector
	vector<int> pattern_day_of_year;
	vector<vector<double>> pattern;
	vector<float> pattern_gpu;

	//timeseries vector
	vector<int> timeseries_day_of_year;
	vector<vector<double>> timeseries;
	vector<float> timeseries_gpu_vector;

	// read pattern
	string pattern_file_path = pattern_queue.front();
	pattern_queue.pop();
	GetTimeSeriesToTest(pattern_file_path, pattern_day_of_year, pattern, pattern_gpu, pattern_chunk_size);

	// read time series data
	string timeseries_file_path = timeseries_queue.front();
	timeseries_queue.pop();
	GetTimeSeriesToTest(timeseries_file_path, timeseries_day_of_year, timeseries, timeseries_gpu_vector, timeseries_chunk_size);

	// run NUM_EXECUTIONS times
	for (int i = 0; i < NUM_EXECUTIONS; i++) {
		vector<double> dtw_cost_result = runDTW(pattern_day_of_year, pattern, pattern_gpu,
				timeseries_day_of_year, timeseries, timeseries_gpu_vector,
				1, pattern_chunk_size, 1, timeseries_chunk_size, use_cpu);
	}

	int end_time = float( clock () - begin_time )  * 1000 /  CLOCKS_PER_SEC;
	return end_time;
}

int main(int argc, char *argv[]) {

	string files_directory;
	string benchmark_file;
	int repeat;
	bool use_cpu = true;
	bool test_large_ts = false;

	try
	  {
		boost::program_options::options_description desc{"Options"};
	    desc.add_options()
	      ("help,h", "Help screen")
	      ("directory,d", boost::program_options::value<string>(&files_directory)->required(), "Files directory")
	      ("benchmark,b", boost::program_options::value<string>(&benchmark_file)->required(), "Benchmark file")
	      ("repeat,r", boost::program_options::value<int>(&repeat)->default_value(1), "Number of repetitions to measure time")
	      ("cpu,c", "Use CPU")
	      ("gpu,g", "Use GPU")
	      ("test,t", "Flag that indicates a test with large time series");

	    boost::program_options::variables_map vm;
	    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	    boost::program_options::notify(vm);

	    if (vm.count("help"))
	      std::cout << desc << '\n';
	    if (vm.count("gpu"))
	    	use_cpu = false;
	    if (vm.count("test"))
	    	test_large_ts = true;

	  }
	  catch (const boost::program_options::error &ex)
	  {
	    std::cerr << ex.what() << '\n';
	  }


	ifstream myfile(benchmark_file.c_str());
	string line;
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			cout << line << endl;
			for (int i = 0; i < repeat; i++) {
				int time = 0;
				if (test_large_ts)
					time = dtwTest(files_directory, benchmark_file, use_cpu, line);
				else
					time = dtwMain(files_directory, benchmark_file, use_cpu, line);
				cout << time << endl;
			}
		}
	}

	return 0;
}
