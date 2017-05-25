#include<iostream>
#include "dt_pid.h"

constexpr int discarded_tail_size = 6;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " model_file.json infile outfile" << std::endl;
    exit(-1);
  }
  DT_PID dt_pid;
  dt_pid.init_from_json_file(argv[1]);
  std::ifstream input_file(argv[2], std::ifstream::in);
  std::ofstream output_file(argv[3], std::ofstream::out);
  std::string line;
  // read the header
  getline(input_file, line);
  while (getline(input_file, line)) {
    std::vector<float> x;
    float value;
    std::stringstream line_as_stream(line);
    while (line_as_stream >> value) {
      x.push_back(value);
    }
    x.erase(x.end() - discarded_tail_size, x.end());
    for (auto& s: dt_pid.predict_scores_single_example(x)) {
	output_file << s << ' ';
    }
    output_file << std::endl;
  }
  input_file.close();
  output_file.close();
  return 0;
}
