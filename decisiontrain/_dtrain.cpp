#include "_dtrain.h"

float DecisionTrainPredictor::predict_score_single_example(
    const std::vector<float>& example) const {
  const std::vector<uint8_t> transformed_example = transformer.transform(example);
  float result = initial_bias;
  uint32_t index = 0;
  // TODO(kazeevn) check depth
  const uint32_t mask = (1 << depth) - 1;
  for (const Estimator& estimator: estimators) {
    index <<= 1;
    // TODO(kazeevn) check types
    index |= (transformed_example[estimator.feature] > estimator.cut);
    result += estimator.leaf_values[index & mask];
  }
  return 1 / (1 + exp(-result));
}

std::vector<uint8_t> BinTransformer::transform(
    const std::vector<float>& example) const {
  if (example.size() != percentiles.size()) {
    throw std::invalid_argument("Example has a wrong number of features");
  }
  std::vector<uint8_t> bin_indices(example.size(), 0);
  for (size_t percentile_index = 0; percentile_index < percentiles.size(); ++percentile_index) {
    bin_indices[percentile_index] = std::lower_bound(percentiles[percentile_index].begin(),
                                                     percentiles[percentile_index].end(),
                                                     example[percentile_index]) - \
                                    percentiles[percentile_index].begin();
  }
  return bin_indices;
}

BinTransformer::BinTransformer(const std::vector<std::vector<float>> input_percentiles):
    percentiles(input_percentiles) {};

// TODO(kazeevn) better model storage. We aren't at ICPC
void DecisionTrainPredictor::init_from_file(const std::string file_name) {
  std::ifstream model_file(file_name, std::ifstream::in);
  model_file >> initial_bias;
  model_file >> depth;
  model_file >> n_features;
  unsigned int n_estimators;
  model_file >> n_estimators;

  std::vector<std::vector<float>> percentiles(n_features);
  unsigned int line_index = 0;
  bool estimator_header = true;
  Estimator current_estimator;
  std::string line;
  model_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while (std::getline(model_file, line)) {
    std::stringstream line_as_stream(line);
    if (line_index < n_features) {
      float value;
      while (line_as_stream >> value) {
        percentiles[line_index].push_back(value);
      }
    } else {
      if (estimator_header) {
        current_estimator = Estimator();
        line_as_stream >> current_estimator.feature;
        line_as_stream >> current_estimator.cut;		
      } else {
        float value;
        while (line_as_stream >> value) {
          current_estimator.leaf_values.push_back(value);
        }
        estimators.push_back(std::move(current_estimator));
      }
      estimator_header = !estimator_header;
    }
    ++line_index;
  }
  transformer = BinTransformer(std::move(percentiles));
  model_file.close();
  std::cout << "Read model: " << n_features << " features; " \
            << estimators.size() << " estimators" << std::endl;
}
    
