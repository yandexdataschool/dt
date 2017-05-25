// Copyright 2017, Nikita Kazeev (Yandex)
#include "./dtrain.h"

float DecisionTrainPredictor::predict_score_single_example(
    const std::vector<float>& example) const {
  const std::vector<uint8_t> transformed_example = \
    transformer.transform(example);
  float result = initial_bias;
  uint32_t index = 0;
  const uint32_t mask = (1 << depth) - 1;
  for (const Estimator& estimator : estimators) {
    index <<= 1;
    index |= static_cast<uint32_t>(transformed_example[estimator.feature] > estimator.cut);
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
  for (size_t percentile_index = 0; percentile_index < percentiles.size();
       ++percentile_index) {
    bin_indices[percentile_index] = std::lower_bound(
        percentiles[percentile_index].begin(),
        percentiles[percentile_index].end(),
        example[percentile_index]) - percentiles[percentile_index].begin();
  }
  return bin_indices;
}


BinTransformer::BinTransformer(
    const std::vector<std::vector<float>> input_percentiles):
    percentiles(input_percentiles) {}


DecisionTrainPredictor::Estimator::Estimator(
    const nlohmann::json estimator_json) {
  feature = estimator_json.at("feature");
  cut = estimator_json.at("cut");
  leaf_values = estimator_json.at("leafs").get<std::vector<float>>();
}


void DecisionTrainPredictor::init_from_json(
    const nlohmann::json model_in_json) {
  initial_bias = model_in_json.at("initial_bias");
  depth = model_in_json.at("depth");
  if (depth > max_depth) {
    throw std::invalid_argument("Invalid depth");
  }
  n_features = model_in_json.at("n_features");
  transformer = BinTransformer(model_in_json.at("transformer_percentiles").get<
                               std::vector<std::vector<float>>>());
  estimators.clear();
  for (const nlohmann::json& estimator_json : model_in_json.at("estimators")) {
    estimators.push_back(Estimator(estimator_json));
  }
}
