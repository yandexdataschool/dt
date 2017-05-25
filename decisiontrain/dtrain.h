// Copyright 2017, Nikita Kazeev (Yandex)
#ifndef DECISIONTRAIN_DTRAIN_H_
#define DECISIONTRAIN_DTRAIN_H_

#include<stdexcept>
#include<vector>
#include<iostream>
#include<fstream>
#include<cstdint>
#include<algorithm>
#include<sstream>
#include<limits>
#include "json.hpp"

class BinTransformer {
 public:
    BinTransformer() = default;
    explicit BinTransformer(const std::vector<std::vector<float> >);
    std::vector<uint8_t> transform(const std::vector<float>& example) const;
 private:
    std::vector<std::vector<float>> percentiles;
};

class DecisionTrainPredictor {
 public:
  // Limited because we use uint32 and (1 << depth)
  static const unsigned int max_depth = 31;
  void init_from_json(const nlohmann::json);
  float predict_score_single_example(const std::vector<float>& example) const;
 private:
  struct Estimator {
    unsigned int feature;
    unsigned int cut;
    std::vector<float> leaf_values;
    explicit Estimator(const nlohmann::json);
  };
  float initial_bias;
  unsigned int depth;
  unsigned int n_features;
  BinTransformer transformer;
  std::vector<Estimator> estimators;
};

#endif  // DECISIONTRAIN_DTRAIN_H_
