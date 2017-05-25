// Copyright 2017, Nikita Kazeev (Yandex)
#ifndef DECISIONTRAIN_DT_PID_H_
#define DECISIONTRAIN_DT_PID_H_

#include <vector>
#include <string>
#include "./dtrain.h"
constexpr size_t N_CLASSES = 6;

class DT_PID {
 public:
  std::array<float, N_CLASSES> predict_scores_single_example(
      const std::vector<float>& example) const;
  float prdict_score_for_class_single_example(
      const unsigned int class_, const std::vector<float>& example) const;
  void init_from_json_file(const std::string);
 private:
  std::array<DecisionTrainPredictor, 6> models;
};

#endif  // DECISIONTRAIN_DT_PID_H_
