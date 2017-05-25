// Copyright 2017, Nikita Kazeev (Yandex)

#include "./dt_pid.h"

void DT_PID::init_from_json_file(const std::string file_name) {
    std::ifstream model_file(file_name, std::ifstream::in);
    nlohmann::json model_json;
    model_file >> model_json;
    model_file.close();
    size_t model_index = 0;
    for (const nlohmann::json& decision_train_json : model_json) {
      models[model_index].init_from_json(decision_train_json);
      ++model_index;
    }
    if (model_index != N_CLASSES) {
      throw std::invalid_argument("The number of estimators should be"
                                  " the same as the number of classes.");
    }
}


std::array<float, N_CLASSES> DT_PID::predict_scores_single_example(
    const std::vector<float>& example) const {
  std::array<float, N_CLASSES> result;
  size_t estimator_index = 0;
  for (const DecisionTrainPredictor& estimator : models) {
    result[estimator_index] = estimator.predict_score_single_example(example);
    ++estimator_index;
  }
  return result;
}


float DT_PID::prdict_score_for_class_single_example(
    const unsigned int class_, const std::vector<float>& example) const {
  return models[class_].predict_score_single_example(example);
}
