// Copyright 2017, Nikita Kazeev (Yandex)
#include <string>
#include "json.hpp"
#include "./dtrain.h"
#include "gtest/gtest.h"

class DecisionTrainPredictorTest : public ::testing::Test {
 public:
  static const std::string toy_model;
  nlohmann::json toy_model_in_json;
  DecisionTrainPredictorTest() {
    std::ifstream toy_model_file(toy_model, std::ifstream::in);
    toy_model_file >> toy_model_in_json;
    toy_model_file.close();
  }
};
const std::string DecisionTrainPredictorTest::toy_model = "toy_model.json";

TEST_F(DecisionTrainPredictorTest, JsonInitWrongDepth) {
  DecisionTrainPredictor dt;
  toy_model_in_json["depth"] = 128;
  ASSERT_THROW(dt.init_from_json(toy_model_in_json), std::invalid_argument);
}


TEST_F(DecisionTrainPredictorTest, JsonInitWrongEstimators) {
  DecisionTrainPredictor dt;
  toy_model_in_json["estimators"] = 1424134;
  ASSERT_ANY_THROW(dt.init_from_json(toy_model_in_json));
}

TEST_F(DecisionTrainPredictorTest, ToyModelLoad) {
  DecisionTrainPredictor dt;
  ASSERT_NO_THROW(dt.init_from_json(toy_model_in_json));
}

TEST_F(DecisionTrainPredictorTest, SinglePredicionTest) {
  DecisionTrainPredictor dt;
  dt.init_from_json(toy_model_in_json);
  const std::vector<float> x = {1, 1, 1, 1};
  ASSERT_FLOAT_EQ(0.551256597042, dt.predict_score_single_example(x));
}

TEST_F(DecisionTrainPredictorTest, SeveralPredicionsTest) {
  DecisionTrainPredictor dt;
  dt.init_from_json(toy_model_in_json);
  std::ifstream x_stream("x_test", std::ifstream::in);
  std::ifstream y_stream("y_test", std::ifstream::in);
  std::string line;
  while (std::getline(x_stream, line)) {
    std::stringstream line_stream(line);
    float y;
    y_stream >> y;
    std::vector<float> x;
    float value;
    while (line_stream >> value) {
      x.push_back(value);
    }
    ASSERT_FLOAT_EQ(y, dt.predict_score_single_example(x));
  }
  x_stream.close();
  y_stream.close();
}


