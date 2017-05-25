#ifndef _DTRAIN_H
#define _DTRAIN_H

#include<stdexcept>
#include<vector>
#include<iostream>
#include<fstream>
#include<cstdint>
#include<algorithm>
#include<sstream>
#include<limits>

class BinTransformer {
 public:
    BinTransformer() = default;
    BinTransformer(const std::vector<std::vector<float> >);
    std::vector<uint8_t> transform(const std::vector<float>& example) const;
 private:
    std::vector<std::vector<float>> percentiles;
};

class DecisionTrainPredictor {
 public:
    void init_from_file(const std::string);
    float predict_score_single_example(const std::vector<float>& example) const;
 private:
    struct Estimator {
	unsigned int feature;
	unsigned int cut;
	std::vector<float> leaf_values;
    };
    float initial_bias;
    unsigned int depth;
    unsigned int n_features;
    BinTransformer transformer;
    std::vector<Estimator> estimators;
};

#endif // _DTRAIN_H
