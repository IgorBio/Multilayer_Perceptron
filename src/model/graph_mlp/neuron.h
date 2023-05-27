#ifndef MLP_MODEL_GRAPH_MLP_NEURON_H_
#define MLP_MODEL_GRAPH_MLP_NEURON_H_

#include "../config.h"
#include "../utility/matrix_operations.h"

namespace s21 {

class Neuron {
 public:
  explicit Neuron(std::size_t layer_size)
      : value_{0.0}, bias_{RandomWeight()}, weights_(layer_size) {
    std::generate(weights_.begin(), weights_.end(), RandomWeight);
  }

  void SetValue(double value) { value_ = value; }
  double GetValue() const { return value_; }
  void SetBias(double bias) { bias_ = bias; }
  double GetBias() const { return bias_; }
  Vector& GetWeights() { return weights_; }
  void UpdateWeights(const Vector& errors, double learning_rate) {
#pragma omp parallel for
    for (std::size_t i = 0u; i < weights_.size(); ++i)
      weights_[i] += learning_rate * errors[i] * value_;
    bias_ += learning_rate * errors.back();
  }

 private:
  double value_;
  double bias_;
  Vector weights_;
};

}  // namespace s21

#endif  // MLP_MODEL_GRAPH_MLP_NEURON_H_
