#include "neuron.h"

namespace s21 {

Neuron::Neuron(std::size_t prev_size)
    : value_(0.0), error_(0.0), bias_(0.0), weights_(prev_size) {
  std::generate(weights_.begin(), weights_.end(), RandomWeight);
}

void Neuron::CalculateValue(const Vector& prev_values) {
  if (prev_values.size() != weights_.size()) {
    throw std::invalid_argument("Next size doesn't match weight size");
  }

  double sum = bias_;
  for (std::size_t i = 0; i < prev_values.size(); ++i) {
    sum += prev_values[i] * weights_[i];
  }
  value_ = ApplyActivation(sum, sigmoid);
}

void Neuron::CalculateError(double err) {
  error_ = err * ApplyActivationDerivative(value_, sigmoid_derivative);
}

void Neuron::UpdateWeights(const Vector& prev_values, double learning_rate) {
  if (prev_values.size() != weights_.size()) {
    throw std::invalid_argument("Next size doesn't match weight size");
  }

  for (std::size_t i = 0; i < weights_.size(); ++i) {
    weights_[i] += learning_rate * error_ * prev_values[i];
  }

  bias_ += learning_rate * error_;
}

}  // namespace s21
