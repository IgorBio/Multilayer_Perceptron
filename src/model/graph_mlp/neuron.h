#ifndef MODEL_GRAPH_MLP_NEURON_H_
#define MODEL_GRAPH_MLP_NEURON_H_

#include "abstract_mlp.h"
#include "matrix_operations.h"

namespace s21 {

/**
 * @class Neuron
 * @brief Represents a single neuron in a neural network.
 *
 * The Neuron class defines the properties and operations of an individual
 * neuron within a neural network. It provides methods for calculating the
 * neuron's value, error, and weight updates during training.
 */
class Neuron {
 public:
  explicit Neuron(std::size_t prev_size = 0);

  void SetValue(double value) { value_ = value; }
  void SetError(double error) { error_ = error; }
  void SetBias(double bias) { bias_ = bias; }
  void SetWeights(const Vector& weights) { weights_ = weights; }

  double GetValue() const { return value_; }
  double GetError() const { return error_; }
  double GetBias() const { return bias_; }
  const Vector& GetWeights() const { return weights_; }
  double GetWeight(std::size_t idx) const { return weights_[idx]; }

  void CalculateValue(const Vector& prev_values);
  void CalculateError(double err);
  void UpdateWeights(const Vector& prev_values, double learning_rate);

 private:
  double value_;
  double error_;
  double bias_;
  Vector weights_;
};

}  // namespace s21

#endif  // MODEL_GRAPH_MLP_NEURON_H_
