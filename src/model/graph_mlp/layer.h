#ifndef MLP_MODEL_GRAPH_MLP_LAYER_H_
#define MLP_MODEL_GRAPH_MLP_LAYER_H_

#include <memory>

#include "../config.h"
#include "../utility/matrix_operations.h"
#include "neuron.h"

namespace s21 {

using Neurons = std::vector<std::unique_ptr<Neuron>>;

enum class LayerType { kInput, kHidden, kOutput };

class Layer {
 public:
  explicit Layer(std::size_t input_size, std::size_t output_size)
      : layer_(output_size) {
    std::generate(layer_.begin(), layer_.end(), [&input_size]() {
      return std::make_unique<Neuron>(input_size);
    });
  }

  void SetType(LayerType type) { type_ = type; }
  LayerType GetLayerType() const { return type_; }
  const Neurons& GetLayer() const { return layer_; }
  Matrix CalculateError(const Vector& expected) const {
    Matrix error(layer_.size(), Vector(expected.size()));
#pragma omp parallel for
    for (std::size_t i{0u}; i < layer_.size(); ++i) {
      auto& neuron = *layer_[i];
      for (std::size_t j{0u}; j < expected.size(); ++j) {
        error[i][j] = expected[j] - neuron.GetValue();
      }
    }
    return error;
  }
  void UpdateWeights(const Matrix errors, double learning_rate) {
#pragma omp parallel for
    for (std::size_t i{0u}; i < layer_.size(); ++i) {
      auto& neuron = *layer_[i];
      neuron.UpdateWeights(errors[i], learning_rate);
    }
  }

 private:
  LayerType type_;
  Neurons layer_;
};

}  // namespace s21

#endif  // MLP_MODEL_GRAPH_MLP_LAYER_H_
