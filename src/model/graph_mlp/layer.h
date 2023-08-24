#ifndef MODEL_GRAPH_MLP_LAYER_H_
#define MODEL_GRAPH_MLP_LAYER_H_

#include <memory>

#include "neuron.h"

namespace s21 {

/**
 * @class Layer
 * @brief Represents a layer of neurons in a Multi-Layer Perceptron (MLP).
 *
 * The Layer class provides methods for setting neuron values, performing
 * feedforward and backpropagation operations, as well as updating weights
 * during training.
 */
class Layer {
 public:
  explicit Layer(std::size_t size, std::shared_ptr<Layer> prev = nullptr);

  void SetValues(const Vector& values);
  void FeedForward();
  void CalculateOutputError(const Vector& expected);
  void CalculateError();
  void UpdateWeights(double learning_rate);

  std::vector<Neuron>& GetLayer() { return layer_; }
  std::size_t GetSize() const { return layer_.size(); }

  void SetNextLayer(std::shared_ptr<Layer> next) { next_layer_ = next; }
  std::shared_ptr<Layer> GetNext() { return next_layer_; }
  std::shared_ptr<Layer> GetPrev() { return prev_layer_; }

 private:
  std::vector<Neuron> layer_;
  std::shared_ptr<Layer> prev_layer_;
  std::shared_ptr<Layer> next_layer_;

  double ErrorSum(std::size_t idx) const;
  Vector GetPrevValues() const;
};

}  // namespace s21

#endif  // MODEL_GRAPH_MLP_LAYER_H_
