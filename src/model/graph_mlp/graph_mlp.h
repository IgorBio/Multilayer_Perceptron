#ifndef MODEL_GRAPH_MLP_GRAPH_MLP_H_
#define MODEL_GRAPH_MLP_GRAPH_MLP_H_

#include "config.h"
#include "layer.h"

namespace s21 {

/**
 * @class GraphMlp
 * @brief Implementation of Multi-Layer Perceptron (MLP) using a graph-based
 * structure.
 *
 * The GraphMlp class represents a Multi-Layer Perceptron (MLP) implemented
 * using a graph-based structure, where each layer is connected to the previous
 * one. It inherits from the AbstractMlp interface and provides methods for
 * setting input layers, performing forward and backward propagations and
 * accessing MLP parameters.
 */
class GraphMlp : public AbstractMlp {
 public:
  explicit GraphMlp(const Topology& topology);

  void SetInputLayer(const Vector& input_values) override;
  void ForwardPropagation() override;
  void BackPropagation(const Vector& expected, double learning_rate) override;
  Vector GetOutput() const override;
  std::pair<const Tensor, const Tensor> GetMlp() const override;
  void SetMlp(const Tensor&, const Tensor&) override;

 private:
  std::vector<std::shared_ptr<Layer>> net_;
};

}  // namespace s21

#endif  // MODEL_GRAPH_MLP_GRAPH_MLP_H_
