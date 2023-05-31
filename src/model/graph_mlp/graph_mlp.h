#ifndef MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_
#define MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_

#include "../config.h"
#include "../mlp_interface.h"
#include "layer.h"

namespace s21 {

using Net = std::vector<std::unique_ptr<Layer>>;

class GraphMlp : public Interface {
 public:
  explicit GraphMlp(Topology);

  void SetInputLayer(const Vector &) override;
  void ForwardPropagation() override;
  void BackPropagation(const Vector &, double) override;
  double CalculateLoss(const Vector &, const Vector &) override;
  Vector Predict(const Vector &) override;
  Vector GetOutput() const override;
  Weights GetWeights() const override;
  void SetWeights(const Weights &) override;

 private:
  void AddLayer(std::size_t, std::size_t);
  void UpdateLayer(const Matrix &, double, std::size_t);

  Net net_;
};

}  // namespace s21

#endif  // MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_
