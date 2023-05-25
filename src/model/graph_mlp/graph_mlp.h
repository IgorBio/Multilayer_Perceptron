#ifndef MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_
#define MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_

#include "../config.h"
#include "../utility/matrix_operations.h"
#include "layer.h"

namespace s21 {

using Net = std::vector<std::unique_ptr<Layer>>;

class GraphMlp {
 public:
  explicit GraphMlp(Topology);

  void SetInputLayer(const Vector &);
  void ForwardPropagation();
  void BackPropagation(const Vector &, double);
  double CalculateLoss(const Matrix &, const Matrix &);
  Vector Predict(const Vector &input) const;
  Vector GetOutput() const;
  Vector GetWeights() const;
  void SetWeights(const Vector &);

 private:
  void AddLayer(std::size_t, std::size_t);
  void UpdateWeights(const Matrix &, double, std::size_t);

  Net net_;
};

}  // namespace s21

#endif  // MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_
