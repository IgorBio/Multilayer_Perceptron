#ifndef MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_
#define MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_

#include "../config.h"
#include "../mlp_interface.h"
#include "../utility/matrix_operations.h"

namespace s21 {

class MatrixMlp : public Interface {
 public:
  explicit MatrixMlp(Topology);

  void SetInputLayer(const Vector &) override;
  void ForwardPropagation() override;
  void BackPropagation(const Vector &, double) override;
  double CalculateLoss(const Vector &, const Vector &) override;
  Vector Predict(const Vector &) override;
  Vector GetOutput() const override;
  Vector GetWeights() const override;
  void SetWeights(const Vector &) override;

 private:
  void AddLayer(std::size_t, std::size_t);
  void UpdateLayer(const Matrix &, double, std::size_t);

  std::vector<Matrix> weights_;
  std::vector<Matrix> neurons_;
  std::vector<Matrix> bias_;
};
}  // namespace s21

#endif  // MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_
