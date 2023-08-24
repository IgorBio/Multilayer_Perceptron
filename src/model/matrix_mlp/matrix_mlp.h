#ifndef MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_
#define MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_

#include "abstract_mlp.h"
#include "config.h"
#include "matrix_operations.h"

namespace s21 {

/**
 * @class MatrixMlp
 * @brief Implementation of Multi-Layer Perceptron (MLP) in matrix form.
 *
 * The MatrixMlp class represents a Multi-Layer Perceptron implemented using
 * matrix operations for efficient forward and backward propagations. It
 * inherits from the AbstractMlp interface and provides methods for setting
 * input layers, performing forward and backward propagations, and accessing MLP
 * parameters.
 */
class MatrixMlp : public AbstractMlp {
 public:
  explicit MatrixMlp(const Topology &);

  void SetInputLayer(const Vector &) override;
  void ForwardPropagation() override;
  void BackPropagation(const Vector &, double) override;
  Vector GetOutput() const override;
  std::pair<const Tensor, const Tensor> GetMlp() const override;
  void SetMlp(const Tensor &, const Tensor &) override;

 private:
  Tensor weights_;
  Tensor biases_;
  Tensor values_;
};
}  // namespace s21

#endif  // MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_
