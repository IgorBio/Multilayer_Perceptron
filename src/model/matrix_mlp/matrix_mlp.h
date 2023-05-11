#ifndef SRC_MODEL_MATRIX_MATRIX_H_
#define SRC_MODEL_MATRIX_MATRIX_H_

#include "../config.h"
#include "../utility/matrix_operations.h"

namespace s21 {

class MatrixMlp {
 public:
  using Matrix = std::vector<std::vector<double>>;
  using Vector = std::vector<double>;

  explicit MatrixMlp(Architecture architecture);

  void SetInput(const Vector& outputs);
  void ForwardPropagation();
  void BackPropagation(const Vector& expected_output, double learning_rate_);
  Vector GetOutput();

  Vector GetWeights();
  void LoadWeights(const Vector& weights);

 private:
  void FillMatrixRandom(Matrix& m);
  Matrix ActivationFuncMatrix(const Matrix& m);
  Matrix DerivativeActivationFuncMatrix(const Matrix& m);
  void AdjustWeights(std::size_t i, double learning_rate, const Matrix& error);

  std::vector<Matrix> values_;
  std::vector<Matrix> weights_;
};

}  // namespace s21

#endif  // SRC_MODEL_MATRIX_MATRIX_H_
