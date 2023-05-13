#ifndef SRC_MODEL_MATRIX_MATRIX_H_
#define SRC_MODEL_MATRIX_MATRIX_H_

#include "../config.h"
#include "../utility/matrix_operations.h"

namespace s21 {

class MatrixMlp {
 public:
  explicit MatrixMlp(Architecture);

  void SetInputLayer(const Vector &);
  void ForwardPropagation();
  void BackPropagation(const Vector &, double);
  Vector GetOutput();
  Vector GetWeights();
  void SetWeights(const Vector &);

 private:
  void AddWheights(std::size_t, std::size_t);
  void AdjustWeights(const Matrix &, double, size_t);

  std::vector<Matrix> weights_;
  std::vector<Matrix> layers_;
};

}  // namespace s21

#endif  // SRC_MODEL_MATRIX_MATRIX_H_
