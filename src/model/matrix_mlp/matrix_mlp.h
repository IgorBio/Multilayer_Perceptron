#ifndef MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_
#define MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_

#include "../config.h"
#include "../utility/matrix_operations.h"

namespace s21 {

class MatrixMlp {
 public:
  explicit MatrixMlp(Topology);

  void SetInputLayer(const Vector &);
  void ForwardPropagation();
  void BackPropagation(const Vector &, double);
  Vector GetOutput() const;
  Vector GetWeights() const;
  void SetWeights(const Vector &);
  Parallel GetParallel() const;
  void SetParallel(Parallel);

 private:
  void AddWheights(std::size_t, std::size_t);
  void UpdateWeights(const Matrix &, double, std::size_t);

  std::vector<Matrix> weights_;
  std::vector<Matrix> neurons_;
  Vector biases_;
  Parallel parallel_ = Parallel::kMaxThreads;
};

}  // namespace s21

#endif  // MLP_MODEL_MATRIX_MLP_MATRIX_MLP_H_
