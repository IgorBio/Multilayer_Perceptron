#include "matrix_mlp.h"

namespace s21 {

MatrixMlp::MatrixMlp(const Topology &topology)
    : weights_(topology.GetLayersCount() - 1),
      biases_(topology.GetLayersCount() - 1),
      values_(topology.GetLayersCount()) {
  for (std::size_t i = 0; i < topology.GetLayersCount() - 1; ++i) {
    weights_[i] =
        Matrix(topology.GetLayerSize(i), Vector(topology.GetLayerSize(i + 1)));
    RandomizeMatrix(weights_[i]);
    biases_[i] = Matrix(1, Vector(topology.GetLayerSize(i + 1)));
    RandomizeMatrix(biases_[i]);
  }
}

void MatrixMlp::SetInputLayer(const Vector &input) {
  values_[0] = Matrix(1, input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i = 0; i < weights_.size(); ++i) {
    values_[i + 1] = Activate(values_[i] * weights_[i] + biases_[i], sigmoid);
  }
}

void MatrixMlp::BackPropagation(const Vector &expected, double lr) {
  Matrix errors =
      MultiplyHadamard(values_.back() - Matrix(1, expected),
                       ActivateDerivative(values_.back(), sigmoid_derivative));

  for (std::size_t i = weights_.size(); i-- > 0;) {
    weights_[i] -= Transpose(values_[i]) * errors * lr;
    biases_[i] -= errors * lr;
    errors =
        MultiplyHadamard(errors * Transpose(weights_[i]),
                         ActivateDerivative(values_[i], sigmoid_derivative));
  }
}

Vector MatrixMlp::GetOutput() const {
  const Matrix &output_matrix = values_.back();
  return Vector{output_matrix.front().cbegin(), output_matrix.front().cend()};
}

std::pair<const Tensor, const Tensor> MatrixMlp::GetMlp() const {
  return {weights_, biases_};
}

void MatrixMlp::SetMlp(const Tensor &weights, const Tensor &biases) {
  weights_ = weights;
  biases_ = biases;
}

}  // namespace s21
