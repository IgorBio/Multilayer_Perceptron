#include "matrix_mlp.h"

namespace s21 {

MatrixMlp::MatrixMlp(Topology topology) : values_(topology.GetLayersCount()) {
  AddLayer(topology.GetInputSize(), topology.GetLayerSize(1));

  for (std::size_t i{1u}; i < topology.GetHiddenCount(); ++i) {
    AddLayer(topology.GetLayerSize(i), topology.GetLayerSize(i + 1));
  }

  AddLayer(topology.GetLastHidden(), topology.GetOutputSize());
}

void MatrixMlp::AddLayer(std::size_t rows, std::size_t cols) {
  weights_.emplace_back(rows, Vector(cols));
  RandomizeMatrix(weights_.back());
}

void MatrixMlp::SetInputLayer(const Vector &input) {
  values_[0] = Matrix(1, input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i{0u}; i < weights_.size(); ++i) {
    values_[i + 1] = Activate(values_[i] * weights_[i], sigmoid);
  }
}

void MatrixMlp::BackPropagation(const Vector &expected, double lr) {
  Matrix errors =
      MultiplyHadamard(values_.back() - Matrix(1, expected),
                       ActivateDerivative(values_.back(), sigmoid_derivative));
  UpdateLayer(errors, lr, weights_.size() - 1);

  for (std::size_t i{values_.size() - 2}; i > 0; --i) {
    errors =
        MultiplyHadamard(errors * Transpose(weights_[i]),
                         ActivateDerivative(values_[i], sigmoid_derivative));

    UpdateLayer(errors, lr, i - 1);
  }
}

void MatrixMlp::UpdateLayer(const Matrix &errors, double lr, std::size_t idx) {
  weights_[idx] = weights_[idx] - (Transpose(values_[idx]) * errors) * lr;
}

double MatrixMlp::CalculateLoss(const Vector &predicted_output,
                                const Vector &expected_output) {
  double loss = 0.0;
  for (std::size_t i{0u}; i < predicted_output.size(); ++i) {
    double diff = expected_output[i] - predicted_output[i];
    loss += diff * diff;
  }
  return loss;
}

Vector MatrixMlp::Predict(const Vector &input) {
  SetInputLayer(input);
  ForwardPropagation();
  return GetOutput();
}

Vector MatrixMlp::GetOutput() const {
  const Matrix &output_matrix = values_.back();
  return Vector{output_matrix.front().cbegin(), output_matrix.front().cend()};
}

Weights MatrixMlp::GetWeights() const { return weights_; }

void MatrixMlp::SetWeights(const Weights &weights) { weights_ = weights; }

}  // namespace s21
