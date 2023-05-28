#include "matrix_mlp.h"

namespace s21 {

MatrixMlp::MatrixMlp(Topology topology) : neurons_(topology.hidden_layers + 2) {
  AddLayer(topology.input_layer, topology.hidden_layer);

  for (std::size_t i{1u}; i < topology.hidden_layers; ++i) {
    AddLayer(topology.hidden_layer, topology.hidden_layer);
  }

  AddLayer(topology.hidden_layer, topology.output_layer);
}

void MatrixMlp::AddLayer(std::size_t rows, std::size_t cols) {
  weights_.emplace_back(rows, Vector(cols));
  RandomizeMatrix(weights_.back());
  bias_.emplace_back(1, Vector(cols, RandomWeight()));
}

void MatrixMlp::SetInputLayer(const Vector &input) {
  neurons_[0] = Matrix(1, input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i{0u}; i < weights_.size(); ++i) {
    neurons_[i + 1] = Activate(neurons_[i] * weights_[i] + bias_[i], sigmoid);
  }
}

void MatrixMlp::BackPropagation(const Vector &expected, double lr) {
  Matrix errors =
      MultiplyHadamard(neurons_.back() - Matrix(1, expected),
                       ActivateDerivative(neurons_.back(), sigmoid_derivative));
  UpdateLayer(errors, lr, weights_.size() - 1);

  for (std::size_t i{neurons_.size() - 2}; i > 0; --i) {
    errors =
        MultiplyHadamard(errors * Transpose(weights_[i]),
                         ActivateDerivative(neurons_[i], sigmoid_derivative));

    UpdateLayer(errors, lr, i - 1);
  }
}

void MatrixMlp::UpdateLayer(const Matrix &errors, double lr, std::size_t idx) {
  weights_[idx] = weights_[idx] - (Transpose(neurons_[idx]) * errors) * lr;
  bias_[idx] = bias_[idx] - errors * lr;
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
  const Matrix &output_matrix = neurons_.back();
  return Vector{output_matrix.front().cbegin(), output_matrix.front().cend()};
}

Vector MatrixMlp::GetWeights() const {
  Vector weights;
  for (auto &matrix : weights_) {
    for (const auto &row : matrix) {
      weights.insert(weights.end(), row.begin(), row.end());
    }
  }
  return weights;
}

void MatrixMlp::SetWeights(const Vector &weights) {
  auto it = weights.cbegin();
  for (auto &matrix : weights_) {
    for (auto &row : matrix) {
      std::copy(it, it + row.size(), row.begin());
      it += row.size();
    }
  }
}

}  // namespace s21
