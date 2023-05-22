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
  bias_.emplace_back(1, Vector(cols, 0.0));
}

void MatrixMlp::SetInputLayer(const Vector &input) {
  neurons_[0] = Matrix(1, input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i{0u}; i < weights_.size(); ++i) {
    neurons_[i + 1] =
        Activate(neurons_[i] * weights_[i] + bias_[i], acivation_);
  }
}

void MatrixMlp::BackPropagation(const Vector &answer, double lr) {
  Matrix errors =
      MultiplyHadamard(neurons_.back() - Matrix(1, answer),
                       ActivateDerivative(neurons_.back(), acivation_));
  UpdateWeights(errors, lr, weights_.size() - 1);

  for (std::size_t i{neurons_.size() - 2}; i > 0; --i) {
    errors = MultiplyHadamard(errors * Transpose(weights_[i]),
                              ActivateDerivative(neurons_[i], acivation_));
    UpdateWeights(errors, lr, i - 1);
  }
}

void MatrixMlp::UpdateWeights(const Matrix &errors, double lr,
                              std::size_t idx) {
  weights_[idx] = weights_[idx] - (Transpose(neurons_[idx - 1]) * errors) * lr;
}

double MatrixMlp::CalculateLoss(const Matrix &inputs, const Matrix &labels) {
  double total_loss{0.0};
  for (std::size_t i{0u}; i < inputs.size(); ++i) {
    SetInputLayer(inputs[i]);
    ForwardPropagation();
    Vector output = GetOutput();
    double loss{0.0};
    for (std::size_t j{0u}; j < output.size(); ++j) {
      loss += std::pow(output[j] - labels[i][j], 2);
    }
    total_loss += loss;
  }
  return total_loss / inputs.size();
}

Vector MatrixMlp::Predict(const Vector &input) {
  SetInputLayer(input);
  ForwardPropagation();
  return GetOutput();
}

Vector MatrixMlp::GetOutput() const {
  const auto &output_matrix = neurons_.back();
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

activation_func MatrixMlp::GetActivationFunction() const { return acivation_; }

void MatrixMlp::SetActivationFunction(activation_func activation) {
  acivation_ = activation;
}

}  // namespace s21
