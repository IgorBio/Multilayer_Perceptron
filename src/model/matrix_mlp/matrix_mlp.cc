#include "matrix_mlp.h"

namespace s21 {

MatrixMlp::MatrixMlp(Topology topology)
    : neurons_(topology.hidden_layers + 2), bias_{0} {
  AddWheights(topology.hidden_layer, topology.input_layer);

  for (std::size_t i{0u}; i < topology.hidden_layers - 1; ++i) {
    AddWheights(topology.hidden_layer, topology.hidden_layer);
  }

  AddWheights(topology.output_layer, topology.hidden_layer);
}

void MatrixMlp::AddWheights(std::size_t rows, std::size_t cols) {
  Matrix wheights(rows, Vector(cols));
  Randomize(wheights);
  weights_.push_back(wheights);
}

void MatrixMlp::SetInputLayer(const Vector &layer) {
  Matrix input = Matrix(layer.size(), Vector(1));
  for (std::size_t i{0u}; i < layer.size(); ++i) {
    input[i][0] = layer[i];
  }
  neurons_.front() = std::move(input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i{0u}; i < weights_.size(); ++i) {
    neurons_[i + 1] =
        Activate(MultiplyWinograd(weights_[i], neurons_[i]), acivation_);
  }
}

void MatrixMlp::BackPropagation(const Vector &answer, double lr) {
  Matrix output = Matrix(answer.size(), Vector(1));
  for (std::size_t i{0u}; i < answer.size(); ++i) {
    output[i][0] = answer[i];
  }
  Matrix errors = Subtraction(neurons_.back(), output);
  Matrix gradient = DeriveActivate(neurons_.back(), acivation_);
  errors = MultiplyHadamard(errors, gradient);
  UpdateWeights(errors, lr, weights_.size() - 1);

  for (int i{static_cast<int>(weights_.size()) - 2}; i >= 0; --i) {
    gradient = DeriveActivate(neurons_[i + 1], acivation_);
    Matrix transposed = Transpose(weights_[i + 1]);
    errors = MultiplyWinograd(transposed, errors);
    errors = MultiplyHadamard(errors, gradient);
    UpdateWeights(errors, lr, i);
  }
}

void MatrixMlp::UpdateWeights(const Matrix &errors, double lr,
                              std::size_t idx) {
  Matrix transposed = Transpose(neurons_[idx]);
  Matrix step = MultiplyNumber(errors, lr);
  Matrix diff = MultiplyWinograd(step, transposed);
  weights_[idx] = Subtraction(weights_[idx], diff);
}

Vector MatrixMlp::GetOutput() const {
  Vector output(neurons_.back().size(), 0);
  for (std::size_t i{0u}; i < neurons_.back().size(); ++i) {
    output[i] = neurons_.back()[i][0];
  }
  return output;
}

Vector MatrixMlp::GetWeights() const {
  Vector weights;
  for (auto &matrix : weights_) {
    for (std::size_t row{0u}; row < matrix.size(); ++row) {
      for (std::size_t col{0u}; col < matrix[0].size(); ++col) {
        weights.push_back(matrix[row][col]);
      }
    }
  }
  return weights;
}

void MatrixMlp::SetWeights(const Vector &weights) {
  std::size_t i{0u};
  for (auto &matrix : weights_) {
    for (std::size_t row{0u}; row < matrix.size(); ++row) {
      for (std::size_t col{0u}; col < matrix[0].size(); ++col) {
        matrix[row][col] = weights[++i];
      }
    }
  }
}

ActivationFunction MatrixMlp::GetActivationFunction() const {
  return acivation_;
}

void MatrixMlp::SetActivationFunction(ActivationFunction activation) {
  acivation_ = activation;
}

}  // namespace s21
