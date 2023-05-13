#include "matrix_mlp.h"

namespace s21 {

MatrixMlp::MatrixMlp(Architecture architecture)
    : layers_(architecture.hidden_layers + 2) {
  AddWheights(architecture.hidden_layer, architecture.input_layer);

  for (std::size_t i{0u}; i < architecture.hidden_layers - 1; ++i) {
    AddWheights(architecture.hidden_layer, architecture.hidden_layer);
  }

  AddWheights(architecture.output_layer, architecture.hidden_layer);
}

void MatrixMlp::AddWheights(std::size_t rows, std::size_t cols) {
  Matrix wheights(rows, Vector(cols));
  RandomMatrixParallel(wheights);
  weights_.push_back(wheights);
}

void MatrixMlp::SetInputLayer(const Vector &layer) {
  Matrix input = Matrix(layer.size(), Vector(1));
  for (std::size_t i{0u}; i < layer.size(); ++i) {
    input[i][0] = layer[i];
  }
  layers_.front() = std::move(input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i{0u}; i < weights_.size(); ++i) {
    layers_[i + 1] = ApplyActivationParallel(
        MultiplyWinogradParallel(weights_[i], layers_[i]));
  }
}

void MatrixMlp::BackPropagation(const Vector &answer, double lr) {
  Matrix output = Matrix(answer.size(), Vector(1));
  for (std::size_t i{0u}; i < answer.size(); ++i) {
    output[i][0] = answer[i];
  }
  Matrix errors = SubtractionParallel(layers_.back(), output);
  Matrix gradient = ApplyDerivativeActivationParallel(layers_.back());
  errors = MultiplyHadamardParallel(errors, gradient);
  AdjustWeights(errors, lr, weights_.size() - 1);

  for (int i{static_cast<int>(weights_.size()) - 2}; i >= 0; --i) {
    gradient = ApplyDerivativeActivationParallel(layers_[i + 1]);
    Matrix transposed = TransposeParallel(weights_[i + 1]);
    errors = MultiplyWinogradParallel(transposed, errors);
    errors = MultiplyHadamardParallel(errors, gradient);
    AdjustWeights(errors, lr, i);
  }
}

void MatrixMlp::AdjustWeights(const Matrix &errors, double lr, size_t idx) {
  Matrix transposed = TransposeParallel(layers_[idx]);
  Matrix step = MultiplyNumberParallel(errors, lr);
  Matrix diff = MultiplyWinogradParallel(step, transposed);
  weights_[idx] = SubtractionParallel(weights_[idx], diff);
}

Vector MatrixMlp::GetOutput() {
  Vector result(layers_.back().size(), 0);
  for (size_t i{0u}; i < layers_.back().size(); ++i) {
    result[i] = layers_.back()[i][0];
  }
  return result;
}

Vector MatrixMlp::GetWeights() {
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

}  // namespace s21
