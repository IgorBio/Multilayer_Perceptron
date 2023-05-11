#include "matrix_mlp.h"

namespace s21 {

MatrixMlp::MatrixMlp(Architecture architecture)
    : values_(architecture.hidden_layers + 2) {
  Matrix weight(architecture.hidden_layer, Vector(architecture.input_layer));

  FillMatrixRandom(weight);
  weights_.push_back(weight);

  for (std::size_t i{0u}; i < architecture.hidden_layers - 1; ++i) {
    weight =
        Matrix(architecture.hidden_layer, Vector(architecture.hidden_layer));
    FillMatrixRandom(weight);
    weights_.push_back(weight);
  }

  weight = Matrix(architecture.output_layer, Vector(architecture.hidden_layer));
  FillMatrixRandom(weight);
  weights_.push_back(weight);
}

void MatrixMlp::SetInput(const Vector &outputs) {
  Matrix input = Matrix(outputs.size(), Vector(1));
  for (std::size_t i{0u}; i < outputs.size(); ++i) {
    input[i][0] = outputs[i];
  }
  values_.front() = std::move(input);
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i{0u}; i < weights_.size(); ++i) {
    values_[i + 1] =
        ActivationFuncMatrix(MultiplyWinogradParallel(weights_[i], values_[i]));
  }
}

void MatrixMlp::BackPropagation(const Vector &expected_output,
                                double learning_rate_) {
  Matrix output = Matrix(expected_output.size(), Vector(1));
  for (std::size_t i{0u}; i < expected_output.size(); ++i) {
    output[i][0] = expected_output[i];
  }
  Matrix error = SubtractionParallel(values_.back(), output);
  error = MultiplyHadamardParallel(
      error, DerivativeActivationFuncMatrix(values_.back()));
  AdjustWeights(weights_.size() - 1, learning_rate_, error);

  for (int i{static_cast<int>(weights_.size()) - 2}; i >= 0; --i) {
    error = MultiplyHadamardParallel(
        MultiplyWinogradParallel(TransposeParallel(weights_[i + 1]), error),
        DerivativeActivationFuncMatrix(values_[i + 1]));
    AdjustWeights(i, learning_rate_, error);
  }
}

void MatrixMlp::AdjustWeights(size_t weight_ind, double learning_rate,
                              const Matrix &error) {
  weights_[weight_ind] = SubtractionParallel(
      weights_[weight_ind],
      MultiplyWinogradParallel(MultiplyNumberParallel(error, learning_rate),
                               TransposeParallel(values_[weight_ind])));
}

Vector MatrixMlp::GetOutput() {
  Vector result(values_.back().size(), 0);

  for (size_t i{0u}; i < values_.back().size(); ++i) {
    result[i] = values_.back()[i][0];
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

void MatrixMlp::LoadWeights(const Vector &weights) {
  std::size_t i{0u};
  for (auto &matrix : weights_) {
    for (std::size_t row{0u}; row < matrix.size(); ++row) {
      for (std::size_t col{0u}; col < matrix[0].size(); ++col) {
        matrix[row][col] = weights[++i];
      }
    }
  }
}

void MatrixMlp::FillMatrixRandom(Matrix &m) {
  for (std::size_t i{0u}; i < m.size(); ++i)
    for (std::size_t j{0}; j < m[0].size(); ++j) {
      m[i][j] = RandomWeight();
    }
}

Matrix MatrixMlp::ActivationFuncMatrix(const Matrix &m) {
  Matrix res(m.size(), Vector(m[0].size()));
  for (std::size_t i{0u}; i < m.size(); ++i)
    for (std::size_t j = 0; j < m[0].size(); ++j) {
      res[i][j] = ActivationFunc(m[i][j]);
    }
  return res;
}

Matrix MatrixMlp::DerivativeActivationFuncMatrix(const Matrix &m) {
  Matrix res(m.size(), Vector(m[0].size()));
  for (std::size_t i{0u}; i < m.size(); ++i)
    for (std::size_t j{0u}; j < m[0].size(); ++j) {
      res[i][j] = DerivativeActivFunc(m[i][j]);
    }
  return res;
}

}  // namespace s21
