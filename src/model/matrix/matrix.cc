#include "matrix.h"

namespace s21 {

MatrixMlp::MatrixMlp(Architecture architecture)
    : values_(architecture.hidden_layers + 2) {
  Matrix weight(architecture.hidden_layer, Vector(architecture.input_layer));

  FillMatrixRandom(weight);
  weights_.push_back(weight);

  for (std::size_t i = 0u; i < architecture.hidden_layers - 1; ++i) {
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
  for (std::size_t i = 0u; i < outputs.size(); ++i) {
    input[i][0] = outputs[i];
  }
  values_.front() = input;
}

void MatrixMlp::ForwardPropagation() {
  for (std::size_t i = 0u; i < weights_.size(); ++i) {
    values_[i + 1] = ActivationFuncMatrix(weights_[i] * values_[i]);
  }
}

MatrixMlp::Matrix Mul(const MatrixMlp::Matrix &m1,
                      const MatrixMlp::Matrix &m2) {
  if (m1.GetColumns() != m2.GetColumns() || m2.GetRows() != m1.GetRows())
    throw std::logic_error("Mul: invalid matrix dims");

  Matrix res(m1.GetRows(), m1.GetColumns());
  for (size_t i = 0; i < m1.GetRows(); ++i)
    for (size_t j = 0; j < m1.GetColumns(); j++) {
      res(i, j) = m1(i, j) * m2(i, j);
    }
  return res;
}

void MatrixMlp::BackPropagation(const Vector &expected_output,
                                double learning_rate_) {
  Matrix error = values_.back() - Matrix(expected_output);
  error = Mul(error, DerivativeActivationFuncMatrix(values_.back()));
  AdjustWeights(weights_.size() - 1, learning_rate_, error);

  for (int i = (int)weights_.size() - 2; i >= 0; i--) {
    error = Mul(weights_[i + 1].Transpose() * error,
                DerivativeActivationFuncMatrix(values_[i + 1]));
    AdjustWeights(i, learning_rate_, error);
  }
}

void MatrixMlp::AdjustWeights(size_t weight_ind, double learning_rate,
                              const Matrix &error) {
  weights_[weight_ind] -=
      learning_rate * error * values_[weight_ind].Transpose();
}

Vector MatrixMlp::GetOutput() {
  Vector result(values_.back().GetRows(), 0);

  for (size_t i = 0; i < values_.back().GetRows(); ++i) {
    result[i] = values_.back()(i, 0);
  }

  return result;
}

Vector MatrixMlp::GetWeights() {
  Vector weights;

  for (auto &matrix : weights_) {
    for (std::size_t row = 0; row < matrix.GetRows(); row++) {
      for (std::size_t column = 0; column < matrix.GetColumns(); column++) {
        weights.push_back(matrix(row, column));
      }
    }
  }

  return weights;
}

void MatrixMlp::LoadWeights(const Vector &weights) {
  std::size_t i = 0;
  for (auto &matrix : weights_) {
    for (std::size_t row = 0; row < matrix.GetRows(); row++) {
      for (std::size_t column = 0; column < matrix.GetColumns(); column++) {
        matrix(row, column) = weights[++i];
      }
    }
  }
}

void MatrixMlp::FillMatrixRandom(Matrix &m) {
  for (size_t i = 0; i < m.GetRows(); ++i)
    for (size_t j = 0; j < m.GetColumns(); j++) {
      m(i, j) = utility::RandomWeight();
    }
}

Matrix MatrixMlp::ActivationFuncMatrix(const Matrix &m) {
  Matrix res(m.GetRows(), m.GetColumns());
  for (size_t i = 0; i < m.GetRows(); ++i)
    for (size_t j = 0; j < m.GetColumns(); j++) {
      res(i, j) = utility::ActivationFunc(m(i, j));
    }
  return res;
}

Matrix MatrixMlp::DerivativeActivationFuncMatrix(const Matrix &m) {
  Matrix res(m.GetRows(), m.GetColumns());
  for (size_t i = 0; i < m.GetRows(); ++i)
    for (size_t j = 0; j < m.GetColumns(); j++) {
      res(i, j) = utility::DerivativeActivFunc(m(i, j));
    }
  return res;
}

}  // namespace s21
