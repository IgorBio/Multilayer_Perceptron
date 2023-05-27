#include "matrix_operations.h"

namespace s21 {

void PrintVector(const Vector v) {
  for (auto elem : v) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;
}

void PrintMatrix(const Matrix m) {
  for (auto vector : m) {
    for (auto elem : vector) {
      std::cout << elem << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename Op>
Matrix BinaryOp(const Matrix& m1, const Matrix& m2, Op op) {
  if (m1.empty() or m2.empty() or m1.size() != m2.size() or
      m1[0].size() != m2[0].size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }
  Matrix matrix(m1.size(), Vector(m1[0].size()));
#pragma omp parallel for
  for (std::size_t i = 0u; i < m1.size(); ++i) {
    const auto &row_m1 = m1[i], &row_m2 = m2[i];
    Vector& row_result = matrix[i];
    for (std::size_t j = 0u; j < row_m1.size(); ++j) {
      row_result[j] = op(row_m1[j], row_m2[j]);
    }
  }

  return matrix;
}

Matrix Addition(const Matrix& m1, const Matrix& m2) {
  auto add = [](double a, double b) { return a + b; };
  return BinaryOp(m1, m2, add);
}

Matrix Subtraction(const Matrix& m1, const Matrix& m2) {
  auto sub = [](double a, double b) { return a - b; };
  return BinaryOp(m1, m2, sub);
}

Matrix MultiplyHadamard(const Matrix& m1, const Matrix& m2) {
  auto mul = [](double a, double b) { return a * b; };
  return BinaryOp(m1, m2, mul);
}

// Randomize a matrix
void RandomizeMatrix(Matrix& matrix) {
#pragma omp parallel for
  for (auto& vector : matrix) {
    RandomizeVector(vector);
  }
}

// Randomize a vector
void RandomizeVector(Vector& vector) {
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
#pragma omp parallel for
  for (double& value : vector) {
    value = dist(gen);
  }
}

double RandomWeight() {
  static std::mt19937_64 gen(std::random_device{}());
  static std::uniform_real_distribution<double> dist(-1.0, 1.0);
  return dist(gen);
}

// Multiply a matrix by a scalar
Matrix MultiplyNumber(const Matrix& matrix, const double d) {
  if (matrix.empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  Matrix result(matrix.size(), Vector(matrix[0].size()));
#pragma omp parallel for
  for (std::size_t i = 0; i < matrix.size(); ++i) {
    std::transform(matrix[i].begin(), matrix[i].end(), result[i].begin(),
                   [&](const auto& x) { return x * d; });
  }
  return result;
}

Matrix Transpose(const Matrix& matrix) {
  if (matrix.empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  const std::size_t rows = matrix[0].size(), cols = matrix.size();
  Matrix result(rows, Vector(cols));
#pragma omp parallel for
  for (std::size_t i = 0u; i < rows; ++i) {
    for (std::size_t j = 0u; j < cols; ++j) {
      result[i][j] = matrix[j][i];
    }
  }

  return result;
}

// Apply an activation function to a matrix
Matrix Activate(const Matrix& matrix, activation_func func) {
  if (matrix.empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  Matrix result(matrix.size(), Vector(matrix[0].size()));
#pragma omp parallel for
  for (std::size_t i = 0; i < matrix.size(); ++i) {
    std::transform(matrix[i].begin(), matrix[i].end(), result[i].begin(),
                   [&](const auto& x) { return ApplyActivation(x, func); });
  }
  return result;
}

// Apply the derivative of an activation function to a matrix
Matrix ActivateDerivative(const Matrix& matrix, activation_derivative func) {
  if (matrix.empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  Matrix result(matrix.size(), Vector(matrix[0].size()));
#pragma omp parallel for
  for (std::size_t i = 0; i < matrix.size(); ++i) {
    std::transform(
        matrix[i].begin(), matrix[i].end(), result[i].begin(),
        [&](const auto& x) { return ApplyActivationDerivative(x, func); });
  }
  return result;
}

Matrix MultiplyWinograd(const Matrix& m1, const Matrix& m2) {
  if (m1.empty() or m2.empty() or m1[0].size() != m2.size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }

  const std::size_t rows_m1 = m1.size(), cols_m2 = m2[0].size();
  Matrix result_matrix(rows_m1, Vector(cols_m2));

  // Compute row-wise factors
  Vector row_factors(rows_m1);
  ComputeRowFactors(m1, row_factors);

  // Compute column-wise factors
  Vector col_factors(cols_m2);
  ComputeColFactors(m2, col_factors);

  // Compute the result matrix in parallel using threads
  const std::size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::size_t chunk_size = rows_m1 / num_threads;
  std::size_t start_row = 0, end_row = chunk_size;
  for (std::size_t t = 0; t < num_threads; ++t) {
    if (t == num_threads - 1) {
      end_row = rows_m1;
    }
    threads[t] = std::thread(ComputeResultMatrix, std::cref(m1), std::cref(m2),
                             std::cref(row_factors), std::cref(col_factors),
                             std::ref(result_matrix), start_row, end_row);
    start_row += chunk_size;
    end_row += chunk_size;
  }
  // Join all threads
  for (auto& thread : threads) {
    thread.join();
  }

  return result_matrix;
}

// Compute row-wise factors
void ComputeRowFactors(const Matrix& m1, Vector& row_factors) {
  const std::size_t half = m1[0].size() / 2;
  for (std::size_t i = 0; i < m1.size(); ++i) {
    double factor = m1[i][0] * m1[i][1];
    for (std::size_t j = 1; j < half; ++j) {
      factor += m1[i][2 * j] * m1[i][2 * j + 1];
    }
    row_factors[i] = factor;
  }
}

// Compute column-wise factors
void ComputeColFactors(const Matrix& m2, Vector& col_factors) {
  const std::size_t half = m2.size() / 2;
  for (std::size_t i = 0; i < m2[0].size(); ++i) {
    double factor = m2[0][i] * m2[1][i];
    for (std::size_t j = 1; j < half; ++j) {
      factor += m2[2 * j][i] * m2[2 * j + 1][i];
    }
    col_factors[i] = factor;
  }
}

// Compute the result matrix
void ComputeResultMatrix(const Matrix& m1, const Matrix& m2,
                         const Vector& row_factors, const Vector& col_factors,
                         Matrix& result_matrix, std::size_t start_row,
                         std::size_t end_row) {
  const std::size_t cols_m2 = m2[0].size();
  const std::size_t half = m1[0].size() / 2;
  for (std::size_t i = start_row; i < end_row; ++i) {
    for (std::size_t j = 0; j < cols_m2; ++j) {
      double dot_product = -row_factors[i] - col_factors[j];
      for (std::size_t k = 0; k < half; ++k) {
        dot_product += (m1[i][2 * k] + m2[2 * k + 1][j]) *
                       (m1[i][2 * k + 1] + m2[2 * k][j]);
      }
      if (m1[0].size() % 2 != 0) {
        dot_product += m1[i][m1[0].size() - 1] * m2[m1[0].size() - 1][j];
      }
      result_matrix[i][j] = dot_product;
    }
  }
}

Matrix Multiply(const Matrix& m1, const Matrix& m2) {
  if (m1.empty() or m2.empty() or m1[0].size() != m2.size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }

  const std::size_t rows_m1 = m1.size(), cols_m2 = m2[0].size();
  if (rows_m1 > 200 and cols_m2 > 200 and m1[0].size() > 200) {
    return MultiplyWinograd(m1, m2);
  }
  Matrix result_matrix(rows_m1, Vector(cols_m2));
#pragma omp parallel for
  for (std::size_t i = 0; i < rows_m1; ++i) {
    for (std::size_t j = 0; j < cols_m2; ++j) {
      result_matrix[i][j] = 0.0;
      for (std::size_t k = 0; k < m1[0].size(); ++k) {
        result_matrix[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
  return result_matrix;
}

Matrix operator+(const Matrix& m1, const Matrix& m2) {
  return Addition(m1, m2);
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
  return Subtraction(m1, m2);
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
  return Multiply(m1, m2);
}

Matrix operator*(const Matrix& m1, const double d) {
  return MultiplyNumber(m1, d);
}

}  // namespace s21
