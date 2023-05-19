#include "matrix_operations.h"

namespace s21 {

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

// Helper function to iterate over a matrix and apply an operation to each
// element
void UnaryOp(const Matrix& m, const std::function<double(double)>& op,
             Matrix& matrix) {
  if (!m.empty()) {
    const std::size_t rows = m.size(), cols = m[0].size();
#pragma omp parallel for
    for (std::size_t i = 0u; i < rows; ++i) {
      for (std::size_t j = 0u; j < cols; ++j) {
        matrix[i][j] = op(m[i][j]);
      }
    }
  }
}

// Randomize a matrix
void Randomize(Matrix& m) {
  UnaryOp(
      m, [](double) { return RandomWeight(); }, m);
}

double RandomWeight() {
  static uint64_t x = std::random_device{}();
  x ^= x >> 12;  // These constants are chosen based on empirical studies
  x ^= x << 25;
  x ^= x >> 27;
  static constexpr double kNorm = 1.0 / UINT64_MAX;   // Normalize to [0,1]
  return static_cast<double>(x) * kNorm * 2.0 - 1.0;  // Scale to [-1,1]
}

// Multiply a matrix by a scalar
Matrix MultiplyNumber(const Matrix& m, const double d) {
  Matrix matrix(m.size(), Vector(m[0].size()));
  UnaryOp(
      m, [d](double x) { return x * d; }, matrix);
  return matrix;
}

Matrix Transpose(const Matrix& m) {
  if (m.empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  const std::size_t rows = m[0].size(), cols = m.size();
  Matrix matrix(rows, Vector(cols));
#pragma omp parallel for
  for (std::size_t i = 0u; i < rows; ++i) {
    for (std::size_t j = 0u; j < cols; ++j) {
      matrix[i][j] = m[j][i];
    }
  }

  return matrix;
}

// Apply an activation function to a matrix
Matrix Activate(const Matrix& m, ActivationFunction func) {
  Matrix matrix(m.size(), Vector(m[0].size()));
  UnaryOp(
      m, [func](double x) { return ApplyActivation(x, func); }, matrix);
  return matrix;
}

// Matrix Activate(const Matrix& m, ActivationFunction func) {
//   Matrix matrix(m.size(), Vector(m[0].size()));
// #pragma omp parallel for
//   for (std::size_t i = 0u; i < m.size(); ++i) {
//     for (std::size_t j = 0u; j < m[0].size(); ++j) {
//       matrix[i][j] = ApplyActivation(m[i][j], func);
//     }
//   }
//   return matrix;
// }

// Apply the derivative of an activation function to a matrix
Matrix DeriveActivate(const Matrix& m, ActivationFunction func) {
  Matrix matrix(m.size(), Vector(m[0].size()));
  UnaryOp(
      m, [func](double x) { return ApplyDerivativeActivation(x, func); },
      matrix);
  return matrix;
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

}  // namespace s21
