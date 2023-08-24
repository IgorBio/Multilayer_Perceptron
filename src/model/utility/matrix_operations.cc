#include "matrix_operations.h"

namespace s21 {

/**
 * Applies a binary operation to two matrices of the same size element-wise.
 *
 * @tparam Op The type of the binary operation functor.
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param op The binary operation functor.
 * @return A new matrix that contains the result of the operation.
 * @throws std::logic_error if the input matrices have inconsistent dimensions.
 */
template <typename Op>
Matrix BinaryOp(const Matrix& m1, const Matrix& m2, Op op) {
  if (m1.empty() or m2.empty() or m1.size() != m2.size() or
      m1[0].size() != m2[0].size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }
  Matrix result_matrix(m1.size(), Vector(m1[0].size()));
  // #pragma omp parallel for
  for (std::size_t i = 0; i < m1.size(); ++i) {
    const Vector &row_m1 = m1[i], &row_m2 = m2[i];
    Vector& row_result = result_matrix[i];
    for (std::size_t j = 0; j < row_m1.size(); ++j) {
      row_result[j] = op(row_m1[j], row_m2[j]);
    }
  }

  return result_matrix;
}

/**
 * Performs matrix addition of two input matrices.
 *
 * @param m1 The first input matrix to be added.
 * @param m2 The second input matrix to be added.
 * @return A new matrix representing the sum of m1 and m2.
 */
Matrix Addition(const Matrix& m1, const Matrix& m2) {
  auto add = [](double a, double b) { return a + b; };
  return BinaryOp(m1, m2, add);
}

/**
 * Performs a subtraction operation between two matrices.
 *
 * @param m1 The first input matrix.
 * @param m2 The second input matrix.
 * @return  A new matrix after performing the subtraction operation.
 */
Matrix Subtraction(const Matrix& m1, const Matrix& m2) {
  auto sub = [](double a, double b) { return a - b; };
  return BinaryOp(m1, m2, sub);
}

/**
 * Multiplies two matrices element-wise using the Hadamard product.
 *
 * @param m1 The first input matrix.
 * @param m2 The second input matrix.
 * @return  A new matrix after performing the element-wise multiplication
 * operation.
 */
Matrix MultiplyHadamard(const Matrix& m1, const Matrix& m2) {
  auto mul = [](double a, double b) { return a * b; };
  return BinaryOp(m1, m2, mul);
}

/**
 * Multiplies two matrices using the standard matrix multiplication algorithm.
 *
 * @param m1 The first input matrix to be multiplied.
 * @param m2 The second input matrix to be multiplied.
 * @return A new matrix after performing the matrix multiplication operation.
 * @throws std::logic_error if matrices have inconsistent dimensions.
 */
Matrix Multiplication(const Matrix& m1, const Matrix& m2) {
  if (m1.empty() or m2.empty() or m1[0].size() != m2.size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }
  const std::size_t rows_m1 = m1.size(), cols_m2 = m2[0].size();
  Matrix result_matrix(rows_m1, Vector(cols_m2));
  // #pragma omp parallel for
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

/**
 * Multiplies a matrix with a scalar value.
 *
 * @param matrix The input matrix to be multiplied.
 * @param d The scalar value with which the matrix is to be multiplied.
 * @return A new matrix after performing the scalar multiplication operation.
 * @throws std::logic_error if the matrix is empty.
 */
Matrix MultiplyNumber(const Matrix& matrix, const double d) {
  if (matrix.empty() or matrix[0].empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  Matrix result_matrix(matrix.size(), Vector(matrix[0].size()));
  // #pragma omp parallel for
  for (std::size_t i = 0; i < matrix.size(); ++i) {
    std::transform(matrix[i].begin(), matrix[i].end(), result_matrix[i].begin(),
                   [&](double x) { return x * d; });
  }

  return result_matrix;
}

/**
 * Transpose a matrix.
 *
 * @param matrix The input matrix to be transposed.
 * @return A new transposed matrix.
 * @throws std::logic_error if the matrix is empty.
 */
Matrix Transpose(const Matrix& matrix) {
  if (matrix.empty() or matrix[0].empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  const std::size_t rows = matrix[0].size(), cols = matrix.size();
  Matrix result_matrix(rows, Vector(cols));
  // #pragma omp parallel for
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      result_matrix[i][j] = matrix[j][i];
    }
  }

  return result_matrix;
}

/**
 * Apply an activation function element-wise to a matrix.
 *
 * @param matrix The input matrix to be activated.
 * @param func The activation function to be applied.
 * @return A new matrix with the activation function applied element-wise.
 * @throws std::logic_error if the matrix is empty.
 */
Matrix Activate(const Matrix& matrix, activation_func func) {
  if (matrix.empty() or matrix[0].empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  Matrix result_matrix(matrix.size(), Vector(matrix[0].size()));
  // #pragma omp parallel for
  for (std::size_t i = 0; i < matrix.size(); ++i) {
    std::transform(matrix[i].begin(), matrix[i].end(), result_matrix[i].begin(),
                   [&](double x) { return ApplyActivation(x, func); });
  }

  return result_matrix;
}

/**
 * Apply the derivative of an activation function element-wise to a matrix.
 *
 * @param matrix The input matrix to be activated.
 * @param func The derivative of the activation function to be applied.
 * @return A new matrix with the derivative of the activation function applied
 * element-wise.
 * @throws std::logic_error if the matrix is empty.
 */
Matrix ActivateDerivative(const Matrix& matrix, activation_derivative func) {
  if (matrix.empty() or matrix[0].empty()) {
    throw std::logic_error("Matrix have inconsistent dimensions");
  }
  Matrix result_matrix(matrix.size(), Vector(matrix[0].size()));
  // #pragma omp parallel for
  for (std::size_t i = 0; i < matrix.size(); ++i) {
    std::transform(
        matrix[i].begin(), matrix[i].end(), result_matrix[i].begin(),
        [&](double x) { return ApplyActivationDerivative(x, func); });
  }

  return result_matrix;
}

/**
 * Multiplies two matrices using the Winograd algorithm.
 *
 * @param m1 The first input matrix to be multiplied.
 * @param m2 The second input matrix to be multiplied.
 * @return A new matrix after performing the multiplication via the Winograd
 * algorithm.
 * @throws std::logic_error if matrices have inconsistent dimensions.
 */
Matrix MultiplyWinograd(const Matrix& m1, const Matrix& m2) {
  if (m1.empty() or m2.empty() or m1[0].size() != m2.size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }

  const std::size_t rows_m1 = m1.size(), cols_m2 = m2[0].size();
  Matrix result_matrix(rows_m1, Vector(cols_m2));

  Vector row_factors(rows_m1);
  ComputeRowFactors(m1, row_factors);

  Vector col_factors(cols_m2);
  ComputeColFactors(m2, col_factors);

  const std::size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::size_t chunk_size = rows_m1 / num_threads;
  std::size_t start_row = 0, end_row = chunk_size;
  for (std::size_t t = 0; t < num_threads; ++t) {
    if (t == num_threads - 1) end_row = rows_m1;
    threads[t] = std::thread(ComputeResultMatrix, std::cref(m1), std::cref(m2),
                             std::cref(row_factors), std::cref(col_factors),
                             std::ref(result_matrix), start_row, end_row);
    start_row += chunk_size;
    end_row += chunk_size;
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return result_matrix;
}

/**
 * Randomizes the elements of a matrix. The input matrix is modified in place.
 *
 * @param matrix The matrix to be randomized.
 */
void RandomizeMatrix(Matrix& matrix) {
  // #pragma omp parallel for
  for (Vector& vector : matrix) {
    RandomizeVector(vector);
  }
}

/**
 * Randomizes the elements of a vector by generating a new random value in the
 * range [-1.0, 1.0] for each element. The input vector is modified in place.
 *
 * @param vector The vector to be randomized.
 */
void RandomizeVector(Vector& vector) {
  // #pragma omp parallel for
  for (double& value : vector) {
    value = RandomWeight();
  }
}

/**
 * Generates a random weight value.
 *
 * @param vector The vector to be randomized in the range [-1.0, 1.0] using a
 * static random number generator and uniform distribution.
 * @return A randomly generated weight value.
 */
double RandomWeight() {
  static std::mt19937_64 gen(std::random_device{}());
  static std::uniform_real_distribution<double> dist(-0.5, 0.5);
  return dist(gen);
}

/**
 * Computes the row factors matrix for Winograd algorithm.
 *
 * @param m1 The first input matrix of the multiplication.
 * @param row_factors The vector of row factors.
 */
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

/**
 * Computes the column factors matrix for Winograd algorithm.
 *
 * @param m2 The second input matrix of the multiplication.
 * @param col_factors The vector of column factors.
 */
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

/**
 * Computes the result matrix by multiplying the two given matrices m1 and m2,
 * and subtracting the row and column factors.
 *
 * @param m1 The first input matrix to be multiplied.
 * @param m2 The second input matrix to be multiplied.
 * @param row_factors The vector of row factors to subtract from each row of the
 * computed matrix.
 * @param col_factors The vector of column factors to subtract from each column
 * of the computed matrix.
 * @param result_matrix The output matrix that will store the computed result.
 * @param start_row The starting row index (inclusive).
 * @param end_row The ending row index (exclusive).
 */
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

/**
 * Multiplies two matrices m1 and m2 using either the standard multiplication
 * algorithm or the Winograd algorithm, depending on the size of the matrices.
 *
 * @param m1 The first input matrix to be multiplied.
 * @param m2 The second input matrix to be multiplied.
 * @return A new matrix after performing the matrix multiplication operation.
 * @throws std::logic_error if matrices have inconsistent dimensions.
 */
Matrix Multiply(const Matrix& m1, const Matrix& m2) {
  if (m1.empty() or m2.empty() or m1[0].size() != m2.size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }

  if (m1.size() > kWinogradThreshold and m2[0].size() > kWinogradThreshold and
      m1[0].size() > kWinogradThreshold) {
    return MultiplyWinograd(m1, m2);
  } else {
    return Multiplication(m1, m2);
  }
}

/**
 * Overloaded operator+ that performs matrix addition.
 *
 * @param m1 The first input matrix to be added.
 * @param m2 The second input matrix to be added.
 * @return A new matrix representing the sum of m1 and m2.
 */
Matrix operator+(const Matrix& m1, const Matrix& m2) {
  return Addition(m1, m2);
}

/**
 * Overloaded operator- that performs matrix subtraction.
 *
 * @param m1 The first input matrix to be subtracted.
 * @param m2 The second input matrix to be subtracted.
 * @return A new matrix after performing the subtraction operation.
 */
Matrix operator-(const Matrix& m1, const Matrix& m2) {
  return Subtraction(m1, m2);
}

/**
 * Overloaded operator* that performs matrix multiplication.
 *
 * @param m1 The first input matrix to be multiplied.
 * @param m2 The second input matrix to be multiplied.
 * @return A new matrix after performing the matrix multiplication operation.
 */
Matrix operator*(const Matrix& m1, const Matrix& m2) {
  return Multiply(m1, m2);
}

/**
 * Overloaded operator* that multiplies a matrix with a scalar value.
 *
 * @param matrix The input matrix to be multiplied.
 * @param d The scalar value with which the matrix is to be multiplied.
 * @return A new matrix after performing the scalar multiplication operation.
 */
Matrix operator*(const Matrix& matrix, const double d) {
  return MultiplyNumber(matrix, d);
}

/**
 * Performs a subtraction operation between two matrices.
 *
 * @param m1 The first input matrix.
 * @param m2 The second input matrix.
 */
void operator-=(Matrix& m1, const Matrix& m2) { m1 = m1 - m2; }

/**
 * Prints all elements of a given vector to the standard output stream.
 *
 * @param vector The vector to print.
 */
void PrintVector(const Vector& vector) {
  for (const auto& elem : vector) {
    std::cout << elem << ' ';
  }
  std::cout << '\n';
}

/**
 * Prints all elements of a given matrix to the standard output stream.
 *
 * @param matrix The matrix to print.
 */
void PrintMatrix(const Matrix& matrix) {
  for (const auto& vector : matrix) {
    PrintVector(vector);
  }
  std::cout << '\n';
}

}  // namespace s21
