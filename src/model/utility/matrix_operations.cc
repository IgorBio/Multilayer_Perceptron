#include "matrix_operations.h"

namespace s21 {

void RandomMatrixParallel(Matrix& m) {
  std::size_t rows = m.size(), cols = m[0].size();
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) { RandomElement(m, idx); };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
}

Matrix SubtractionParallel(const Matrix& m1, const Matrix& m2) {
  if (m1.size() != m2.size() or m1[0].size() != m2[0].size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }
  std::size_t rows = m1.size(), cols = m1[0].size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) { SubtractElement(matrix, m1, m2, idx); };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  return matrix;
}

Matrix TransposeParallel(const Matrix& m) {
  std::size_t rows = m[0].size(), cols = m.size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) { TransposeElement(matrix, m, idx); };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  return matrix;
}

Matrix MultiplyNumberParallel(const Matrix& m, const double d) {
  std::size_t rows = m.size(), cols = m[0].size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) { MultiplyNumberElement(matrix, m, d, idx); };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  return matrix;
}

Matrix MultiplyHadamardParallel(const Matrix& m1, const Matrix& m2) {
  if (m1.size() != m2.size() or m1[0].size() != m2[0].size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }
  std::size_t rows = m1.size(), cols = m1[0].size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) { MultiplyElement(matrix, m1, m2, idx); };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  return matrix;
}

Matrix MultiplyWinogradParallel(const Matrix& m1, const Matrix& m2) {
  if (m1[0].size() != m2.size()) {
    throw std::logic_error("Matrices have inconsistent dimensions");
  }
  std::size_t rows = m1.size(), cols = m2[0].size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Vector row_factor = Vector(rows);
  Vector col_factor = Vector(cols);
  Threads threads = Threads(2);

  threads[0] = std::move(std::thread{[&]() {
    for (std::size_t idx{0u}; idx < rows; ++idx) {
      GetRowFactor(m1, row_factor, idx);
    }
  }});
  threads[1] = std::move(std::thread{[&]() {
    for (std::size_t idx{0}; idx < cols; ++idx) {
      GetColFactor(m2, col_factor, idx);
    }
  }});

  threads[0].join();
  threads[1].join();

  threads.resize(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) {
    GetResultMatrix(matrix, m1, m2, row_factor, col_factor, idx);
  };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  OddMatrixProcessing(matrix, m1, m2);
  return matrix;
}

Matrix ApplyActivationParallel(const Matrix& m) {
  std::size_t rows = m.size(), cols = m[0].size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) { ApplyActivationElement(matrix, m, idx); };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  return matrix;
}

Matrix ApplyDerivativeActivationParallel(const Matrix& m) {
  std::size_t rows = m.size(), cols = m[0].size();
  Matrix matrix = Matrix(rows, Vector(cols));
  Threads threads = Threads(std::thread::hardware_concurrency());

  auto job = [&](std::size_t idx) {
    ApplyDerivativeActivationElement(matrix, m, idx);
  };

  for (std::size_t idx{0u}; idx < rows; ++idx) {
    if (threads[idx % threads.size()].joinable()) {
      threads[idx % threads.size()].join();
    }
    threads[idx % threads.size()] = std::move(std::thread{job, idx});
  }
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
  return matrix;
}

void RandomElement(Matrix& m, const std::size_t i) {
  for (std::size_t j{0}; j < m[0].size(); ++j) {
    m[i][j] = RandomWeight();
  }
}

double RandomWeight() {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::random_device rd;
  return dist(rd);
}

void SubtractElement(Matrix& res, const Matrix& m1, const Matrix& m2,
                     const std::size_t i) {
  for (std::size_t j{0}; j < res[0].size(); ++j) {
    res[i][j] = m1[i][j] - m2[i][j];
  }
}

void TransposeElement(Matrix& res, const Matrix& m, const std::size_t i) {
  for (std::size_t j{0}; j < res[0].size(); ++j) {
    res[i][j] = m[j][i];
  }
}

void MultiplyNumberElement(Matrix& res, const Matrix& m, double d,
                           const std::size_t i) {
  for (std::size_t j{0}; j < res[0].size(); ++j) {
    res[i][j] = m[i][j] * d;
  }
}

void MultiplyElement(Matrix& res, const Matrix& m1, const Matrix& m2,
                     const std::size_t i) {
  for (std::size_t j{0}; j < res[0].size(); ++j) {
    res[i][j] = m1[i][j] * m2[i][j];
  }
}

void GetRowFactor(const Matrix& m, Vector& v, const std::size_t i) {
  v[i] = m[i][0] * m[i][1];
  for (std::size_t j{1}; j < m[0].size() / 2; ++j) {
    v[i] += m[i][2 * j] * m[i][2 * j + 1];
  }
}

void GetColFactor(const Matrix& m, Vector& v, const std::size_t i) {
  v[i] = m[0][i] * m[1][i];
  for (std::size_t j{1}; j < m.size() / 2; ++j) {
    v[i] += m[2 * j][i] * m[2 * j + 1][i];
  }
}

void GetResultMatrix(Matrix& res, const Matrix& m1, const Matrix& m2,
                     const Vector rf, const Vector cf, const std::size_t i) {
  for (std::size_t j{0}; j < m2[0].size(); ++j) {
    res[i][j] = -rf[i] - cf[j];
    for (std::size_t k{0}; k < m1[0].size() / 2; ++k) {
      res[i][j] +=
          (m1[i][2 * k] + m2[2 * k + 1][j]) * (m1[i][2 * k + 1] + m2[2 * k][j]);
    }
  }
}

void OddMatrixProcessing(Matrix& res, const Matrix& m1, const Matrix& m2) {
  if (m1[0].size() / 2 % 2) {
    for (std::size_t i{0}; i < m1.size(); ++i) {
      for (std::size_t j{0}; j < m2[0].size(); ++j) {
        res[i][j] += m1[i][m1[0].size() - 1] * m2[m2.size() - 1][j];
      }
    }
  }
}

void ApplyActivationElement(Matrix& res, const Matrix& m, const std::size_t i) {
  for (std::size_t j{0}; j < res[0].size(); ++j) {
    res[i][j] = Sigmoid(m[i][j]);
  }
}

void ApplyDerivativeActivationElement(Matrix& res, const Matrix& m,
                                      const std::size_t i) {
  for (std::size_t j{0}; j < res[0].size(); ++j) {
    res[i][j] = DerivativeSigmoid(m[i][j]);
  }
}

}  // namespace s21
