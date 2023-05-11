#ifndef SRC_MODEL_UTILITY_WINOGRAD_H_
#define SRC_MODEL_UTILITY_WINOGRAD_H_

#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace s21 {
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

Matrix MultiplyMaxThreads(const Matrix&, const Matrix&);
Matrix MultiplyPipeline(const Matrix&, const Matrix&);
void GetRowFactor(const Matrix&, Vector&, const std::size_t);
void GetColFactor(const Matrix&, Vector&, const std::size_t);
void GetResultMatrix(Matrix&, const Matrix&, const Matrix&, const Vector,
                     const Vector, const std::size_t);
void OddMatrixProcessing(Matrix&, const Matrix&, const Matrix&);
}  // namespace s21

#endif  // SRC_MODEL_UTILITY_WINOGRAD_H_
