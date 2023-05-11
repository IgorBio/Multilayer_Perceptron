#ifndef SRC_MODEL_UTILITY_MATRIX_OPERATIONS_H_
#define SRC_MODEL_UTILITY_MATRIX_OPERATIONS_H_

#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace s21 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

Matrix SubtractionParallel(const Matrix&, const Matrix&);
Matrix TransposeParallel(const Matrix&);
Matrix MultiplyNumberParallel(const Matrix&, const double);
Matrix MultiplyHadamardParallel(const Matrix&, const Matrix&);
Matrix MultiplyWinogradParallel(const Matrix&, const Matrix&);
void SubtractElement(Matrix&, const Matrix&, const Matrix&, const std::size_t);
void TransposeElement(Matrix&, const Matrix&, const std::size_t);
void MultiplyNumberElement(Matrix&, const Matrix&, double, const std::size_t);
void MultiplyElement(Matrix&, const Matrix&, const Matrix&, const std::size_t);
void GetRowFactor(const Matrix&, Vector&, const std::size_t);
void GetColFactor(const Matrix&, Vector&, const std::size_t);
void GetResultMatrix(Matrix&, const Matrix&, const Matrix&, const Vector,
                     const Vector, const std::size_t);
void OddMatrixProcessing(Matrix&, const Matrix&, const Matrix&);

}  // namespace s21

#endif  // SRC_MODEL_UTILITY_MATRIX_OPERATIONS_H_
