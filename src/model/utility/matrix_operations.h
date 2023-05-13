#ifndef SRC_MODEL_UTILITY_MATRIX_OPERATIONS_H_
#define SRC_MODEL_UTILITY_MATRIX_OPERATIONS_H_

#include <functional>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "activation_functions.h"

namespace s21 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

void RandomMatrixParallel(Matrix &);
Matrix SubtractionParallel(const Matrix &, const Matrix &);
Matrix TransposeParallel(const Matrix &);
Matrix MultiplyNumberParallel(const Matrix &, const double);
Matrix MultiplyHadamardParallel(const Matrix &, const Matrix &);
Matrix MultiplyWinogradParallel(const Matrix &, const Matrix &);
Matrix ApplyActivationParallel(const Matrix &);
Matrix ApplyDerivativeActivationParallel(const Matrix &);

void RandomElement(Matrix &, const std::size_t);
double RandomWeight();
void SubtractElement(Matrix &, const Matrix &, const Matrix &,
                     const std::size_t);
void TransposeElement(Matrix &, const Matrix &, const std::size_t);
void MultiplyNumberElement(Matrix &, const Matrix &, double, const std::size_t);
void MultiplyElement(Matrix &, const Matrix &, const Matrix &,
                     const std::size_t);
void GetRowFactor(const Matrix &, Vector &, const std::size_t);
void GetColFactor(const Matrix &, Vector &, const std::size_t);
void GetResultMatrix(Matrix &, const Matrix &, const Matrix &, const Vector,
                     const Vector, const std::size_t);
void OddMatrixProcessing(Matrix &, const Matrix &, const Matrix &);
void ApplyActivationElement(Matrix &, const Matrix &, const std::size_t);
void ApplyDerivativeActivationElement(Matrix &, const Matrix &,
                                      const std::size_t);

}  // namespace s21

#endif  // SRC_MODEL_UTILITY_MATRIX_OPERATIONS_H_
