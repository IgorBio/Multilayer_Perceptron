#ifndef MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
#define MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_

#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace s21 {

enum class Parallel { kMaxThreads, kOpenMP };

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

void RandomMatrix(Matrix &, Parallel);
Matrix Subtraction(const Matrix &, const Matrix &, Parallel);
Matrix Transpose(const Matrix &, Parallel);
Matrix MultiplyNumber(const Matrix &, const double, Parallel);
Matrix MultiplyHadamard(const Matrix &, const Matrix &, Parallel);
Matrix MultiplyWinograd(const Matrix &, const Matrix &, Parallel);
Matrix Activate(const Matrix &, Parallel);
Matrix DeriveActivate(const Matrix &, Parallel);

void RandomMatrixThreads(Matrix &);
void RandomMatrixOmp(Matrix &);
Matrix SubtractionThreads(const Matrix &, const Matrix &);
Matrix SubtractionOmp(const Matrix &, const Matrix &);
Matrix TransposeThreads(const Matrix &);
Matrix TransposeOmp(const Matrix &);
Matrix MultiplyNumberThreads(const Matrix &, const double);
Matrix MultiplyNumberOmp(const Matrix &, const double);
Matrix MultiplyHadamardThreads(const Matrix &, const Matrix &);
Matrix MultiplyHadamardOmp(const Matrix &, const Matrix &);
Matrix MultiplyWinogradThreads(const Matrix &, const Matrix &);
Matrix MultiplyWinogradOmp(const Matrix &, const Matrix &);
Matrix ApplyActivationThreads(const Matrix &);
Matrix ApplyActivationOmp(const Matrix &);
Matrix ApplyDerivativeActivationThreads(const Matrix &);
Matrix ApplyDerivativeActivationOmp(const Matrix &);

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

#endif  // MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
