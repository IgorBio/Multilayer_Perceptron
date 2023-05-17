#ifndef MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
#define MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_

#include <algorithm>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace s21 {

enum class Parallel { kMaxThreads, kOpenMP };

enum class ActivationFunction;

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

void Randomize(Matrix &);
double RandomWeight();

Matrix Subtraction(const Matrix &, const Matrix &);
Matrix Transpose(const Matrix &);
Matrix MultiplyNumber(const Matrix &, const double);
Matrix MultiplyHadamard(const Matrix &, const Matrix &);

Matrix MultiplyWinograd(const Matrix &, const Matrix &, Parallel);
Matrix Activate(const Matrix &, Parallel, ActivationFunction);
Matrix DeriveActivate(const Matrix &, Parallel, ActivationFunction);

Matrix MultiplyWinogradThreads(const Matrix &, const Matrix &);
Matrix MultiplyWinogradOmp(const Matrix &, const Matrix &);
Matrix ActivateThreads(const Matrix &, ActivationFunction);
Matrix ActivateOmp(const Matrix &, ActivationFunction);
Matrix DeriveActivateThreads(const Matrix &, ActivationFunction);
Matrix DeriveActivateOmp(const Matrix &, ActivationFunction);

void GetRowFactor(const Matrix &, Vector &, const std::size_t);
void GetColFactor(const Matrix &, Vector &, const std::size_t);
void GetResultMatrix(Matrix &, const Matrix &, const Matrix &, const Vector,
                     const Vector, const std::size_t);
void OddMatrixProcessing(Matrix &, const Matrix &, const Matrix &);
void ActivateElement(Matrix &, const Matrix &, const std::size_t,
                     ActivationFunction);
void DeriveActivateElement(Matrix &, const Matrix &, const std::size_t,
                           ActivationFunction);
}  // namespace s21

#endif  // MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
