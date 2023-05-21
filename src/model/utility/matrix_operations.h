#ifndef MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
#define MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_

#include <algorithm>
#include <future>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "activation_functions.h"

namespace s21 {

enum class ActivationFunction;

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

template <typename Op>
Matrix BinaryOp(const Matrix &, const Matrix &, Op);
Matrix Addition(const Matrix &, const Matrix &);
Matrix Subtraction(const Matrix &, const Matrix &);
Matrix MultiplyHadamard(const Matrix &, const Matrix &);
Matrix MultiplyNumber(const Matrix &, const double);
void RandomizeMatrix(Matrix &);
void RandomizeVector(Vector &);
Matrix Transpose(const Matrix &);
Matrix Activate(const Matrix &, ActivationFunction);
Matrix DeriveActivate(const Matrix &, ActivationFunction);
Matrix MultiplyWinograd(const Matrix &, const Matrix &);

void ComputeRowFactors(const Matrix &, Vector &);
void ComputeColFactors(const Matrix &, Vector &);
void ComputeResultMatrix(const Matrix &, const Matrix &, const Vector &,
                         const Vector &, Matrix &, std::size_t, std::size_t);
}  // namespace s21

#endif  // MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
