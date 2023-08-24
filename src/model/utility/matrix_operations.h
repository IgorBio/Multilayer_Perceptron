#ifndef MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
#define MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "activation_functions.h"

namespace s21 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Threads = std::vector<std::thread>;

// Use Winograd algorithm for large matrices to improve performance.
constexpr int kWinogradThreshold = 200;

template <typename Op>
Matrix BinaryOp(const Matrix &, const Matrix &, Op);
Matrix Addition(const Matrix &, const Matrix &);
Matrix Subtraction(const Matrix &, const Matrix &);
Matrix Multiplication(const Matrix &, const Matrix &);
Matrix MultiplyHadamard(const Matrix &, const Matrix &);
Matrix MultiplyNumber(const Matrix &, const double);
Matrix Transpose(const Matrix &);
Matrix Activate(const Matrix &, activation_func);
Matrix ActivateDerivative(const Matrix &, activation_derivative);
Matrix Multiply(const Matrix &, const Matrix &);
Matrix MultiplyWinograd(const Matrix &, const Matrix &);
void RandomizeMatrix(Matrix &);
void RandomizeVector(Vector &);
double RandomWeight();

Matrix operator+(const Matrix &, const Matrix &);
Matrix operator-(const Matrix &, const Matrix &);
Matrix operator*(const Matrix &, const Matrix &);
Matrix operator*(const Matrix &, const double);
void operator-=(Matrix &, const Matrix &);

void ComputeRowFactors(const Matrix &, Vector &);
void ComputeColFactors(const Matrix &, Vector &);
void ComputeResultMatrix(const Matrix &, const Matrix &, const Vector &,
                         const Vector &, Matrix &, std::size_t, std::size_t);

void PrintVector(const Vector &);
void PrintMatrix(const Matrix &);

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_MATRIX_OPERATIONS_H_
