#ifndef MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
#define MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_

#include <cmath>
#include <functional>

namespace s21 {

using activation_func = double (*)(double);
using activation_derivative = double (*)(double);

// Activation functions
constexpr double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
constexpr double tanh(double x) { return std::tanh(x); }
constexpr double relu(double x) { return (x > 0.0) ? x : 0.0; }

// Activation function derivatives
constexpr double sigmoid_derivative(double x) { return x * (1.0 - x); }
constexpr double tanh_derivative(double x) {
  return 1.0 - std::tanh(x) * std::tanh(x);
}
constexpr double relu_derivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }

// Apply an activation function to a single value
inline double ApplyActivation(double x, activation_func func) {
  return (*func)(x);
}

// Apply derivative of an activation function to a single value
inline double ApplyActivationDerivative(double x, activation_derivative func) {
  return (*func)(x);
}
}  // namespace s21

#endif  // MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
