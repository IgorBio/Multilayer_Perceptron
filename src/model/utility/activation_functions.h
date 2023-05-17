#ifndef MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
#define MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_

#include <array>
#include <cmath>

namespace s21 {

enum class ActivationFunction { kSigmoid, kTanh, kRelu };

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double DerivativeSigmoid(double x) { return x * (1.0 - x); }

inline double Tanh(double x) { return std::tanh(x); }
inline double DerivativeTanh(double x) {
  return 1.0 - std::tanh(x) * std::tanh(x);
}

inline double Relu(double x) { return (x > 0.0) ? x : 0.0; }
inline double DerivativeRelu(double x) { return (x > 0.0) ? 1.0 : 0.0; }

const std::array<double, 20001> sigmoid_table = []() {
  std::array<double, 20001> table;
  for (int i = -10000; i <= 10000; ++i) {
    table[i + 10000] = Sigmoid(i / 1000.0);
  }
  return table;
}();

double ApplyActivation(double x, ActivationFunction func) {
  if (func == ActivationFunction::kSigmoid) {
    if (x >= -10.0 and x <= 10.0) {
      return sigmoid_table[static_cast<int>(x * 1000) + 10000];
    }
    return Sigmoid(x);
  }
  if (func == ActivationFunction::kTanh) return Tanh(x);
  if (func == ActivationFunction::kRelu) return Relu(x);
  throw std::invalid_argument("Invalid activation function type");
}

double ApplyDerivativeActivation(double x, ActivationFunction func) {
  if (func == ActivationFunction::kSigmoid) return DerivativeSigmoid(x);
  if (func == ActivationFunction::kTanh) return DerivativeTanh(x);
  if (func == ActivationFunction::kRelu) return DerivativeRelu(x);
  throw std::invalid_argument("Invalid activation function type");
}

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
