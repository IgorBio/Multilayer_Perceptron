#ifndef MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
#define MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_

#include <array>
#include <cmath>

namespace s21 {

enum class ActivationFunction { kSigmoid, kTanh, kRelu };

const int kSigmoidTableSize = 20001;
const double kSigmoidTableStep = 0.001;
const double kSigmoidTableOffset = 10.0;

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double DerivativeSigmoid(double x) { return x * (1.0 - x); }

inline double Tanh(double x) { return std::tanh(x); }
inline double DerivativeTanh(double x) {
  return 1.0 - std::tanh(x) * std::tanh(x);
}

inline double Relu(double x) { return (x > 0.0) ? x : 0.0; }
inline double DerivativeRelu(double x) { return (x > 0.0) ? 1.0 : 0.0; }

const std::array<double, kSigmoidTableSize> sigmoid_table = []() {
  std::array<double, kSigmoidTableSize> table;
  for (std::size_t i = 0; i < kSigmoidTableSize; ++i) {
    table[i] = Sigmoid((i - kSigmoidTableSize / 2) * kSigmoidTableStep);
  }
  return table;
}();

inline double SigmoidInterpolated(double x) {
  if (x < -kSigmoidTableOffset or x > kSigmoidTableOffset) {
    return Sigmoid(x);  // fallback to original implementation
  }

  const double i = (x + kSigmoidTableOffset) / kSigmoidTableStep;
  const int j = static_cast<int>(i);
  const double a = sigmoid_table[j];
  const double b = sigmoid_table[j + 1];

  return a + (b - a) * (i - j);
}

inline double ApplyActivation(double x, ActivationFunction func) {
  if (func == ActivationFunction::kSigmoid) return SigmoidInterpolated(x);
  if (func == ActivationFunction::kTanh) return Tanh(x);
  if (func == ActivationFunction::kRelu) return Relu(x);
  throw std::invalid_argument("Invalid activation function type");
}

inline double ApplyDerivativeActivation(double x, ActivationFunction func) {
  if (func == ActivationFunction::kSigmoid) return DerivativeSigmoid(x);
  if (func == ActivationFunction::kTanh) return DerivativeTanh(x);
  if (func == ActivationFunction::kRelu) return DerivativeRelu(x);
  throw std::invalid_argument("Invalid activation function type");
}

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
