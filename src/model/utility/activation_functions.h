#ifndef MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
#define MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_

#include <array>
#include <cmath>

namespace s21 {

// Enum for different activation functions
enum class ActivationFunction { kSigmoid, kTanh, kRelu };

// Size of the sigmoid table and step size
const int kSigmoidTableSize = 20001;
const double kSigmoidTableStep = 0.001;

// Offset used to calculate values outside the range of the sigmoid table
const double kSigmoidTableOffset = 10.0;

// Basic sigmoid function
inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double DerivativeSigmoid(double x) { return x * (1.0 - x); }

// Hyperbolic tangent function
inline double Tanh(double x) { return std::tanh(x); }
inline double DerivativeTanh(double x) {
  return 1.0 - std::tanh(x) * std::tanh(x);
}

// Rectified linear unit function
inline double Relu(double x) { return (x > 0.0) ? x : 0.0; }
inline double DerivativeRelu(double x) { return (x > 0.0) ? 1.0 : 0.0; }

// Array of precalculated sigmoid values
const std::array<double, kSigmoidTableSize> sigmoid_table = []() {
  std::array<double, kSigmoidTableSize> table;
  std::generate(table.begin(), table.end(),
                [i = -kSigmoidTableSize / 2]() mutable {
                  return Sigmoid((i++ * kSigmoidTableStep));
                });
  return table;
}();

// Interpolate the sigmoid value of a given input
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

// Apply an activation function to a single value
inline double ApplyActivation(double x, ActivationFunction func) {
  switch (func) {
    case ActivationFunction::kSigmoid:
      return SigmoidInterpolated(x);
    case ActivationFunction::kTanh:
      return Tanh(x);
    case ActivationFunction::kRelu:
      return Relu(x);
    default:
      throw std::invalid_argument("Invalid activation function type");
  }
}

// Apply derivative of an activation function to a single value
inline double ApplyDerivativeActivation(double x, ActivationFunction func) {
  switch (func) {
    case ActivationFunction::kSigmoid:
      return DerivativeSigmoid(x);
    case ActivationFunction::kTanh:
      return DerivativeTanh(x);
    case ActivationFunction::kRelu:
      return DerivativeRelu(x);
    default:
      throw std::invalid_argument("Invalid activation function type");
  }
}

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
