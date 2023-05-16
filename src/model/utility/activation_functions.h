#ifndef MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
#define MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_

#include <cmath>

namespace s21 {

enum class ActivationFunction { kSigmoid, kTanh, kRelu };

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double DerivativeSigmoid(double x) { return x * (1.0 - x); }

double Tanh(double x) { return tanh(x); }
double DerivativeTanh(double x) { 1.0 - tanh(x) * tanh(x); }

double Relu(double x) { return (x > 0.0) ? x : 0.0; }
double DerivativeRelu(double x) { return (x > 0.0) ? 1.0 : 0.0; }

double ApplyActivate(double x, ActivationFunction func) {
  if (func == ActivationFunction::kSigmoid) return Sigmoid(x);
  if (func == ActivationFunction::kTanh) return Tanh(x);
  if (func == ActivationFunction::kRelu) return Relu(x);
}

double ApplyDerivativeActivate(double x, ActivationFunction func) {
  if (func == ActivationFunction::kSigmoid) return DerivativeSigmoid(x);
  if (func == ActivationFunction::kTanh) return DerivativeTanh(x);
  if (func == ActivationFunction::kRelu) return DerivativeRelu(x);
}

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
