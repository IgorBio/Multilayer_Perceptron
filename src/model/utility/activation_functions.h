#ifndef MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
#define MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_

#include <cmath>

namespace s21 {

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double DerivativeSigmoid(double x) { return x * (1.0 - x); }

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_ACTIVATION_FUNCTIONS_H_
