#ifndef MLP_MODEL_MLP_INTERFACE_H_
#define MLP_MODEL_MLP_INTERFACE_H_

#include <vector>

namespace s21 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Weights = std::vector<Matrix>;

class Interface {
 public:
  virtual ~Interface() {}

  virtual void SetInputLayer(const Vector &) = 0;
  virtual void ForwardPropagation() = 0;
  virtual void BackPropagation(const Vector &, double) = 0;
  virtual double CalculateLoss(const Vector &, const Vector &) = 0;
  virtual Vector Predict(const Vector &) = 0;
  virtual Vector GetOutput() const = 0;
  virtual Weights GetWeights() const = 0;
  virtual void SetWeights(const Weights &) = 0;
};
}  // namespace s21

#endif  // MLP_MODEL_MLP_INTERFACE_H_
