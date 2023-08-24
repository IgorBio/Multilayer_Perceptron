#ifndef MLP_MODEL_ABSTRACT_MLP_H_
#define MLP_MODEL_ABSTRACT_MLP_H_

#include <vector>

namespace s21 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Tensor = std::vector<Matrix>;

/**
 * @class AbstractMlp
 * @brief Abstract class for Multi-Layer Perceptrons (MLPs).
 *
 * This class defines the interface for Multi-Layer Perceptrons (MLPs) and
 * provides methods for setting the input layer, performing forward and
 * backward propagation, obtaining the output of the MLP, and getting/setting
 * the MLP's weights and biases. This class is abstract, and its methods must
 * be implemented in derived classes.
 */
class AbstractMlp {
 public:
  virtual ~AbstractMlp() = default;

  virtual void SetInputLayer(const Vector &) = 0;
  virtual void ForwardPropagation() = 0;
  virtual void BackPropagation(const Vector &, double) = 0;
  virtual Vector GetOutput() const = 0;
  virtual std::pair<const Tensor, const Tensor> GetMlp() const = 0;
  virtual void SetMlp(const Tensor &, const Tensor &) = 0;
};
}  // namespace s21

#endif  // MLP_MODEL_ABSTRACT_MLP_H_
