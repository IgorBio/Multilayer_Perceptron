#ifndef MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_
#define MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_

#include "../config.h"
#include "../utility/matrix_operations.h"
#include "layer.h"

namespace s21 {

using Net = std::vector<std::unique_ptr<Layer>>;

class GraphMlp {
 public:
  explicit GraphMlp(Topology);

 private:
  Net net_;
  activation_func acivation_{sigmoid};
};

}  // namespace s21

#endif  // MLP_MODEL_GRAPH_MLP_GRAPH_MLP_H_
