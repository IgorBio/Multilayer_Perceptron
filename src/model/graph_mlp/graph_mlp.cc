#include "graph_mlp.h"

namespace s21 {

GraphMlp::GraphMlp(Topology topology) : net_{topology.hidden_layers + 2} {}

}  // namespace s21