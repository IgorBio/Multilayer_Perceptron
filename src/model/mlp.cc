#include "mlp.h"

namespace s21 {

MLP::MLP(ModelType type, Topology topology) : type_(type), topology_(topology) {
  switch (type) {
    case ModelType::kMatrix:
      mlp_ = std::make_unique<MatrixMlp>(topology);
      break;
    case ModelType::kGraph:
      mlp_ = std::make_unique<GraphMlp>(topology);
      break;
  }
}

void TrainEpoch(const std::vector<Image>& dataset, double learning_rate,
                bool exit);
}  // namespace s21
