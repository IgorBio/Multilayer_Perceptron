#include "mlp.h"
// #include "../config.h"
#include <iostream>

// #include "graph_mlp/neuron.h"
// #include "graph_mlp/layer.h"

using namespace s21;

void PrintVector(const Vector v) {
  for (auto elem : v) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;
}

void PrintMatrix(const Matrix m) {
  for (auto vector : m) {
    for (auto elem : vector) {
      std::cout << elem << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  // Neuron neuron(8);
  // PrintVector(neuron.GetWeights());

  // Layer layer(10, 10);

  Topology topology{.hidden_layers = 2u,
                    .input_layer = 784u,
                    .hidden_layer = 240u,
                    .output_layer = 26u};

  MLP mlp{Config::ModelType::kMatrix, topology};
  mlp.SetTrainDataset("../datasets/emnist-letters/sample-train.csv");
  mlp.SetVerbose(true);
  mlp.SetTrainType(Config::TrainType::kCrossValidation);
  // mlp.SetTestDataset("../datasets/emnist-letters/emnist-letters-test.csv");
  mlp.Train();
  return 0;
}