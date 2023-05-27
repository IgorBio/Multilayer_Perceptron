#ifndef MLP_MODEL_MLP_H_
#define MLP_MODEL_MLP_H_

#include <iostream>

#include "config.h"
#include "graph_mlp/graph_mlp.h"
#include "io/io.h"
#include "matrix_mlp/matrix_mlp.h"

namespace s21 {

class MLP {
 public:
  explicit MLP(const Config::ModelType type, const Topology topology);

  void Train();
  void Test();
  Vector Predict(const Image& image);

  Vector GetWeights() const;
  void SetWeights(const Vector& weights);

  void SetTrainDataset(const std::string& path) { train_ = ParseEmnist(path); }
  void SetTestDataset(const std::string& path) { test_ = ParseEmnist(path); }

  Config::ModelType GetType() const { return type_; }
  void SetType(Config::ModelType type) { type_ = type; }

  const Topology& GetTopology() const { return topology_; }

  bool Verbose() const { return config_.GetVerbose(); }
  void SetVerbose(bool verbose) { config_.SetVerbose(verbose); }

  void SetTrainType(Config::TrainType type) { config_.SetTrainType(type); }

 private:
  Vector ExpectedOutput(const Image& image);
  void Report(const std::size_t epoch, const std::size_t fold_index);
  void TrainEpochs();
  void CrossValidate();

  Config config_;
  Config::ModelType type_;
  Topology topology_;
  std::unique_ptr<Interface> mlp_;
  Dataset train_;
  Dataset test_;
  Metrics metrics_;
};

}  // namespace s21

#endif  // MLP_MODEL_MLP_H_
