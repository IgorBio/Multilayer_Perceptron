#ifndef MLP_MODEL_MLP_H_
#define MLP_MODEL_MLP_H_

#include <iomanip>

#include "config.h"
#include "graph_mlp/graph_mlp.h"
#include "io/io.h"
#include "matrix_mlp/matrix_mlp.h"
#include "metrics.h"

namespace s21 {

class MLP {
 public:
  explicit MLP(const Topology);

  void Train();
  void Test();
  Vector Predict(const Vector&);
  char Predict(const Image&);
  std::size_t PredictLabel(const Image&);
  void Save();
  void Load(const std::string& path);

  Tensor GetWeights() const { return mlp_->GetWeights(); }

  void SetTrainDataset(const std::string& path) { train_ = ParseEmnist(path); }
  void SetTrainDataset(const Dataset& dataset) { train_ = dataset; };
  void SetTestDataset(const std::string& path) { test_ = ParseEmnist(path); }
  void SetTestDataset(const Dataset& dataset) { test_ = dataset; };

  Config::ModelType GetType() const { return config_.GetModelType(); }
  void SetType(Config::ModelType type);

  void SetVerbose(bool verbose) { config_.SetVerbose(verbose); }
  void SetTrainType(Config::TrainType type) { config_.SetTrainType(type); }
  void SetEpochs(std::size_t epochs) { config_.SetEpochs(epochs); }
  void SetLearningRate(double rate) { config_.SetLearningRate(rate); }
  void SetTestSample(double sample) { config_.SetTestSample(sample); }
  void SetKFolds(std::size_t k_folds) { config_.SetKFolds(k_folds); }
  void SetActivateThreshold(double thr) { config_.SetActivateThreshold(thr); }

 private:
  Vector ExpectedOutput(const Image&);
  void Report();
  void TrainEpochs();
  void CrossValidate();
  void UpdateTopology(Tensor&);

  Config config_;
  Topology topology_;
  std::unique_ptr<Interface> mlp_;
  Dataset train_;
  Dataset test_;
  Metrics metrics_;
};

}  // namespace s21

#endif  // MLP_MODEL_MLP_H_
