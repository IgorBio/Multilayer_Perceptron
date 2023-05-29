#ifndef MLP_MODEL_MLP_H_
#define MLP_MODEL_MLP_H_

#include "config.h"
#include "graph_mlp/graph_mlp.h"
#include "io/io.h"
#include "matrix_mlp/matrix_mlp.h"

namespace s21 {

class MLP {
 public:
  explicit MLP(const Topology topology);

  void Train();
  void Test();
  char Predict(const Image&);
  std::size_t PredictLabel(const Image&);

  Vector GetWeights() const { return mlp_->GetWeights(); }
  void SetWeights(const Vector& weights);
  void SetWeights(const std::string& path);

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
  Vector ExpectedOutput(const Image& image);
  void Report(const std::size_t epoch, const std::size_t fold_index);
  void TrainEpochs();
  void CrossValidate();

  Config config_;
  Topology topology_;
  std::unique_ptr<Interface> mlp_;
  Dataset train_;
  Dataset test_;
  Metrics metrics_;
};

}  // namespace s21

#endif  // MLP_MODEL_MLP_H_
