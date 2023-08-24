#ifndef MLP_MODEL_MLP_H_
#define MLP_MODEL_MLP_H_

#include "config.h"
#include "graph_mlp.h"
#include "io.h"
#include "matrix_mlp.h"
#include "metrics.h"

namespace s21 {

/**
 * @class MLP
 * @brief Multi-Layer Perceptron (MLP).
 *
 * The MLP class represents a Multi-Layer Perceptron, capable of training,
 * testing, and making predictions. It provides methods to set datasets,
 * configure parameters, train the model, perform testing, and predict outputs.
 */
class MLP {
 public:
  explicit MLP(const Topology&);

  void Train();
  void Test();
  Vector Predict(const Vector&);
  char Predict(const Image&);
  std::size_t PredictLabel(const Image&);
  void Save(const std::string&);
  void Load(const std::string&);
  void UpdateMlp(const Tensor&, const Tensor&);
  void UpdateTopology(std::size_t hidden, std::size_t size);

  void SetTrainDataset(const std::string& path) { train_ = ParseEmnist(path); }
  void SetTrainDataset(const Dataset& dataset) { train_ = dataset; };
  void SetTestDataset(const std::string& path) { test_ = ParseEmnist(path); }
  void SetTestDataset(const Dataset& dataset) { test_ = dataset; };

  std::size_t GetEpochs() const { return config_.GetEpochs(); }
  double GetTestSample() const { return config_.GetTestSample(); }
  Config::ModelType GetType() const { return config_.GetModelType(); }
  void SetType(Config::ModelType);
  std::size_t GetTrainDatasetSize() { return train_.size(); }
  std::size_t GetTestDatasetSize() { return test_.size(); }
  Topology& GetTopology() { return topology_; }
  Metrics& GetMetrics() { return metrics_; }

  void SetVerbose(bool verbose) { config_.SetVerbose(verbose); }
  void SetTrainType(Config::TrainType type) { config_.SetTrainType(type); }
  void SetEpochs(std::size_t epochs) { config_.SetEpochs(epochs); }
  void SetLearningRate(double rate) { config_.SetLearningRate(rate); }
  void SetTestSample(double sample) { config_.SetTestSample(sample); }
  void SetKFolds(std::size_t k_folds) { config_.SetKFolds(k_folds); }

  void SetMFunc(std::function<void(Metrics)> func) { ptr_metrics_ = func; }
  void SetPFunc(std::function<void(int)> func) { ptr_progress_ = func; }
  void SetFPFunc(std::function<void(double)> func) {
    ptr_full_progress_ = func;
  }

 private:
  Vector ExpectedOutput(const Image&);
  void TrainEpoch(const Dataset&);
  void TrainEpochs();
  void Test(const Dataset&);
  void CrossValidate();

  std::function<void(Metrics&)> ptr_metrics_;
  std::function<void(int)> ptr_progress_;
  std::function<void(double)> ptr_full_progress_;

  Config config_;
  Topology topology_;
  std::unique_ptr<AbstractMlp> mlp_;
  Dataset train_;
  Dataset test_;
  Metrics metrics_;
};

}  // namespace s21

#endif  // MLP_MODEL_MLP_H_
