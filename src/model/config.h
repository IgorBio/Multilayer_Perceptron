#ifndef MLP_MODEL_CONFIG_H_
#define MLP_MODEL_CONFIG_H_

#include <cstddef>
#include <vector>

namespace s21 {

/**
 * @class Topology
 * @brief Represents the network topology or architecture.
 *
 * The Topology class provides methods to retrieve and modify the architecture
 * of a neural network, including the number of layers, sizes of layers,
 * input and output sizes, and hidden layer sizes.
 */
class Topology {
 public:
  Topology() : sizes_{784, 100, 100, 26} {}
  explicit Topology(std::size_t hidden_num) : sizes_{784} {
    sizes_.insert(sizes_.end(), hidden_num, 100);
    sizes_.push_back(26);
  }
  explicit Topology(std::initializer_list<std::size_t> sizes) : sizes_{sizes} {}
  explicit Topology(const std::vector<std::size_t>& sizes) : sizes_{sizes} {}

  std::size_t GetInputSize() const { return sizes_.front(); }
  void SetInputSize(std::size_t size) { sizes_.front() = size; }
  std::size_t GetOutputSize() const { return sizes_.back(); }
  void SetOutputSize(std::size_t size) { sizes_.back() = size; }
  std::size_t GetLayersCount() const { return sizes_.size(); }
  std::size_t GetHiddenCount() const { return sizes_.size() - 2; }
  std::size_t GetLayerSize(std::size_t idx) const { return sizes_[idx]; }
  void SetLayerSize(std::size_t size, std::size_t idx) { sizes_[idx] = size; }
  std::size_t GetLastHidden() const { return sizes_[sizes_.size() - 2]; }
  void SetTopology(const std::vector<std::size_t>& sizes) { sizes_ = sizes; }

 private:
  std::vector<std::size_t> sizes_;
};

/**
 * @class Config
 * @brief Configuration settings for the neural network.
 *
 * The Config class holds various configuration settings that define how the
 * neural network behaves during training and testing. It provides methods to
 * get and set values for different configuration options.
 */
class Config {
 public:
  enum class ModelType { kMatrix, kGraph };
  enum class TrainType { kTrain, kCrossValidation };

  explicit Config()
      : model_type_{ModelType::kMatrix},
        train_type_{TrainType::kTrain},
        test_sample_{1.0},
        k_folds_{3},
        epochs_{5},
        learning_rate_{0.1},
        activate_threshold_{0.5},
        verbose_{false} {}

  ModelType GetModelType() const { return model_type_; }
  void SetModelType(ModelType type) { model_type_ = type; }
  TrainType GetTrainType() const { return train_type_; }
  void SetTrainType(TrainType type) { train_type_ = type; }
  double GetTestSample() const { return test_sample_; }
  void SetTestSample(double sample) { test_sample_ = sample; }
  std::size_t GetKFolds() const { return k_folds_; }
  void SetKFolds(std::size_t k_folds) { k_folds_ = k_folds; }
  std::size_t GetEpochs() const { return epochs_; }
  void SetEpochs(std::size_t epochs) { epochs_ = epochs; }
  double GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(double rate) { learning_rate_ = rate; }
  bool GetVerbose() const { return verbose_; }
  void SetVerbose(bool verbose) { verbose_ = verbose; }
  double GetActivateThreshold() const { return activate_threshold_; }
  void SetActivateThreshold(double thr) { activate_threshold_ = thr; }

 private:
  ModelType model_type_;
  TrainType train_type_;
  double test_sample_;
  std::size_t k_folds_;
  std::size_t epochs_;
  double learning_rate_;
  double activate_threshold_;
  bool verbose_;
};

}  // namespace s21

#endif  // MLP_MODEL_CONFIG_H_
