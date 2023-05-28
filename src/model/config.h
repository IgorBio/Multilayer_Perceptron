#ifndef MLP_MODEL_CONFIG_H_
#define MLP_MODEL_CONFIG_H_

#include <cstddef>

namespace s21 {

struct Metrics {
  double accuracy = 0.0;
  double precision = 0.0;
  double recall = 0.0;
  double f_measure = 0.0;
  double loss = 0.0;
  std::size_t time = 0u;
};

struct Topology {
  std::size_t hidden_layers = 2u;
  std::size_t input_layer = 784u;
  std::size_t hidden_layer = 240u;
  std::size_t output_layer = 26u;
};

class Config {
 public:
  enum class ModelType { kMatrix, kGraph };
  enum class TrainType { kTrain, kCrossValidation };

  explicit Config()
      : model_type_{ModelType::kMatrix},
        train_type_{TrainType::kTrain},
        test_sample_{1.0},
        k_folds_{3u},
        epochs_{5u},
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
