#ifndef MLP_MODEL_CONFIG_H_
#define MLP_MODEL_CONFIG_H_

#include <vector>

namespace s21 {

enum class ModelType { kMatrix, kGraph };

// struct Metrics {
//   double accuracy = 0.0;
//   double precision = 0.0;
//   double recall = 0.0;
//   double f_measure = 0.0;
//   std::size_t time = 0u;
// };

struct Topology {
  std::size_t hidden_layers = 2u;
  std::size_t input_layer = 784u;
  std::size_t hidden_layer = 240u;
  std::size_t output_layer = 26u;
};

class Config {
 public:
  enum class TrainType { kTrain, kCrossValidation };

  ModelType GetModelType() const { return model_type_; }
  void SetModelType(ModelType type) { model_type_ = type; }

  std::size_t GetHiddenLayers() const { return hidden_layers_; }
  void SetHiddenLayers(std::size_t layers) { hidden_layers_ = layers; }

  TrainType GetTrainType() const { return train_type_; }
  void SetTrainType(TrainType type) { train_type_ = type; }

  double GetTestSample() const { return test_sample_; }
  void SetTestSample(double sample) { test_sample_ = sample; }

  std::size_t GetKFolds() const { return k_folds_; }
  void SetKFolds(std::size_t folds) { k_folds_ = folds; }

  std::size_t GetEpochs() const { return epochs_; }
  void SetEpochs(std::size_t epochs) { epochs_ = epochs; }

  double GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(double rate) { learning_rate_ = rate; }

 private:
  ModelType model_type_ = ModelType::kMatrix;
  std::size_t hidden_layers_ = 2u;
  TrainType train_type_ = TrainType::kTrain;
  double test_sample_ = 1.0;
  std::size_t k_folds_ = 1u;
  std::size_t epochs_ = 5u;
  double learning_rate_ = 0.1;
  double activation_threshold_ = 0.5;
};

}  // namespace s21

#endif  // MLP_MODEL_CONFIG_H_
