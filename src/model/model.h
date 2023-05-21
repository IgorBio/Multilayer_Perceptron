#ifndef MLP_MODEL_MODEL_H_
#define MLP_MODEL_MODEL_H_

#include <atomic>
// #include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

// #include "neural_network/io/weight_reader.h"
#include "./io/emnist_parser.h"
#include "./matrix_mlp/matrix_mlp.h"

namespace s21 {

class Model {
 public:
  Model(ModelType type, Topology topology);

  const Config& GetConfig() const { return config_; }
  void SetConfig(const Config& config) { config_ = config; }

  const ModelType& GetType() const { return type_; }
  void SetType(const ModelType& type) { type_ = type; }

  const Topology& GetTopology() const { return topology_; }
  void SetTopology(const Topology& topology) { topology_ = topology; }

  Vector& GetWeights() const;
  void SetWeights(const Vector& weights);
  void SetWeights(const std::string& filename);

  void SetTrainDataset(const std::string& filename);
  void SetTestDataset(const std::string& filename);

  void Train();
  void StopTrain();

  void Test();
  void StopTest();

  void TrainCrossValidation();

 private:
  void Normalize(std::vector<Image>* images);

 private:
  Config config_;
  Topology topology_;
  ModelType type_;
  std::unique_ptr<MatrixMlp> m_mlp_;
  std::unique_ptr<EmnistParser> parser_ = std::make_unique<EmnistParser>();
  std::vector<Image> train_;
  std::vector<Image> test_;
};

}  // namespace s21

#endif  // MLP_MODEL_MODEL_H_
