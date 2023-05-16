#ifndef MLP_MODEL_MLP_H_
#define MLP_MODEL_MLP_H_

#include <atomic>
// #include <ctime>
// #include <fstream>
#include <algorithm>
#include <functional>
// #include <iomanip>
#include <chrono>
#include <list>
#include <memory>
// #include <sstream>
#include <vector>

#include "../io/image.h"
// #include "../reader/csv_reader.h"
// #include "graph_network/graph_network.h"
// #include "io/weight_writer.h"
#include "matrix_mlp/matrix_mlp.h"
// #include "network_interface.h"
#include "config.h"

namespace s21 {

class MLP {
 public:
  MLP(ModelType type, Topology topology) : type_(type), topology_(topology) {
    switch (type) {
      case ModelType::kMatrix:
        mlp_ = std::make_unique<MatrixMlp>(topology);
        break;
      case ModelType::kGraph:
        mlp_ = std::make_unique<GraphNetwork>(topology);
        break;
    }
  }

  void Train(const std::list<Image>& data, std::size_t epochs,
             double learning_rate = 0.15,
             std::function<void()> start_callback = nullptr,
             std::function<void(std::size_t)> epoch_progress_callback = nullptr,
             std::function<void(std::size_t)> epoch_end_callback = nullptr,
             std::function<void()> end_callback = nullptr,
             const std::atomic_bool& exit = std::atomic_bool(false));
  void TrainEpoch(const std::list<Image>& data, double learning_rate,
                  std::function<void(std::size_t)> epoch_progress_callback,
                  const std::atomic_bool& exit);

  Metrics Test(const std::list<Image>& data, double part,
               std::function<void()> start_callback = nullptr,
               std::function<void(std::size_t)> progress_callback = nullptr,
               std::function<void(Metrics)> end_callback = nullptr,
               const std::atomic_bool& exit = std::atomic_bool(false));

  std::vector<double> Prediction(const Image& image);
  std::pair<std::size_t, double> Predict(const Image& image);

  std::vector<double> GetWeights() const;
  void SetWeights(const std::vector<double>& weights);

  ModelType GetType() const { return type_; }
  const Topology& GetSettings() const { return topology_; }

 private:
  std::vector<double> ExpectedOutput(const Image& image);

  ModelType type_;
  Topology topology_;
  std::unique_ptr<MatrixMlp> mlp_;
};

}  // namespace s21

#endif  // MLP_MODEL_MLP_H_
