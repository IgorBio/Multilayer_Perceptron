#ifndef SRC_MODEL_MLP_H_
#define SRC_MODEL_MLP_H_

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

// #include "../image.h"
// #include "../reader/csv_reader.h"
// #include "graph_network/graph_network.h"
// #include "io/weight_writer.h"
// #include "matrix_network/matrix_network.h"
// #include "network_interface.h"
#include "config.h"

namespace s21 {

class MLP {
 public:
  MLP(ModelType type, Architecture architecture)
      : type_(type), architecture_(architecture) {
    switch (type) {
      case ModelType::kMatrix:
        network_ = std::make_unique<MatrixNetwork>(architecture);
        break;
      case ModelType::kGraph:
        network_ = std::make_unique<GraphNetwork>(architecture);
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
  const Architecture& GetSettings() const { return architecture_; }

 private:
  std::vector<double> ExpectedOutput(const Image& image);

  ModelType type_;
  Architecture architecture_;
  std::unique_ptr<NetworkInterface> network_;
};

}  // namespace s21

#endif  // SRC_MODEL_MLP_H_
