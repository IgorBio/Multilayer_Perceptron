#ifndef MLP_CONTROLLER_H_
#define MLP_CONTROLLER_H_

#include "../model/mlp.h"
#include "image.h"

namespace s21 {
class Controller {
 public:
  Controller() {
    Topology topology;
    model_ = new MLP{topology};
  }
  Controller(const Controller &) = delete;
  Controller(Controller &&) = delete;
  Controller operator=(const Controller &) = delete;
  Controller operator=(Controller &&) = delete;
  ~Controller() {
    if (model_) delete model_;
  }

  void SetMFunc(std::function<void(Metrics)> func);
  void SetPFunc(std::function<void(int)> func);
  void SetFPFunc(std::function<void(double)> func);

  void SetType(int idx);
  void UpdateTopology(int hidden_num);

  void LoadWeights(const std::string &);
  size_t GetHiddenLayersNum();
  void SaveWeights(const std::string &);

  void SetTrainDataset(const std::string &);
  void SetExperimentDataset(const std::string &);
  size_t GetSizeOfTrainDataset();
  size_t GetSizeOfExperimentDataset();

  void Train(s21::Config::TrainType, int, double);
  void Test(double);

  char GetPredict(std::vector<double> &image);

 private:
  MLP *model_ = nullptr;
};
}  // namespace s21

#endif  // MLP_CONTROLLER_H_
