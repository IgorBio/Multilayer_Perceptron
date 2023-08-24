#include "controller.h"

namespace s21 {

void Controller::SetMFunc(std::function<void(Metrics)> func) {
  model_->SetMFunc(func);
}

void Controller::SetPFunc(std::function<void(int)> func) {
  model_->SetPFunc(func);
}

void Controller::SetFPFunc(std::function<void(double)> func) {
  model_->SetFPFunc(func);
}

void Controller::SetType(int idx) {
  idx == 0 ? model_->SetType(s21::Config::ModelType::kMatrix)
           : model_->SetType(s21::Config::ModelType::kGraph);
}

void Controller::UpdateTopology(int hidden_num) {
  model_->UpdateTopology(hidden_num, 100);
}

void Controller::LoadWeights(const std::string &filepath) {
  model_->Load(filepath);
}

size_t Controller::GetHiddenLayersNum() {
  return model_->GetTopology().GetHiddenCount();
}

void Controller::SaveWeights(const std::string &filepath) {
  model_->Save(filepath);
}

void Controller::SetTrainDataset(const std::string &filepath) {
  model_->SetTrainDataset(filepath);
}

void Controller::SetExperimentDataset(const std::string &filepath) {
  model_->SetTestDataset(filepath);
}

size_t Controller::GetSizeOfTrainDataset() {
  return model_->GetTrainDatasetSize();
}

size_t Controller::GetSizeOfExperimentDataset() {
  return model_->GetTestDatasetSize();
}

void Controller::Train(s21::Config::TrainType type, int epoch_num,
                       double leaning_rate) {
  model_->SetTrainType(type);
  model_->SetEpochs(epoch_num);
  model_->SetKFolds(epoch_num);
  model_->SetLearningRate(leaning_rate);
  model_->Train();
}

void Controller::Test(double sample_part) {
  model_->SetTestSample(sample_part);
  model_->Test();
}

char Controller::GetPredict(std::vector<double> &image) {
  Image result(image);
  return model_->Predict(result);
}

}  // namespace s21
