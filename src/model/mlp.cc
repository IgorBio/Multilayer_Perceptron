#include "mlp.h"

namespace s21 {

MLP::MLP(const Config::ModelType type, const Topology topology)
    : type_{type}, topology_{topology} {
  if (type_ == Config::ModelType::kMatrix) {
    mlp_ = std::make_unique<MatrixMlp>(topology_);
  } else if (type_ == Config::ModelType::kGraph) {
    mlp_ = std::make_unique<GraphMlp>(topology_);
  }
}

void MLP::Train() {
  if (config_.GetTrainType() == Config::TrainType::kTrain) {
    TrainEpochs();
  } else if (config_.GetTrainType() == Config::TrainType::kCrossValidation) {
    CrossValidate();
  }
}

void MLP::TrainEpochs() {
  auto start_time = std::chrono::steady_clock::now();
  for (std::size_t epoch{0u}; epoch < config_.GetEpochs(); ++epoch) {
    double total_loss{0.0};
    std::random_shuffle(train_.begin(), train_.end());
    for (const auto& image : train_) {
      mlp_->SetInputLayer(image.GetPixels());
      mlp_->ForwardPropagation();
      auto expected_output = ExpectedOutput(image);
      auto predicted_output = mlp_->GetOutput();
      total_loss += mlp_->CalculateLoss(predicted_output, expected_output);
      mlp_->BackPropagation(expected_output, config_.GetLearningRate());
    }
    metrics_.time = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - start_time)
                        .count();
    metrics_.loss = total_loss / static_cast<double>(train_.size());
    if (Verbose()) Report(epoch, {});
  }
}

void MLP::CrossValidate() {
  auto start_time = std::chrono::steady_clock::now();
  std::vector<Dataset> folds(config_.GetKFolds());
  std::size_t fold_size = train_.size() / config_.GetKFolds();
  std::vector<std::size_t> indices(train_.size());
  std::iota(indices.begin(), indices.end(), 0u);
  std::random_shuffle(indices.begin(), indices.end());
  for (std::size_t i{0u}; i < indices.size(); ++i) {
    folds[i % config_.GetKFolds()].push_back(train_[indices[i]]);
  }

  for (std::size_t epoch{0u}; epoch < config_.GetEpochs(); ++epoch) {
    double total_loss{0.0};
    for (std::size_t fold_index{0u}; fold_index < config_.GetKFolds();
         ++fold_index) {
      Dataset validation_fold(folds[fold_index]);
      Dataset train_fold;
      for (std::size_t i{0u}; i < folds.size(); ++i) {
        if (i != fold_index) {
          train_fold.insert(train_fold.end(), folds[i].begin(), folds[i].end());
        }
      }

      std::random_shuffle(train_fold.begin(), train_fold.end());
      for (const auto& image : train_fold) {
        mlp_->SetInputLayer(image.GetPixels());
        mlp_->ForwardPropagation();
        auto expected_output = ExpectedOutput(image);
        auto predicted_output = mlp_->GetOutput();
        total_loss +=
            mlp_->CalculateLoss({predicted_output}, {expected_output}) /
            train_fold.size();

        mlp_->BackPropagation(expected_output, config_.GetLearningRate());
      }

      metrics_.time = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - start_time)
                          .count();
      metrics_.loss = total_loss;
      if (Verbose()) Report(epoch, fold_index);
    }
  }
}

/**
 * Reports training and validation metrics for a given epoch and fold.
 *
 * @param epoch The current epoch number.
 * @param fold_index The index of the current fold (if using cross-validation).
 */
void MLP::Report(const std::size_t epoch, const std::size_t fold_index = 0u) {
  std::string fold_info;
  if (config_.GetTrainType() == Config::TrainType::kCrossValidation) {
    fold_info = " (fold " + std::to_string(fold_index + 1) + ")";
  }

  std::cout << "Epoch " << epoch + 1 << fold_info << ":\n"
            << "\tTime: " << metrics_.time << " sec\n"
            << "\tLoss: " << metrics_.loss << std::endl;
}

Vector MLP::ExpectedOutput(const Image& image) {
  Vector expected_output(topology_.output_layer, 0.0);
  expected_output[image.GetLabel()] = 1.0;
  return expected_output;
}

}  // namespace s21
