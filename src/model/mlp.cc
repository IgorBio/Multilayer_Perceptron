#include "mlp.h"

namespace s21 {

MLP::MLP(const Topology topology)
    : topology_{topology}, metrics_{topology_.GetOutputSize()} {
  mlp_ = std::make_unique<MatrixMlp>(topology_);
}

void MLP::Train() {
  if (config_.GetTrainType() == Config::TrainType::kTrain) {
    TrainEpochs();
  } else if (config_.GetTrainType() == Config::TrainType::kCrossValidation) {
    CrossValidate();
  }
}

void MLP::TrainEpochs() {
  for (std::size_t epoch{0u}; epoch < config_.GetEpochs(); ++epoch) {
    std::random_shuffle(train_.begin(), train_.end());

    auto start_time = std::chrono::steady_clock::now();

    for (const Image& image : train_) {
      mlp_->SetInputLayer(image.GetPixels());
      mlp_->ForwardPropagation();
      const Vector expected_output = ExpectedOutput(image);
      mlp_->BackPropagation(expected_output, config_.GetLearningRate());
    }

    auto end_time = std::chrono::steady_clock::now();
    auto epoch_time =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count();
    double average_epoch_time =
        static_cast<double>(epoch_time) / static_cast<double>(epoch + 1);
    auto remaining_time = static_cast<long long>(
        (config_.GetEpochs() - epoch - 1) * average_epoch_time);
    if (config_.GetVerbose()) {
      std::cout << "Epoch: " << epoch + 1 << std::endl;
      std::cout << "\nTime Elapsed: " << epoch_time << " seconds\n";
      std::cout << "Time Remaining: " << remaining_time << " seconds\n\n";
    }
    Test();
  }
}

void MLP::Test() {
  auto start_time = std::chrono::steady_clock::now();

  std::vector<std::size_t> indices(test_.size());
  std::iota(indices.begin(), indices.end(), 0u);
  std::random_shuffle(indices.begin(), indices.end());
  std::size_t test_size =
      static_cast<std::size_t>(test_.size() * config_.GetTestSample());

  double total_loss = 0.0;
  for (std::size_t i{0u}; i < test_size; ++i) {
    const Image& image = test_[indices[i]];
    mlp_->SetInputLayer(image.GetPixels());
    mlp_->ForwardPropagation();
    Vector expected_output = ExpectedOutput(image);
    Vector predicted_output = mlp_->GetOutput();

    total_loss += mlp_->CalculateLoss(predicted_output, expected_output);

    std::size_t predicted_label = PredictLabel(image);
    std::size_t true_label = image.GetLabel();
    if (predicted_label == true_label) {
      metrics_.AddTruePositive(true_label);
    } else {
      metrics_.AddFalsePositive(predicted_label);
      metrics_.AddFalseNegative(true_label);
    }
  }
  auto end_time = std::chrono::steady_clock::now();
  metrics_.SetTime(
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
          .count());
  metrics_.SetLoss(total_loss / test_size);
  if (config_.GetVerbose()) Report();
}

void MLP::CrossValidate() {
  // auto start_time = std::chrono::steady_clock::now();
  std::vector<Dataset> folds(config_.GetKFolds());
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
      for (const Image& image : train_fold) {
        mlp_->SetInputLayer(image.GetPixels());
        mlp_->ForwardPropagation();
        Vector expected_output = ExpectedOutput(image);
        Vector predicted_output = mlp_->GetOutput();
        total_loss +=
            mlp_->CalculateLoss({predicted_output}, {expected_output}) /
            train_fold.size();

        mlp_->BackPropagation(expected_output, config_.GetLearningRate());
      }
      if (config_.GetVerbose()) Report();
    }
  }
}

/**
 * Reports training and validation metrics for a given epoch and fold.
 *
 * @param epoch The current epoch number.
 * @param fold_index The index of the current fold (if using cross-validation).
 */
void MLP::Report() {
  std::cout << "Test on " << config_.GetTestSample() * 100 << " %\n";
  std::cout << "\tLoss: " << metrics_.GetLoss() << std::endl;
  std::cout << "\tAccuracy: " << metrics_.GetAccuracy() << std::endl;
  std::cout << "\tPrecision: " << metrics_.GetPrecision() << std::endl;
  std::cout << "\tRecall: " << metrics_.GetRecall() << std::endl;
  std::cout << "\tF1 Score: " << metrics_.GetF1Score() << std::endl;
  std::cout << "\tTotal time: " << metrics_.GetTime() << std::endl;
}

Vector MLP::ExpectedOutput(const Image& image) {
  Vector expected_output(topology_.GetOutputSize(), 0.0);
  expected_output[image.GetLabel()] = 1.0;
  return expected_output;
}

Vector MLP::Predict(const Vector& vector) { return mlp_->Predict(vector); }

char MLP::Predict(const Image& image) {
  return static_cast<char>(PredictLabel(image) - 1) + 'A';
}

std::size_t MLP::PredictLabel(const Image& image) {
  Vector vector = image.GetPixels();
  Vector output = Predict(vector);
  auto it = std::max_element(output.begin(), output.end());
  return std::distance(output.begin(), it);
}

void MLP::SetType(Config::ModelType type) {
  config_.SetModelType(type);
  if (type == Config::ModelType::kMatrix) {
    mlp_ = std::make_unique<MatrixMlp>(topology_);
  } else if (type == Config::ModelType::kGraph) {
    mlp_ = std::make_unique<GraphMlp>(topology_);
  }
}

void MLP::Save() {
  std::stringstream ss;
  ss << "./weights/w_" << topology_.GetLayersCount() << "l_"
     << config_.GetEpochs() << "e_" << metrics_.GetLoss() << "loss";
  auto time = std::time(nullptr);
  ss << "_" << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M") << ".bin";

  SaveWeights(GetWeights(), ss.str());
}

void MLP::Load(const std::string& path) {
  Tensor weights = LoadWeights(path);
  UpdateTopology(weights);
  mlp_->SetWeights(weights);
}

void MLP::UpdateTopology(Tensor& weights) {
  std::vector<std::size_t> layer_sizes;
  layer_sizes.push_back(weights[0][0].size());
  for (std::size_t i{0}; i < weights.size(); ++i) {
    layer_sizes.push_back(weights[i][1].size());
  }
  topology_ = Topology{layer_sizes};
  if (config_.GetModelType() == Config::ModelType::kMatrix) {
    mlp_ = std::make_unique<MatrixMlp>(topology_);
  } else if (config_.GetModelType() == Config::ModelType::kGraph) {
    mlp_ = std::make_unique<GraphMlp>(topology_);
  }
  metrics_ = Metrics{topology_.GetOutputSize()};
}

}  // namespace s21
