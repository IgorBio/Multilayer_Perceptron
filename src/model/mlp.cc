#include "mlp.h"

namespace s21 {

MLP::MLP(const Topology& topology)
    : topology_{topology}, metrics_{topology_.GetOutputSize()} {
  mlp_ = std::make_unique<MatrixMlp>(topology_);
}

void MLP::Train() {
  if (train_.empty()) {
    throw std::runtime_error("Train dataset not loaded.");
  }

  switch (config_.GetTrainType()) {
    case Config::TrainType::kTrain:
      TrainEpochs();
      break;
    case Config::TrainType::kCrossValidation:
      CrossValidate();
      break;
    default:
      throw std::runtime_error("Invalid training type.");
  }
}

void MLP::TrainEpoch(const Dataset& train) {
  std::size_t percent = static_cast<std::size_t>(train.size() / 100.0);

  for (std::size_t i = 0; i < train.size(); ++i) {
    mlp_->SetInputLayer(train[i].GetPixels());
    mlp_->ForwardPropagation();
    const Vector expected_output = ExpectedOutput(train[i]);
    mlp_->BackPropagation(expected_output, config_.GetLearningRate());
    metrics_.AddLoss(mlp_->GetOutput(), expected_output);

    if (i % percent == 0) {
      ptr_progress_((i / percent) + 1);
    }
  }
}

void MLP::TrainEpochs() {
  double percent = static_cast<double>(100.0 / config_.GetEpochs());

  metrics_.StartMeasure(train_.size());
  for (std::size_t epoch = 0; epoch < config_.GetEpochs(); ++epoch) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(train_.begin(), train_.end(), gen);

    TrainEpoch(train_);

    if (config_.GetVerbose()) {
      metrics_.TrainReport(config_.GetEpochs(), epoch);
    }

    ptr_full_progress_((epoch * percent) + percent);
    ptr_metrics_(metrics_);
    metrics_.SetLoss(0);
  }
}

void MLP::Test(const Dataset& test) {
  std::vector<std::size_t> indices(test.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
  std::size_t test_size =
      static_cast<std::size_t>(test.size() * config_.GetTestSample());
  std::size_t percent = static_cast<std::size_t>(test_size / 100.0);

  metrics_.StartMeasure(test_size);
  for (std::size_t i = 0; i < test_size; ++i) {
    const Image& image = test[indices[i]];
    mlp_->SetInputLayer(image.GetPixels());
    mlp_->ForwardPropagation();

    metrics_.AddLoss(mlp_->GetOutput(), ExpectedOutput(image));
    metrics_.AddPrediction(PredictLabel(image), image.GetLabel());

    if (i % percent == 0) {
      ptr_progress_((i / percent) + 1);
    }
  }
  metrics_.StopMeasure();
  if (config_.GetVerbose()) {
    metrics_.TestReport();
  }
  ptr_metrics_(metrics_);
}

void MLP::Test() {
  if (test_.empty()) {
    throw std::runtime_error("Test dataset not loaded.");
  }

  Test(test_);
}

void MLP::CrossValidate() {
  std::vector<Dataset> folds(config_.GetKFolds());
  std::vector<std::size_t> indices(train_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
  double percent = static_cast<double>(100.0 / config_.GetKFolds());

  for (std::size_t i = 0; i < indices.size(); ++i) {
    folds[i % config_.GetKFolds()].push_back(train_[indices[i]]);
  }

  for (std::size_t fold = 0; fold < config_.GetKFolds(); ++fold) {
    Dataset validation = folds[fold];
    Dataset train;
    for (std::size_t i = 0; i < folds.size(); ++i) {
      if (i != fold) {
        train.insert(train.end(), folds[i].begin(), folds[i].end());
      }
    }

    std::shuffle(train.begin(), train.end(), std::default_random_engine());
    metrics_.StartMeasure(train.size());

    TrainEpoch(train);

    if (config_.GetVerbose()) {
      metrics_.TrainReport(config_.GetKFolds(), fold);
    }

    Test(validation);

    ptr_full_progress_((fold * percent) + percent);
  }
}

Vector MLP::ExpectedOutput(const Image& image) {
  Vector expected_output(topology_.GetOutputSize(), 0.0);
  expected_output[image.GetLabel() - 1] = 1.0;
  return expected_output;
}

Vector MLP::Predict(const Vector& input) {
  mlp_->SetInputLayer(input);
  mlp_->ForwardPropagation();
  return mlp_->GetOutput();
}

char MLP::Predict(const Image& image) {
  return static_cast<char>(PredictLabel(image) + 'A' - 1);
}

std::size_t MLP::PredictLabel(const Image& image) {
  Vector input = image.GetPixels();
  Vector predicted = Predict(input);
  auto it = std::max_element(predicted.begin(), predicted.end());
  return std::distance(predicted.begin(), it) + 1;
}

void MLP::SetType(Config::ModelType type) {
  config_.SetModelType(type);
  if (type == Config::ModelType::kMatrix) {
    mlp_ = std::make_unique<MatrixMlp>(topology_);
  } else if (type == Config::ModelType::kGraph) {
    mlp_ = std::make_unique<GraphMlp>(topology_);
  }
}

void MLP::Save(const std::string& path) {
  std::stringstream ss(path);
  std::ofstream file(ss.str(), std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + ss.str());
  }

  const auto& [weights, biases] = mlp_->GetMlp();

  // Write the number of layers
  std::size_t num_layers = weights.size();
  file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

  for (std::size_t i = 0; i < num_layers; ++i) {
    const Matrix& layer_weights = weights[i];
    const Matrix& layer_biases = biases[i];

    // Write the dimensions of the weight matrix
    std::size_t rows = layer_weights.size();
    std::size_t cols = layer_weights[0].size();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Write the weight matrix
    for (const Vector& row : layer_weights) {
      file.write(reinterpret_cast<const char*>(row.data()),
                 sizeof(double) * cols);
    }

    // Write the bias vector
    for (const Vector& bias : layer_biases) {
      file.write(reinterpret_cast<const char*>(bias.data()),
                 sizeof(double) * bias.size());
    }
  }
}

void MLP::Load(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  // Read the number of layers
  std::size_t num_layers;
  file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

  // Read each layer's weights and biases
  Tensor weights(num_layers);
  Tensor biases(num_layers);
  for (std::size_t i = 0; i < num_layers; ++i) {
    // Read the dimensions of the weight matrix
    std::size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Read the weight matrix
    Matrix layer_weights(rows, Vector(cols));
    for (Vector& row : layer_weights) {
      file.read(reinterpret_cast<char*>(row.data()), sizeof(double) * cols);
    }
    weights[i] = std::move(layer_weights);

    // Read the bias matrix
    Matrix layer_biases(1, Vector(cols));
    file.read(reinterpret_cast<char*>(layer_biases[0].data()),
              sizeof(double) * cols);
    biases[i] = std::move(layer_biases);
  }

  UpdateMlp(weights, biases);
}

void MLP::UpdateMlp(const Tensor& weights, const Tensor& biases) {
  std::vector<std::size_t> layer_sizes;
  layer_sizes.push_back(weights[0].size());

  for (const auto& layer : weights) {
    layer_sizes.push_back(layer[0].size());
  }

  topology_.SetTopology(layer_sizes);
  SetType(config_.GetModelType());
  mlp_->SetMlp(weights, biases);
  metrics_ = Metrics{topology_.GetOutputSize()};
}

void MLP::UpdateTopology(std::size_t hidden, std::size_t size) {
  std::vector<std::size_t> layer_sizes;
  layer_sizes.push_back(topology_.GetInputSize());

  for (std::size_t i = 0; i < hidden; ++i) {
    layer_sizes.push_back(size);
  }

  layer_sizes.push_back(topology_.GetOutputSize());
  topology_.SetTopology(layer_sizes);
  SetType(config_.GetModelType());
  metrics_ = Metrics{topology_.GetOutputSize()};
}

}  // namespace s21
