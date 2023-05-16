#include "model.h"

namespace s21 {

void Model::SetConfiguration(const Config& config) {
  if (config.GetModelType() != config_.GetModelType() and mlp_) {
    auto weights = mlp_->GetWeights();
    mlp_ = std::make_unique<MLP>(config.GetModelType(), mlp_->GetSettings());
    mlp_->SetWeights(weights);
  }
  config_ = config;
}

void Model::SetWeights(
    const std::string& filename,
    std::function<void(Topology, std::size_t, std::size_t)> success_callback,
    std::function<void(const std::string&)> error_callback) {
  try {
    auto data = WeightReader::Read(filename);

    mlp_ = std::make_unique<MLP>(config_.GetModelType(), data.topology);
    mlp_->SetWeights(data.weights);

    if (success_callback)
      success_callback(mlp_->GetSettings(), data.epoch, data.accuracy);
  } catch (const std::runtime_error& e) {
    if (error_callback) error_callback(e.what());
  }
}

void Model::SetTrainDataset(
    const std::string& filename,
    std::function<void(std::string, std::size_t)> success_callback,
    std::function<void(const std::string&)> error_callback) {
  if (train_dataset_filename_ != filename) {
    std::thread([this, filename, success_callback, error_callback]() -> void {
      try {
        train_dataset_ = reader_->Read(filename);
        NormalizeData(&train_dataset_);
        train_dataset_filename_ = filename;

        if (success_callback) success_callback(filename, train_dataset_.size());
      } catch (const std::runtime_error& e) {
        if (error_callback) error_callback(e.what());
      }
    }).detach();
  } else {
    success_callback(filename, train_dataset_.size());
  }
}

void Model::SetTestDataset(
    const std::string& filename,
    std::function<void(std::string, std::size_t)> success_callback,
    std::function<void(const std::string&)> error_callback) {
  if (test_dataset_filename_ != filename) {
    std::thread([this, filename, success_callback, error_callback]() -> void {
      try {
        test_dataset_ = reader_->Read(filename);
        NormalizeData(&test_dataset_);
        test_dataset_filename_ = filename;

        if (success_callback) success_callback(filename, test_dataset_.size());
      } catch (const std::runtime_error& e) {
        if (error_callback) error_callback(e.what());
      }
    }).detach();
  } else {
    success_callback(filename, test_dataset_.size());
  }
}

void Model::Train(std::function<void()> start_callback,
                  std::function<void(std::size_t)> epoch_progress_callback,
                  std::function<void(std::size_t)> epoch_end_callback,
                  std::function<void()> test_start_callback,
                  std::function<void(std::size_t)> test_progress_callback,
                  std::function<void(Metrics, std::size_t)> test_end_callback,
                  std::function<void()> end_callback) {
  if (train_dataset_.size() == 0)
    throw std::runtime_error("тренировочный набор данных отсутствует");
  if (test_dataset_.size() == 0)
    throw std::runtime_error("тестовый набор данных отсутствует");

  train_exit_flag_.exchange(false);
  std::thread th([this, start_callback, epoch_progress_callback,
                  epoch_end_callback, test_start_callback,
                  test_progress_callback, test_end_callback,
                  end_callback]() -> void {
    Topology topology;
    topology.hidden_layers = config_.GetHiddenLayers();
    mlp_ = std::make_unique<MLP>(config_.GetModelType(), topology);

    mlp_->Train(
        train_dataset_, config_.GetEpochs(), config_.GetLearningRate(),
        start_callback, epoch_progress_callback,
        [this, test_start_callback, test_progress_callback, test_end_callback,
         epoch_end_callback](std::size_t epoch) -> void {
          if (train_exit_flag_ == false) {
            auto metrics = mlp_->Test(
                test_dataset_, 1, test_start_callback, test_progress_callback,
                [test_end_callback, epoch](Metrics metrics_) -> void {
                  if (test_end_callback) test_end_callback(metrics_, epoch);
                },
                train_exit_flag_);
            if (epoch_end_callback) epoch_end_callback(epoch);
          }
        },
        end_callback, train_exit_flag_);
  });
  th.detach();
}

void Model::StopTrain() { train_exit_flag_.exchange(true); }

void Model::Test(double part, std::function<void()> start_callback,
                 std::function<void(std::size_t)> progress_callback,
                 std::function<void(Metrics)> end_callback) {
  if (!mlp_) throw std::runtime_error("веса сети отсутствуют");
  if (test_dataset_.size() == 0)
    throw std::runtime_error("тестовый набор данных отсутствует");

  test_exit_flag_.exchange(false);
  std::thread th(
      [this, part, start_callback, progress_callback, end_callback]() -> void {
        mlp_->Test(test_dataset_, part, start_callback, progress_callback,
                   end_callback, test_exit_flag_);
      });
  th.detach();
}

void Model::StopTest() { test_exit_flag_.exchange(true); }

void Model::TrainCrossValidation(
    std::size_t epochs, std::size_t k, std::function<void()> start_callback,
    std::function<void(std::size_t)> progress_callback,
    std::function<void(Metrics)> end_callback) {
  if (train_dataset_.size() == 0)
    throw std::runtime_error("тренировочный набор данных отсутствует");
  // if (test_dataset_.size() == 0)
  //   throw std::runtime_error("тестовый набор данных отсутствует");

  test_exit_flag_.exchange(false);
  std::thread th([this, epochs, k, start_callback, progress_callback,
                  end_callback]() {
    if (start_callback) start_callback();

    std::unique_ptr<MLP> network_cv;
    std::unique_ptr<MLP> network_best;
    Metrics best_metrics;

    ModelType type = config_.GetModelType();
    Topology topology;
    topology.hidden_layers = config_.GetHiddenLayers();

    size_t block_size = train_dataset_.size() / k;
    // std::cout << block_size << std::endl;

    // std::size_t prev_progress = std::string::npos;
    for (std::size_t i = 0; i < k and !test_exit_flag_; i++) {
      std::list<Image> test_data;
      test_data.splice(test_data.begin(), train_dataset_,
                       train_dataset_.begin(),
                       std::next(train_dataset_.begin(), block_size));
      network_cv = std::make_unique<MLP>(type, topology);
      network_cv->Train(train_dataset_, epochs, config_.GetLearningRate(),
                        nullptr, progress_callback, nullptr, nullptr,
                        test_exit_flag_);
      Metrics metrics = network_cv->Test(
          test_data, 1, nullptr, progress_callback, nullptr, test_exit_flag_);

      if (metrics.f_measure > best_metrics.f_measure) {
        best_metrics = metrics;
        network_best = std::move(network_cv);
      }
      train_dataset_.splice(train_dataset_.end(), test_data);

      // std::size_t progress = static_cast<std::size_t>(
      //     static_cast<double>(i + 1) / static_cast<double>(k) * 100);
      // if (progress_callback and prev_progress != progress) {
      //   prev_progress = progress;
      //   progress_callback(progress);
      // }
    }

    mlp_ = std::move(network_best);

    if (end_callback) end_callback(best_metrics);
  });
  th.detach();
}

char Model::AnalyzeRawImage(const std::vector<double>& data) {
  char letter = 0;
  if (IsNetworkCreated()) {
    auto image = Image(data);
    image.NormalizeData();
    letter = static_cast<char>(Predict(image).first);
  }
  return static_cast<char>(letter) + 'A';
}

void Model::NormalizeData(std::list<Image>* images) {
  std::for_each(images->begin(), images->end(),
                [](Image& image) -> void { image.NormalizeData(); });
}

}  // namespace s21
