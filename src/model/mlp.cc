#include "mlp.h"

namespace s21 {

void MLP::Train(const std::list<Image>& data, std::size_t epochs,
                double learning_rate, std::function<void()> start_callback,
                std::function<void(std::size_t)> epoch_progress_callback,
                std::function<void(std::size_t)> epoch_end_callback,
                std::function<void()> end_callback,
                const std::atomic_bool& exit) {
  if (epochs != 0) {
    if (start_callback) start_callback();

    for (std::size_t epoch = 0; epoch < epochs and !exit; epoch++) {
      TrainEpoch(data, learning_rate, epoch_progress_callback, exit);
      if (epoch == 2 or epoch == 3 or epoch == 4) learning_rate /= 2;
      if (epoch_end_callback) epoch_end_callback(epoch + 1);
    }
  }
  if (end_callback and !exit) end_callback();
}

void MLP::TrainEpoch(const std::list<Image>& data, double learning_rate,
                     std::function<void(std::size_t)> epoch_progress_callback,
                     const std::atomic_bool& exit) {
  std::size_t count = 1;
  std::size_t data_size = data.size();

  std::size_t prev_progress = std::string::npos;
  for (const Image& image : data) {
    if (exit) break;

    mlp_->SetInput(image.GetPixels());
    mlp_->ForwardPropagation();
    mlp_->BackPropagation(ExpectedOutput(image), learning_rate);

    std::size_t progress = static_cast<std::size_t>(
        static_cast<double>(count) / static_cast<double>(data_size) * 100);
    if (epoch_progress_callback and prev_progress != progress) {
      prev_progress = progress;
      epoch_progress_callback(progress);
    }
    count++;
  }
}

Metrics MLP::Test(const std::list<Image>& data, double part,
                  std::function<void()> start_callback,
                  std::function<void(std::size_t)> progress_callback,
                  std::function<void(Metrics)> end_callback,
                  const std::atomic_bool& exit) {
  if (start_callback) start_callback();

  Metrics metrics;

  std::size_t true_pos = 0;
  std::size_t true_neg = 0;
  std::size_t false_pos = 0;
  std::size_t false_neg = 0;

  const double kActivationThreshold = 0.5;

  auto timestamp = std::chrono::high_resolution_clock::now();

  std::size_t count = 0;
  std::size_t partition = (std::size_t)(part * (double)data.size());
  for (const Image& image : data) {
    if (exit) break;

    std::pair<int, double> res = Predict(image);
    if (res.first == image.GetLabel() - 1) {
      if (res.second > kActivationThreshold)
        true_pos++;
      else
        true_neg++;
    } else {
      if (res.second > kActivationThreshold)
        false_pos++;
      else
        false_neg++;
    }
    count++;
    if (progress_callback)
      progress_callback(static_cast<std::size_t>(
          static_cast<double>(count) / static_cast<double>(partition) * 100));
    if (count > partition) break;
  }

  metrics.accuracy = (double)(true_pos + true_neg) /
                     (double)(false_pos + false_neg + true_pos + true_neg);
  metrics.precision = (double)true_pos / (double)(true_pos + false_pos);
  metrics.recall = (double)true_pos / (double)(true_pos + false_neg);
  metrics.f_measure = 2 * (metrics.precision * metrics.recall /
                           (metrics.precision + metrics.recall));

  metrics.time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::high_resolution_clock::now() - timestamp)
                     .count();

  if (end_callback) end_callback(metrics);

  return metrics;
}

std::vector<double> MLP::ExpectedOutput(const Image& image) {
  std::vector<double> expected_output(26, 0);
  expected_output[image.GetNumber() - 1] = 1;
  return expected_output;
}

std::vector<double> MLP::Prediction(const Image& image) {
  mlp_->SetInput(image.GetData());
  mlp_->ForwardPropagation();
  return mlp_->GetOutput();
}

std::pair<std::size_t, double> MLP::Predict(const Image& image) {
  std::vector<double> output = Prediction(image);
  auto it = std::max_element(output.begin(), output.end());
  std::size_t max_ind = std::distance(output.begin(), it);
  return {max_ind, *it};
}

void MLP::SetWeights(const std::vector<double>& weights) {
  mlp_->LoadWeights(weights);
}

std::vector<double> MLP::GetWeights() const { return mlp_->GetWeights(); }

}  // namespace s21
