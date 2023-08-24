#ifndef MLP_MODEL_METRICS_H_
#define MLP_MODEL_METRICS_H_

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "abstract_mlp.h"

namespace s21 {

/**
 * @class Metrics
 * @brief Class for calculating various evaluation metrics and tracking
 * performance.
 *
 * The Metrics class provides methods to calculate evaluation metrics such as
 * accuracy, precision, recall, and F1-score for multi-class classification.
 * It also allows tracking loss and time for evaluation.
 */
class Metrics {
 public:
  explicit Metrics(std::size_t num_classes)
      : tp_(num_classes, 0),
        fp_(num_classes, 0),
        tn_(num_classes, 0),
        fn_(num_classes, 0),
        loss_(0.0),
        time_(0),
        size_(1) {}

  void AddTruePositive(std::size_t label) { ++tp_[label - 1]; }
  void AddFalsePositive(std::size_t label) { ++fp_[label - 1]; }
  void AddTrueNegative(std::size_t label) { ++tn_[label - 1]; }
  void AddFalseNegative(std::size_t label) { ++fn_[label - 1]; }

  double GetAccuracy() const {
    double total_tp = std::accumulate(tp_.begin(), tp_.end(), 0);
    double total_fp = std::accumulate(fp_.begin(), fp_.end(), 0);
    double total_tn = std::accumulate(tn_.begin(), tn_.end(), 0);
    double total_fn = std::accumulate(fn_.begin(), fn_.end(), 0);
    return (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn);
  }

  double Precision(std::size_t label) const {
    if (tp_[label] == 0 and fp_[label] == 0) return 0.0;
    double tp = static_cast<double>(tp_[label]);
    double fp = static_cast<double>(fp_[label]);
    return tp / (tp + fp);
  }

  double GetPrecision() const {
    double total_precision = 0.0;
    for (std::size_t i = 0; i < tp_.size(); ++i) {
      total_precision += Precision(i);
    }
    return total_precision / tp_.size();
  }

  double Recall(std::size_t label) const {
    if (tp_[label] == 0 and fn_[label] == 0) return 0.0;
    double tp = static_cast<double>(tp_[label]);
    double fn = static_cast<double>(fn_[label]);
    return tp / (tp + fn);
  }

  double GetRecall() const {
    double total_recall = 0.0;
    for (std::size_t i = 0; i < tp_.size(); ++i) {
      total_recall += Recall(i);
    }
    return total_recall / tp_.size();
  }

  double F1Score(std::size_t label) const {
    double precision = Precision(label);
    double recall = Recall(label);
    if (std::fabs(precision + recall) < 1e-7) return 0.0;
    return 2 * precision * recall / (precision + recall);
  }

  double GetF1Score() const {
    double total_f1score = 0.0;
    for (std::size_t i = 0; i < tp_.size(); ++i) {
      total_f1score += F1Score(i);
    }
    return total_f1score / tp_.size();
  }

  static double GetMSE(const Vector& predict, const Vector& expect) {
    double loss = 0.0;
    for (std::size_t i = 0; i < predict.size(); ++i) {
      double diff = expect[i] - predict[i];
      loss += diff * diff;
    }
    return loss;
  }

  double GetLoss() const { return loss_ / size_; }
  void SetLoss(double loss) { loss_ = loss; }
  void AddLoss(const Vector& predict, const Vector& expect) {
    loss_ += GetMSE(predict, expect);
  }

  long long GetTotalTime() const { return time_; }
  void SetTime(long long time) { time_ += time; }

  void AddPrediction(std::size_t pred, std::size_t real) {
    if (pred == real) {
      ++tp_[real - 1];
    } else {
      ++fp_[pred - 1];
      ++fn_[real - 1];
    }
  }

  void StartMeasure(std::size_t size) {
    Clear();
    size_ = size;
    StartTimer();
  }

  void Clear() {
    std::fill(tp_.begin(), tp_.end(), 0.0);
    std::fill(fp_.begin(), fp_.end(), 0.0);
    std::fill(tn_.begin(), tn_.end(), 0.0);
    std::fill(fn_.begin(), fn_.end(), 0.0);
    loss_ = 0.0;
    time_ = 0;
    size_ = 1;
  }

  void StartTimer() { start_time_ = std::chrono::high_resolution_clock::now(); }

  void StopMeasure() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time_);
    time_ += duration.count();
  }

  void TestReport() const {
    std::cout << "Test on " << size_ << " images\n";
    std::cout << "\tLoss: " << GetLoss() << std::endl;
    std::cout << "\tAccuracy: " << GetAccuracy() << std::endl;
    std::cout << "\tPrecision: " << GetPrecision() << std::endl;
    std::cout << "\tRecall: " << GetRecall() << std::endl;
    std::cout << "\tF1 Score: " << GetF1Score() << std::endl;
    std::cout << "\tTotal time: " << GetTotalTime() << " seconds\n";
  }

  void TrainReport(std::size_t epochs, std::size_t epoch) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto epoch_time =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_)
            .count();
    double average_epoch_time =
        static_cast<double>(epoch_time) / static_cast<double>(epoch + 1);
    auto remaining_time =
        static_cast<long long>((epochs - epoch - 1) * average_epoch_time);
    std::cout << "\nEpoch: " << epoch + 1 << std::endl;
    std::cout << "\nTime Elapsed: " << epoch_time << " seconds\n";
    std::cout << "Time Remaining: " << remaining_time << " seconds\n";
    std::cout << "Loss: " << GetLoss() << "\n\n";
  }

 private:
  std::vector<std::size_t> tp_, fp_, tn_, fn_;
  double loss_;
  long long time_;
  std::size_t size_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

}  // namespace s21

#endif  // MLP_MODEL_METRICS_H_
