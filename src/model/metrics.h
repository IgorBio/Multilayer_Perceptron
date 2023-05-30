#ifndef MLP_MODEL_METRICS_H_
#define MLP_MODEL_METRICS_H_

#include <numeric>
#include <vector>

namespace s21 {

class Metrics {
 public:
  Metrics(std::size_t num_classes)
      : tp_(num_classes, 0u),
        fp_(num_classes, 0u),
        tn_(num_classes, 0u),
        fn_(num_classes, 0u),
        loss_(0.0),
        time_{0} {}

  void AddTruePositive(std::size_t label) { tp_[label]++; }
  void AddFalsePositive(std::size_t label) { fp_[label]++; }
  void AddTrueNegative(std::size_t label) { tn_[label]++; }
  void AddFalseNegative(std::size_t label) { fn_[label]++; }

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

  double GetPrecision() {
    double total_precision = 0.0;
    for (std::size_t i{0u}; i < tp_.size(); ++i) {
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

  double GetRecall() {
    double total_recall = 0.0;
    for (std::size_t i{0u}; i < tp_.size(); ++i) {
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

  double GetF1Score() {
    double total_f1score = 0.0;
    for (std::size_t i{0u}; i < tp_.size(); ++i) {
      total_f1score += F1Score(i);
    }
    return total_f1score / tp_.size();
  }

  double GetLoss() const { return loss_; }
  void SetLoss(double loss) { loss_ = loss; }

  long long GetTime() const { return time_; }
  void SetTime(long long time) { time_ += time; }

 private:
  std::vector<std::size_t> tp_, fp_, tn_, fn_;
  double loss_;
  long long time_;
};

}  // namespace s21

#endif  // MLP_MODEL_METRICS_H_
