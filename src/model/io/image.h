#ifndef MLP_MODEL_IO_IMAGE_H_
#define MLP_MODEL_IO_IMAGE_H_

#include <algorithm>
#include <vector>

namespace s21 {

class Image {
 public:
  using Pixels = std::vector<double>;

  constexpr static const double kMaxPixel = 255.0;
  static const int kPixels = 784;

  Image() : label_(0) { pixels_.reserve(kPixels); }
  explicit Image(const Pixels& pixels) : label_(-1), pixels_(pixels) {}

  int GetLabel() const { return label_; }
  void SetLabel(int label) { label_ = label; }
  const Pixels& GetPixels() const { return pixels_; }
  void AddPixel(double pixel) { pixels_.push_back(pixel); }

  void Normalize() {
    std::transform(pixels_.begin(), pixels_.end(), pixels_.begin(),
                   [](double d) -> double { return d / kMaxPixel; });
  }

 private:
  int label_;
  Pixels pixels_;
};

}  // namespace s21

#endif  // MLP_MODEL_IO_IMAGE_H_
