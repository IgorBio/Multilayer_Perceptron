#ifndef MLP_MODEL_IO_IMAGE_H_
#define MLP_MODEL_IO_IMAGE_H_

#include <algorithm>
#include <vector>

namespace s21 {

class Image {
 public:
  using Pixels = std::vector<double>;

  static constexpr const double kMaxPixel = 255.0;
  static constexpr const int kPixels = 784;

  Image() : label_{0u} { pixels_.reserve(kPixels); }
  explicit Image(const Pixels& pixels) : label_(0u), pixels_(pixels) {}
  Image(const Pixels& pixels, std::size_t label)
      : label_(label), pixels_(pixels) {}

  std::size_t GetLabel() const { return label_; }
  void SetLabel(std::size_t label) { label_ = label; }
  const Pixels& GetPixels() const { return pixels_; }
  void AddPixel(double pixel) { pixels_.push_back(pixel); }
  char GetLetter() { return static_cast<char>(label_) - 1 + 'A'; }

  void Normalize() {
    std::transform(pixels_.begin(), pixels_.end(), pixels_.begin(),
                   [](double d) -> double { return d / kMaxPixel; });
  }

 private:
  std::size_t label_;
  Pixels pixels_;
};

}  // namespace s21

#endif  // MLP_MODEL_IO_IMAGE_H_
