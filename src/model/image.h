#ifndef MLP_MODEL_IMAGE_H_
#define MLP_MODEL_IMAGE_H_

#include <algorithm>
#include <iostream>
#include <vector>

namespace s21 {

/**
 * @class Image
 * @brief Represents an image with pixel values and associated label.
 *
 * The Image class encapsulates an image along with its pixel values and a
 * label indicating its corresponding classification. It provides methods to
 * access and modify the label, pixel values, and perform normalization.
 */
class Image {
 public:
  using Pixels = std::vector<double>;

  static constexpr const double kMaxPixel = 255.0;
  static constexpr const std::size_t kHeight = 28;
  static constexpr const std::size_t kWidth = 28;
  static constexpr const std::size_t kPixels = kHeight * kWidth;

  Image() : label_{0u} { pixels_.reserve(kPixels); }
  explicit Image(const Pixels& pixels) : label_(0u), pixels_(pixels) {}
  Image(const Pixels& pixels, std::size_t label)
      : label_(label), pixels_(pixels) {
    Normalize();
  }

  std::size_t GetLabel() const { return label_; }
  void SetLabel(std::size_t label) { label_ = label; }
  const Pixels& GetPixels() const { return pixels_; }
  void AddPixel(double pixel) { pixels_.push_back(pixel); }
  char GetLetter() const { return static_cast<char>(label_) + 'A' - 1; }

  void Normalize() {
    std::transform(pixels_.begin(), pixels_.end(), pixels_.begin(),
                   [](double d) -> double { return d / kMaxPixel; });
  }

  void Transform() {
    Pixels pixels(kPixels);

    for (std::size_t i = 0; i < kWidth; ++i) {
      for (std::size_t j = 0; j < kHeight; ++j) {
        pixels[i * kHeight + j] = pixels_[(kHeight - j - 1) * kWidth + i];
      }
    }

    for (std::size_t i = 0; i < kHeight; ++i) {
      for (std::size_t j = 0; j < kWidth / 2; ++j) {
        std::swap(pixels[i * kWidth + j],
                  pixels[i * kWidth + (kWidth - j - 1)]);
      }
    }

    pixels_ = std::move(pixels);
  }

  void InverseTransform() {
    Pixels pixels(kPixels);

    for (std::size_t i = 0; i < kHeight; ++i) {
      for (std::size_t j = 0; j < kWidth / 2; ++j) {
        std::swap(pixels_[i * kWidth + j],
                  pixels_[i * kWidth + (kWidth - j - 1)]);
      }
    }

    for (std::size_t i = 0; i < kWidth; ++i) {
      for (std::size_t j = 0; j < kHeight; ++j) {
        pixels[i * kHeight + j] = pixels_[j * kWidth + (kWidth - i - 1)];
      }
    }

    pixels_ = std::move(pixels);
  }

  void PrintImage() const {
    std::string symbols = "# .";

    for (std::size_t row = 0; row < kWidth; ++row) {
      for (std::size_t col = 0; col < kHeight; ++col) {
        double pixel = pixels_[row * kWidth + col];
        std::size_t idx =
            static_cast<std::size_t>(pixel * (symbols.size() - 1));
        std::cout << symbols[idx];
      }
      std::cout << '\n';
    }
  }

 private:
  std::size_t label_;
  Pixels pixels_;
};

}  // namespace s21

#endif  // MLP_MODEL_IMAGE_H_
