#include "emnist_parser.h"

namespace s21 {
EmnistParser::Dataset EmnistParser::ParseEmnist(const std::string& path) {
  Dataset dataset;
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }
  std::string line, pixel_str;
  while (std::getline(file, line)) {
    Image image;
    std::istringstream iss(line);
    std::string token;
    std::getline(iss, token, ',');
    image.SetLabel(std::stoi(token));
    for (std::size_t i{0u}; i < Image::kPixels; ++i) {
      std::getline(iss, token, ',');
      image.AddPixel(static_cast<double>(std::stoi(token)) / Image::kMaxPixel);
    }
    dataset.push_back(image);
  }
  return dataset;
}

}  // namespace s21
