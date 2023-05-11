#include "emnist_parser.h"

namespace s21 {
EmnistParser::Dataset EmnistParser::ParseEmnist(const std::string& path) {
  Dataset dataset;
  std::ifstream emnist(path);
  if (emnist.is_open()) {
    std::string line;
    while (std::getline(emnist, line)) {
      Image image;
      std::size_t idx = 0u;
      image.SetLabel(ParseLabel(line, idx, &idx));
      while (idx < line.length()) {
        image.AddPixel(ParsePixel(line, idx, &idx));
      }
      dataset.push_back(image);
    }
    emnist.close();
  } else {
    throw std::runtime_error("Incorrect path to emnist");
  }
  return dataset;
}

int EmnistParser::ParseLabel(const std::string& line, std::size_t in,
                             std::size_t* out) {
  std::size_t pos = 0u;
  int label;
  try {
    label = std::stoi(line.substr(in), &pos);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error("Incorrect emnist format");
  }
  *out = in + pos + 1;
  return label;
}

double EmnistParser::ParsePixel(const std::string& line, std::size_t in,
                                std::size_t* out) {
  std::size_t pos = 0u;
  double pixel;
  try {
    pixel = std::stod(line.substr(in), &pos);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error("Incorrect emnist format");
  }
  *out = in + pos + 1;
  return pixel;
}
}  // namespace s21
