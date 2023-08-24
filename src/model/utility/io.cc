#include "io.h"

namespace s21 {

Dataset ParseEmnist(const std::string& path) {
  Dataset dataset;
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::string line;
  while (std::getline(file, line)) {
    Image image;
    std::istringstream iss(line);
    std::string token;
    std::getline(iss, token, ',');
    image.SetLabel(std::stoi(token));
    for (std::size_t i = 0; i < Image::kPixels; ++i) {
      std::getline(iss, token, ',');
      image.AddPixel(static_cast<double>(std::stoi(token)) / Image::kMaxPixel);
    }
    dataset.push_back(std::move(image));
  }

  file.close();

  return dataset;
}

std::string GetColor(Color color) {
  switch (color) {
    case Color::kRed:
      return "\u001b[41;1m";
    case Color::kGreen:
      return "\u001b[42;1m";
    case Color::kYellow:
      return "\u001b[43;1m";
    case Color::kBlue:
      return "\u001b[44;1m";
    case Color::kMagenta:
      return "\u001b[45;1m";
    case Color::kCyan:
      return "\u001b[46;1m";
    case Color::kGrey:
      return "\u001b[47;1m";
    case Color::kEnd:
      return "\u001b[0m";
    default:
      return "";
  }
}

std::string Align(const std::string& str) {
  std::ostringstream aligned;
  int padding = static_cast<int>(kStringWidth) - static_cast<int>(str.size());
  int left_padding = padding / 2;
  int right_padding = padding - left_padding;

  aligned << std::string(left_padding, ' ') << str
          << std::string(right_padding, ' ');
  return aligned.str();
}

}  // namespace s21
