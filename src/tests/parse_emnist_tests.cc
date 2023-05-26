#include <chrono>
#include <iostream>

#include "../model/io/emnist_parser.h"

constexpr size_t kWidth = 60u;

enum class Color { kRed, kGreen, kBlue, kYellow, kGrey, kCyan, kMagenta, kEnd };
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

std::string Align(const std::string &str) {
  std::string aligned;
  std::string addition((kWidth - str.size()) / 2, ' ');
  aligned.append(addition);
  aligned.append(str);
  aligned.append(addition);
  while (aligned.size() < kWidth) aligned.append(" ");
  return aligned;
}

int main() {
  s21::EmnistParser parser;

  std::cout << "\n"
            << GetColor(Color::kMagenta) << Align("EMNIST PARSING TEST")
            << GetColor(Color::kEnd) << "\n\n";

  auto start = std::chrono::steady_clock::now();

  s21::EmnistParser::Dataset dataset = parser.ParseEmnist(
      "../datasets/emnist-letters/"
      "emnist-letters-train.csv");

  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "\tElapsed Time : " << std::to_string(elapsed.count()) << " sec"
            << "\n";

  std::cout << "\tLabels of first 5 letters: ";
  for (std::size_t i{0u}; i < 5; ++i) {
    std::cout << dataset[i].GetLabel() << " ";
  }
  std::cout << "\n";
  // for (auto pixel : dataset[0].GetPixels()) {
  //   std::cout << pixel << " ";
  // }
  // std::cout << std::endl;
  std::cout << "\tSize of parsed pixels : " << dataset[0].GetPixels().size()
            << "\n\n";

  std::cout << GetColor(Color::kMagenta) << Align(" ") << GetColor(Color::kEnd)
            << "\n\n";
  return 0;
}