#include <chrono>
#include <iostream>

#include "io.h"

using namespace s21;

int main() {
  system("clear");
  std::cout << GetColor(Color::kMagenta) << Align("EMNIST PARSING TEST")
            << GetColor(Color::kEnd) << "\n\n";

  auto start = std::chrono::high_resolution_clock::now();

  Dataset dataset = ParseEmnist("../datasets/emnist-letters-train.csv");

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed Time : " << std::to_string(elapsed.count()) << " sec"
            << "\n";

  std::cout << "Labels of first 5 letters: ";
  for (std::size_t i = 0; i < 5; ++i) {
    std::cout << dataset[i].GetLabel() << " ";
  }

  std::cout << "\nSize of parsed pixels : " << dataset[0].GetPixels().size()
            << "\n\n";

  dataset[0].Transform();
  dataset[0].PrintImage();
  std::cout << "\tLetter 1: " << dataset[0].GetLetter() << "\n\n";

  std::cout << GetColor(Color::kMagenta) << Align(" ") << GetColor(Color::kEnd)
            << "\n\n";

  return 0;
}
