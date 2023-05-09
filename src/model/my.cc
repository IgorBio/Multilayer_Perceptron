#include <iostream>

#include "parser/emnist_parser.h"

int main() {
  s21::EmnistParser parser;
  s21::EmnistParser::Dataset dataset = parser.ParseEmnist(
      "/home/igor/Projects/CPP7_MLP-0-master/src/resources/emnist-letters/"
      "emnist-letters-train.csv");
  std::cout << dataset[0].GetLabel() << std::endl;
  //   for (auto pixel : dataset[0].GetPixels()) {
  //     std::cout << pixel;
  //   }
  std::cout << dataset[0].GetPixels().size() << std::endl;
  return 0;
}