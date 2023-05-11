#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// #include "io/emnist_parser.h"
#include "utility/winograd.h"

s21::Matrix RandomMatrix(int rows, int cols);
void PrintMatrix(s21::Matrix m);
s21::Matrix RandomMatrix(int rows, int cols) {
  s21::Matrix matrix(rows, s21::Vector(cols));
  std::random_device random_device;
  std::mt19937 random_generator{random_device()};
  std::uniform_real_distribution<double> distribution{1, 9};
  for (int i{0}; i < rows; ++i) {
    for (int j{0}; j < cols; ++j) {
      matrix[i][j] = distribution(random_generator);
    }
  }
  return matrix;
}

void PrintMatrix(s21::Matrix m) {
  for (std::size_t i = 0; i < m.size(); i++) {
    for (std::size_t j = 0; j < m[i].size(); j++) {
      std::cout << m[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  // s21::EmnistParser parser;
  // s21::EmnistParser::Dataset dataset = parser.ParseEmnist(
  //     "/home/igor/Projects/CPP7_MLP-0-master/src/resources/emnist-letters/"
  //     "emnist-letters-train.csv");
  // std::cout << dataset[0].GetLabel() << std::endl;
  //   for (auto pixel : dataset[0].GetPixels()) {
  //     std::cout << pixel;
  //   }
  // std::cout << dataset[0].GetPixels().size() << std::endl;

  // s21::Matrix m1 = {{2, 3, 3, 10}, {2, 3, 4, 20}};
  // s21::Matrix m2 = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {10, 20, 30,
  // 40}};
  s21::Matrix m1 = RandomMatrix(784, 784);
  s21::Matrix m2 = RandomMatrix(784, 784);
  auto start = std::chrono::high_resolution_clock::now();
  s21::Matrix res = s21::MultiplyMaxThreads(m1, m2);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Elapsed Time : " << elapsed.count() << " ms" << std::endl;

  // PrintMatrix(m1);
  // PrintMatrix(m2);
  // PrintMatrix(res);
  return 0;
}