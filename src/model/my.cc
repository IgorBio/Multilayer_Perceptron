#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// #include "io/emnist_parser.h"
#include "utility/matrix_operations.h"

s21::Matrix RandomMatrixStandart(int rows, int cols);
void PrintMatrix(s21::Matrix m);

s21::Matrix RandomMatrixStandart(int rows, int cols) {
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

  // s21::Matrix m1 = s21::Matrix(1000, s21::Vector(1000));
  // s21::Matrix m2 = s21::Matrix(1000, s21::Vector(1000));
  // s21::Randomize(m1);
  // s21::Randomize(m2);

  // s21::Matrix m1 = {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  // s21::Matrix m2 = {{-1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  // s21::Matrix m2 = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {5, 6, 7}, {6, 7, 8}};
  double d = 5;
  s21::Matrix m1 = RandomMatrixStandart(1000, 1000);
  s21::Matrix m2 = RandomMatrixStandart(1000, 1000);

  // s21::Matrix res = s21::Matrix(10, s21::Vector(10));

  auto start = std::chrono::steady_clock::now();

  // s21::Randomize(res);
  // s21::Matrix res = s21::Subtraction(m1, m2);
  // s21::Matrix res = s21::Transpose(m1);
  // s21::Matrix res = s21::MultiplyNumber(m1, d);
  // s21::Matrix res = s21::MultiplyHadamard(m1, m2);
  // s21::Matrix res = s21::MultiplyWinograd(m1, m2);
  s21::Matrix res = s21::Activate(m1, s21::ActivationFunction::kSigmoid);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed Time : " << std::to_string(elapsed.count()) << " sec"
            << std::endl;

  // s21::Matrix res = RandomMatrixStandart(1000, 1000);
  // for (size_t i = 0; i < res.size(); ++i) {
  //   for (size_t j = 0; j < res[0].size(); ++j) {
  //     for (size_t k = 0; k < m2.size(); k++) {
  //       res[i][j] += m1[i][k] * m2[k][j];
  //     }
  //   }
  // }
  // PrintMatrix(m1);
  // PrintMatrix(m2);
  // PrintMatrix(res);
  return 0;
}